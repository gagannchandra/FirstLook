"""E5 leave-one-hospital-out + E6 conformal under shift and the local-adaptation curve.

For each held-out site H and tier (T1, T2), using the selected model family and its
tuned hyper-parameters (configs/selection.json, logs/best_*.json):
  1. train on the two other sites (S2 train), early stopping on S2 val;
  2. fit calibrator + Mondrian conformal (alpha) on S2 cal;
  3. target pool = all rows of H outside the S1 test set (DECISIONS D11);
  4. n = 0: evaluate on the whole pool (no adaptation);
  5. for n in N_LOCAL, 20 random draws: take n local visits, evaluate on the rest:
       (i)   none           source calibrator + source conformal
       (ii)  recal          local Platt recalibration + source conformal thresholds
       (iii) recal+conf     local Platt + local Mondrian conformal
       (iv)  retrain (ref)  model refit on source train + n local rows (3 draws, n in RETRAIN_N)
The model itself is never retrained in arms (i)-(iii).

Usage: python -m src.experiments.loho
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling import calibrate, conformal
from src.modelling.evaluate import calibration_slope_intercept, ece, fast_auc
from src.data.splits import loho
from src.modelling.train import fit_model, load_tier, predict

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
PRED = ROOT / "data" / "preds"
SITES = ["A", "B", "C"]
N_LOCAL = [100, 250, 500, 1000, 2000, 5000]
N_DRAWS = 20
RETRAIN_N = [500, 2000, 5000]
RETRAIN_DRAWS = 3
ALPHA = 0.10
SEED = 42


def metrics(y, p, cfg) -> dict:
    slope, citl = calibration_slope_intercept(y, p)
    c = conformal.evaluate(p, y, cfg)
    return {"auc": fast_auc(y, p), "ece": ece(y, p), "cal_slope": slope, "citl": citl,
            "coverage": c["coverage"], "coverage_admit": c["coverage_admit"],
            "coverage_discharge": c["coverage_discharge"], "abstention": c["abstention"],
            "prevalence": float(np.mean(y)), "mean_p": float(np.mean(p))}


def run_tier(tier: str, sel: dict, X: pd.DataFrame, meta: pd.DataFrame) -> list[dict]:
    model_name = sel["model"]
    params = json.loads((LOGS / f"best_{tier}_{model_name}.json").read_text())["best_params"]
    y = meta.y.to_numpy()
    out = []
    for H in SITES:
        t0 = time.time()
        sp = loho(meta, H)
        m, info = fit_model(model_name, params, X.iloc[sp["train"]], y[sp["train"]], X.iloc[sp["val"]], y[sp["val"]])
        raw_cal = predict(m, X.iloc[sp["cal"]])
        raw_pool = predict(m, X.iloc[sp["test"]])
        y_pool = y[sp["test"]]
        src_cal = calibrate.make(sel["calibrator"]).fit(raw_cal, y[sp["cal"]])
        src_cfg = conformal.fit(src_cal.predict(raw_cal), y[sp["cal"]], ALPHA)
        # save pool predictions for figures / E11 (S2)
        pd.DataFrame({"row": sp["test"], "raw": raw_pool.astype(np.float32),
                      "p_source": src_cal.predict(raw_pool).astype(np.float32)}) \
            .to_parquet(PRED / f"loho_{tier}_{H}.parquet", index=False)
        # in-distribution reference on source val
        raw_val = predict(m, X.iloc[sp["val"]])
        ref = metrics(y[sp["val"]], src_cal.predict(raw_val), src_cfg)
        out.append({"tier": tier, "site": H, "arm": "source_val (in-distribution)", "n": 0, "draw": 0, **ref})
        base = metrics(y_pool, src_cal.predict(raw_pool), src_cfg)
        out.append({"tier": tier, "site": H, "arm": "none", "n": 0, "draw": 0, **base})
        rng = np.random.default_rng(SEED)
        for n in N_LOCAL:
            for d in range(N_DRAWS):
                perm = rng.permutation(len(y_pool))
                loc, ev = perm[:n], perm[n:]
                yl, ye = y_pool[loc], y_pool[ev]
                p_src_ev = src_cal.predict(raw_pool[ev])
                out.append({"tier": tier, "site": H, "arm": "none", "n": n, "draw": d, **metrics(ye, p_src_ev, src_cfg)})
                loc_cal = calibrate.Platt().fit(raw_pool[loc], yl)
                p_loc_ev = loc_cal.predict(raw_pool[ev])
                out.append({"tier": tier, "site": H, "arm": "recal", "n": n, "draw": d, **metrics(ye, p_loc_ev, src_cfg)})
                loc_cfg = conformal.fit(loc_cal.predict(raw_pool[loc]), yl, ALPHA)
                out.append({"tier": tier, "site": H, "arm": "recal+conformal", "n": n, "draw": d,
                            **metrics(ye, p_loc_ev, loc_cfg)})
        # (iv) retrain reference: fixed number of trees from the source fit (no early stopping on H)
        fixed = dict(params)
        it = info.get("best_iteration")
        if it:
            fixed["n_estimators"] = fixed["iterations"] = max(int(it), 10)
        rng2 = np.random.default_rng(SEED + 1)
        pool_rows = sp["test"]
        for n in RETRAIN_N:
            for d in range(RETRAIN_DRAWS):
                perm = rng2.permutation(len(pool_rows))
                loc, ev = perm[:n], perm[n:]
                tr_rows = np.concatenate([sp["train"], pool_rows[loc]])
                m2, _ = fit_model(model_name, fixed, X.iloc[tr_rows], y[tr_rows], None, None)
                p_cal2 = predict(m2, X.iloc[sp["cal"]])
                c2 = calibrate.make(sel["calibrator"]).fit(p_cal2, y[sp["cal"]])
                cfg2 = conformal.fit(c2.predict(p_cal2), y[sp["cal"]], ALPHA)
                p_ev = c2.predict(predict(m2, X.iloc[pool_rows[ev]]))
                out.append({"tier": tier, "site": H, "arm": "retrain (ref)", "n": n, "draw": d,
                            **metrics(y[pool_rows[ev]], p_ev, cfg2)})
        print(f"[{tier}] held-out {H}: AUC {base['auc']:.3f} ECE {base['ece']:.3f} cov {base['coverage']:.3f} "
              f"(in-dist AUC {ref['auc']:.3f}) {time.time()-t0:.0f}s", flush=True)
    return out


def main() -> None:
    sel = json.loads((ROOT / "configs" / "selection.json").read_text())["tiers"]
    rows = []
    for tier in ("t1", "t2"):
        X, meta = load_tier(tier)
        rows += run_tier(tier, sel[tier], X, meta)
        del X
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "E6_local_adaptation_raw.csv", index=False)
    print("saved", len(df), "rows")


if __name__ == "__main__":
    main()
