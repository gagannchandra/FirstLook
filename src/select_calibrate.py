"""Phase 2 wrap-up: E2 validation table, E4 calibration, model selection, thresholds, E3 (val).

Pre-declared rules (DECISIONS.md D18), all on S1 validation / S3 calibration only:
  * calibrators (Platt, isotonic) are fit on S3 'cal' rows, compared on 'val' rows;
  * best model per tier = lowest val log-loss after its best calibrator;
  * calibrator per model = lowest val ECE (Platt wins ties within 0.001);
  * t_low = largest threshold with NPV >= 0.95 on val; t_high = smallest with PPV >= 0.80.

Usage: python -m src.select_calibrate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import calibrate, viz
from src.evaluate import core_metrics, delong_test, ece, paired_bootstrap_delta_auc, reliability

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "preds"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
SEL = ROOT / "configs" / "selection.json"
TIERS = ["t1", "t2", "t3"]
MODELS = ["lr", "rf", "xgb", "lgbm", "cat"]


def load_preds(tier, model, y):
    f = PRED / f"{tier}_{model}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["y"] = y[d.row.to_numpy()]
    return d


def choose_thresholds(y, p, npv_target=0.95, ppv_target=0.80):
    order = np.argsort(p); ps = p[order]; ys = y[order]
    # t_low: predicted-negative set = p < t; NPV = share of y=0 among them
    cum_neg = np.cumsum(1 - ys); cnt = np.arange(1, len(ys) + 1)
    npv = cum_neg / cnt
    ok = np.flatnonzero(npv >= npv_target)
    t_low = float(ps[ok.max() + 1]) if len(ok) and ok.max() + 1 < len(ps) else float(ps[0])
    # t_high: predicted-positive set = p >= t; PPV among them
    rev_pos = np.cumsum(ys[::-1])[::-1]; rev_cnt = cnt[::-1]
    ppv = rev_pos / rev_cnt
    ok2 = np.flatnonzero(ppv >= ppv_target)
    t_high = float(ps[ok2.min()]) if len(ok2) else 1.0
    return t_low, t_high


def main() -> None:
    viz.setup()
    meta = pd.read_parquet(ROOT / "data" / "processed" / "meta.parquet", columns=["y"])
    y = meta.y.to_numpy()
    rows, cal_rows = [], []
    calibrated = {}
    for t in TIERS:
        for m in MODELS:
            d = load_preds(t, m, y)
            if d is None:
                continue
            va, ca = d[d.split == "val"], d[d.split == "cal"]
            best_info = json.loads((LOGS / f"best_{t}_{m}.json").read_text())
            raw = core_metrics(va.y, va.p)
            rows.append({"tier": t, "model": m, **{k: v for k, v in raw.items()},
                         "n_trials": best_info["n_trials"], "tune_rows": best_info["tune_rows"]})
            best = None
            for kind in ("none", "platt", "isotonic"):
                c = calibrate.make(kind).fit(ca.p.to_numpy(), ca.y.to_numpy())
                pv = c.predict(va.p.to_numpy())
                met = core_metrics(va.y, pv)
                cal_rows.append({"tier": t, "model": m, "calibrator": kind, "ece": met["ece"], "brier": met["brier"],
                                 "logloss": met["logloss"], "cal_slope": met["cal_slope"], "citl": met["citl"]})
                if kind == "none":
                    continue
                if best is None or met["ece"] < best[1]["ece"] - 0.001:
                    best = (kind, met, c, pv)
            calibrated[(t, m)] = best
    e2 = pd.DataFrame(rows)
    e2.to_csv(TABLES / "E2_val_raw.csv", index=False)
    e4 = pd.DataFrame(cal_rows)
    e4.to_csv(TABLES / "E4_calibration_val.csv", index=False)

    selection = {"rule": "lowest val log-loss after calibration; calibrator by val ECE; thresholds NPV>=0.95 / PPV>=0.80 on val",
                 "date_frozen": None, "tiers": {}}
    summary = []
    for t in TIERS:
        cands = [(m, calibrated[(t, m)]) for m in MODELS if (t, m) in calibrated]
        if not cands:
            continue
        m_best, (kind, met, cal, pv) = min(cands, key=lambda x: x[1][1]["logloss"])
        yv = load_preds(t, m_best, y).query("split=='val'").y.to_numpy()
        t_low, t_high = choose_thresholds(yv, pv)
        selection["tiers"][t] = {"model": m_best, "calibrator": kind, "calibrator_params": cal.to_dict(),
                                 "t_low": t_low, "t_high": t_high, "val_auc": met["auc"], "val_ece": met["ece"],
                                 "val_logloss": met["logloss"]}
        for m, (k, mt, _, _) in cands:
            summary.append({"tier": t, "model": m, "calibrator": k, "val_auc": mt["auc"], "val_prauc": mt["prauc"],
                            "val_brier": mt["brier"], "val_ece": mt["ece"], "val_logloss": mt["logloss"],
                            "selected": m == m_best})
    SEL.write_text(json.dumps(selection, indent=1))
    s = pd.DataFrame(summary)
    s.to_csv(TABLES / "E2_val_calibrated.csv", index=False)
    (TABLES / "E2_val_calibrated.md").write_text(s.round(4).to_markdown(index=False))
    print(s.round(4).to_string(index=False))
    print(json.dumps({t: {k: v for k, v in d.items() if k != "calibrator_params"} for t, d in selection["tiers"].items()}, indent=1))

    # E3 on validation (preliminary; final E3 uses the test set in Phase 5)
    e3 = []
    pv = {}
    for t in selection["tiers"]:
        sel = selection["tiers"][t]
        d = load_preds(t, sel["model"], y).query("split=='val'").sort_values("row")
        pv[t] = (d.row.to_numpy(), d.y.to_numpy(), calibrate.from_dict(sel["calibrator_params"]).predict(d.p.to_numpy()))
    for a, b in [("t1", "t2"), ("t2", "t3"), ("t1", "t3")]:
        if a in pv and b in pv:
            assert (pv[a][0] == pv[b][0]).all()
            yy = pv[a][1]
            auc_a, auc_b, p = delong_test(yy, pv[a][2], pv[b][2])
            d, lo, hi = paired_bootstrap_delta_auc(yy, pv[a][2], pv[b][2], n_boot=1000)
            e3.append({"comparison": f"{a} -> {b}", "auc_a": auc_a, "auc_b": auc_b, "delta_auc": d,
                       "ci_low": lo, "ci_high": hi, "delong_p": p})
    e3 = pd.DataFrame(e3)
    e3.to_csv(TABLES / "E3_value_of_information_val.csv", index=False)
    print(e3.round(4).to_string(index=False))

    # Reliability diagrams (val): best model per tier, before vs after calibration
    fig, axes = viz.plt.subplots(1, len(pv), figsize=(4 * len(pv), 3.8), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), pv):
        sel = selection["tiers"][t]
        d = load_preds(t, sel["model"], y).query("split=='val'").sort_values("row")
        ax.plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
        for lab, pp, col in [("Before", d.p.to_numpy(), viz.NEUTRAL), (f"After ({sel['calibrator']})", pv[t][2], viz.TIER_COLOR[t])]:
            mp, fy, _ = reliability(d.y.to_numpy(), pp)
            ax.plot(mp, fy, marker="o", ms=4, color=col, label=f"{lab}  ECE {ece(d.y.to_numpy(), pp):.3f}")
        ax.set_title(f"{viz.TIER_NAME[t]} · {viz.MODEL_NAME[sel['model']]}")
        ax.set_xlabel("Predicted probability"); ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Observed admission rate")
    fig.suptitle("Reliability on validation (S1): calibrators fit on S3", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, "fig04_reliability_val.png")


if __name__ == "__main__":
    main()
