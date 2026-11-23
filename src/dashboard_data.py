"""Aggregate numbers for the app's research dashboard (no row-level data).

Usage:
  python -m src.dashboard_data --split val                      # development
  FIRSTLOOK_OPEN_TEST=1 python -m src.dashboard_data --split test   # Phase 5
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from src import conformal
from src.common import PROC, ROOT, calibrator, load_model, meta, rows_for, selection
from src.evaluate import bootstrap_ci, reliability

OUT = ROOT / "results" / "dashboard" / "dashboard.json"
TABLES = ROOT / "results" / "tables"
NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    sel = selection()["tiers"]
    d = {"eval_label": "S1 test set (frozen, opened once)" if a.split == "test" else "S1 validation (development — test set still sealed)",
         "metrics": [], "calibration": {}, "risk_coverage": {}}
    for t in ("t1", "t2", "t3"):
        if t not in sel:
            continue
        X = pd.read_parquet(PROC / f"{t}.parquet")
        rows = rows_for(a.split)
        p = calibrator(t).predict(load_model(t, sel[t]["model"]).predict_proba(X.iloc[rows])[:, 1])
        y = meta().y.to_numpy()[rows]
        ci = bootstrap_ci(y, p, n_boot=a.n_boot)
        d["metrics"].append({"tier": t, "model": NAMES[sel[t]["model"]], **{k: list(v) for k, v in ci.items()}})
        mp, fy, _ = reliability(y, p)
        d["calibration"][t] = {"pred": mp.round(4).tolist(), "obs": fy.round(4).tolist()}
        cov, err, aurc = conformal.risk_coverage(p, y)
        k = np.linspace(0, len(cov) - 1, 120).astype(int)
        d["risk_coverage"][t] = {"coverage": cov[k].round(4).tolist(), "error": err[k].round(4).tolist(), "aurc": aurc}
        del X
    dc = pd.read_csv(TABLES / f"E12_decision_curve_{a.split}.csv")
    d["decision_curve"] = {c: dc[c].round(5).tolist() for c in dc.columns}
    la_f = TABLES / "E6_local_adaptation_raw.csv"
    if la_f.exists():
        la = pd.read_csv(la_f)
        la = la[(la.tier == "t1") & la.arm.isin(["recal+conformal", "none"])]
        sites = {}
        for s in ("A", "B", "C"):
            n0 = la[(la.site == s) & (la.arm == "none") & (la.n == 0)]
            g = la[(la.site == s) & (la.arm == "recal+conformal")].groupby("n").coverage_admit.agg(["mean", "std"])
            sites[s] = {"n": [50] + g.index.tolist(), "coverage_admit": [float(n0.coverage_admit.iloc[0])] + g["mean"].round(4).tolist(),
                        "coverage_admit_sd": [0.0] + g["std"].round(4).tolist()}
        d["local_adaptation"] = {"sites": sites, "caption": "Mean ± SD over 20 random local samples; the model is never retrained. "
                                 "First point (plotted at 50) = no adaptation (source-hospital conformal thresholds)."}
    bench = []
    for f in sorted(glob.glob(str(ROOT / "results" / "bench" / "bench_*.json"))):
        b = json.load(open(f))
        for slot, v in b["bundles"].items():
            bench.append({"setting": b["setting"], "model": slot, "size_kb": v["model_onnx_kb"], "p50_ms": v["e2e_p50_ms"],
                          "p95_ms": v["e2e_p95_ms"], "peak_rss_mb": b["peak_rss_mb"], "net_bytes": b["net_bytes_sent_non_loopback"]})
    d["benchmark"] = bench
    d["footnote"] = ("All numbers are computed by the FirstLook pipeline from the Yale ED dataset (Hong et al., 2018); "
                     "95% CIs from 1,000 bootstrap resamples. Local adaptation uses S2 (leave-one-hospital-out). "
                     "Latency is end-to-end (form → result incl. explanation).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d))
    print("wrote", OUT, "metrics:", [(m["tier"], round(m["auc"][0], 4)) for m in d["metrics"]])


if __name__ == "__main__":
    main()
