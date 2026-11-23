"""E1: baseline reproduction (LR + XGBoost, fixed settings) on T2 and T3, S1 validation.

Sanity check against Hong et al. (2018): XGB triage-only ~0.87, full ~0.92 (their test set).
Usage: python -m src.e1_baseline
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluate import core_metrics
from src.train import LOGS, ROOT, fit_model, load_tier, predict
from src.splits import s1

ROWS = []


def main() -> None:
    for tier in ("t2", "t3"):
        X, meta = load_tier(tier)
        idx = s1(meta); y = meta.y.to_numpy()
        tr, va = idx["train"], idx["val"]
        xgb_params = {"learning_rate": 0.05, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0,
                      "min_child_weight": 1, "reg_lambda": 1.0, "n_estimators": 500}
        for name, params in [("lr", {"C": 1.0}), ("xgb", xgb_params)]:
            m, info = fit_model(name, params, X.iloc[tr], y[tr], X.iloc[va], y[va])
            p = predict(m, X.iloc[va])
            met = core_metrics(y[va], p)
            ROWS.append({"tier": tier, "model": name, **{k: round(v, 4) for k, v in met.items()}, **info})
            print(ROWS[-1], flush=True)
        del X
    df = pd.DataFrame(ROWS)
    LOGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "tables" / "E1_baseline_val.csv", index=False)
    (ROOT / "results" / "tables" / "E1_baseline_val.md").write_text(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
