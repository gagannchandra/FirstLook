"""Apply the pre-declared edge-model rule (DECISIONS D19) to the E8 results.

Rule: among exported artifacts (V3 ONNX, V4 LR — the things that actually ship), pick the
smallest whose validation AUC is >= V0 AUC - 0.01 for that tier. If none qualifies, take the
highest-AUC export and record that the rule failed. Also builds the app's T2 bundle with the
same recipe as the chosen T1 edge model plus ESI.

Writes configs/edge_selection.json. Usage: python -m src.select_edge
"""
from __future__ import annotations

import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd

from src.common import PROC, ROOT, meta, rows_for
from src.evaluate import fast_auc
from src.train import fit_model

TABLES = ROOT / "results" / "tables"
ART = ROOT / "models" / "variants"
TOL = 0.01


def pick(d: pd.DataFrame) -> tuple[pd.Series, bool]:
    v0 = d[d.variant == "V0 full"].val_auc.iloc[0]
    ships = d[d.variant.str.startswith(("V3", "V4"))].copy()
    ok = ships[ships.val_auc >= v0 - TOL]
    if len(ok):
        return ok.sort_values(["size_kb", "val_auc"], ascending=[True, False]).iloc[0], True
    return ships.sort_values("val_auc", ascending=False).iloc[0], False


def variant_file(tier: str, r: pd.Series) -> str:
    if r.variant.startswith("V3"):
        return f"models/variants/{tier}_V2_k{int(r.k)}_d{int(r.depth)}_t{int(r.trees)}.joblib"
    return f"models/variants/{tier}_V4_k{int(r.k)}.joblib"


def main():
    e8 = pd.read_csv(TABLES / "E8_compression_val.csv")
    out = {"rule": f"smallest exported artifact with val AUC >= V0 - {TOL}", "date": dt.date.today().isoformat(),
           "bundles": [], "table_E": [], "choices": {}}
    for tier, slot in (("t1", "t1"), ("t3", "t3lite")):
        d = e8[e8.tier == tier]
        r, ok = pick(d)
        v0 = d[d.variant == "V0 full"].iloc[0]
        out["choices"][slot] = {"variant": r.variant, "k": int(r.k), "size_kb": float(r.size_kb), "val_auc": float(r.val_auc),
                                "v0_val_auc": float(v0.val_auc), "v0_size_kb": float(v0.size_kb), "rule_met": ok,
                                "size_ratio_vs_v0": float(v0.size_kb / r.size_kb)}
        out["bundles"].append({"slot": slot, "variant_file": variant_file(tier, r), "base_tier": tier, "version": "v1"})
        # Table E rows: V0, best V1 at chosen k, chosen V2/V3, V4 at chosen k
        k = int(r.k)
        _, cols = joblib.load(ROOT / variant_file(tier, r))
        out["table_E"].append({"tier": tier, "name": "V0 full", "base_tier": tier, "file": "V0", "features": None})
        out["table_E"].append({"tier": tier, "name": f"V1 top-{k}", "base_tier": tier,
                               "file": f"models/variants/{tier}_V1_k{k}.joblib", "features": cols})
        if r.variant.startswith("V3"):
            out["table_E"].append({"tier": tier, "name": f"V2 shallow d{int(r.depth)} t{int(r.trees)} top-{k}", "base_tier": tier,
                                   "file": variant_file(tier, r), "features": cols})
            out["table_E"].append({"tier": tier, "name": f"V3 ONNX d{int(r.depth)} t{int(r.trees)} top-{k}", "base_tier": tier,
                                   "file": f"models/variants/{tier}_V3_k{k}_d{int(r.depth)}_t{int(r.trees)}.onnx", "features": cols})
        out["table_E"].append({"tier": tier, "name": f"V4 LR top-{k}", "base_tier": tier,
                               "file": f"models/variants/{tier}_V4_k{k}.joblib", "features": cols})
        print(slot, out["choices"][slot])

    # T2 app model: the chosen T1 recipe + ESI (same features + 'esi', same model type/hyper-parameters)
    c = out["choices"]["t1"]
    m1, cols1 = joblib.load(ROOT / out["bundles"][0]["variant_file"])
    cols2 = list(cols1) + (["esi"] if "esi" not in cols1 else [])
    X = pd.read_parquet(PROC / "t2.parquet", columns=cols2)
    y = meta().y.to_numpy(); tr, va = rows_for("train"), rows_for("val")
    if c["variant"].startswith("V3"):
        r = e8[(e8.tier == "t1") & (e8.variant == c["variant"])].iloc[0]
        sp = {"learning_rate": 0.1, "num_leaves": 2 ** int(r.depth), "max_depth": int(r.depth), "min_child_samples": 100,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_estimators": int(r.trees)}
        m2, _ = fit_model("lgbm", sp, X.iloc[tr], y[tr], None, None)
    else:
        from src.train import make_lr
        m2 = make_lr({"C": 1.0}).fit(X.iloc[tr], y[tr])
    auc2 = fast_auc(y[va], m2.predict_proba(X.iloc[va])[:, 1])
    f2 = "models/variants/t2_edge.joblib"
    joblib.dump((m2, cols2), ROOT / f2)
    out["bundles"].insert(1, {"slot": "t2", "variant_file": f2, "base_tier": "t2", "version": "v1"})
    out["choices"]["t2"] = {"variant": f"T1 edge recipe + ESI ({len(cols2)} features)", "val_auc": float(auc2)}
    print("t2", out["choices"]["t2"])
    (ROOT / "configs" / "edge_selection.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
