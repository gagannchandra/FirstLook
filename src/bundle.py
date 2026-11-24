"""Build deployable model bundles (guide 04 §7) from a chosen variant.

Calibrator, conformal thresholds and risk-band thresholds are re-fit for the exported
model: calibrator + conformal on S3 'cal', thresholds on S1 'val'. No test rows are used.

Usage: python -m src.bundle            (reads configs/edge_selection.json)
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import calibrate, conformal
from src.common import PROC, ROOT, meta, rows_for
from src.compress import onnx_export, onnx_session, onnx_proba
from src.evaluate import core_metrics
from src.features import VITAL_RANGES, safe_name
from src.runtime import complaint_label
from src.select_calibrate import choose_thresholds

BUNDLES = ROOT / "models" / "bundles"
EDGE_SEL = ROOT / "configs" / "edge_selection.json"
SLOT_NAME = {"t1": "Nurse-only (T1)", "t2": "Nurse + ESI (T2)", "t3lite": "Nurse + ESI + history (T3-lite)"}


def explain_payload(model, out: Path) -> str:
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(out / "explain_lgbm.txt"))
        return "lgbm_treeshap"
    if hasattr(model, "named_steps"):
        imp, sc, lr = model.named_steps["impute"], model.named_steps["scale"], model.named_steps["lr"]
        d = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
             "intercept": float(lr.intercept_[0]), "median": imp.statistics_.tolist(),
             "indicator_features": imp.indicator_.features_.tolist() if imp.indicator_ is not None else []}
        (out / "explain.json").write_text(json.dumps(d))
        return "linear_exact"
    return "none"


def build(slot: str, variant_file: str, base_tier: str, version: str) -> dict:
    model, cols = joblib.load(ROOT / variant_file)
    out = BUNDLES / f"{slot}_{version}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    md = meta(); y = md.y.to_numpy()
    X = pd.read_parquet(PROC / f"{base_tier}.parquet", columns=cols)
    va, ca, tr = rows_for("val"), rows_for("cal"), rows_for("train")

    onx = onnx_export(model, len(cols))
    (out / "model.onnx").write_bytes(onx.SerializeToString())
    sess = onnx_session(onx)
    raw_va = onnx_proba(sess, X.iloc[va].to_numpy(np.float32))
    raw_ca = onnx_proba(sess, X.iloc[ca].to_numpy(np.float32))
    parity = float(np.max(np.abs(raw_va - model.predict_proba(X.iloc[va])[:, 1])))
    assert parity < 1e-4, f"ONNX parity failed: {parity}"

    best = None
    for kind in ("platt", "isotonic"):
        c = calibrate.make(kind).fit(raw_ca, y[ca])
        m = core_metrics(y[va], c.predict(raw_va))
        if best is None or m["ece"] < best[1]["ece"] - 0.001:
            best = (c, m)
    cal, val_metrics = best
    p_ca, p_va = cal.predict(raw_ca), cal.predict(raw_va)
    conf = conformal.fit(p_ca, y[ca], 0.10)
    conf["alternatives"] = {str(a): conformal.fit(p_ca, y[ca], a) for a in (0.05, 0.20)}
    t_low, t_high = choose_thresholds(y[va], p_va)

    base_spec = json.loads((ROOT / "configs" / "feature_spec.json").read_text())
    Xtr = pd.read_parquet(PROC / f"{base_tier}.parquet", columns=cols).iloc[tr]
    cc_freq = Xtr[[c for c in cols if c.startswith("cc_")]].sum() if any(c.startswith("cc_") for c in cols) else pd.Series(dtype=float)
    all_cc = base_spec["cc_columns"]
    raw_by_safe = {safe_name(h): h for h in base_spec["history_columns"]}
    hist_safe = [c for c in cols if c in raw_by_safe]          # model feature names
    hist_used = [raw_by_safe[c] for c in hist_safe]             # raw names = app form keys
    hist_default_zero = {raw_by_safe[c]: bool(not Xtr[c].isna().any() and np.nanmin(Xtr[c].to_numpy()) == 0) for c in hist_safe}
    spec = {
        "version": version, "app_slot": slot, "slot_name": SLOT_NAME[slot], "base_tier": base_tier,
        "model_features": cols, "cc_columns": all_cc, "history_columns": base_spec["history_columns"] if base_tier == "t3" else [],
        "history_inputs": hist_used, "history_default_zero": hist_default_zero,
        "history_input_medians": {raw_by_safe[c]: float(np.nanmedian(Xtr[c])) for c in hist_safe},
        "history_input_kind": {raw_by_safe[c]: ("lab" if Xtr[c].isna().any() else
                                                "binary" if set(np.unique(Xtr[c].dropna())) <= {0.0, 1.0} else "count")
                                for c in hist_safe},
        "uses_previousdispo": any(c.startswith("prevdispo=") for c in cols),
        "complaints_in_model": [c for c in cols if c.startswith("cc_")],
        "complaint_labels": {c: complaint_label(c) for c in all_cc},
        "complaint_order": sorted(all_cc, key=lambda c: -float(cc_freq.get(c, 0))),
        "levels": base_spec["levels"], "vital_ranges": {k: list(v) for k, v in VITAL_RANGES.items()},
        "age_range": [18, 110], "temperature_unit": "F", "spo2_100_maps_to": 99,
        "training_medians": {c: float(np.nanmedian(Xtr[c])) for c in cols if Xtr[c].isna().any()},
    }
    (out / "feature_spec.json").write_text(json.dumps(spec, indent=1))
    (out / "calibrator.json").write_text(json.dumps(cal.to_dict()))
    (out / "conformal.json").write_text(json.dumps(conf, indent=1))
    (out / "thresholds.json").write_text(json.dumps({"t_low": t_low, "t_high": t_high,
                                                      "rule": "t_low: NPV>=0.95, t_high: PPV>=0.80 on S1 validation"}, indent=1))
    ce = conformal.evaluate(p_va, y[va], conf)
    metrics = {"validation": {k: float(v) for k, v in val_metrics.items()},
               "validation_conformal_alpha_0.10": ce, "onnx_parity_max_abs_diff": parity,
               "test": None, "note": "test metrics are added once, in Phase 5"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    expl = explain_payload(model, out)
    write_card(out, spec, metrics, cal.kind, expl, len(cols))
    print(f"bundle {out.name}: {len(cols)} features, onnx {(out/'model.onnx').stat().st_size/1024:.1f} KB, "
          f"val AUC {val_metrics['auc']:.4f} ECE {val_metrics['ece']:.4f}, t_low {t_low:.3f} t_high {t_high:.3f}, "
          f"conformal val cov {ce['coverage']:.3f} abst {ce['abstention']:.3f}")
    return {"slot": slot, "path": str(out.relative_to(ROOT))}


def write_card(out: Path, spec: dict, metrics: dict, cal_kind: str, expl: str, nfeat: int) -> None:
    v = metrics["validation"]
    t = metrics.get("test")
    perf = (f"Test (S1, frozen): AUC {t['auc']:.3f} (95% CI {t['auc_ci'][0]:.3f}–{t['auc_ci'][1]:.3f}), "
            f"ECE {t['ece']:.3f}" if t else "Test metrics: added in Phase 5 (test set sealed during development).")
    card = f"""# Model card — FirstLook {spec['slot_name']} · {spec['version']}

| Field | Content |
|---|---|
| Model | {spec['slot_name']}, {nfeat} input features, exported to ONNX; calibrator: {cal_kind}; explanation: {expl} |
| Intended use | Research demonstration of admission-risk estimation at ED triage |
| Not intended for | Clinical decisions, diagnosis, triage-level assignment, use on real patients |
| Training data | Yale ED admission dataset (Hong et al., 2018), 2014–2017, 3 US EDs, S1 training split |
| Inputs | {', '.join(sorted(set(f.split('=')[0] for f in spec['model_features'] if not f.startswith('cc_'))))}; chief complaint flags: {len(spec['complaints_in_model'])} |
| Output | Calibrated admission probability, Low/Medium/High band, conformal set (alpha = 0.10), local explanation |
| Validation performance | AUC {v['auc']:.3f}, PR-AUC {v['prauc']:.3f}, Brier {v['brier']:.3f}, ECE {v['ece']:.3f} (S1 validation) |
| Test performance | {perf} |
| Transfer | See results/tables/E5_loho.md — calibration and coverage shift at an unseen hospital |
| Known failure modes | Unseen hospital types (needs local recalibration), missing vitals, rare complaints, SpO₂ 100 encoded as 99 |
| Date / author | {dt.date.today().isoformat()} / Gagan Chandra (built with FirstLook pipeline) |

Decision support only. Not a diagnosis. Clinical judgement decides.
"""
    (out / "model_card.md").write_text(card)


def main() -> None:
    sel = json.loads(EDGE_SEL.read_text())
    out = [build(s["slot"], s["variant_file"], s["base_tier"], s["version"]) for s in sel["bundles"]]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
