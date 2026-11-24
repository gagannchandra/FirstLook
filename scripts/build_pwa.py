"""Export the shipped T1/T2 logistic-regression bundles to JSON for the offline PWA (L3).

Usage: python scripts/build_pwa.py
Writes app/pwa/models.json. The PWA computes exactly the same function as the ONNX graph
(impute -> standardise -> linear -> sigmoid -> calibrator -> conformal); tests/test_pwa.py
checks parity against the Python runtime.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import ARRIVAL_LEVELS, HOUR_BINS, VITALS  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from examples import EXAMPLES  # noqa: E402  (synthetic patients only)
from validation import RANGES, SOFT  # noqa: E402
from random_patient import HOUR_BINS as RP_HOURS, MISSABLE, PROFILES  # noqa: E402

OUT = ROOT / "app" / "pwa" / "models.json"


def main():
    out = {"version": None, "arrival_levels": ARRIVAL_LEVELS, "hour_bins": HOUR_BINS, "vitals": VITALS, "models": {}}
    for slot in ("t1", "t2"):
        d = ROOT / "models" / "bundles" / f"{slot}_v1"
        spec = json.loads((d / "feature_spec.json").read_text())
        ex = json.loads((d / "explain.json").read_text())
        out["version"] = spec["version"]
        out["models"][slot] = {
            "name": spec["slot_name"], "features": spec["model_features"], "explain": ex,
            "calibrator": json.loads((d / "calibrator.json").read_text()),
            "conformal": {k: v for k, v in json.loads((d / "conformal.json").read_text()).items() if k != "alternatives"},
            "thresholds": json.loads((d / "thresholds.json").read_text()),
            "metrics": json.loads((d / "metrics.json").read_text()).get("test"),
        }
        if slot == "t1":
            out["complaint_order"] = spec["complaint_order"]
            out["complaint_labels"] = spec["complaint_labels"]
            out["vital_ranges"] = spec["vital_ranges"]
    out["examples"] = {k: {"label": v["label"], "form": v["form"]} for k, v in EXAMPLES.items()}
    out["ranges"] = {k: list(v) for k, v in RANGES.items()}
    out["soft"] = {k: list(v) for k, v in SOFT.items()}
    # random-patient profiles: one source of truth (app/random_patient.py), consumed by model.js randomPatient()
    out["random"] = {"profiles": PROFILES, "hour_bins": RP_HOURS, "missable": MISSABLE}
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
