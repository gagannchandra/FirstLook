"""Export everything the offline web app (frontend/) needs as static JSON.

Usage: python -m backend.export
Writes
  frontend/data/models.json     T1/T2 logistic-regression bundles (run in the browser), examples, validation
                                ranges, random-patient profiles, and the T3-lite history form (served by the API)
  frontend/data/dashboard.json  research results for the Evidence page (copy of results/dashboard/dashboard.json)
The browser model computes exactly the same function as the ONNX graph
(impute -> standardise -> linear -> sigmoid -> calibrator -> conformal); tests/test_pwa.py checks parity.
"""
import json
from pathlib import Path

from app.examples import EXAMPLES  # synthetic patients only
from app.random_patient import HOUR_BINS as RP_HOURS, MISSABLE, PROFILES
from app.validation import RANGES, SOFT
from src.data.features import ARRIVAL_LEVELS, HOUR_BINS, VITALS
from src.edge.runtime import complaint_label

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "frontend" / "data" / "models.json"
DASH_IN = ROOT / "results" / "dashboard" / "dashboard.json"
DASH_OUT = ROOT / "frontend" / "data" / "dashboard.json"

HIST_LABEL = {"n_edvisits": "ED visits in the past year", "n_admissions": "Hospital admissions in the past year",
              "n_surgeries": "Past surgeries / procedures", "asthma": "Asthma", "hyperlipidem": "Hyperlipidaemia",
              "alcoholrelateddisorders": "Alcohol-related disorder", "neutrophils_median": "Neutrophils (median, past year)",
              "rdw_last": "RDW (most recent, past year)", "meds_elect/caloric/h2o": "Electrolytes / caloric / water"}
PREV_DISPO = ["No previous dispo", "Discharge", "Admit", "Transfer to Another Facility", "LWBS after Triage", "AMA",
              "Eloped", "LWBS before Triage", "Observation", "Send to L&D"]


def cc_labels(labels: dict) -> dict:
    return {k: complaint_label(k) for k in labels}


def hist_label(h: str) -> str:
    if h in HIST_LABEL:
        return HIST_LABEL[h]
    if h.startswith("meds_"):
        return h[5:].replace("drugs", " drugs").replace("/", " / ").capitalize()
    return h.replace("_", " ").capitalize()


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
            out["complaint_labels"] = cc_labels(spec["complaint_labels"])
            out["vital_ranges"] = spec["vital_ranges"]
    out["examples"] = {k: {"label": v["label"], "form": v["form"]} for k, v in EXAMPLES.items()}
    out["ranges"] = {k: list(v) for k, v in RANGES.items()}
    out["soft"] = {k: list(v) for k, v in SOFT.items()}
    # random-patient profiles: one source of truth (app/random_patient.py), consumed by model.js randomPatient()
    out["random"] = {"profiles": PROFILES, "hour_bins": RP_HOURS, "missable": MISSABLE}
    # T3-lite (tree ensemble) runs only in the local backend; the browser needs its form and headline metrics
    d3 = ROOT / "models" / "bundles" / "t3lite_v1"
    if d3.exists():
        s3 = json.loads((d3 / "feature_spec.json").read_text())
        kinds = s3.get("history_input_kind", {})
        out["history"] = {"slot": "t3lite", "name": s3["slot_name"], "prev_dispo": PREV_DISPO if s3.get("uses_previousdispo") else [],
                          "fields": [{"key": h, "label": hist_label(h), "kind": kinds.get(h, "count"),
                                      "group": "key" if h in ("n_edvisits", "n_admissions", "n_surgeries") else kinds.get(h, "count")}
                                     for h in s3["history_inputs"]],
                          "metrics": json.loads((d3 / "metrics.json").read_text()).get("test"),
                          "features": len(s3["model_features"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if DASH_IN.exists():
        DASH_OUT.write_text(json.dumps(json.loads(DASH_IN.read_text())))
        print(f"wrote {DASH_OUT} ({DASH_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
