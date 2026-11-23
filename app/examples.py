"""Synthetic example patients for the live demo.

These are invented by hand (guide 04 §8). They are NOT rows from the dataset.
tests/test_app.py checks that each example lands in the state its button promises
under the shipped model bundle.
"""
from __future__ import annotations

EXAMPLES = {
    "high": {
        "label": "High risk",
        "hint": "82 y, ambulance, breathless, SpO₂ 88%",
        "form": {"age": 82, "gender": "Female", "arrivalmode": "ambulance", "arrivalhour_bin": "11-14",
                 "triage_vital_hr": 118, "triage_vital_sbp": 96, "triage_vital_dbp": 58, "triage_vital_rr": 28,
                 "triage_vital_o2": 88, "triage_vital_o2_device": 1, "triage_vital_temp": 100.8,
                 "complaints": ["cc_shortnessofbreath"]},
    },
    "low": {
        "label": "Low risk",
        "hint": "24 y, walk-in, sore throat, normal vitals",
        "form": {"age": 24, "gender": "Male", "arrivalmode": "Walk-in", "arrivalhour_bin": "15-18",
                 "triage_vital_hr": 78, "triage_vital_sbp": 122, "triage_vital_dbp": 76, "triage_vital_rr": 16,
                 "triage_vital_o2": 99, "triage_vital_o2_device": 0, "triage_vital_temp": 98.6,
                 "complaints": ["cc_sorethroat"]},
    },
    "uncertain": {
        "label": "Uncertain",
        "hint": "Mid-risk profile the model cannot call",
        "form": None,  # filled by scripts/find_uncertain_example.py output (synthetic search), see below
    },
    "missing_spo2": {
        "label": "SpO₂ not measured",
        "hint": "67 y, chest pain, no pulse oximeter",
        "form": {"age": 67, "gender": "Male", "arrivalmode": "Car", "arrivalhour_bin": "07-10",
                 "triage_vital_hr": 96, "triage_vital_sbp": 152, "triage_vital_dbp": 88, "triage_vital_rr": 20,
                 "triage_vital_o2": None, "triage_vital_o2_device": None, "triage_vital_temp": 98.2,
                 "complaints": ["cc_chestpain"]},
    },
}

# A hand-written mid-risk profile. Its exact values were adjusted by hand (synthetic,
# never copied from data) until the shipped T1 bundle returns "uncertain";
# tests/test_app.py guards this.
EXAMPLES["uncertain"]["form"] = {
    "age": 58, "gender": "Female", "arrivalmode": "Car", "arrivalhour_bin": "19-22",
    "triage_vital_hr": 98, "triage_vital_sbp": 138, "triage_vital_dbp": 82, "triage_vital_rr": 18,
    "triage_vital_o2": 97, "triage_vital_o2_device": 0, "triage_vital_temp": 99.1,
    "complaints": ["cc_abdominalpain"],
}
