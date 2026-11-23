"""Input validation for the patient form (pure functions, unit-tested)."""
from __future__ import annotations

# Plausible ranges = the original dataset's cleaning limits (DECISIONS D16).
RANGES = {
    "age": (18, 110, "Age", "years"),
    "triage_vital_hr": (30, 299, "Heart rate", "bpm"),
    "triage_vital_sbp": (30, 399, "Systolic BP", "mmHg"),
    "triage_vital_dbp": (25, 299, "Diastolic BP", "mmHg"),
    "triage_vital_rr": (8, 69, "Respiratory rate", "breaths/min"),
    "triage_vital_o2": (60, 100, "SpO₂", "%"),
    "triage_vital_temp": (90.0, 109.9, "Temperature", "°F"),
}
SOFT = {  # unusual but possible: warn, don't block
    "triage_vital_hr": (40, 180), "triage_vital_sbp": (70, 230), "triage_vital_rr": (10, 40),
    "triage_vital_o2": (80, 100), "triage_vital_temp": (95.0, 104.0),
}


def c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def validate(form: dict) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors block prediction; warnings are shown only."""
    errors, warnings = [], []
    age = form.get("age")
    if age is None:
        errors.append("Please enter the patient's age.")
    for key, (lo, hi, name, unit) in RANGES.items():
        v = form.get(key)
        if v is None:
            continue
        if not (lo <= v <= hi):
            if key == "age":
                errors.append(f"Age must be between {lo} and {hi} — this tool was built on adult visits only.")
            else:
                errors.append(f"{name} {v:g} {unit} is outside the plausible range ({lo:g}–{hi:g} {unit}). "
                              "Please re-check the reading, or mark it as not measured.")
        elif key in SOFT and not (SOFT[key][0] <= v <= SOFT[key][1]):
            warnings.append(f"{name} {v:g} {unit} is unusual — please double-check it.")
    sbp, dbp = form.get("triage_vital_sbp"), form.get("triage_vital_dbp")
    if sbp is not None and dbp is not None and dbp >= sbp:
        errors.append(f"Diastolic BP ({dbp:g}) should be lower than systolic BP ({sbp:g}).")
    if not form.get("gender"):
        errors.append("Please select the patient's sex.")
    if not form.get("complaints"):
        errors.append("Please choose at least one chief complaint (use “Other (not listed)” if none fits).")
    if len(form.get("complaints", [])) > 3:
        errors.append("Choose at most 3 chief complaints.")
    if form.get("use_esi") and form.get("esi") is None:
        errors.append("ESI is switched on — please pick an ESI level, or switch it off.")
    return errors, warnings
