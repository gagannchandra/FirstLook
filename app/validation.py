"""Input validation for the patient form (pure functions, unit-tested).

validate_fields() is the single source of truth: each message carries the field it belongs to,
so both apps can show it next to that field (Vercel guidelines: errors beside fields, focus the
first error). The offline web app has a line-for-line JS port (frontend/js/model.js: validateFields);
tests/test_pwa.py checks the two return identical messages.
Messages say how to fix the problem; numbers and units are joined by a non-breaking space.
"""
from __future__ import annotations

NBSP = " "

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


def fmt(v: float) -> str:
    """Number formatting shared with the JS port: integers without decimals, else up to 1 decimal."""
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _u(unit: str) -> str:
    return unit if unit == "%" else NBSP + unit


def validate_fields(form: dict) -> list[dict]:
    """[{field, level: 'error'|'warning', msg}] in form order. Errors block prediction."""
    out: list[dict] = []
    add = lambda f, lvl, m: out.append({"field": f, "level": lvl, "msg": m})  # noqa: E731
    if form.get("age") is None:
        add("age", "error", "Enter the patient’s age in years.")
    if not form.get("gender"):
        add("gender", "error", "Select the patient’s sex.")
    cc = form.get("complaints") or []
    if not cc:
        add("complaints", "error", "Choose at least one chief complaint — use “Other” if none fits.")
    elif len(cc) > 3:
        add("complaints", "error", "Choose at most 3 chief complaints.")
    for key, (lo, hi, name, unit) in RANGES.items():
        v = form.get(key)
        if v is None:
            continue
        if not (lo <= v <= hi):
            if key == "age":
                add(key, "error", f"Enter an age from {lo} to {hi} — this tool was built on adult visits only.")
            else:
                add(key, "error", f"{name} {fmt(v)}{_u(unit)} is outside {fmt(lo)}–{fmt(hi)}{_u(unit)}. "
                                  "Re-check the reading, or tick “Not measured”.")
        elif key in SOFT and not (SOFT[key][0] <= v <= SOFT[key][1]):
            add(key, "warning", f"{name} {fmt(v)}{_u(unit)} is unusual — double-check it before continuing.")
    sbp, dbp = form.get("triage_vital_sbp"), form.get("triage_vital_dbp")
    if sbp is not None and dbp is not None and dbp >= sbp:
        add("triage_vital_dbp", "error", f"Diastolic BP ({fmt(dbp)}) must be lower than systolic BP ({fmt(sbp)}).")
    if form.get("use_esi") and form.get("esi") is None:
        add("esi", "error", "Pick an ESI level, or switch ESI off.")
    return out


def validate(form: dict) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors block prediction; warnings are shown only."""
    v = validate_fields(form)
    return [d["msg"] for d in v if d["level"] == "error"], [d["msg"] for d in v if d["level"] == "warning"]
