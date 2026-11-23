"""Single feature pipeline shared by training and the app.

The *raw schema* is the set of columns in the original Yale dataframe that are
model inputs. `build_features` turns a raw-schema DataFrame (560k rows from
Parquet, or one row built from the app form) into the numeric model matrix.
Training and the app both call this function, so the encoding is identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "feature_spec.json"

VITALS = [
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_o2_device",
    "triage_vital_temp",
]

# Valid ranges used by the original cleaning script (9_edvitalsclean.R):
# values with x < min or x >= max were set to missing. Temperature in °F.
VITAL_RANGES = {
    "triage_vital_hr": (30, 300),
    "triage_vital_sbp": (30, 400),
    "triage_vital_dbp": (25, 300),
    "triage_vital_rr": (8, 70),
    "triage_vital_o2": (60, 100),
    "triage_vital_temp": (90, 110),
}

GENDER_LEVELS = ["Female", "Male"]
ARRIVAL_LEVELS = ["Car", "ambulance", "Walk-in", "Other", "Public Transportation", "Wheelchair", "Police"]
HOUR_BINS = ["23-02", "03-06", "07-10", "11-14", "15-18", "19-22"]
PREVDISPO_LEVELS = [
    "Discharge", "No previous dispo", "Admit", "Transfer to Another Facility",
    "LWBS after Triage", "AMA", "Eloped", "LWBS before Triage", "Observation", "Send to L&D",
]


def load_spec(path: Path = SPEC_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _onehot(series: pd.Series, levels: list[str], prefix: str) -> pd.DataFrame:
    s = series.astype("object")
    out = {f"{prefix}={lv}": (s == lv).astype(np.float32).to_numpy() for lv in levels}
    return pd.DataFrame(out, index=series.index)


def clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original dataset's outlier rule, so app input matches training.

    SpO2 = 100 is a special case: the original rule (x >= 100 -> NA) removed every
    SpO2 of 100, so the training data never contains 100. We map 100 -> 99
    (nearest value seen in training) instead of treating it as missing.
    See DECISIONS.md D9.
    """
    df = df.copy()
    if "triage_vital_o2" in df:
        o2 = pd.to_numeric(df["triage_vital_o2"], errors="coerce")
        df["triage_vital_o2"] = o2.where(o2 != 100, 99.0)
    for col, (lo, hi) in VITAL_RANGES.items():
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where((v >= lo) & (v < hi))
    return df


def feature_names(tier: str, cc_cols: list[str], hist_cols: list[str]) -> list[str]:
    """Ordered output columns of build_features (used to write the spec)."""
    names = ["age", "sex_male"]
    names += [f"arrival={lv}" for lv in ARRIVAL_LEVELS]
    names += [f"hour={lv}" for lv in HOUR_BINS]
    names += VITALS + list(cc_cols)
    if tier in ("t2", "t3"):
        names.append("esi")
    if tier == "t3":
        names += [f"prevdispo={lv}" for lv in PREVDISPO_LEVELS] + list(hist_cols)
    return names


def build_features(raw: pd.DataFrame, tier: str, spec: dict | None = None) -> pd.DataFrame:
    """Raw-schema rows -> numeric feature matrix (float32) for a tier.

    tier: 't1', 't2' or 't3'. Column order is taken from the spec when given
    (always the case in the app), so the output order is guaranteed.
    """
    spec = spec or load_spec()
    raw = clean_vitals(raw)
    parts: list[pd.DataFrame] = []
    idx = raw.index
    parts.append(pd.DataFrame({"age": pd.to_numeric(raw["age"], errors="coerce").astype(np.float32)}, index=idx))
    parts.append(pd.DataFrame({"sex_male": (raw["gender"].astype("object") == "Male").astype(np.float32)}, index=idx))
    parts.append(_onehot(raw["arrivalmode"], ARRIVAL_LEVELS, "arrival"))
    parts.append(_onehot(raw["arrivalhour_bin"], HOUR_BINS, "hour"))
    parts.append(raw[VITALS].apply(pd.to_numeric, errors="coerce").astype(np.float32))
    cc_cols = spec["cc_columns"]
    cc = raw.reindex(columns=cc_cols).fillna(0).astype(np.float32)
    parts.append(cc)
    if tier in ("t2", "t3"):
        esi = pd.to_numeric(raw["esi"].astype("object"), errors="coerce").astype(np.float32)
        parts.append(pd.DataFrame({"esi": esi}, index=idx))
    if tier == "t3":
        parts.append(_onehot(raw["previousdispo"], PREVDISPO_LEVELS, "prevdispo"))
        hist_cols = spec["history_columns"]
        parts.append(raw.reindex(columns=hist_cols).apply(pd.to_numeric, errors="coerce").astype(np.float32))
    X = pd.concat(parts, axis=1)
    order = spec["tiers"][tier] if "tiers" in spec else spec["model_features"]
    missing = [c for c in order if c not in X.columns]
    if missing:
        raise ValueError(f"feature pipeline missing columns: {missing[:5]}")
    return X[order]
