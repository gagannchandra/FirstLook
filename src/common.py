"""Helpers to load the selected (frozen) models and score rows, with a test-set guard."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import calibrate

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEL = ROOT / "configs" / "selection.json"


class TestSetSealed(RuntimeError):
    pass


def selection() -> dict:
    return json.loads(SEL.read_text())


def assert_test_allowed() -> None:
    """The S1 test set may be scored only after choices are frozen (Phase 5)."""
    sel = selection()
    if not sel.get("date_frozen") or os.environ.get("FIRSTLOOK_OPEN_TEST") != "1":
        raise TestSetSealed("S1 test set is sealed: freeze selection.json (date_frozen) and set FIRSTLOOK_OPEN_TEST=1")


@lru_cache(maxsize=None)
def meta() -> pd.DataFrame:
    return pd.read_parquet(PROC / "meta.parquet")


def rows_for(split: str) -> np.ndarray:
    if split == "test":
        assert_test_allowed()
    return np.flatnonzero(meta()["s1"].to_numpy() == split)


def load_model(tier: str, model: str):
    return joblib.load(ROOT / "models" / "raw" / f"{tier}_{model}.joblib")


def calibrator(tier: str):
    return calibrate.from_dict(selection()["tiers"][tier]["calibrator_params"])


def score(tier: str, split: str, X: pd.DataFrame | None = None, model_name: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rows, y, calibrated p) for the selected model of `tier` on an S1 split."""
    rows = rows_for(split)
    sel = selection()["tiers"][tier]
    name = model_name or sel["model"]
    if X is None:
        X = pd.read_parquet(PROC / f"{tier}.parquet")
    m = load_model(tier, name)
    raw = m.predict_proba(X.iloc[rows])[:, 1]
    p = calibrator(tier).predict(raw) if name == sel["model"] else raw
    return rows, meta()["y"].to_numpy()[rows], p
