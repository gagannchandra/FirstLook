"""Split helpers shared by all experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def s1(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    s = meta["s1"].to_numpy()
    return {k: np.flatnonzero(s == k) for k in ("train", "cal", "val", "test")}


def loho(meta: pd.DataFrame, held_out: str, seed: int = SEED) -> dict[str, np.ndarray]:
    """S2 leave-one-hospital-out.

    Train/cal/val come from the two other sites (80/10/10 of their non-test rows);
    'test' is every row of the held-out site. Rows in the S1 test set are excluded
    everywhere, so the S1 test set stays sealed until Phase 5 (DECISIONS.md D11).
    """
    site = meta["site"].to_numpy()
    keep = meta["s1"].to_numpy() != "test"
    y = meta["y"].to_numpy()
    src = np.flatnonzero(keep & (site != held_out))
    tr, tmp = train_test_split(src, test_size=0.2, stratify=y[src], random_state=seed)
    va, ca = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    return {"train": tr, "cal": ca, "val": va, "test": np.flatnonzero(keep & (site == held_out))}
