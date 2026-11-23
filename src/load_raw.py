"""Phase 1 step 1: load the original RData file and save it as Parquet.

Usage: python -m src.load_raw
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "5v_cleandf.RData"
OUT = ROOT / "data" / "interim" / "full.parquet"


def main() -> None:
    t0 = time.time()
    res = pyreadr.read_r(str(RAW))
    key = list(res.keys())[0]
    df = res[key]
    print(f"object '{key}' loaded in {time.time() - t0:.1f}s, shape={df.shape}")
    # float64 -> float32 to save memory (guide 02 §6)
    f64 = df.select_dtypes("float64").columns
    df[f64] = df[f64].astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
