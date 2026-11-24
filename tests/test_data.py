"""Data pipeline tests: shapes, no leakage, no split overlap."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROC / "meta.parquet").exists(), reason="processed data not built")

SENSITIVE = ["ethnicity", "race", "lang", "religion", "maritalstatus", "employstatus", "insurance_status"]


@pytest.fixture(scope="module")
def meta():
    return pd.read_parquet(PROC / "meta.parquet")


@pytest.fixture(scope="module")
def spec():
    import json
    return json.loads((ROOT / "configs" / "feature_spec.json").read_text())


def test_raw_shape_and_prevalence(meta):
    assert len(meta) == 560486
    assert abs(meta.y.mean() - 0.297) < 0.002


@pytest.mark.parametrize("tier,ncols", [("t1", 222), ("t2", 223), ("t3", 976)])
def test_tier_shapes(tier, ncols, spec):
    import pyarrow.parquet as pq
    sch = pq.read_schema(PROC / f"{tier}.parquet")
    names = [n for n in sch.names if not n.startswith("__index")]
    assert len(names) == ncols
    assert names == spec["tiers"][tier]
    assert pq.read_metadata(PROC / f"{tier}.parquet").num_rows == 560486


def test_tiers_nested(spec):
    t1, t2, t3 = (set(spec["tiers"][t]) for t in ("t1", "t2", "t3"))
    assert t1 < t2 < t3
    assert t2 - t1 == {"esi"}


def test_no_sensitive_or_site_or_target_inputs(spec):
    forbidden = set(SENSITIVE) | {"dep_name", "disposition", "arrivalmonth", "arrivalday"}
    for t, cols in spec["tiers"].items():
        assert not forbidden & set(cols), t
        assert not any(c.startswith(("race", "insurance", "religion")) for c in cols)


def test_splits_partition_and_no_overlap(meta):
    s = meta.s1
    assert set(s.unique()) == {"train", "cal", "val", "test"}
    # every row in exactly one split (a single label column guarantees disjointness)
    assert s.notna().all()
    counts = s.value_counts()
    assert abs(counts["test"] / len(s) - 0.10) < 0.002
    assert abs(counts["val"] / len(s) - 0.10) < 0.002
    assert abs(counts["cal"] / len(s) - 0.08) < 0.002


def test_split_is_deterministic(meta):
    from src.data.build import make_splits
    again = make_splits(meta.y.to_numpy(), meta.site.to_numpy())
    assert (again.s1.to_numpy() == meta.s1.to_numpy()).all()


def test_prevalence_stratified(meta):
    rates = meta.groupby("s1").y.mean()
    assert (rates.max() - rates.min()) < 0.002


def test_loho_excludes_s1_test(meta):
    from src.data.splits import loho
    for h in ("A", "B", "C"):
        d = loho(meta, h)
        assert not (meta.s1.to_numpy()[d["test"]] == "test").any()
        assert (meta.site.to_numpy()[d["test"]] == h).all()
        for k in ("train", "cal", "val"):
            assert (meta.site.to_numpy()[d[k]] != h).all()
        allidx = np.concatenate([d[k] for k in ("train", "cal", "val", "test")])
        assert len(allidx) == len(set(allidx))


def test_leakage_audit_passed():
    sfa = pd.read_csv(ROOT / "results" / "tables" / "leakage_single_feature_auc.csv")
    assert sfa.auc.max() < 0.95
    d = pd.read_csv(ROOT / "results" / "tables" / "data_dictionary.csv")
    assert len(d) == 972
    assert (d.available_at_triage != "").all()


def test_tiers_yaml_exists():
    y = yaml.safe_load((ROOT / "configs" / "tiers.yaml").read_text())
    assert set(y) >= {"t1", "t2", "t3", "excluded_from_inputs"}
