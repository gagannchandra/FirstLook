"""Phase 1: data dictionary, leakage audit, tiers T1/T2/T3, splits S1/S2/S3.

Usage: python -m src.data
Outputs (data/ is git-ignored):
  data/processed/{t1,t2,t3}.parquet   feature matrices, shared row index
  data/processed/meta.parquet         y, site, S1 split, sensitive audit attrs
  configs/feature_spec.json           ordered feature lists + category levels
  configs/tiers.yaml                  human-readable tier definitions
  results/tables/data_dictionary.csv  one row per raw column
  results/tables/leakage_single_feature_auc.csv
  results/tables/split_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import (ARRIVAL_LEVELS, GENDER_LEVELS, HOUR_BINS, PREVDISPO_LEVELS, VITAL_RANGES, VITALS,
                          build_features, feature_names)

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "interim" / "full.parquet"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
SEED = 42

SENSITIVE = ["ethnicity", "race", "lang", "religion", "maritalstatus", "employstatus", "insurance_status"]
SITE = "dep_name"
TARGET = "disposition"
# Site labels are anonymised (A/B/C) in the public file. Mapping to hospital type
# is inferred from volume and admission rate; see DECISIONS.md D8.
SITE_TYPE = {"A": "Academic (inferred)", "B": "Community (inferred)", "C": "Free-standing (inferred)"}


def column_groups(cols: list[str]) -> dict[str, str]:
    i = cols.index
    g: dict[str, str] = {}
    for c in cols[: i("previousdispo") + 1]:
        g[c] = "base"
    for c in cols[i("2ndarymalig"): i("whtblooddx") + 1]:
        g[c] = "pmh_ccs"
    for c in cols[i("whtblooddx") + 1: i("triage_vital_hr")]:
        g[c] = "hist_labs" if c not in ("n_edvisits", "n_admissions") else "prior_utilisation"
    for c in VITALS:
        g[c] = "triage_vitals"
    for c in cols[i("triage_vital_temp") + 1: i("cc_abdominalcramping")]:
        if c.startswith("meds_"):
            g[c] = "outpatient_meds"
        elif c.endswith("_count") and c.split("_")[0] in {"cxr", "echo", "ekg", "headct", "mri", "otherct", "otherimg", "otherus", "otherxr"}:
            g[c] = "prior_imaging_ekg"
        elif c == "n_surgeries":
            g[c] = "prior_procedures"
        else:
            g[c] = "hist_vitals"
    for c in cols:
        if c.startswith("cc_"):
            g[c] = "chief_complaint"
    return g


BASE_TRIAGE = {
    "dep_name": ("site", "yes", "excluded (site id; LOHO only)"),
    "esi": ("triage", "yes (see note: most recent ESI kept by original script)", "T2,T3"),
    "age": ("demographic", "yes", "T1,T2,T3"),
    "gender": ("demographic", "yes", "T1,T2,T3"),
    "disposition": ("outcome", "no — outcome", "target"),
    "arrivalmode": ("arrival", "yes", "T1,T2,T3"),
    "arrivalmonth": ("arrival", "yes", "not used (not in guide's T1)"),
    "arrivalday": ("arrival", "yes", "not used (not in guide's T1)"),
    "arrivalhour_bin": ("arrival", "yes", "T1,T2,T3"),
    "previousdispo": ("prior_utilisation", "yes (previous visit)", "T3"),
}
GROUP_INFO = {
    "pmh_ccs": ("yes (EHR problem list at time of visit)", "T3"),
    "hist_labs": ("yes (labs from previous visits within 1 yr)", "T3"),
    "prior_utilisation": ("yes (counts over previous year)", "T3"),
    "triage_vitals": ("yes (recorded before rooming)", "T1,T2,T3"),
    "hist_vitals": ("yes (vitals from previous visits within 1 yr)", "T3"),
    "prior_imaging_ekg": ("yes (orders from previous visits within 1 yr)", "T3"),
    "outpatient_meds": ("yes (outpatient med list)", "T3"),
    "prior_procedures": ("yes (EHR surgical history)", "T3"),
    "chief_complaint": ("yes (coded at triage)", "T1,T2,T3"),
}


def build_dictionary(df: pd.DataFrame, constant: list[str]) -> pd.DataFrame:
    cols = list(df.columns)
    groups = column_groups(cols)
    rows = []
    for c in cols:
        grp = groups.get(c, "unknown")
        if c in SENSITIVE:
            grp, avail, tier = "sensitive_demographic", "yes", "excluded (fairness audit only)"
        elif c in BASE_TRIAGE:
            grp, avail, tier = BASE_TRIAGE[c]
        else:
            avail, tier = GROUP_INFO[grp]
        if c in constant:
            tier = "excluded (constant column)"
        s = df[c]
        rows.append({
            "column": c, "dtype": str(s.dtype), "group": grp,
            "pct_missing": round(float(s.isna().mean()) * 100, 2),
            "n_unique": int(s.nunique()), "available_at_triage": avail, "tier": tier,
        })
    return pd.DataFrame(rows)


def single_feature_auc(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Leakage red-flag test: AUC of each feature alone (NaN -> median)."""
    out = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=np.float64)
        if np.all(np.isnan(v)) or np.nanstd(v) == 0:
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        a = roc_auc_score(y, v)
        out.append({"feature": c, "auc": max(a, 1 - a)})
    return pd.DataFrame(out).sort_values("auc", ascending=False)


def make_splits(y: np.ndarray, site: np.ndarray) -> pd.DataFrame:
    """S1: stratified 80/10/10 train/val/test; S3: 10% of S1-train as 'cal'.

    S2 (leave-one-hospital-out) is defined on the fly from `site`; it uses only
    rows that are NOT in the S1 test set, so the S1 test set stays sealed.
    """
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
    tr, ca = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
    s1 = np.empty(len(y), dtype=object)
    s1[tr], s1[ca], s1[va], s1[te] = "train", "cal", "val", "test"
    return pd.DataFrame({"s1": s1})


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FULL)
    assert df.shape == (560486, 972), df.shape
    cols = list(df.columns)

    y = (df[TARGET].astype(str) == "Admit").astype(np.int8).to_numpy()
    print(f"rows={len(df):,} cols={df.shape[1]} admit rate={y.mean():.4f}")

    # Patient-ID check (guide 02 §5)
    id_like = [c for c in cols if c.lower() in {"pat_mrn_id", "mrn", "patient_id", "pat_id", "pat_enc_csn_id"}]
    print("patient id columns:", id_like or "none")

    constant = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    print("constant columns:", constant)
    print("exact duplicate rows:", int(df.duplicated().sum()))

    dictionary = build_dictionary(df, constant)
    dictionary.to_csv(TABLES / "data_dictionary.csv", index=False)
    print(dictionary.groupby("group").size())

    # Category levels must match what the pipeline expects
    assert set(df["gender"].dropna().astype(str)) <= set(GENDER_LEVELS)
    assert set(df["arrivalmode"].dropna().astype(str)) <= set(ARRIVAL_LEVELS)
    assert set(df["arrivalhour_bin"].dropna().astype(str)) <= set(HOUR_BINS)
    assert set(df["previousdispo"].dropna().astype(str)) <= set(PREVDISPO_LEVELS)

    cc_cols = [c for c in cols if c.startswith("cc_")]
    groups = column_groups(cols)
    hist_groups = {"pmh_ccs", "hist_labs", "prior_utilisation", "hist_vitals", "prior_imaging_ekg",
                   "outpatient_meds", "prior_procedures"}
    hist_cols = [c for c in cols if groups.get(c) in hist_groups and c not in constant]

    # Training-range info for the app (computed on S1 train later is unnecessary:
    # ranges are the fixed physiological limits of the original cleaning).
    spec = {
        "version": 1,
        "cc_columns": cc_cols,
        "history_columns": hist_cols,
        "levels": {"gender": GENDER_LEVELS, "arrivalmode": ARRIVAL_LEVELS, "arrivalhour_bin": HOUR_BINS,
                   "previousdispo": PREVDISPO_LEVELS},
        "vital_ranges": {k: list(v) for k, v in VITAL_RANGES.items()},
        "temperature_unit": "F",
        "tiers": {t: feature_names(t, cc_cols, hist_cols) for t in ("t1", "t2", "t3")},
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "feature_spec.json").write_text(json.dumps(spec, indent=1))
    tiers_yaml = {
        "t1": {"description": "Nurse-only: age, sex, arrival mode, arrival hour bin, 7 triage vitals, chief complaint",
               "raw_inputs": ["age", "gender", "arrivalmode", "arrivalhour_bin"] + VITALS + ["cc_* (200)"],
               "n_features": len(spec["tiers"]["t1"])},
        "t2": {"description": "T1 + ESI", "raw_inputs": ["T1", "esi"], "n_features": len(spec["tiers"]["t2"])},
        "t3": {"description": "T2 + all EHR history groups", "raw_inputs": ["T2", "previousdispo"] + sorted(hist_groups),
               "n_features": len(spec["tiers"]["t3"])},
        "excluded_from_inputs": {"sensitive": SENSITIVE, "site": [SITE], "outcome": [TARGET],
                                 "not_in_guide_tiers": ["arrivalmonth", "arrivalday"], "constant": constant},
    }
    (ROOT / "configs" / "tiers.yaml").write_text(yaml.safe_dump(tiers_yaml, sort_keys=False))

    # Split
    site = df[SITE].astype(str).to_numpy()
    splits = make_splits(y, site)

    meta = pd.DataFrame({"y": y, "site": site, "s1": splits["s1"].to_numpy()})
    for c in SENSITIVE:
        meta[c] = df[c].astype("object").astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    meta["age"] = df["age"].to_numpy()
    meta["gender"] = df["gender"].astype(str).to_numpy()
    meta["esi"] = df["esi"].astype("object").astype(str).to_numpy()
    meta["arrivalmode"] = df["arrivalmode"].astype("object").astype(str).to_numpy()
    meta.to_parquet(PROC / "meta.parquet")

    for t in ("t1", "t2", "t3"):
        X = build_features(df, t, spec)
        assert len(X) == len(df)
        X.to_parquet(PROC / f"{t}.parquet")
        print(f"{t}: {X.shape}")

    # Leakage red-flag test on S1 train only (T3 includes every input feature)
    X3 = pd.read_parquet(PROC / "t3.parquet")
    tr = meta["s1"].to_numpy() == "train"
    sfa = single_feature_auc(X3[tr], y[tr])
    sfa.to_csv(TABLES / "leakage_single_feature_auc.csv", index=False)
    print("top single-feature AUCs:\n", sfa.head(10).to_string(index=False))
    flagged = sfa[sfa.auc > 0.95]
    print("features with single AUC > 0.95:", flagged.feature.tolist() or "none")

    # Split summary (no metric is computed on test; counts/prevalence only)
    rows = []
    for name, m in [("all", np.ones(len(y), bool))] + [(s, meta.s1.to_numpy() == s) for s in ("train", "cal", "val", "test")]:
        rows.append({"split": f"S1:{name}", "n": int(m.sum()), "admit_rate": round(float(y[m].mean()), 4)})
    for s in sorted(set(site)):
        m = site == s
        rows.append({"split": f"site:{s} {SITE_TYPE[s]}", "n": int(m.sum()), "admit_rate": round(float(y[m].mean()), 4)})
        for part in ("train", "cal", "val", "test"):
            mm = m & (meta.s1.to_numpy() == part)
            rows.append({"split": f"site:{s} S1:{part}", "n": int(mm.sum()), "admit_rate": round(float(y[mm].mean()), 4)})
    summ = pd.DataFrame(rows)
    summ.to_csv(TABLES / "split_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
