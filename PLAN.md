# FirstLook (EdgeTriage) — Build Plan

Status key: ✅ done · 🟡 in progress / partial · ❌ not done · ⬜ not started

## Phases

| Phase | Goal | Milestone | Status |
|-------|------|-----------|--------|
| P1 Setup & data | Repo, env, dataset, dictionary, leakage audit, tiers, splits | M1 Data ready | ✅ 560,486×972, 29.73% admit, no leakage flag |
| P2 Models | E1 baseline, E2 5×3 models (Optuna on val), calibration, thresholds, E3 | M2 Models ready | ✅ val AUC T1 .857 / T2 .876 / T3 .928 |
| P3 Research | Conformal, E5 LOHO, **E6 local-adaptation**, E7 SHAP, E10, E11, E12 | M3 Research results | ✅ (val; test numbers in P5) |
| P4 Edge & app | E8 Pareto, ONNX/m2cgen, bundles, E9 Docker benchmark, Streamlit L1, Docker L2, (PWA L3) | M4 Prototype | ✅ L1 Streamlit + L2 Docker + L3 offline web app (frontend/ T1/T2 in-browser + backend/ local API incl. T3-lite, D37) |
| P5 Final eval | Freeze, open S1 test once, Tables A–H, figures | M5 Submission | ✅ test opened 2026-11-24; Tables A–G in results/RESULTS.md; H not run (needs humans) |

## Experiments

| ID | Experiment | Status | Note |
|----|------------|--------|------|
| E1 | Baseline LR + XGB on T2/T3 (vs Hong et al.) | ✅ | val XGB T2 0.873 / T3 0.925 vs Hong 0.87 / 0.92 |
| E2 | 5 models × T1/T2/T3, Optuna on val | ✅ | 15 studies; selected LGBM/LGBM/CatBoost (D18) |
| E3 | Value of ESI/history (ΔAUC, DeLong) | ✅ | test: +0.019 (ESI, CI 0.017–0.020), +0.053 (history); DeLong p<1e-100 |
| E4 | Calibration before/after (Platt, isotonic on S3) | ✅ | val ECE ≤ 0.005 for all selected |
| E5 | Leave-one-hospital-out + local recal | ✅ | T1 AUC 0.82–0.87 at unseen site; ECE up to 0.128 (site C) → ≤0.018 after 1,000 local visits |
| E6 | Conformal (random split + under shift) + local-adaptation curve n=0…5000 ×20 | ✅ | test coverage 0.903/0.900 (T1/T2, α=.1), all α within ±0.02; under shift repaired by 250–1000 local visits |
| E7 | SHAP global/local + stability | ✅ | top-10 stable across seeds (ρ≥0.98); less across sites (ρ 0.68–0.96) |
| E8 | Compression Pareto front | ✅ | T1: no export within 0.01 → LR (9 KB, Δ0.011, D27); T3-lite LGBM d4 t200 top-50 (215 KB, Δ0.0095) |
| E9 | Edge benchmark (size/latency/RAM/net), Docker --network none | ✅ | Docker 2 CPU/2 GB: p95 9.4–28 ms, 188 MB RSS, 0 bytes; ORT telemetry found + disabled (D28) |
| E10 | Missing-vitals robustness | ✅ | test: random loss of 1–3 vitals ≤0.005 AUC; median imputation ~0.007 worse |
| E11 | Subgroup / fairness | ✅ | test: flagged groups listed in Table G (results/RESULTS.md) |
| E12 | Decision-curve analysis | ✅ | test: all tiers beat treat-all/none over 0.10–0.70 |
| E13 | Usability pilot (time-to-enter, SUS) | ❌ | Needs human participants — cannot be run by the agent; protocol + app ready |

## Deliverables

- [ ] `results/tables/data_dictionary.csv`, leakage audit
- [ ] `data/processed/` tier files + splits
- [ ] `src/` pipeline scripts, `pytest` suite
- [ ] Model bundles in `models/` (§7 format) + model card
- [ ] Streamlit app (4 screens), Docker image
- [x] Tables A–G + criteria in `results/RESULTS.md`, figures in `results/figures/` (H: not run)
- [x] README with run instructions

## Milestones

- M1 Data ready ✅ (supervisor sign-off of leakage audit pending — human step)
- M2 Models ready ✅
- M3 Research results ready ✅
- M4 Prototype ready ✅
- M5 Final evaluation ✅ (usability pilot outstanding)
