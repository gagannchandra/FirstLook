# FirstLook (EdgeTriage) — Build Plan

Status key: ✅ done · 🟡 in progress / partial · ❌ not done · ⬜ not started

## Phases

| Phase | Goal | Milestone | Status |
|-------|------|-----------|--------|
| P1 Setup & data | Repo, env, dataset, dictionary, leakage audit, tiers, splits | M1 Data ready | ⬜ |
| P2 Models | E1 baseline, E2 5×3 models (Optuna on val), calibration, thresholds, E3 | M2 Models ready | ⬜ |
| P3 Research | Conformal, E5 LOHO, **E6 local-adaptation**, E7 SHAP, E10, E11, E12 | M3 Research results | ⬜ |
| P4 Edge & app | E8 Pareto, ONNX/m2cgen, bundles, E9 Docker benchmark, Streamlit L1, Docker L2, (PWA L3) | M4 Prototype | ⬜ |
| P5 Final eval | Freeze, open S1 test once, Tables A–H, figures | M5 Submission | ⬜ |

## Experiments

| ID | Experiment | Status | Note |
|----|------------|--------|------|
| E1 | Baseline LR + XGB on T2/T3 (vs Hong et al.) | ⬜ | |
| E2 | 5 models × T1/T2/T3, Optuna on val | ⬜ | |
| E3 | Value of ESI/history (ΔAUC, DeLong) | ⬜ | |
| E4 | Calibration before/after (Platt, isotonic on S3) | ⬜ | |
| E5 | Leave-one-hospital-out + local recal | ⬜ | |
| E6 | Conformal (random split + under shift) + local-adaptation curve n=0…5000 ×20 | ⬜ | |
| E7 | SHAP global/local + stability | ⬜ | |
| E8 | Compression Pareto front | ⬜ | |
| E9 | Edge benchmark (size/latency/RAM/net), Docker --network none | ⬜ | |
| E10 | Missing-vitals robustness | ⬜ | |
| E11 | Subgroup / fairness | ⬜ | |
| E12 | Decision-curve analysis | ⬜ | |
| E13 | Usability pilot (time-to-enter, SUS) | ⬜ | Needs human participants — cannot be run by the agent |

## Deliverables

- [ ] `results/tables/data_dictionary.csv`, leakage audit
- [ ] `data/processed/` tier files + splits
- [ ] `src/` pipeline scripts, `pytest` suite
- [ ] Model bundles in `models/` (§7 format) + model card
- [ ] Streamlit app (4 screens), Docker image
- [ ] Tables A–H, figures 1–12 in `results/`
- [ ] README with run instructions

## Milestones

- M1 Data ready ⬜
- M2 Models ready ⬜
- M3 Research results ready ⬜
- M4 Prototype ready ⬜
- M5 Final evaluation ⬜
