# 06 · 12-Week Plan, Milestones and Risks

## 1. Phase overview

| Phase | Weeks | Goal | Gate |
|-------|-------|------|------|
| P1 Foundation | 1–3 | Literature, data loaded, dictionary, leakage audit, tiers and splits | **M1 Data ready** (end W3) |
| P2 Modelling | 4–6 | Baselines, 5 models × 3 tiers, tuning, calibration | **M2 Models ready** (end W6) |
| P3 Trust & transfer | 7–8 | Conformal, SHAP, leave-one-hospital-out, robustness, fairness | **M3 Research results ready** (end W8) |
| P4 Edge & app | 9–10 | Compression, export, benchmark, offline app | **M4 Prototype ready** (end W10) |
| P5 Finish | 11–12 | Test-set run, final tables, dissertation, demo, viva prep | **M5 Submission** (end W12) |

Writing happens **every week** (see last column), so Week 12 is polishing, not starting.

## 2. Week-by-week plan

| Week | Main work | Deliverable | Write-up this week |
|------|-----------|-------------|--------------------|
| 1 | Read Hong et al., **Abdulai et al. (2025)** and **Dave et al. (2026)** closely + 8–10 related papers (admission prediction, conformal prediction, edge ML in health). Set up repo, environment, download dataset and original code. Meet supervisor with `01_proposal.md`. | Repo skeleton, dataset downloaded, literature notes | Ch. 2 notes |
| 2 | Load RData → Parquet. Build data dictionary (column, group, type, % missing, available-at-triage?). Leakage audit. Check for patient ID. | `data_dictionary.csv`, leakage audit note | Ch. 3.1 Dataset |
| 3 | Cleaning, range checks, EDA figures. Build T1/T2/T3 tier files and S1/S2/S3 splits. | Tier Parquet files, EDA notebook, split file | Ch. 3.2 Preprocessing |
| 4 | Baselines: LR and XGBoost on T2/T3 (reproduce Hong et al. ballpark). Evaluation module (metrics + bootstrap CIs). | E1 results, `evaluate.py` | Ch. 3.3 Models |
| 5 | RF, LightGBM, CatBoost on all tiers. Optuna tuning on validation. | E2 validation results | Ch. 4 draft tables |
| 6 | Calibration (Platt/isotonic), thresholds and risk bands, value-of-information analysis. | E3, E4 results, chosen models per tier | Ch. 4.1–4.2 |
| 7 | Conformal prediction (own implementation), risk–coverage. Leave-one-hospital-out. **Local-adaptation curve** (n = 0…5000 local visits, 20 draws each). | E5, E6 results, signature figure | Ch. 4.3–4.4 |
| 8 | SHAP global/local + stability. Missing-vitals robustness. Subgroup analysis. Decision curves. | E7, E10, E11, E12 results | Ch. 4.5–4.7 |
| 9 | SHAP-based feature reduction, shallow models, ONNX/m2cgen export, Pareto plot. Model bundle format. | E8 results, `models/` bundles | Ch. 5.1 Edge design |
| 10 | Streamlit app (L1), ONNX runtime + Docker limited/no-network (L2), edge benchmarks. PWA (L3) only if ahead. | Working offline app, E9 results | Ch. 5.2 System |
| 11 | **Freeze everything → run final test set once.** Fill all tables and figures. Usability pilot (5–8 participants, ~20 min each). Record demo video as backup. | Final results, SUS results, demo video | Ch. 4 final, Ch. 6 Discussion |
| 12 | Dissertation polish, abstract, conclusion, limitations, slides, viva practice. | Final dissertation, slides, code release | Ch. 1, 7, abstract |

## 3. Milestone checklists

**M1 — Data ready (end W3)**
- [ ] RData loaded; row and column counts match 560,486 × 972
- [ ] Target defined; admission rate ≈ 29.7%
- [ ] Data dictionary complete
- [ ] Leakage audit signed off with supervisor
- [ ] Patient-ID question answered; split strategy fixed
- [ ] T1/T2/T3 files and S1/S2/S3 splits saved with seeds

**M2 — Models ready (end W6)**
- [ ] 5 models × 3 tiers trained and tuned (validation only)
- [ ] Baseline within reasonable range of the original study
- [ ] Calibration applied; reliability diagrams produced
- [ ] Best model per tier selected with written justification

**M3 — Research results ready (end W8)**
- [ ] Conformal coverage verified on validation
- [ ] Leave-one-hospital-out table complete
- [ ] SHAP figures done
- [ ] Robustness, fairness, decision curves done

**M4 — Prototype ready (end W10)**
- [ ] Edge model selected by pre-declared rule
- [ ] App works end-to-end with networking disabled
- [ ] Benchmark table complete (size, latency, RAM, bytes)

**M5 — Submission (end W12)**
- [ ] Test set opened once; date logged
- [ ] All tables/figures final
- [ ] Dissertation, slides, demo video, README, model card

## 4. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | RData fails to load in Python | Low | High | Use `pyreadr`; fallback: install R, `load()` then `arrow::write_parquet()` |
| R2 | Not enough RAM for T3 | Medium | Medium | float32, tier files, Colab, subsample for tuning, LightGBM |
| R3 | Column meanings unclear | Medium | Medium | Read original R scripts and paper S1 Table; ask supervisor |
| R4 | No patient ID → repeat-patient leakage | High | Medium | Visit-level split as in original + leave-one-hospital-out as stricter test; state limitation |
| R5 | Hospital identifier not usable | Low | High | If missing, replace E5 with an arrival-time-based split (e.g., by month) and say so |
| R6 | Tuning takes too long | Medium | Medium | Limit Optuna trials, subsample, early stopping, run overnight |
| R7 | ONNX export fails for a model | Medium | Low | Use `m2cgen`, or keep native LightGBM/XGBoost loaders; LR always exports |
| R8 | PWA (L3) takes too long | High | Low | It is a stretch goal; L1 + L2 fully satisfy RQ4 |
| R9 | Results weaker than hoped | Medium | Low | Honest negative findings (e.g., calibration breaks across sites) are valid research results |
| R10 | Scope creep (LLMs, chatbots, extra datasets) | Medium | High | Anything not in `03_methodology.md` goes to "Future work" |
| R11 | Laptop failure / data loss | Low | High | Git + cloud backup of code; data re-downloadable; weekly backup of results |
| R12 | Examiner says "this is already done" (Abdulai 2025 used the same data + conformal) | Medium | High | Cite it up front; frame E6 random-split result as a replication; put the novelty in hospital shift, local adaptation, nurse-only tier and edge deployment |
| R13 | One hospital has too few visits or very different admission rate | Medium | Medium | Report per-site n and prevalence in EDA; use wider CIs; discuss prevalence shift explicitly |
| R14 | Cannot recruit usability participants | Medium | Low | Fall back to timed self-entry + 3 classmates; report as limitation |

## 5. Weekly supervisor update template

```
EdgeTriage — Week N update (date)

Done this week:
- …

Key result (one number or one figure):
- …

Problems / decisions needed:
- …

Plan for next week:
- …

Milestone status: M1 ✅  M2 🟡  M3 ⬜  M4 ⬜  M5 ⬜
```

## 6. Cut list (if behind schedule, drop in this order)

1. PWA (L3)
2. EBM glass-box model
3. Random Forest on T3 (slowest, least likely to win)
4. 3-seed repeats (keep 1 seed + bootstrap CIs)
5. Audit-only fairness attributes (keep age, sex, site)

Never cut: leakage audit, calibration, leave-one-hospital-out, conformal + local-adaptation curve, offline demo.
