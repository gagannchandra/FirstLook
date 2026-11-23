# 02 · Dataset and Features

## 1. Primary dataset

**Yale Emergency Department Admission Prediction Dataset** (Hong, Haimovich & Taylor, 2018)

| Property | Detail |
|----------|--------|
| Records | 560,486 adult ED visits (one row per visit) |
| Variables | 972 |
| Unique patients | 202,953 (reported in paper) |
| Sites | 3 EDs in one US health system: academic level-I trauma centre, community hospital ED, suburban free-standing ED |
| Period | March 2014 – July 2017 (history look-back from March 2013) |
| Outcome | Hospital admission vs. discharge |
| Admission rate | ≈29.7% |
| Format | R dataframe `5v_cleandf.RData` inside a ~100 MB ZIP (~1.7 GB as CSV) |
| Download | Zenodo record 1308993 |
| Code | github.com/yaleemmlc/admissionprediction (R scripts: cleaning, XGBoost, DNN, LR) |
| Licence | Not clearly stated — use for research, cite the paper, do not redistribute the data |

**Citation required:** Hong WS, Haimovich AD, Taylor RA (2018). Predicting hospital admission at emergency department triage using machine learning. *PLoS ONE* 13(7): e0201016.

## 2. What is in the 972 variables

| Group | Content | Notes |
|-------|---------|-------|
| Site | Name of presenting hospital (3 values) | Used for leave-one-hospital-out, **not** as a model input in the transfer experiment |
| Demographics | Age, sex, plus race, ethnicity, language, religion, marital status, employment, insurance | Sensitive / US-specific fields are **excluded from inputs** (see §4) |
| Arrival | Arrival mode, month, day, hour / 4-hour bin | Arrival mode is available at triage |
| Triage | ESI level, triage vital signs (HR, SBP, DBP, RR, SpO₂, O₂ device, temperature; mean if multiple sets) | Outliers already set to missing by original scripts |
| Chief complaint | Top 200 complaints as coded categories, the rest in one "other" bin | **Not free text** |
| Past medical history | Conditions grouped into AHRQ CCS categories (ICD-9 based) | Tier 3 only |
| Prior utilisation | ED visits and admissions in previous year, previous ED disposition, prior procedures/surgeries | Tier 3 only |
| Historical labs | Previous lab values (last, min, max, median within one year) | Tier 3 only |
| Historical vitals | Previous-visit vitals summaries | Tier 3 only |
| Prior imaging / ECG | Counts of earlier orders | Tier 3 only |
| Medications | Outpatient medication counts in 48 therapeutic classes | Tier 3 only (top predictor in original study) |

> Exact column names must be confirmed in Week 2. Expected names include `disposition`, `esi`, `age`, `gender`, `dep_name`, `arrivalmode`, `triage_vital_*`, `cc_*`, `n_edvisits`, `n_admissions`, `previousdispo`. Build the data dictionary from the real file, not from this list.

## 3. Input tiers (core experimental design)

| Tier | Name | Inputs | Approx. # features | Real-world meaning |
|------|------|--------|--------------------|--------------------|
| **T1** | Nurse-only | Age, sex, arrival mode, arrival hour bin, 7 triage vitals, chief complaint | ~15 raw (~215 after one-hot) | Any ED, no triage scale, no EHR — **main India-relevant model** |
| **T2** | Nurse + ESI | T1 + ESI level | T1 + 1 | ED that uses a structured triage scale |
| **T3** | Full | T2 + all history groups | ~900+ | ED with a mature EHR |
| **T3-lite** | Compact full | T2 + top-k history features chosen by SHAP on the training set | ~20–40 | Edge model when partial history exists |

T1 is what EdgeTriage's app asks for by default. T2 and T3 are research comparisons that answer "what is the value of ESI and of EHR history?"

## 4. Feature inclusion and exclusion rules

**Exclude from all model inputs**
- Race, ethnicity, religion, language, insurance status, employment status, marital status — sensitive and/or US-specific; would not transfer to India and risk encoding bias. Keep them only in a separate audit table for fairness analysis.
- Hospital/site identifier — excluded so the model learns patient signals, not "which hospital"; also required for a fair leave-one-hospital-out test. (An "including site" variant can be reported as a secondary result.)
- Anything that is the outcome or derived from it.

**Leakage audit (do this in Week 2 and record results)**
- The original dataset was constructed from variables available at triage, but verify it yourself: for every column group, write down "available at triage? yes/no/unclear".
- Check for any current-visit labs, imaging results, ED length of stay, treatment orders or final diagnosis. Drop if found.
- Red flag test: if any single feature gives AUC > 0.95 alone, investigate it as possible leakage.

## 5. Data splits

Three split designs, each answering a different question:

| Split | How | Used for |
|-------|-----|----------|
| **S1 — Random** | Stratified 80 / 10 / 10 (train / validation / test), fixed seed | Main comparison; comparable with Hong et al. |
| **S2 — Leave-one-hospital-out** | Train+validate on 2 sites, test on the 3rd; repeat for all 3 | RQ2: transfer to unseen hospital type |
| **S3 — Calibration set** | Carve a further slice from S1 training data (e.g., 10%) | Fit isotonic/Platt calibration and conformal thresholds without touching the test set |

**Repeat patients.** 202,953 patients made 560,486 visits, so random visit-level splits put the same patient in train and test, which can inflate scores. Plan:
1. Check whether the public file contains a patient identifier.
2. If yes → use grouped splits (`GroupShuffleSplit`) on patient ID for S1.
3. If no → keep the visit-level split (same as the original paper), state it as a limitation, and note that S2 (different hospitals) gives a stricter, more honest estimate.

**Test set discipline.** The S1 test set is opened once, after all model, threshold, calibration and compression choices are frozen. Record the date you opened it.

## 6. Preprocessing plan

| Step | Action |
|------|--------|
| Load | `pyreadr` → pandas → save as Parquet (float32 for numerics to fit in RAM) |
| Target | Map disposition to 1 = admitted, 0 = discharged; confirm prevalence ≈29.7% |
| Duplicates | Check exact duplicate rows |
| Missing values | Report % missing per column and per tier; keep NaN for tree models (native handling); median-impute + missing-indicator for LR |
| Outliers | Original scripts already set implausible vitals to missing; re-check ranges (e.g., HR 20–250, SpO₂ 50–100, temp 30–43 °C) |
| Categoricals | One-hot for LR/RF; native categorical for CatBoost/LightGBM |
| Scaling | Standardise numerics for LR only |
| Class imbalance | Do **not** resample by default; use `class_weight`/`scale_pos_weight` as a tuned option and evaluate with PR-AUC and calibration |
| Feature-tier files | Save `t1.parquet`, `t2.parquet`, `t3.parquet` with a shared row index and split column |

## 7. Exploratory analysis to include in the report

- Admission rate overall, by site, by ESI, by age band, by arrival mode.
- Missingness heatmap for triage vitals.
- Distributions of each vital sign for admitted vs discharged.
- Top 20 chief complaints by frequency and by admission rate.
- Correlation of history counts with admission.

## 8. Hardware note

560k × 972 at float32 ≈ 2.2 GB in memory. A 16 GB laptop or Google Colab is enough. Work on tier files (T1 is tiny) to iterate quickly; only T3 needs the full width.

## 9. Optional secondary dataset (only if time allows)

For an external sanity check outside Yale, **MIMIC-IV-ED** (PhysioNet, credentialed access, several hundred thousand ED stays) contains triage vitals, chief complaint, acuity and disposition. Getting access needs CITI training and approval, which can take weeks — apply in Week 1 only if your supervisor wants external validation. The project does **not** depend on it.
