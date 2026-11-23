# 01 · Project Proposal

## 1. Title

**EdgeTriage: An Offline-First, Uncertainty-Aware Edge-AI System for Predicting Hospital Admission at Emergency Department Triage**

Short title: *EdgeTriage*

## 2. Abstract

Emergency departments (EDs) must make fast decisions with incomplete information, and early knowledge of which patients are likely to need a hospital bed helps staff plan beds, staffing and transfers. Published machine-learning models for admission prediction usually assume three things that many hospitals in India and other low-resource settings do not have: a structured triage scale such as the Emergency Severity Index (ESI), a rich electronic health record (EHR) history, and a reliable internet connection to a cloud model.

This project proposes **EdgeTriage**, an offline-first decision-support system that estimates the probability of hospital admission from information available at initial triage, runs entirely on a local low-cost device, explains each estimate, and flags low-confidence cases for clinician review. Using the public Yale ED Admission Prediction dataset (560,486 adult visits, 972 variables, three hospitals of different types, 2014–2017), the project builds three input tiers: (1) a nurse-entered tier with only demographics, vital signs, arrival information and chief complaint; (2) Tier 1 plus ESI; and (3) Tier 2 plus EHR history. Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost are compared.

Beyond discrimination (ROC-AUC, PR-AUC), the study evaluates probability calibration, clinical net benefit (decision-curve analysis), robustness to missing vital signs, and subgroup performance. Split conformal prediction produces an "uncertain" output. Unlike prior conformal work on this dataset, which used a random split, the study tests whether calibration and conformal coverage survive transfer to an unseen hospital (leave-one-hospital-out), and measures how many local visits are needed to repair them without retraining. The selected model is compressed and exported for edge inference, and its file size, latency, memory use and network traffic are measured on a resource-limited device. The final prototype is demonstrated live with networking disabled.

EdgeTriage is a research prototype for decision support. It does not diagnose, assign treatment or replace clinical judgement, and local clinical validation would be required before any real use.

**Keywords:** emergency department, hospital admission prediction, edge AI, offline-first, conformal prediction, calibration, explainable AI, gradient boosting.

## 3. Problem statement

At triage, a nurse has a few minutes and a handful of measurements. Whether the patient will eventually be admitted is not known until hours later. Knowing it early helps the department:

- request inpatient beds and staff earlier,
- start transfer arrangements in facilities without the required inpatient services,
- prioritise attention for patients likely to deteriorate.

Existing admission-prediction research (e.g., Hong et al., 2018) achieved high ROC-AUC (≈0.92) using 972 variables including ESI and detailed EHR history, inside a well-connected US health system. Three practical gaps remain for low-resource settings:

1. **Input gap** — many Indian EDs do not use ESI and have little or no digitised history. How well does prediction work with only what the nurse can enter?
2. **Transfer gap** — models are usually tested on the same hospital they were trained on. Do they hold up in a different *type* of hospital?
3. **Deployment gap** — cloud models need connectivity and send patient data off-site. Can a compact model run fully offline on cheap hardware, and does the model know when its estimate is unreliable?

## 4. Research gap vs. closest prior work

Novelty check (November 2026) found three directly related studies:

- **Hong et al. (2018)** — created the Yale dataset; LR/XGBoost/DNN on triage-only, history-only and full sets.
- **Abdulai, Storm & Ehrlich (2025)**, *Int. J. Med. Inform.* — **same Yale dataset**, XGBoost made uncertainty-aware with conformal prediction and a "Don't know" output. Tested only on a random sample of the same data; **no external-site or distribution-shift test**.
- **Dave et al. (2026)**, *JMIR Med. Inform.* — admission prediction at triage at 2 McGill hospitals with calibration and explainability; stratified random split.

So "conformal prediction on Yale" and "calibrated, explainable admission models" **are already published**. EdgeTriage must not claim these as new. Its novelty is what none of them test:

| Question | Hong 2018 | Abdulai 2025 | Dave 2026 | **EdgeTriage** |
|----------|-----------|--------------|-----------|----------------|
| Nurse-only tier (no ESI, no EHR history) | — | — | — | ✅ **core** |
| Value of ESI and history quantified | Partly (ESI inside triage set) | — | — | ✅ T1→T2→T3 |
| Calibration | — | — | ✅ | ✅ |
| Conformal "uncertain" output | — | ✅ | — | ✅ (builds on Abdulai) |
| **Does uncertainty survive a new hospital?** | — | — | — | ✅ **leave-one-hospital-out** |
| **How many local patients to repair it?** | — | — | — | ✅ **local-adaptation curve** |
| Missing-vitals robustness | — | — | — | ✅ |
| Decision-curve analysis | — | — | — | ✅ |
| Offline edge deployment, measured | — | — | — | ✅ |
| Usability / time-to-enter test | — | — | — | ✅ small pilot |

Hong et al. also reported that on triage-only data, Logistic Regression and XGBoost both reached ≈0.87 AUC. That suggests a small, simple model may be enough for the triage tier — which is exactly the edge-deployment hypothesis this project tests.

**The central research story in one sentence:** *A model and its uncertainty estimate built in one hospital are moved, offline, to a different and poorer-resourced one — what breaks, and how little local data fixes it?*

## 5. Aim

To design, build and evaluate an offline, explainable and uncertainty-aware admission-risk model that works with minimal triage inputs and runs on low-cost hardware, and to quantify what is lost (and gained) at each step of simplification.

## 6. Objectives

| # | Objective | Measurable output |
|---|-----------|-------------------|
| O1 | Build a leakage-free, documented dataset with three input tiers | Data dictionary, tier definitions, clean Parquet files |
| O2 | Train and compare five model families on each tier | Results table (AUC, PR-AUC, Brier, ECE) with 95% CIs |
| O3 | Test generalisation to an unseen hospital | Leave-one-hospital-out results table |
| O4 | Make outputs trustworthy: calibration, explanation, abstention | Calibration plots, SHAP analysis, conformal coverage and risk–coverage curves |
| O5 | Compress and deploy the model for offline edge inference | Size/latency/RAM/network benchmark table, working offline app |
| O6 | Stress-test robustness and fairness | Missing-vital curves, subgroup performance table |

## 7. Research questions and hypotheses

**RQ1 — Minimal inputs.** How well can hospital admission be predicted using only nurse-entered information (no ESI, no EHR history), and how much do ESI and history add?
- *H1:* The nurse-only tier retains most of the discriminative ability of the full model; the largest single gain comes from adding EHR history. (Reference point: Hong et al. report ≈0.87 AUC triage-only with ESI and ≈0.92 full.)

**RQ2 — Transfer.** Does performance and calibration hold when the model is tested on a hospital it never saw during training?
- *H2:* Discrimination (AUC) transfers reasonably, but calibration degrades and needs local recalibration.

**RQ3 — Uncertainty under hospital shift.** Abdulai et al. (2025) showed conformal "Don't know" outputs work on the Yale data under a random split. When the model is moved to a hospital it was not trained on, does conformal coverage still hold — and how many local patient visits are needed to restore calibration and coverage?
- *H3a:* Coverage and calibration fall below target at an unseen hospital.
- *H3b:* Recalibrating only the calibrator and conformal threshold with a small local sample (hundreds to a few thousand visits, no retraining) restores most of the lost coverage.

**RQ4 — Edge efficiency.** What is the trade-off between predictive performance and deployment cost (size, latency, memory, network) as the model is compressed, and can it run with zero network traffic on low-end hardware?
- *H4:* A compact model under 1 MB loses only a small amount of AUC versus the full model and predicts in milliseconds on a constrained CPU.

**Secondary analyses:** robustness to missing vital signs; performance across age group, sex and hospital site.

## 8. Expected contributions

1. A **three-tier benchmark** quantifying admission-prediction performance from nurse-only inputs up to full EHR history, including a no-ESI tier relevant to settings without formal triage scales.
2. A **cross-hospital evaluation** (leave-one-hospital-out) of discrimination and calibration across three different ED types.
3. The first test (to our knowledge) of **conformal uncertainty for ED admission prediction under hospital shift**, extending Abdulai et al. (2025), plus a **local-adaptation curve** showing how many local visits a new hospital needs to repair calibration and coverage without retraining.
4. A **measured edge-deployment study**: performance vs. size/latency/memory, with verified zero-network inference.
5. An **open, reproducible prototype** (code + offline app) suitable for a live demonstration, with a small usability pilot measuring time-to-enter and System Usability Scale (SUS) score.

## 9. Scope

**In scope**
- Binary prediction: admitted vs. discharged, at initial triage.
- Structured inputs: demographics, vitals, arrival mode/time, coded chief complaint, ESI (Tier 2+), EHR history (Tier 3).
- Five classical ML model families, calibration, conformal prediction, SHAP.
- Offline local app (Streamlit) and, as a stretch goal, an installable browser app (PWA).

**Out of scope**
- ESI or acuity-level prediction (different outcome).
- Diagnosis, treatment recommendation, chatbots or LLM-based dialogue.
- Real-time integration with any hospital system, or any use on real patients.
- Clinical validation on Indian data (proposed as future work).

## 10. Why this is a strong M.Tech project

- **Real data at scale:** 560k real de-identified visits, not a toy or synthetic set.
- **Clear research questions** with a published baseline to compare against.
- **Engineering depth:** compression, ONNX export, resource-limited benchmarking, offline app.
- **Trustworthy-AI depth:** calibration, conformal prediction, SHAP, fairness, decision curves.
- **Memorable demo:** switch Wi-Fi off, enter a patient, get a risk estimate, explanation and uncertainty flag in milliseconds.
- **Feasible in 12 weeks** for one student: tabular ML, no GPU required.
