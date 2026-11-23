# 03 · Methodology

## 1. End-to-end pipeline

```
Yale ED dataset (RData, 560k × 972)
        │
        ▼
[A] Load → Parquet → data dictionary → leakage audit
        │
        ▼
[B] Build tiers T1 / T2 / T3 / T3-lite  +  splits S1 / S2 / S3
        │
        ▼
[C] Train 5 model families × tiers  (Optuna tuning on validation)
        │
        ▼
[D] Calibrate (Platt / isotonic)  →  choose thresholds & risk bands
        │
        ▼
[E] Conformal prediction  →  "Low / High / Uncertain" output
        │
        ▼
[F] Explain (TreeSHAP global + local)  →  SHAP-based feature reduction
        │
        ▼
[G] Compress + export (ONNX / m2cgen)  →  edge benchmark
        │
        ▼
[H] Robustness (missing vitals) + fairness + decision curves
        │
        ▼
[I] Offline EdgeTriage app  →  live demo
```

## 2. Models

| Model | Why it is included | Key hyper-parameters to tune |
|-------|--------------------|------------------------------|
| Logistic Regression (L2 / L1) | Transparent baseline; matched XGBoost on triage-only data in the original study; tiny edge artifact | `C`, penalty, class weight |
| Random Forest | Classic non-linear baseline | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| XGBoost | Strong tabular learner; original study's best | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `scale_pos_weight` |
| LightGBM | Fast training, small models, native categoricals — likely best size/accuracy trade-off | `num_leaves`, `max_depth`, `learning_rate`, `n_estimators`, `min_data_in_leaf` |
| CatBoost | Handles high-cardinality categoricals (chief complaint) well | `depth`, `iterations`, `learning_rate`, `l2_leaf_reg` |
| *EBM (optional)* | Glass-box model: exact per-feature contributions, easy to run in a browser | `max_bins`, `interactions` |

**Tuning:** Optuna, 30–50 trials per model per tier on the validation set, objective = validation log-loss (rewards both ranking and calibration). Early stopping for boosting models. Fix seeds and log everything in MLflow (or a CSV log).

**Training budget tip:** tune on T1/T2 with full data (few columns, fast); for T3, tune on a 20% training subsample, then retrain the best config on all training data.

## 3. Calibration

A risk percentage shown to a nurse must mean what it says: among patients shown "70%", about 70% should be admitted.

1. Plot reliability diagrams (10 bins) for every model on the validation set.
2. Fit **Platt scaling** and **isotonic regression** on calibration split S3.
3. Report Brier score and Expected Calibration Error (ECE) before/after.
4. In the leave-one-hospital-out experiment, test **local recalibration**: refit only the calibrator on a small sample (e.g., 1,000 visits) from the new hospital and measure how much calibration recovers. This simulates "deploy in a new hospital with a little local data".

## 4. Thresholds and risk bands

Pick operating points on the **validation** set only:

| Band | Rule (example — final values set from validation data) |
|------|---------------------------------------------------------|
| Low | calibrated p < t_low, chosen so NPV ≥ 95% |
| Medium | t_low ≤ p < t_high |
| High | p ≥ t_high, chosen so PPV ≥ 80% |
| Uncertain | conformal set = {admit, discharge} (see §5) |

Also report sensitivity at 90% specificity and specificity at 90% sensitivity for comparability.

## 5. Uncertainty: split conformal prediction

Conformal prediction turns any classifier into one that outputs a **set** of labels with a guaranteed coverage level, without distributional assumptions (only exchangeability).

Procedure (binary, per-class / Mondrian variant to protect the minority class):
1. On calibration split S3, compute non-conformity scores `s = 1 − p̂(true class)`.
2. For error rate α (e.g., 0.10), compute the per-class quantile q̂.
3. For a new patient, include each label whose score ≤ q̂.
4. Output:
   - `{admit}` → "Likely admission"
   - `{discharge}` → "Likely discharge"
   - `{admit, discharge}` → **"Uncertain — clinician review"**
   - `{}` (rare) → treat as uncertain

Evaluate:
- **Empirical coverage** on the test set vs. the target 1 − α (should match).
- **Abstention rate** (share of "uncertain" outputs) at α = 0.05, 0.10, 0.20.
- **Risk–coverage curve**: error rate on answered cases vs. fraction answered.
- **Who gets flagged**: are uncertain cases concentrated in certain complaints, ages or sites?
- **Conformal under shift (core novelty):** coverage when calibrated on 2 hospitals and tested on the 3rd.

### 5.1 Local-adaptation curve (main new experiment)

Prior work (Abdulai et al., 2025) applied conformal prediction to this dataset with a random split only. EdgeTriage asks what happens at a new hospital and how cheaply it can be repaired.

For each held-out hospital H:
1. Train model + calibrator + conformal thresholds on the other two hospitals.
2. Draw a small "local" sample of n visits from H: n ∈ {0, 100, 250, 500, 1000, 2000, 5000}. The rest of H is the test set.
3. Using only that local sample, refit (a) the calibrator and (b) the conformal thresholds. **The model itself is never retrained** — this is what a hospital with no ML team and no internet could realistically do.
4. Measure ECE, calibration-in-the-large, conformal coverage and abstention rate on H's test portion.
5. Repeat each n with 20 random draws; plot mean ± SD against n.

Comparison arms: (i) no adaptation, (ii) local recalibration only, (iii) local recalibration + local conformal, (iv) full retrain on 2 sites + local data (upper reference).

Output: a curve answering *"A new hospital needs about N labelled visits to get trustworthy probabilities and uncertainty."* Also check for **prevalence shift**: admission rates likely differ between the trauma centre and the free-standing ED, which affects calibration more than ranking.

Tools: ~20 lines of NumPy (recommended — easy to explain in the viva) or the `MAPIE` library.

## 6. Explainability

- **Global:** TreeSHAP mean |SHAP| bar chart and beeswarm per tier; compare top features with the original study (ESI, medications, prior usage, demographics).
- **Local:** per-patient waterfall plot showing which inputs pushed risk up or down.
- **Stability:** compare top-10 features across random seeds and across sites (rank correlation).
- **Glass-box comparison (optional):** compare EBM/LR contributions with SHAP of the boosted model on the same patients (agreement on top-3 features).
- **Wording rule in app and report:** "features associated with higher model-estimated risk", never "causes".

## 7. Compression for edge deployment

| Variant | How it is built |
|---------|-----------------|
| V0 Full | Best tuned model on the tier, no size limit |
| V1 Feature-reduced | Retrain on top-k features by mean |SHAP| (k = 10, 15, 20, 30, 50) |
| V2 Shallow | V1 with constrained `max_depth` (3–4) and fewer trees (50–200) |
| V3 Quantised / exported | V2 exported to ONNX (`onnxmltools`/`skl2onnx`) or compiled to dependency-free code (`m2cgen` → Python/JS/C) |
| V4 Linear | Calibrated Logistic Regression on the same top-k features |

Plot **AUC vs. model size** and **AUC vs. p95 latency** for all variants (Pareto front). The selected edge model is the smallest one within a pre-declared tolerance of V0 (e.g., ΔAUC ≤ 0.01).

## 8. Robustness: missing vital signs

In low-resource settings, some vitals are often not recorded (no pulse oximeter, no thermometer reading).
1. On the test set, mask each vital individually, then 1, 2, 3 random vitals at 10/30/50% of rows.
2. Compare three strategies: native NaN handling (boosting), median imputation, missing-indicator features.
3. Report AUC and ECE drop per scenario; highlight which single vital is most costly to lose.
4. In the app, allow "not measured" for each vital and show a note when inputs are incomplete.

## 9. Fairness and subgroup analysis

Report AUC, sensitivity, PPV and calibration-in-the-large for:
- Age bands (18–39, 40–64, 65–79, 80+)
- Sex
- Hospital site
- Audit-only attributes (race, insurance) — **not model inputs**, only to check for performance gaps

Flag any subgroup with AUC more than 0.05 below overall, or with calibration-in-the-large far from 0.

## 10. Clinical usefulness: decision-curve analysis

Plot **net benefit** across threshold probabilities (0.1–0.7) for: the T1 model, T2, T3, "treat all as admit", and "treat none". This shows over which range of thresholds the model is actually useful for planning, which AUC alone cannot show.

## 11. Usability pilot (small, but closes an untested claim)

The app claims Tier-1 entry in under 60 seconds. Test it:
- 5–8 participants (nurses, interns or final-year medical/nursing students; classmates only as last resort).
- Each enters 5 written synthetic patient cases into the app; record time-to-enter per case and input errors.
- Then fill the 10-item **System Usability Scale** (SUS) and answer 3 open questions (Was the explanation understandable? Would the "uncertain" flag change what you do? What is missing?).
- No real patient data is involved. Check with your department whether a consent form or ethics note is needed for a usability survey.

## 12. Statistical rigour

- 95% confidence intervals for every metric via 1,000 bootstrap resamples of the test set.
- Paired comparison of AUCs between tiers/models with **DeLong's test** (or paired bootstrap).
- Repeat key experiments with 3 seeds; report mean ± SD.
- Follow the **TRIPOD+AI** reporting checklist in the dissertation.

## 13. Full experiment matrix

| ID | Experiment | Tiers | Split | Answers |
|----|------------|-------|-------|---------|
| E1 | Baseline reproduction (LR + XGBoost) | T2, T3 | S1 | Sanity check vs Hong et al. |
| E2 | 5-model comparison | T1, T2, T3 | S1 | RQ1 |
| E3 | Value of ESI and history (ΔAUC, DeLong) | T1→T2→T3 | S1 | RQ1 |
| E4 | Calibration before/after | Best per tier | S1 + S3 | RQ1, RQ3 |
| E5 | Leave-one-hospital-out (+ local recalibration) | T1, T2 | S2 | RQ2 |
| E6 | Conformal coverage + abstention (random split, replicates Abdulai 2025 setting) and under hospital shift; **local-adaptation curve** | Best T1, T2 | S1, S2 | RQ3 |
| E7 | SHAP global/local + stability | All | S1 | Explainability |
| E8 | Compression Pareto front | T1, T3-lite | S1 | RQ4 |
| E9 | Edge benchmark (size/latency/RAM/network) | Selected models | — | RQ4 |
| E10 | Missing-vitals robustness | T1, T2 | S1 | Secondary |
| E11 | Subgroup / fairness | Best T1, T2 | S1, S2 | Secondary |
| E12 | Decision-curve analysis | T1, T2, T3 | S1 | Clinical usefulness |
| E13 | Usability pilot (time-to-enter, SUS) | App (T1) | — | Practicality |

**Priority if time runs short:** E1, E2, E3, E4, E5, E6, E8, E9 are core — E5 + E6 (hospital shift and local adaptation) are the main novelty and must never be cut. E7 is required for the app. E10–E13 can be reduced in depth but should not be dropped entirely.
