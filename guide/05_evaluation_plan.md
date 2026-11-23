# 05 · Evaluation Plan

## 1. Predictive performance metrics

| Metric | What it tells you | Why it matters here |
|--------|-------------------|---------------------|
| ROC-AUC | Ranking: admitted vs discharged | Main comparison with Hong et al. |
| PR-AUC | Performance on the minority class (~30% admitted) | Accuracy is misleading under imbalance |
| Brier score | Overall probability error | Rewards good calibration |
| ECE + reliability diagram | Do probabilities mean what they say? | App shows a percentage |
| Calibration slope / intercept | Over/under-confidence, shift | Key for leave-one-hospital-out |
| Sensitivity, specificity, PPV, NPV | Behaviour at chosen thresholds | Clinical interpretation |
| Sens @ 90% spec, Spec @ 90% sens | Fixed operating points | Fair comparison across models |
| F1 | Precision–recall balance | Reported for completeness |
| Accuracy | Overall correctness | Reported, never used alone |
| Net benefit (decision curve) | Usefulness across thresholds | Clinical value beyond AUC |

## 2. Uncertainty metrics (RQ3)

| Metric | Target / expectation |
|--------|---------------------|
| Empirical coverage | ≈ 1 − α on in-distribution test (e.g., ≥ 0.89 for α = 0.10) |
| Per-class coverage | Close to target for both admitted and discharged |
| Abstention rate | Reported for α = 0.05, 0.10, 0.20 |
| Error on answered cases | Lower than error on all cases |
| Risk–coverage AURC | Lower is better |
| Coverage under hospital shift | Reported honestly; expected to drop |

## 3. Edge / deployment metrics (RQ4)

| Metric | How to measure | Project target |
|--------|----------------|----------------|
| Artifact size | File size of model bundle | ≤ 1 MB (T1 edge model) |
| Single-prediction latency | 1,000 single-row predictions; p50 and p95 with `perf_counter` | p95 ≤ 50 ms on constrained CPU |
| Cold-start time | App launch → first prediction | Report |
| Peak RAM | `psutil` / `tracemalloc` during inference | Report; ≤ 300 MB app total target |
| Network bytes per prediction | `tcpdump` on the interface, or DevTools Network tab (PWA) | **0** |
| Works with no network | Docker `--network none`, airplane mode | Yes / No |
| Install size | Docker image or PWA cache size | Report |

Test hardware: (a) your normal laptop, (b) the same laptop in Docker limited to 2 CPUs / 2 GB RAM, (c) a phone browser (L3) or Raspberry Pi if available.

*Targets are this project's own design goals, not published values.*

## 4. Success criteria (decide and record before opening the test set)

| Criterion | Pass condition |
|-----------|----------------|
| Reproduction | T2/T3 XGBoost AUC within ~0.02 of Hong et al.'s reported values on S1 |
| Tier-1 usefulness | T1 best model AUC with 95% CI reported; net benefit > "treat all/none" over a meaningful threshold range |
| Calibration | Post-calibration ECE ≤ 0.05 on S1 test |
| Conformal | Coverage within ±0.02 of target on S1 test |
| Compression | Edge model ΔAUC ≤ 0.01 vs V0 at ≥ 10× smaller size |
| Offline | Full demo works with networking disabled; 0 bytes measured |
| Local adaptation | Smallest n at which coverage returns within ±0.02 of target is identified for each held-out hospital (a negative result — "never" — is also a valid finding) |
| Usability | Median Tier-1 entry time and SUS reported for ≥ 5 participants |

## 5. Result tables to fill in

### Table A — Model comparison (S1 test, 95% CI)

| Tier | Model | ROC-AUC | PR-AUC | Brier | ECE | Sens@90Spec |
|------|-------|---------|--------|-------|-----|-------------|
| T1 | LR | | | | | |
| T1 | RF | | | | | |
| T1 | XGBoost | | | | | |
| T1 | LightGBM | | | | | |
| T1 | CatBoost | | | | | |
| T2 | … | | | | | |
| T3 | … | | | | | |
| Ref | Hong et al. XGB triage / full | 0.87 / 0.92 | — | — | — | — |

### Table B — Value of information (best model per tier)

| Comparison | ΔAUC | 95% CI | DeLong p |
|------------|------|--------|----------|
| T1 → T2 (add ESI) | | | |
| T2 → T3 (add history) | | | |
| T1 → T3 | | | |

### Table C — Leave-one-hospital-out

| Test site | Tier | AUC | Calib. slope | ECE (before) | ECE (after local recal.) | Conformal coverage |
|-----------|------|-----|--------------|--------------|--------------------------|--------------------|
| Academic | T1 | | | | | |
| Community | T1 | | | | | |
| Free-standing | T1 | | | | | |

### Table C2 — Local-adaptation curve (per held-out hospital, mean ± SD over 20 draws)

| Local visits n | ECE (recal only) | Coverage (recal + local conformal) | Abstention % | AUC (unchanged) |
|----------------|------------------|-----------------------------------|--------------|-----------------|
| 0 | | | | |
| 100 | | | | |
| 250 | | | | |
| 500 | | | | |
| 1000 | | | | |
| 2000 | | | | |
| 5000 | | | | |

### Table D — Conformal prediction (random split; compare with Abdulai et al. 2025)

| α | Coverage | Abstention % | Error (answered) | Error (all) |
|---|----------|--------------|------------------|-------------|
| 0.05 | | | | |
| 0.10 | | | | |
| 0.20 | | | | |

### Table E — Compression / edge benchmark

| Variant | # features | Size (KB) | p50 ms | p95 ms | Peak RAM (MB) | AUC | Net bytes |
|---------|-----------|-----------|--------|--------|---------------|-----|-----------|
| V0 Full | | | | | | | |
| V1 top-20 | | | | | | | |
| V2 shallow | | | | | | | |
| V3 ONNX | | | | | | | |
| V4 LR | | | | | | | |

### Table F — Missing vitals

| Missing | Strategy | AUC | ΔAUC | ECE |
|---------|----------|-----|------|-----|
| SpO₂ only | Native NaN | | | |
| 2 random vitals @ 30% rows | Median impute | | | |
| … | | | | |

### Table G — Subgroups

| Group | n | Admit % | AUC | Sens | PPV | Calib-in-the-large |
|-------|---|---------|-----|------|-----|--------------------|
| Age 18–39 | | | | | | |
| Age 80+ | | | | | | |
| Female / Male | | | | | | |

### Table H — Usability pilot

| Participant role | Mean time-to-enter (s) | Input errors | SUS score |
|------------------|------------------------|--------------|-----------|
| | | | |
| **Mean** | | | (SUS > 68 is usually read as above average) |

## 6. Required figures

1. Pipeline diagram
2. Admission rate by ESI / age / site
3. ROC and PR curves per tier (best models)
4. Reliability diagrams before/after calibration
5. SHAP global beeswarm (T1 and T3)
6. Example SHAP waterfall (synthetic patient)
7. Decision curves (T1, T2, T3, treat-all, treat-none)
8. Risk–coverage curve, and **local-adaptation curve** (coverage and ECE vs. number of local visits) — the signature figure of the project
9. AUC vs. model size Pareto plot
10. Missing-vitals degradation plot
11. Leave-one-hospital-out calibration plots
12. App screenshots (4 screens)

## 7. Statistical procedure

- Bootstrap 1,000× on the test set for all CIs.
- DeLong test (or paired bootstrap) for AUC differences.
- 3 seeds for model training; report mean ± SD for the main table.
- No metric is computed on the test set until all choices are frozen (log the date).
