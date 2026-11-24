# FirstLook — final results

S1 test set opened once on **2026-11-24** after all choices were frozen (`docs/frozen_choices.md`). Every number below is read from files written by the pipeline (`results/tables/`). 95% CIs: 1,000 bootstrap resamples of the test set (n = 56,049).

## Pre-registered success criteria

| criterion                                                   | result                                                                                                 | pass                  |
|:------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:----------------------|
| Reproduction (T2, T3 within 0.02 of 0.87 / 0.92)            | T2 0.874, T3 0.928                                                                                     | ✅ pass                |
| Tier-1 usefulness (net benefit > treat-all/none, 0.10–0.70) | T1 AUC 0.856 (0.852–0.859)                                                                             | ✅ pass                |
| Calibration (ECE ≤ 0.05, selected models)                   | T1 0.005, T2 0.003, T3 0.003                                                                           | ✅ pass                |
| Conformal (coverage within ±0.02 of target)                 | T1 α=0.05: 0.954, T1 α=0.1: 0.903, T1 α=0.2: 0.812, T2 α=0.05: 0.951, T2 α=0.1: 0.900, T2 α=0.2: 0.798 | ✅ pass                |
| Compression (edge ΔAUC ≤ 0.01 at ≥ 10× smaller)             | T1 LR ΔAUC 0.0129 (×1105 smaller); T3-lite ΔAUC 0.0092 (×58)                                           | T1 FAIL, T3-lite pass |
| Offline (0 bytes, works with networking disabled)           | Docker --network none: 0 bytes; app sockets audited (D28)                                              | ✅ pass                |
| Local adaptation (smallest n identified per site)           | T1 A: n*=500; T1 B: n*=1000; T1 C: n*=1000; T2 A: n*=500; T2 B: n*=250; T2 C: n*=1000                  | ✅ pass                |
| Usability (≥ 5 participants)                                | not run (needs humans)                                                                                 | NOT RUN               |

## Table A — Model comparison (S1 test, 95% CI)

| tier   | model               | selected   | calibrator   | ROC-AUC             | PR-AUC              | Brier               | ECE                 | Sens@90Spec         |
|:-------|:--------------------|:-----------|:-------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|
| T1     | Logistic Regression | False      | isotonic     | 0.843 (0.839–0.846) | 0.685 (0.678–0.693) | 0.143 (0.141–0.144) | 0.005 (0.004–0.009) | 0.502 (0.495–0.540) |
| T1     | Random Forest       | False      | platt        | 0.840 (0.837–0.844) | 0.692 (0.684–0.700) | 0.143 (0.141–0.145) | 0.006 (0.005–0.010) | 0.530 (0.520–0.541) |
| T1     | XGBoost             | False      | isotonic     | 0.855 (0.852–0.859) | 0.709 (0.702–0.716) | 0.137 (0.135–0.139) | 0.006 (0.004–0.009) | 0.522 (0.515–0.531) |
| T1     | LightGBM            | True       | isotonic     | 0.856 (0.852–0.859) | 0.709 (0.702–0.716) | 0.137 (0.135–0.139) | 0.005 (0.004–0.009) | 0.547 (0.540–0.555) |
| T1     | CatBoost            | False      | isotonic     | 0.856 (0.852–0.859) | 0.709 (0.702–0.716) | 0.137 (0.135–0.139) | 0.005 (0.004–0.009) | 0.558 (0.525–0.569) |
| T2     | Logistic Regression | False      | isotonic     | 0.866 (0.863–0.869) | 0.719 (0.712–0.725) | 0.133 (0.132–0.135) | 0.006 (0.004–0.009) | 0.579 (0.566–0.585) |
| T2     | Random Forest       | False      | isotonic     | 0.867 (0.864–0.870) | 0.719 (0.712–0.726) | 0.133 (0.131–0.135) | 0.006 (0.005–0.009) | 0.571 (0.535–0.578) |
| T2     | XGBoost             | False      | platt        | 0.874 (0.871–0.877) | 0.741 (0.735–0.748) | 0.129 (0.128–0.131) | 0.004 (0.003–0.008) | 0.596 (0.586–0.605) |
| T2     | LightGBM            | True       | platt        | 0.874 (0.871–0.877) | 0.742 (0.735–0.749) | 0.129 (0.128–0.131) | 0.003 (0.003–0.007) | 0.596 (0.587–0.606) |
| T2     | CatBoost            | False      | platt        | 0.874 (0.871–0.877) | 0.741 (0.735–0.748) | 0.129 (0.128–0.131) | 0.003 (0.003–0.007) | 0.597 (0.588–0.607) |
| T3     | Logistic Regression | False      | isotonic     | 0.918 (0.916–0.921) | 0.844 (0.839–0.849) | 0.100 (0.098–0.102) | 0.004 (0.003–0.007) | 0.738 (0.731–0.756) |
| T3     | Random Forest       | False      | platt        | 0.917 (0.915–0.920) | 0.846 (0.841–0.851) | 0.100 (0.099–0.102) | 0.004 (0.003–0.007) | 0.755 (0.747–0.763) |
| T3     | XGBoost             | False      | platt        | 0.927 (0.925–0.930) | 0.867 (0.863–0.872) | 0.093 (0.092–0.095) | 0.003 (0.003–0.007) | 0.780 (0.772–0.788) |
| T3     | LightGBM            | False      | platt        | 0.928 (0.925–0.930) | 0.867 (0.863–0.872) | 0.093 (0.092–0.095) | 0.003 (0.003–0.006) | 0.780 (0.773–0.788) |
| T3     | CatBoost            | True       | platt        | 0.928 (0.925–0.930) | 0.868 (0.864–0.873) | 0.093 (0.092–0.095) | 0.003 (0.003–0.006) | 0.782 (0.774–0.789) |
| Ref | Hong et al. XGB triage / full | 0.87 / 0.92 | — | — | — | — |


Selected = model chosen per tier on validation log-loss (D18). Optuna trials per study are in `results/logs/best_*.json` (T3 XGBoost: 6 trials within its time cap, D24).

## Table B — Value of information (selected model per tier, S1 test)

| comparison   |   AUC from |   AUC to |   ΔAUC | 95% CI           |   DeLong p |
|:-------------|-----------:|---------:|-------:|:-----------------|-----------:|
| T1 → T2      |     0.8555 |   0.8743 | 0.0188 | 0.0172 to 0.0202 |  3.87e-137 |
| T2 → T3      |     0.8743 |   0.9277 | 0.0535 | 0.0514 to 0.0556 |  0         |
| T1 → T3      |     0.8555 |   0.9277 | 0.0722 | 0.0697 to 0.0748 |  0         |

## Table C — Leave-one-hospital-out (S2; S1 test rows excluded)

| held-out site               | tier   |   prevalence |   in-dist AUC (source val) |   AUC |   calib slope |   CITL |   ECE before |   ECE after local recal (n=1000) |   coverage (source conformal) |   coverage admitted |   coverage discharged |   coverage after local conformal (n=1000) |   coverage admitted after (n=1000) |
|:----------------------------|:-------|-------------:|---------------------------:|------:|--------------:|-------:|-------------:|---------------------------------:|------------------------------:|--------------------:|----------------------:|------------------------------------------:|-----------------------------------:|
| A (academic, inferred)      | T1     |        0.319 |                      0.868 | 0.822 |         0.814 |  0.369 |        0.051 |                            0.015 |                         0.9   |               0.866 |                 0.916 |                                     0.904 |                              0.904 |
| B (community, inferred)     | T1     |        0.317 |                      0.843 | 0.869 |         1.086 |  0.038 |        0.016 |                            0.018 |                         0.898 |               0.928 |                 0.885 |                                     0.903 |                              0.903 |
| C (free-standing, inferred) | T1     |        0.156 |                      0.859 | 0.825 |         0.932 | -1.034 |        0.128 |                            0.013 |                         0.895 |               0.912 |                 0.892 |                                     0.9   |                              0.906 |
| A (academic, inferred)      | T2     |        0.319 |                      0.887 | 0.856 |         0.872 |  0.047 |        0.022 |                            0.015 |                         0.875 |               0.886 |                 0.87  |                                     0.903 |                              0.901 |
| B (community, inferred)     | T2     |        0.317 |                      0.866 | 0.886 |         1.099 |  0.236 |        0.033 |                            0.014 |                         0.911 |               0.911 |                 0.911 |                                     0.904 |                              0.905 |
| C (free-standing, inferred) | T2     |        0.156 |                      0.875 | 0.852 |         0.901 | -0.586 |        0.062 |                            0.012 |                         0.917 |               0.855 |                 0.928 |                                     0.901 |                              0.906 |

Site types are inferred from volume and admission rate (D8). 'ECE after local recal' and 'coverage after local conformal' are means over 20 draws of 1,000 local visits.

## Table C2 — Local-adaptation curve (mean ± SD over 20 draws; model never retrained)

| tier   | site   |    n | ECE (recal only)   | coverage (recal + local conformal)   | coverage admitted   | coverage discharged   | abstention %   | AUC (unchanged model)   |
|:-------|:-------|-----:|:-------------------|:-------------------------------------|:--------------------|:----------------------|:---------------|:------------------------|
| T1     | A      |    0 | 0.051              | 0.900                                | 0.866               | 0.916                 | 35.2           | 0.822                   |
| T1     | A      |  100 | 0.042 ± 0.024      | 0.940 ± 0.021                        | 0.956 ± 0.033       | 0.932 ± 0.025         | 58.8 ± 10.7    | 0.822 ± 0.000           |
| T1     | A      |  250 | 0.024 ± 0.012      | 0.920 ± 0.014                        | 0.929 ± 0.018       | 0.915 ± 0.021         | 47.4 ± 4.9     | 0.822 ± 0.000           |
| T1     | A      |  500 | 0.022 ± 0.011      | 0.909 ± 0.013                        | 0.911 ± 0.023       | 0.908 ± 0.018         | 42.3 ± 5.4     | 0.822 ± 0.000           |
| T1     | A      | 1000 | 0.015 ± 0.006      | 0.904 ± 0.009                        | 0.904 ± 0.017       | 0.904 ± 0.010         | 40.1 ± 3.6     | 0.822 ± 0.000           |
| T1     | A      | 2000 | 0.012 ± 0.005      | 0.904 ± 0.007                        | 0.907 ± 0.010       | 0.902 ± 0.008         | 40.3 ± 2.3     | 0.822 ± 0.000           |
| T1     | A      | 5000 | 0.008 ± 0.002      | 0.900 ± 0.005                        | 0.900 ± 0.008       | 0.900 ± 0.006         | 38.5 ± 1.5     | 0.822 ± 0.000           |
| T1     | B      |    0 | 0.016              | 0.898                                | 0.928               | 0.885                 | 29.4           | 0.869                   |
| T1     | B      |  100 | 0.038 ± 0.016      | 0.924 ± 0.034                        | 0.954 ± 0.027       | 0.911 ± 0.042         | 42.1 ± 12.5    | 0.869 ± 0.000           |
| T1     | B      |  250 | 0.029 ± 0.010      | 0.908 ± 0.017                        | 0.914 ± 0.034       | 0.905 ± 0.025         | 30.7 ± 5.6     | 0.869 ± 0.000           |
| T1     | B      |  500 | 0.022 ± 0.007      | 0.912 ± 0.010                        | 0.921 ± 0.017       | 0.908 ± 0.013         | 31.8 ± 3.6     | 0.869 ± 0.000           |
| T1     | B      | 1000 | 0.018 ± 0.003      | 0.903 ± 0.009                        | 0.903 ± 0.015       | 0.902 ± 0.011         | 27.8 ± 2.8     | 0.869 ± 0.000           |
| T1     | B      | 2000 | 0.017 ± 0.004      | 0.903 ± 0.010                        | 0.908 ± 0.011       | 0.900 ± 0.012         | 28.2 ± 2.9     | 0.869 ± 0.000           |
| T1     | B      | 5000 | 0.015 ± 0.001      | 0.901 ± 0.004                        | 0.902 ± 0.006       | 0.901 ± 0.004         | 27.2 ± 1.1     | 0.869 ± 0.000           |
| T1     | C      |    0 | 0.128              | 0.895                                | 0.912               | 0.892                 | 38.0           | 0.825                   |
| T1     | C      |  100 | 0.032 ± 0.015      | 0.905 ± 0.026                        | 0.936 ± 0.068       | 0.899 ± 0.027         | 49.0 ± 15.3    | 0.825 ± 0.000           |
| T1     | C      |  250 | 0.025 ± 0.012      | 0.914 ± 0.018                        | 0.930 ± 0.041       | 0.911 ± 0.018         | 47.4 ± 13.4    | 0.825 ± 0.000           |
| T1     | C      |  500 | 0.017 ± 0.010      | 0.905 ± 0.014                        | 0.928 ± 0.025       | 0.901 ± 0.015         | 43.3 ± 6.3     | 0.825 ± 0.000           |
| T1     | C      | 1000 | 0.013 ± 0.005      | 0.900 ± 0.009                        | 0.906 ± 0.022       | 0.898 ± 0.012         | 38.0 ± 3.5     | 0.825 ± 0.000           |
| T1     | C      | 2000 | 0.011 ± 0.003      | 0.904 ± 0.009                        | 0.907 ± 0.020       | 0.903 ± 0.010         | 38.8 ± 4.0     | 0.825 ± 0.000           |
| T1     | C      | 5000 | 0.008 ± 0.001      | 0.899 ± 0.003                        | 0.904 ± 0.011       | 0.898 ± 0.003         | 37.4 ± 2.0     | 0.825 ± 0.001           |
| T2     | A      |    0 | 0.022              | 0.875                                | 0.886               | 0.870                 | 24.5           | 0.856                   |
| T2     | A      |  100 | 0.044 ± 0.024      | 0.939 ± 0.026                        | 0.959 ± 0.039       | 0.929 ± 0.034         | 50.3 ± 11.8    | 0.856 ± 0.000           |
| T2     | A      |  250 | 0.023 ± 0.013      | 0.915 ± 0.022                        | 0.926 ± 0.024       | 0.909 ± 0.028         | 36.9 ± 6.4     | 0.856 ± 0.000           |
| T2     | A      |  500 | 0.023 ± 0.012      | 0.908 ± 0.015                        | 0.912 ± 0.022       | 0.906 ± 0.019         | 33.6 ± 4.8     | 0.856 ± 0.000           |
| T2     | A      | 1000 | 0.015 ± 0.005      | 0.903 ± 0.009                        | 0.901 ± 0.019       | 0.903 ± 0.013         | 31.4 ± 3.0     | 0.856 ± 0.000           |
| T2     | A      | 2000 | 0.012 ± 0.005      | 0.904 ± 0.008                        | 0.909 ± 0.013       | 0.901 ± 0.009         | 32.3 ± 2.7     | 0.856 ± 0.000           |
| T2     | A      | 5000 | 0.008 ± 0.002      | 0.900 ± 0.004                        | 0.899 ± 0.008       | 0.900 ± 0.005         | 30.5 ± 1.5     | 0.856 ± 0.000           |
| T2     | B      |    0 | 0.033              | 0.911                                | 0.911               | 0.911                 | 26.3           | 0.886                   |
| T2     | B      |  100 | 0.034 ± 0.014      | 0.930 ± 0.028                        | 0.956 ± 0.024       | 0.919 ± 0.039         | 37.5 ± 9.5     | 0.886 ± 0.000           |
| T2     | B      |  250 | 0.029 ± 0.010      | 0.903 ± 0.020                        | 0.907 ± 0.031       | 0.901 ± 0.030         | 24.9 ± 5.3     | 0.886 ± 0.000           |
| T2     | B      |  500 | 0.018 ± 0.008      | 0.910 ± 0.014                        | 0.916 ± 0.023       | 0.908 ± 0.017         | 26.8 ± 4.2     | 0.886 ± 0.000           |
| T2     | B      | 1000 | 0.014 ± 0.005      | 0.904 ± 0.009                        | 0.905 ± 0.015       | 0.903 ± 0.010         | 24.4 ± 2.6     | 0.886 ± 0.000           |
| T2     | B      | 2000 | 0.011 ± 0.004      | 0.902 ± 0.008                        | 0.907 ± 0.011       | 0.899 ± 0.010         | 24.0 ± 2.2     | 0.886 ± 0.000           |
| T2     | B      | 5000 | 0.008 ± 0.002      | 0.900 ± 0.003                        | 0.900 ± 0.006       | 0.901 ± 0.005         | 23.3 ± 0.9     | 0.886 ± 0.000           |
| T2     | C      |    0 | 0.062              | 0.917                                | 0.855               | 0.928                 | 26.9           | 0.852                   |
| T2     | C      |  100 | 0.027 ± 0.013      | 0.913 ± 0.024                        | 0.931 ± 0.064       | 0.909 ± 0.025         | 39.9 ± 13.3    | 0.852 ± 0.000           |
| T2     | C      |  250 | 0.024 ± 0.011      | 0.915 ± 0.017                        | 0.930 ± 0.045       | 0.912 ± 0.016         | 39.1 ± 13.0    | 0.852 ± 0.000           |
| T2     | C      |  500 | 0.016 ± 0.011      | 0.908 ± 0.012                        | 0.921 ± 0.032       | 0.905 ± 0.015         | 33.5 ± 5.0     | 0.852 ± 0.000           |
| T2     | C      | 1000 | 0.012 ± 0.006      | 0.901 ± 0.009                        | 0.906 ± 0.021       | 0.900 ± 0.011         | 30.0 ± 3.3     | 0.852 ± 0.000           |
| T2     | C      | 2000 | 0.008 ± 0.004      | 0.904 ± 0.009                        | 0.907 ± 0.016       | 0.904 ± 0.009         | 30.5 ± 2.9     | 0.852 ± 0.000           |
| T2     | C      | 5000 | 0.006 ± 0.002      | 0.899 ± 0.004                        | 0.901 ± 0.010       | 0.899 ± 0.004         | 28.9 ± 1.4     | 0.852 ± 0.001           |

### Smallest number of local visits that restores the target

| tier   | site   |   n* marginal coverage within ±0.02 |   n* admitted-class coverage within ±0.02 |   n* discharged-class coverage within ±0.02 |   n* ECE ≤ 0.02 (recal only) |   coverage at n=0 |   admitted coverage at n=0 |
|:-------|:-------|------------------------------------:|------------------------------------------:|--------------------------------------------:|-----------------------------:|------------------:|---------------------------:|
| T1     | A      |                                 250 |                                       500 |                                         250 |                         1000 |          0.900108 |                   0.866121 |
| T1     | B      |                                 250 |                                      1000 |                                         100 |                         1000 |          0.898183 |                   0.927683 |
| T1     | C      |                                 100 |                                      1000 |                                         100 |                          500 |          0.895034 |                   0.911659 |
| T2     | A      |                                 250 |                                       500 |                                         250 |                         1000 |          0.875284 |                   0.886421 |
| T2     | B      |                                 250 |                                       250 |                                         100 |                          500 |          0.910797 |                   0.910552 |
| T2     | C      |                                 100 |                                      1000 |                                         100 |                          500 |          0.916638 |                   0.855253 |

n* = smallest n whose mean stays within ±0.02 of 0.90 (coverage) or ≤ 0.02 (ECE) for all larger n. Reference arm (full retrain with local data) in `results/tables/E6_retrain_reference.csv`.

## Table D — Conformal prediction, random split (S1 test; compare Abdulai et al. 2025)

| tier   |   alpha |   target |   coverage |   coverage admitted |   coverage discharged |   abstention % |   error (answered) |   error (all, p≥0.5) |
|:-------|--------:|---------:|-----------:|--------------------:|----------------------:|---------------:|-------------------:|---------------------:|
| T1     |    0.05 |     0.95 |     0.9536 |              0.9508 |                0.9548 |        50.4737 |             0.0937 |               0.2018 |
| T1     |    0.1  |     0.9  |     0.9026 |              0.8985 |                0.9044 |        30.8373 |             0.1408 |               0.2018 |
| T1     |    0.2  |     0.8  |     0.8116 |              0.8228 |                0.8069 |         8.7994 |             0.2065 |               0.2018 |
| T2     |    0.05 |     0.95 |     0.951  |              0.9504 |                0.9513 |        42.8803 |             0.0857 |               0.1904 |
| T2     |    0.1  |     0.9  |     0.8998 |              0.8974 |                0.9009 |        25.9844 |             0.1353 |               0.1904 |
| T2     |    0.2  |     0.8  |     0.7985 |              0.794  |                0.8004 |         2.3818 |             0.2065 |               0.1904 |

## Table E — Compression / edge benchmark

| tier         | variant                    |   # features |   size KB |   p50 ms (model, in-process) |   p95 ms |   peak traced RAM MB |   val AUC | test AUC (95% CI)   |
|:-------------|:---------------------------|-------------:|----------:|-----------------------------:|---------:|---------------------:|----------:|:--------------------|
| T1           | V0 full                    |          222 |   10236.4 |                        3.637 |    3.838 |                 0.52 |    0.8575 | 0.856 (0.852–0.859) |
| T1           | V1 top-222                 |          222 |   10236.4 |                        3.334 |    4.003 |                 0.52 |    0.8575 | 0.856 (0.852–0.859) |
| T1           | V4 LR top-222              |          222 |       9.3 |                        4.526 |    4.743 |                 2.57 |    0.8461 | 0.843 (0.839–0.846) |
| T3 / T3-lite | V0 full                    |          976 |   12451.7 |                       76.537 |   80.283 |                84.89 |    0.9279 | 0.928 (0.925–0.930) |
| T3 / T3-lite | V1 top-50                  |           50 |    8591.1 |                        5.019 |    5.339 |                 8.92 |    0.9215 | 0.921 (0.919–0.924) |
| T3 / T3-lite | V2 shallow d4 t200 top-50  |           50 |     359.9 |                        1.784 |    1.809 |                 0.49 |    0.9184 | 0.919 (0.916–0.921) |
| T3 / T3-lite | V3 ONNX d4 t200 top-50     |           50 |     214.9 |                        0.034 |    0.043 |                 0.59 |    0.9184 | 0.919 (0.916–0.921) |
| T3 / T3-lite | V4 LR top-50               |           50 |       3.4 |                        2.612 |    2.762 |                 1.22 |    0.9097 | 0.910 (0.908–0.913) |
| T1           | V2 shallow d4 t200 top-222 |          222 |     323.4 |                        3.472 |    3.605 |                 0.51 |    0.8412 | 0.839 (0.836–0.843) |
| T1           | V3 ONNX d4 t200 top-222    |          222 |     187.6 |                        0.033 |    0.041 |                 0.59 |    0.8412 | 0.839 (0.836–0.843) |

In-process latencies in this table are from E8 on the host (single-row `predict_proba`; ONNX rows use ONNX Runtime). The shipped bundles measured inside Docker (`--network none --cpus 2 --memory 2g`, cgroup limits verified):

| bundle   |   ONNX KB |   bundle KB |   end-to-end p50 ms |   end-to-end p95 ms |   model-only p95 ms |
|:---------|----------:|------------:|--------------------:|--------------------:|--------------------:|
| t1       |       9.3 |        59.2 |                8.73 |                9.35 |               0.053 |
| t2       |      10.2 |        61.9 |                9.01 |                9.64 |               0.052 |
| t3lite   |     214.9 |       621.8 |               26.62 |               28.03 |               0.065 |

Docker: interfaces ['lo'], peak RSS 188 MB, cold start 686 ms (3 bundles), **network bytes sent 0**, app external connections 0.

Edge choice (D19/D27): T1 → V4 LR top-222 (rule met: False); T3-lite → V3 ONNX d4 t200 top-50 (rule met: True).

Shipped bundles on the S1 test set (own calibrator + conformal thresholds):

| bundle    | AUC                 | PR-AUC              | Brier               | ECE                 |   coverage (α=0.10) |   abstention % |
|:----------|:--------------------|:--------------------|:--------------------|:--------------------|--------------------:|---------------:|
| t1_v1     | 0.843 (0.839–0.846) | 0.685 (0.678–0.692) | 0.143 (0.141–0.144) | 0.005 (0.004–0.009) |              0.9114 |           37.3 |
| t2_v1     | 0.866 (0.863–0.869) | 0.718 (0.711–0.725) | 0.133 (0.132–0.135) | 0.006 (0.004–0.009) |              0.9094 |           29.8 |
| t3lite_v1 | 0.918 (0.916–0.921) | 0.846 (0.841–0.851) | 0.099 (0.098–0.101) | 0.005 (0.004–0.008) |              0.906  |           18.6 |

## Table F — Missing vitals (S1 test; research models)

| tier   | scenario                       | strategy          |    auc |    ΔAUC |    ece |
|:-------|:-------------------------------|:------------------|-------:|--------:|-------:|
| t1     | none (as recorded)             | native NaN        | 0.8555 |  0      | 0.0051 |
| t1     | none (as recorded)             | median impute     | 0.847  |  0      | 0.0413 |
| t1     | none (as recorded)             | missing indicator | 0.8555 |  0      | 0.0071 |
| t1     | hr missing (100% rows)         | native NaN        | 0.8499 | -0.0056 | 0.0153 |
| t1     | hr missing (100% rows)         | median impute     | 0.844  | -0.003  | 0.0547 |
| t1     | hr missing (100% rows)         | missing indicator | 0.8498 | -0.0057 | 0.0147 |
| t1     | o2 missing (100% rows)         | native NaN        | 0.8542 | -0.0013 | 0.0055 |
| t1     | o2 missing (100% rows)         | median impute     | 0.8462 | -0.0008 | 0.0442 |
| t1     | o2 missing (100% rows)         | missing indicator | 0.8543 | -0.0012 | 0.0065 |
| t1     | temp missing (100% rows)       | native NaN        | 0.8535 | -0.0021 | 0.0249 |
| t1     | temp missing (100% rows)       | median impute     | 0.8469 | -0.0001 | 0.0453 |
| t1     | temp missing (100% rows)       | missing indicator | 0.8535 | -0.002  | 0.0249 |
| t1     | all vitals missing (100% rows) | native NaN        | 0.8394 | -0.0161 | 0.0313 |
| t1     | all vitals missing (100% rows) | median impute     | 0.8372 | -0.0097 | 0.0768 |
| t1     | all vitals missing (100% rows) | missing indicator | 0.8397 | -0.0158 | 0.0306 |
| t1     | 2 random vital(s) @ 30% rows   | native NaN        | 0.8536 | -0.0019 | 0.0065 |
| t1     | 2 random vital(s) @ 30% rows   | median impute     | 0.8462 | -0.0008 | 0.0443 |
| t1     | 2 random vital(s) @ 30% rows   | missing indicator | 0.8537 | -0.0018 | 0.0079 |
| t1     | 3 random vital(s) @ 50% rows   | native NaN        | 0.8506 | -0.0049 | 0.015  |
| t1     | 3 random vital(s) @ 50% rows   | median impute     | 0.8452 | -0.0018 | 0.049  |
| t1     | 3 random vital(s) @ 50% rows   | missing indicator | 0.8508 | -0.0047 | 0.0143 |
| t2     | none (as recorded)             | native NaN        | 0.8743 |  0      | 0.0032 |
| t2     | none (as recorded)             | median impute     | 0.8695 |  0      | 0.0352 |
| t2     | none (as recorded)             | missing indicator | 0.8745 |  0      | 0.0046 |
| t2     | hr missing (100% rows)         | native NaN        | 0.8712 | -0.0031 | 0.0085 |
| t2     | hr missing (100% rows)         | median impute     | 0.8682 | -0.0013 | 0.0458 |
| t2     | hr missing (100% rows)         | missing indicator | 0.8713 | -0.0032 | 0.0073 |
| t2     | o2 missing (100% rows)         | native NaN        | 0.8735 | -0.0008 | 0.0026 |
| t2     | o2 missing (100% rows)         | median impute     | 0.8691 | -0.0004 | 0.0374 |
| t2     | o2 missing (100% rows)         | missing indicator | 0.8737 | -0.0008 | 0.0029 |
| t2     | temp missing (100% rows)       | native NaN        | 0.8734 | -0.0009 | 0.0099 |
| t2     | temp missing (100% rows)       | median impute     | 0.8697 |  0.0002 | 0.0398 |
| t2     | temp missing (100% rows)       | missing indicator | 0.8736 | -0.0009 | 0.0096 |
| t2     | all vitals missing (100% rows) | native NaN        | 0.8659 | -0.0084 | 0.0179 |
| t2     | all vitals missing (100% rows) | median impute     | 0.8652 | -0.0043 | 0.0646 |
| t2     | all vitals missing (100% rows) | missing indicator | 0.866  | -0.0085 | 0.0182 |
| t2     | 2 random vital(s) @ 30% rows   | native NaN        | 0.8735 | -0.0008 | 0.0033 |
| t2     | 2 random vital(s) @ 30% rows   | median impute     | 0.8691 | -0.0004 | 0.0376 |
| t2     | 2 random vital(s) @ 30% rows   | missing indicator | 0.8736 | -0.0009 | 0.0045 |
| t2     | 3 random vital(s) @ 50% rows   | native NaN        | 0.8718 | -0.0025 | 0.0073 |
| t2     | 3 random vital(s) @ 50% rows   | median impute     | 0.8686 | -0.0009 | 0.0415 |
| t2     | 3 random vital(s) @ 50% rows   | missing indicator | 0.8719 | -0.0026 | 0.0071 |

Full grid (every vital, 1–3 vitals at 10/30/50% of rows): `results/tables/E10_missing_vitals_test.csv`.

## Table G — Subgroups (S1 test)

| tier   | attribute              | group                     |     n |   admit_rate |   auc |   sens@t_high |   ppv@t_high |   citl | flag           |
|:-------|:-----------------------|:--------------------------|------:|-------------:|------:|--------------:|-------------:|-------:|:---------------|
| t1     | overall                | all                       | 56049 |        0.297 | 0.856 |         0.383 |        0.791 |  0.009 | nan            |
| t1     | age band               | 80+                       |  5770 |        0.622 | 0.769 |         0.666 |        0.802 |  0.005 | AUC gap > 0.05 |
| t1     | age band               | 40–64                     | 21796 |        0.293 | 0.813 |         0.249 |        0.763 | -0.012 | nan            |
| t1     | age band               | 18–39                     | 20462 |        0.13  | 0.827 |         0.053 |        0.765 |  0.055 | nan            |
| t1     | age band               | 65–79                     |  8021 |        0.503 | 0.817 |         0.561 |        0.801 |  0.001 | nan            |
| t1     | sex                    | Female                    | 30966 |        0.289 | 0.855 |         0.375 |        0.786 |  0.005 | nan            |
| t1     | sex                    | Male                      | 25083 |        0.307 | 0.856 |         0.393 |        0.796 |  0.014 | nan            |
| t1     | site (S1)              | A                         | 32151 |        0.32  | 0.848 |         0.374 |        0.801 |  0.115 | nan            |
| t1     | site (S1)              | B                         | 16670 |        0.314 | 0.874 |         0.432 |        0.804 |  0.022 | nan            |
| t1     | site (S1)              | C                         |  7228 |        0.159 | 0.821 |         0.242 |        0.606 | -0.596 | |CITL| > 0.2   |
| t1     | race (audit only)      | Other                     |  8957 |        0.21  | 0.834 |         0.261 |        0.768 | -0.068 | nan            |
| t1     | race (audit only)      | Patient Refused           |   534 |        0.206 | 0.896 |         0.355 |        0.796 | -0.232 | |CITL| > 0.2   |
| t1     | race (audit only)      | White or Caucasian        | 29918 |        0.359 | 0.846 |         0.431 |        0.799 |  0.061 | nan            |
| t1     | race (audit only)      | Black or African American | 15755 |        0.237 | 0.865 |         0.314 |        0.769 | -0.06  | nan            |
| t1     | race (audit only)      | Asian                     |   625 |        0.203 | 0.854 |         0.276 |        0.778 | -0.007 | nan            |
| t1     | insurance (audit only) | Medicare                  | 12166 |        0.527 | 0.797 |         0.548 |        0.804 |  0.078 | AUC gap > 0.05 |
| t1     | insurance (audit only) | Other                     |  5081 |        0.072 | 0.86  |         0.167 |        0.565 | -0.859 | |CITL| > 0.2   |
| t1     | insurance (audit only) | Medicaid                  | 19469 |        0.21  | 0.828 |         0.187 |        0.739 | -0.005 | nan            |
| t1     | insurance (audit only) | Commercial                | 19162 |        0.294 | 0.851 |         0.357 |        0.797 |  0.046 | nan            |
| t1     | ethnicity (audit only) | Hispanic or Latino        |  9401 |        0.207 | 0.83  |         0.249 |        0.758 | -0.081 | nan            |
| t1     | ethnicity (audit only) | Non-Hispanic              | 46237 |        0.316 | 0.856 |         0.401 |        0.793 |  0.024 | nan            |
| t1     | ethnicity (audit only) | Unknown                   |   280 |        0.225 | 0.926 |         0.429 |        0.844 | -0.035 | nan            |
| t2     | overall                | all                       | 56049 |        0.297 | 0.874 |         0.431 |        0.795 |  0.011 | nan            |
| t2     | age band               | 80+                       |  5770 |        0.622 | 0.791 |         0.706 |        0.81  |  0.016 | AUC gap > 0.05 |
| t2     | age band               | 40–64                     | 21796 |        0.293 | 0.839 |         0.31  |        0.772 | -0.006 | nan            |
| t2     | age band               | 18–39                     | 20462 |        0.13  | 0.855 |         0.072 |        0.746 |  0.042 | nan            |
| t2     | age band               | 65–79                     |  8021 |        0.503 | 0.839 |         0.616 |        0.804 |  0.005 | nan            |
| t2     | sex                    | Female                    | 30966 |        0.289 | 0.873 |         0.425 |        0.796 |  0.005 | nan            |
| t2     | sex                    | Male                      | 25083 |        0.307 | 0.876 |         0.439 |        0.795 |  0.018 | nan            |
| t2     | site (S1)              | A                         | 32151 |        0.32  | 0.864 |         0.446 |        0.791 |  0.013 | nan            |
| t2     | site (S1)              | B                         | 16670 |        0.314 | 0.89  |         0.454 |        0.817 |  0.103 | nan            |
| t2     | site (S1)              | C                         |  7228 |        0.159 | 0.853 |         0.196 |        0.676 | -0.269 | |CITL| > 0.2   |
| t2     | race (audit only)      | Other                     |  8957 |        0.21  | 0.857 |         0.3   |        0.766 | -0.093 | nan            |
| t2     | race (audit only)      | Patient Refused           |   534 |        0.206 | 0.911 |         0.418 |        0.807 | -0.256 | |CITL| > 0.2   |
| t2     | race (audit only)      | White or Caucasian        | 29918 |        0.359 | 0.866 |         0.482 |        0.806 |  0.071 | nan            |
| t2     | race (audit only)      | Black or African American | 15755 |        0.237 | 0.88  |         0.356 |        0.771 | -0.057 | nan            |
| t2     | race (audit only)      | Asian                     |   625 |        0.203 | 0.883 |         0.339 |        0.768 | -0.027 | nan            |
| t2     | insurance (audit only) | Medicare                  | 12166 |        0.527 | 0.822 |         0.6   |        0.809 |  0.074 | AUC gap > 0.05 |
| t2     | insurance (audit only) | Other                     |  5081 |        0.072 | 0.895 |         0.211 |        0.57  | -0.787 | |CITL| > 0.2   |
| t2     | insurance (audit only) | Medicaid                  | 19469 |        0.21  | 0.85  |         0.222 |        0.739 | -0     | nan            |
| t2     | insurance (audit only) | Commercial                | 19162 |        0.294 | 0.871 |         0.412 |        0.805 |  0.036 | nan            |
| t2     | ethnicity (audit only) | Hispanic or Latino        |  9401 |        0.207 | 0.853 |         0.289 |        0.745 | -0.104 | nan            |
| t2     | ethnicity (audit only) | Non-Hispanic              | 46237 |        0.316 | 0.875 |         0.45  |        0.8   |  0.03  | nan            |
| t2     | ethnicity (audit only) | Unknown                   |   280 |        0.225 | 0.93  |         0.476 |        0.811 |  0.005 | nan            |

Sens/PPV at the High-band threshold. Flags: AUC > 0.05 below overall, or |calibration-in-the-large| > 0.2. Race, ethnicity and insurance are audit-only (never model inputs); groups with n < 200 omitted here (in CSV).

## Table H — Usability pilot

**Not run.** E13 requires 5–8 human participants (timed entry + SUS). The protocol is in `guide/03_methodology.md` §11; the app and synthetic cases are ready for it.

## Figures

- `results/figures/fig01_pipeline.png`
- `results/figures/fig02_admission_rate_by_esi_age_site.png`
- `results/figures/fig03_roc_pr_test.png`
- `results/figures/fig04_reliability_val.png`
- `results/figures/fig04b_reliability_test.png`
- `results/figures/fig05_shap_beeswarm_t1.png`
- `results/figures/fig05_shap_beeswarm_t2.png`
- `results/figures/fig05_shap_beeswarm_t3.png`
- `results/figures/fig05b_shap_bar_t1.png`
- `results/figures/fig05b_shap_bar_t2.png`
- `results/figures/fig05b_shap_bar_t3.png`
- `results/figures/fig06_local_explanation_synthetic.png`
- `results/figures/fig07_decision_curves_test.png`
- `results/figures/fig07_decision_curves_val.png`
- `results/figures/fig08_local_adaptation_t1.png`
- `results/figures/fig08_local_adaptation_t2.png`
- `results/figures/fig08b_risk_coverage_test.png`
- `results/figures/fig09_pareto_auc_size.png`
- `results/figures/fig10_missing_vitals_test.png`
- `results/figures/fig10_missing_vitals_val.png`
- `results/figures/fig11_loho_calibration.png`
- `results/figures/fig_eda_vital_missingness.png`
- App screenshots: `results/figures/app/`
