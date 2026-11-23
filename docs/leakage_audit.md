# Leakage audit (guide 02 §4)

Date: 2026-11-24. Source: column definitions in the original R processing scripts (`Scripts/R/*.R` from the Zenodo record) plus checks on the real data. Per-column detail: `results/tables/data_dictionary.csv` (column `available_at_triage`).

| Column group | # cols | Available at triage? | Evidence | Decision |
|---|---|---|---|---|
| Site (`dep_name`) | 1 | yes | — | Excluded from inputs (used only for LOHO) |
| ESI | 1 | yes (see note) | `1_cleanbase…R`, `20_cleanmerged.R`: most recent ESI kept if assigned twice | T2/T3 input; note re-triage risk (D10) |
| Age, gender | 2 | yes | — | T1+ |
| Race, ethnicity, language, religion, marital, employment, insurance | 7 | yes | — | **Excluded** (sensitive); fairness audit only |
| Disposition | 1 | no — outcome | — | Target |
| Arrival mode, hour bin | 2 | yes | `2_cleantime.R` | T1+ |
| Arrival month, day | 2 | yes | `2_cleantime.R` | Not used (not in guide tiers) |
| Previous disposition | 1 | yes | `3_dispo_timeseries.R`: disposition of the *previous* visit | T3 |
| Past medical history (CCS) | 281 | yes | `4_pmhclean_all.R`: ICD-9 codes on EHR at time of visit | T3 |
| Historical labs | 379 | yes | `8_…`, `10_…`: labs ordered at previous visits within 1 year | T3 |
| Prior ED visits / admissions | 2 | yes | `7b_countadmissions.R`: previous visits within 1 year | T3 |
| Triage vitals | 7 | yes | `9b_gettriagevitals.R`: `RECORDED_TIME <= roomtime` | T1+ |
| Historical vitals | 28 | yes | `9c_…`: previous visits within 1 year | T3 |
| Prior imaging / EKG counts | 9 | yes | `12_countprevimg.R`: previous visits within 1 year | T3 |
| Outpatient medications | 48 | yes | `13_cleanmeds.R` | T3 |
| Surgeries count | 1 | yes | `14_pshclean.R` | T3 |
| Chief complaint | 200 | yes | `15_chiefcomplaint.R` | T1+ |

**Current-visit labs, imaging results, ED length of stay, treatment orders, final diagnosis:** none present. Lab, vital, imaging and EKG columns all summarise *previous* visits (suffixes `_last/_min/_max/_median/_count/_npos`); `arrivaltime`/`roomtime` were deleted by the original script.

**Red-flag test:** single-feature AUC on the S1 training rows for all 976 T3 inputs (`results/tables/leakage_single_feature_auc.csv`). Highest: ESI 0.768, age 0.732, cardiovascular meds 0.684. **No feature exceeds 0.95.**

**Patient ID:** none in the public file (D6).

Signed off by: _supervisor sign-off pending (guide M1 item; needs a human)._
