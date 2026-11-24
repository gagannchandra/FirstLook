# Model card — FirstLook Nurse + ESI + history (T3-lite) · v1

| Field | Content |
|---|---|
| Model | Nurse + ESI + history (T3-lite), 50 input features, exported to ONNX; calibrator: isotonic; explanation: lgbm_treeshap |
| Intended use | Research demonstration of admission-risk estimation at ED triage |
| Not intended for | Clinical decisions, diagnosis, triage-level assignment, use on real patients |
| Training data | Yale ED admission dataset (Hong et al., 2018), 2014–2017, 3 US EDs, S1 training split |
| Inputs | age, alcoholrelateddisorders, arrival, asthma, esi, hour, hyperlipidem, meds_analgesics, meds_antiarthritics, meds_antiasthmatics, meds_antibiotics, meds_anticoagulants, meds_antihyperglycemics, meds_antiplateletdrugs, meds_cardiacdrugs, meds_cardiovascular, meds_cnsdrugs, meds_diuretics, meds_elect/caloric/h2o, meds_gastrointestinal, meds_hormones, meds_psychotherapeuticdrugs, meds_thyroidpreps, meds_unclassifieddrugproducts, meds_vitamins, n_admissions, n_edvisits, n_surgeries, neutrophils_median, prevdispo, rdw_last, sex_male, triage_vital_dbp, triage_vital_hr, triage_vital_o2, triage_vital_o2_device, triage_vital_sbp, triage_vital_temp; chief complaint flags: 8 |
| Output | Calibrated admission probability, Low/Medium/High band, conformal set (alpha = 0.10), local explanation |
| Validation performance | AUC 0.918, PR-AUC 0.844, Brier 0.099, ECE 0.004 (S1 validation) |
| Test performance | Test (S1, frozen): AUC 0.918 (95% CI 0.916–0.921), ECE 0.005 |
| Transfer | See results/tables/E5_loho.md — calibration and coverage shift at an unseen hospital |
| Known failure modes | Unseen hospital types (needs local recalibration), missing vitals, rare complaints, SpO₂ 100 encoded as 99 |
| Date / author | 2026-11-24 / Gagan Chandra (built with FirstLook pipeline) |

Decision support only. Not a diagnosis. Clinical judgement decides.
