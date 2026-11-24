# Model card — FirstLook Nurse + ESI (T2) · v1

| Field | Content |
|---|---|
| Model | Nurse + ESI (T2), 223 input features, exported to ONNX; calibrator: isotonic; explanation: linear_exact |
| Intended use | Research demonstration of admission-risk estimation at ED triage |
| Not intended for | Clinical decisions, diagnosis, triage-level assignment, use on real patients |
| Training data | Yale ED admission dataset (Hong et al., 2018), 2014–2017, 3 US EDs, S1 training split |
| Inputs | age, arrival, esi, hour, sex_male, triage_vital_dbp, triage_vital_hr, triage_vital_o2, triage_vital_o2_device, triage_vital_rr, triage_vital_sbp, triage_vital_temp; chief complaint flags: 200 |
| Output | Calibrated admission probability, Low/Medium/High band, conformal set (alpha = 0.10), local explanation |
| Validation performance | AUC 0.868, PR-AUC 0.726, Brier 0.132, ECE 0.004 (S1 validation) |
| Test performance | Test metrics: added in Phase 5 (test set sealed during development). |
| Transfer | See results/tables/E5_loho.md — calibration and coverage shift at an unseen hospital |
| Known failure modes | Unseen hospital types (needs local recalibration), missing vitals, rare complaints, SpO₂ 100 encoded as 99 |
| Date / author | 2026-11-24 / Gagan Chandra (built with FirstLook pipeline) |

Decision support only. Not a diagnosis. Clinical judgement decides.
