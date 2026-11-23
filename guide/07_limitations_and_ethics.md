# 07 · Limitations, Ethics and Responsible Use

## 1. Limitations

| Limitation | Consequence | How the project handles it |
|------------|-------------|----------------------------|
| Data from one US health system (3 EDs) | Patient mix, admission practices and resources differ from Indian hospitals | Leave-one-hospital-out shows how much performance shifts between site types; local recalibration experiment; Indian validation proposed as future work |
| Data from 2014–2017 | Practice patterns have changed (incl. post-COVID); no calendar year in public file for temporal validation | Stated explicitly; site-based shift used as the available proxy |
| Admission ≠ clinical need | Admission depends on bed availability, insurance, local policy, not only severity | The model predicts "likely to be admitted in this system", described as such; not a severity score |
| Not ESI / acuity prediction | Cannot assign triage level | Clearly separated outcomes in all text |
| Chief complaint is coded (top 200) | No free-text or local-language complaints | App uses dropdown; free text / Hindi input is future work |
| Possible repeat patients across splits | Optimistic random-split scores | Leave-one-hospital-out as stricter estimate; limitation stated |
| Retrospective only | No evidence of clinical benefit | No claims of effectiveness or safety |
| Conformal guarantees assume exchangeability | Coverage can fail under distribution shift | Measured and reported in leave-one-hospital-out |
| SHAP shows model associations | Not causal | Wording rule: "associated with higher estimated risk" |
| Edge benchmarks on simulated constrained hardware | Real devices may differ | Docker limits + a real phone/Pi where possible; hardware listed in report |

## 2. What EdgeTriage does not do

- Does not diagnose disease.
- Does not recommend treatment or disposition.
- Does not assign ESI or triage colour.
- Does not replace a nurse's or doctor's judgement.
- Is not approved or validated for use with real patients.

## 3. Ethical considerations

**Data protection**
- The dataset is de-identified and public, but still health data: store it only on your own machine/drive, never commit it to GitHub, never paste rows into slides, reports or AI tools.
- Use synthetic example patients in all demos and screenshots.

**Licence and citation**
- The Zenodo record and GitHub repository do not clearly state a licence. Use the data for research only, cite Hong et al. (2018) as the repository requests, and do not redistribute the data or derived row-level files.
- Ask your department whether an ethics/IRB note is required for secondary analysis of public de-identified data, and record the answer.

**Fairness**
- Sensitive attributes (race, religion, insurance, etc.) are not model inputs.
- Subgroup performance is audited and gaps are reported, not hidden.
- Age is a legitimate clinical input but large age-group performance gaps must be discussed.

**Automation bias**
- The "Uncertain — clinician review" state and the always-visible disclaimer reduce over-reliance.
- The app never shows a binary "Admit / Discharge" command.

**Privacy by design**
- No network calls, no cloud, no telemetry; logs are optional, local and contain no identifiers.

## 4. Responsible-AI statement (for the dissertation)

> EdgeTriage is a research prototype developed on retrospective, de-identified data from a single US health system. Its outputs are estimates of the probability of hospital admission under that system's practices and must not be interpreted as diagnoses, triage levels or treatment recommendations. Performance, calibration and fairness have not been validated in Indian or other healthcare settings. Any future clinical use would require local data, recalibration, prospective validation, clinician involvement in design, and approval from the relevant ethics and regulatory bodies.

## 5. Model card (to ship with the app)

| Field | Content |
|-------|---------|
| Model name / version | e.g. EdgeTriage-T1-LGBM v1.0 |
| Intended use | Research demonstration of admission-risk estimation at triage |
| Not intended for | Clinical decisions, diagnosis, real patients |
| Training data | Yale ED dataset, 2014–2017, 3 US EDs |
| Inputs | T1 fields (list) |
| Output | Calibrated probability, band, conformal set, explanation |
| Performance | Test metrics with 95% CIs (from Table A) |
| Transfer performance | Leave-one-hospital-out (from Table C) |
| Subgroups | From Table G |
| Known failure modes | Missing vitals, unseen hospital types, rare complaints |
| Date / author | — / Gagan Chandra |
