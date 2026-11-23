# Decisions log

Each entry: date · decision · reason.

## 2026-11-24 — Setup

- **D1 Python 3.12 (via uv), not system 3.14.** System Python is 3.14; several required libraries (catboost, shap/numba) do not reliably ship 3.14 wheels. The guide says 3.11; 3.12 is the closest version that is well supported.
- **D2 Repo layout.** The guide's `edgetriage/` structure is used at the repo root (`FirstLook/`). The guide pack stays in `guide/` (the guide calls it `docs/`); `docs/` holds generated project docs.
- **D3 Container runtime.** `docker` on this machine is podman's Docker-compatible CLI. `--network none --cpus 2 --memory 2g` work the same way.
- **D4 Dataset source.** Downloaded Zenodo record 1308993 (`admissionprediction-v1.0.zip`, MD5 `73bad18992d2ae4ae86122cd4e8305bd`, verified). The original R scripts were extracted to `data/raw/extracted/` as reference only and were never executed.
- **D5 MLflow dropped.** The guide allows "MLflow (or CSV logs)". CSV/JSON logs in `results/logs/` are used to keep the environment small.

## 2026-11-24 — Phase 1 (data)

- **D6 No patient identifier.** The public file has no patient ID (the original `20_cleanmerged.R` deletes `pat_mrn_id` and `pat_enc_csn_id`; the column `pid` is the CCS code for pelvic inflammatory disease). S1 is therefore a **visit-level** stratified split, as in Hong et al. Repeat patients may appear in train and test, which can inflate S1 scores. This is stated as a limitation; S2 (leave-one-hospital-out) is the stricter estimate.
- **D7 Tier contents.** T1 follows the guide exactly: age, sex, arrival mode, arrival 4-h bin, 7 triage vitals, 200 chief-complaint flags (222 features after one-hot). `arrivalmonth`/`arrivalday` are available at triage but are not in the guide's T1, so they are not used in any tier. T2 = T1 + ESI (223). T3 = T2 + `previousdispo` + all history groups (PMH, historical labs/vitals, prior imaging/EKG, outpatient meds, surgeries, ED visit/admission counts) = 976 features. 5 constant columns are dropped.
- **D8 Site labels.** Sites are anonymised as A/B/C. Volume and admission rate (A: 322k visits, 31.9%; B: 166k, 31.6%; C: 72k, 15.6%) are used to *infer* A = academic trauma centre, B = community, C = free-standing ED. Tables always say "Site A/B/C" and mark the type as inferred.
- **D9 SpO₂ = 100.** The original cleaning removed values `x >= max` with max = 100, so no SpO₂ of 100 exists in the data (max = 99) and many 100s became missing (SpO₂ missing 48%). The feature pipeline maps an entered SpO₂ of 100 to 99, the nearest value the model has seen. Consequence: "not measured" in the training data is a mix of truly missing and SpO₂ = 100; this is stated as a limitation and analysed in E10.
- **D10 ESI encoding.** ESI is one ordinal numeric feature (1–5, NaN if missing), not one-hot. Note: the original script kept the *most recent* ESI when a visit had two; a re-triage could happen after the first assessment. This affects few visits; it is marked "yes (see note)" in the leakage audit.
- **D11 S2 never touches the S1 test set.** Leave-one-hospital-out uses only rows outside the S1 test set: the two source sites are split 80/10/10 into train/val/cal, and the held-out site's remaining rows are the target pool (local-adaptation samples + evaluation). This keeps the S1 test set sealed until Phase 5.
- **D12 S1 proportions.** 80/10/10 train/val/test, then 10% of train is carved out as the S3 calibration set: train 72%, cal 8%, val 10%, test 10% (seed 42, stratified on outcome).
- **D13 Duplicates.** 2 exact duplicate rows are kept: without visit IDs they cannot be told apart from two genuine identical visits, and 2 rows cannot change any result.
- **D14 Temperature unit.** Triage temperature is in °F (valid range 90–110 °F, from the original cleaning script), not °C as the guide's example suggests. The app accepts °F or °C and converts to °F.
- **D15 Chief complaint.** The dataset stores chief complaint as 200 binary `cc_*` flags (including `cc_other`), and a visit can have several (87% have one, 11% two). The app lets the nurse choose up to 3 complaints from a searchable list.
- **D16 Vital ranges in the app.** Valid ranges are the original cleaning limits (e.g. HR 30–299, SpO₂ 60–100, RR 8–69, temperature 90–109.9 °F). Values outside them are rejected with a friendly message instead of being silently set to missing.

## 2026-11-24 — Phase 2/4 engineering

- **D17 Streamlit bound to localhost.** By default Streamlit looks up the machine's public IP over the network at start-up (to print an "External URL"); the app screenshot test caught this as one external connection. `.streamlit/config.toml` now sets `server.address = "localhost"` and `browser.gatherUsageStats = false`. After the change the app process opens no external sockets during the full demo flow, and the browser makes 0 requests to non-local hosts (`scripts/screenshot_app.py` logs them).
- **D18 Pre-declared selection rules (before E2 results were seen).** Best model per tier = lowest validation log-loss after its calibrator; calibrator (Platt vs isotonic, fit on S3) = lowest validation ECE, Platt on ties within 0.001; t_low = largest threshold with NPV ≥ 0.95 on validation; t_high = smallest threshold with PPV ≥ 0.80 on validation.
- **D19 Pre-declared edge-model rule.** For the app, the smallest exported artifact whose validation AUC is within 0.01 of V0 (full selected model) of the same tier.
- **D20 LR export.** skl2onnx mis-sizes `SimpleImputer(add_indicator=True)`, so the logistic-regression pipeline is exported as a hand-built ONNX graph (IsNaN → Where → MatMul → Sigmoid). Every bundle build asserts ONNX-vs-native parity < 1e-4 on all validation rows.
- **D21 Test-set guard in code.** `src/common.py` raises `TestSetSealed` for any request of S1 test rows unless `configs/selection.json` has `date_frozen` set *and* `FIRSTLOOK_OPEN_TEST=1`.
- **D22 Local adaptation uses Platt.** In E6 the local recalibrator is always Platt (2 parameters), because isotonic regression is unstable with 100–500 visits. Because Platt is monotone, Mondrian conformal sets depend only on the order of raw scores within each class, so fitting the calibrator and the conformal thresholds on the same local sample does not bias coverage.
- **D23 T3-lite.** Interpreted as the top-k features (k = 10–50) of the T3 SHAP ranking, which naturally mixes ESI, age, vitals and history fields; the app's history toggle asks only for the history fields that model uses.
