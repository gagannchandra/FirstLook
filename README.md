# FirstLook (EdgeTriage)

**An offline-first, uncertainty-aware model that estimates the probability of hospital admission at emergency-department triage — and says "Uncertain — clinician review" when it should not be trusted.**

M.Tech project · built on the public Yale ED dataset (Hong, Haimovich & Taylor, 2018; 560,486 visits, 972 variables, 3 hospitals). The full specification is in [`guide/`](guide/README.md); build status is in [`PLAN.md`](PLAN.md); every judgement call is logged in [`DECISIONS.md`](DECISIONS.md).

> Decision support only. Not a diagnosis. Clinical judgement decides. This is a research prototype and must not be used with real patients.

## Headline results (S1 test set, opened once on 2026-11-24)

Best model per tier (95% bootstrap CI, n = 56,049 test visits):

| Tier | Model | ROC-AUC | ECE |
|---|---|---|---|
| T1 | LightGBM | 0.856 (0.852–0.859) | 0.005 |
| T2 | LightGBM | 0.874 (0.871–0.877) | 0.003 |
| T3 | CatBoost | 0.928 (0.925–0.930) | 0.003 |

Shipped edge bundles (what the apps run):

| bundle    | AUC                 | PR-AUC              | Brier               | ECE                 |   coverage (α=0.10) |   abstention % |
|:----------|:--------------------|:--------------------|:--------------------|:--------------------|--------------------:|---------------:|
| t1_v1     | 0.843 (0.839–0.846) | 0.685 (0.678–0.692) | 0.143 (0.141–0.144) | 0.005 (0.004–0.009) |              0.9114 |           37.3 |
| t2_v1     | 0.866 (0.863–0.869) | 0.718 (0.711–0.725) | 0.133 (0.132–0.135) | 0.006 (0.004–0.009) |              0.9094 |           29.8 |
| t3lite_v1 | 0.918 (0.916–0.921) | 0.846 (0.841–0.851) | 0.099 (0.098–0.101) | 0.005 (0.004–0.008) |              0.906  |           18.6 |

### Pre-registered success criteria

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


All tables (A–G), the criteria and the figure list: [`results/RESULTS.md`](results/RESULTS.md). App screenshots: [`results/figures/app/`](results/figures/app/).


## How to run everything

Requirements: Linux/macOS, [uv](https://docs.astral.sh/uv/) (or Python 3.12 + pip), ~12 GB RAM for T3 training, Docker or Podman for L2.

```bash
# 0. environment
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# 1. data (Phase 1) — download Zenodo record 1308993 into data/raw/ (never committed)
mkdir -p data/raw && curl -L -o data/raw/admissionprediction-v1.0.zip \
  "https://zenodo.org/api/records/1308993/files/yaleemmlc/admissionprediction-v1.0.zip/content"
unzip -j data/raw/admissionprediction-v1.0.zip '*/Results/5v_cleandf.RData' -d data/raw/
.venv/bin/python -m src.load_raw      # RData -> data/interim/full.parquet (checks 560,486 x 972)
.venv/bin/python -m src.data          # dictionary, leakage audit, tiers T1/T2/T3, splits S1/S2/S3
.venv/bin/python -m src.eda           # EDA figures + aggregate tables

# 2. models (Phase 2)
.venv/bin/python -m src.e1_baseline   # E1: LR + XGBoost sanity check vs Hong et al.
./scripts/run_e2.sh                   # E2: 5 models x 3 tiers, Optuna on validation (hours)
.venv/bin/python -m src.select_calibrate   # E4 calibration, model selection, thresholds, E3 (val)

# 3. research experiments (Phase 3)
.venv/bin/python -m src.loho && .venv/bin/python -m src.report_loho   # E5 + E6 local-adaptation curve
.venv/bin/python -m src.explain                                        # E7 SHAP
.venv/bin/python -m src.secondary --split val                          # E10, E11, E12 (validation)

# 4. edge model + app (Phase 4)
.venv/bin/python -m src.compress      # E8 compression variants + Pareto plot
.venv/bin/python -m src.bundle        # model bundles in models/bundles/ (reads configs/edge_selection.json)
.venv/bin/python -m src.benchmark --out results/bench/bench_host.json --setting "host laptop"
docker build -t firstlook -f docker/Dockerfile .
docker run --rm --network none --cpus 2 --memory 2g -v "$PWD/results/bench:/out" firstlook \
    python -m src.benchmark --out /out/bench_docker.json --setting "Docker --network none, 2 CPU, 2 GB"

# 5. final evaluation (Phase 5) — opens the S1 test set once; refuses unless selection.json is frozen
FIRSTLOOK_OPEN_TEST=1 .venv/bin/python -m src.final_eval
FIRSTLOOK_OPEN_TEST=1 .venv/bin/python -m src.secondary --split test
FIRSTLOOK_OPEN_TEST=1 .venv/bin/python -m src.dashboard_data --split test
.venv/bin/python -m src.make_report   # results/RESULTS.md (Tables A–H + success criteria), fig01, fig06
.venv/bin/python scripts/build_pwa.py # export T1/T2 bundles for the PWA

# tests
.venv/bin/python -m pytest -q
```

### Run the app (L1, fully offline)

```bash
./scripts/serve_app.sh            # http://localhost:8599  (bound to localhost only)
# or: .venv/bin/streamlit run app/streamlit_app.py
```

Switch Wi-Fi off and use it: nothing in the prediction path touches the network. The dashboard shows the live network state measured from the OS. Screenshots: `python scripts/screenshot_app.py --url http://localhost:8599`.

### Run the app in Docker (L2)

```bash
docker run --rm -p 8501:8501 --cpus 2 --memory 2g firstlook      # UI (port mapping needs a network namespace)
docker run --rm --network none --cpus 2 --memory 2g firstlook python -m src.benchmark   # offline proof
```

### Run the offline web app (L3 PWA)

```bash
./scripts/serve_pwa.sh            # http://localhost:8600 — open once, then it works with the network off
python scripts/screenshot_pwa.py  # proves it: installs the service worker, cuts the network, reloads, predicts
```

Plain HTML + JavaScript, no libraries or CDN. Runs the nurse-only (T1) and +ESI (T2) models; the EHR-history model runs in L1/L2 only (D33).

## Not completed, and why

| Item | Status | Why |
|---|---|---|
| E13 usability pilot (Table H) | not run | Needs 5–8 human participants (timed entry, SUS). Protocol in `guide/03_methodology.md` §11; app and synthetic cases are ready. |
| Supervisor sign-off of the leakage audit | pending | Human step (`docs/leakage_audit.md`). |
| T1 compression criterion (ΔAUC ≤ 0.01) | narrowly missed | No compressed T1 export stayed within 0.01 AUC; shipped LR has test ΔAUC 0.013 at 1/1,105 the size (D27). T3-lite meets it. |
| 3-seed repeats of Table A | not run | Cut-list item (D32); CIs are 1,000-resample bootstrap on one training seed. |
| PWA with EHR-history model | not built | Would need onnxruntime-web (multi-MB WASM); T1/T2 work offline in the PWA (D33). |

Finding worth reporting: ONNX Runtime 1.32 sends telemetry to Microsoft by default on Linux even after `disable_telemetry_events()`. It was found by the OS-level socket audit and is disabled with `ORT_DISABLE_TELEMETRY=1`; a regression test guards it (D28).

## Repository layout

```
src/            pipeline: load_raw, data, features (shared train/app), splits, train, evaluate,
                calibrate, conformal (own NumPy), select_calibrate, loho, report_loho, explain,
                secondary (E10–E12), compress, bundle, runtime (app inference), benchmark,
                dashboard_data, final_eval
app/            Streamlit app (4 screens), input validation, synthetic example patients, network status;
                app/pwa/ offline PWA (L3)
configs/        feature_spec.json, tiers.yaml, selection.json (frozen choices), edge_selection.json
models/bundles/ deployable bundles (model.onnx + calibrator/conformal/thresholds/spec/metrics/model card)
results/        tables/ (markdown copies of every table; CSVs are git-ignored), figures/, bench/, dashboard/
docs/           leakage audit and other generated docs
tests/          pytest: data pipeline, conformal coverage, train/app feature parity, validation, test-set guard
guide/          the project specification (unchanged)
data/           NEVER committed (git-ignored)
```

## Data and ethics

The dataset is de-identified and public but has no clear licence: it is used for research only, cited, and never redistributed (`data/`, `*.RData`, `*.parquet`, `*.csv` are git-ignored). Race, ethnicity, language, religion, marital status, employment and insurance are **never model inputs**; they are used only in the fairness audit. All demo patients are synthetic and hand-written (`app/examples.py`); no dataset row appears in the app, screenshots or docs.

**Citation:** Hong WS, Haimovich AD, Taylor RA (2018). Predicting hospital admission at emergency department triage using machine learning. *PLoS ONE* 13(7): e0201016.
