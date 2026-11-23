# FirstLook (EdgeTriage)

**An offline-first, uncertainty-aware model that estimates the probability of hospital admission at emergency-department triage — and says "Uncertain — clinician review" when it should not be trusted.**

M.Tech project · built on the public Yale ED dataset (Hong, Haimovich & Taylor, 2018; 560,486 visits, 972 variables, 3 hospitals). The full specification is in [`guide/`](guide/README.md); build status is in [`PLAN.md`](PLAN.md); every judgement call is logged in [`DECISIONS.md`](DECISIONS.md).

> Decision support only. Not a diagnosis. Clinical judgement decides. This is a research prototype and must not be used with real patients.

<!-- RESULTS -->

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

## Repository layout

```
src/            pipeline: load_raw, data, features (shared train/app), splits, train, evaluate,
                calibrate, conformal (own NumPy), select_calibrate, loho, report_loho, explain,
                secondary (E10–E12), compress, bundle, runtime (app inference), benchmark,
                dashboard_data, final_eval
app/            Streamlit app (4 screens), input validation, synthetic example patients, network status
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
