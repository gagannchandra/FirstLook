# 04 · System Design (Offline-First Application)

## 1. Design principles

1. **Offline by default** — after installation, no prediction step touches the network.
2. **Data stays on the device** — no external API, no telemetry, no cloud logging.
3. **Fast entry** — Tier-1 form completable in under 60 seconds.
4. **Honest output** — calibrated probability, risk band, explanation, and an explicit "uncertain" state.
5. **Graceful with missing data** — every vital has a "not measured" option.
6. **Small footprint** — model artifact under 1 MB target; runs on a 2-core / 2 GB machine.

## 2. Architecture (four layers)

```
┌──────────────────────────────────────────────────────────┐
│  Presentation layer                                       │
│  Patient form · Result card · Explanation · Dashboard     │
├──────────────────────────────────────────────────────────┤
│  Decision-support layer                                   │
│  Calibrator → risk band → conformal set → "Uncertain" flag│
│  Local explanation (TreeSHAP or glass-box contributions)  │
├──────────────────────────────────────────────────────────┤
│  Inference layer                                          │
│  Input validation → feature pipeline → compact model      │
│  (native / ONNX Runtime / m2cgen code)                    │
├──────────────────────────────────────────────────────────┤
│  Local storage layer                                      │
│  SQLite (desktop) or IndexedDB (browser): optional        │
│  anonymised prediction log, model version, benchmark log  │
└──────────────────────────────────────────────────────────┘
          No network calls at any layer during prediction
```

## 3. Deployment levels (build in this order)

| Level | Priority | What | Runs on | Offline proof |
|-------|----------|------|---------|---------------|
| **L1** | Must | Streamlit app + Python model loaded with joblib | Laptop / lab PC | Disable Wi-Fi, run full demo |
| **L2** | Should | Same app but model served via **ONNX Runtime**; packaged in Docker with `--network none --cpus 2 --memory 2g` | Any PC, simulates a low-end device | Docker has no network interface at all |
| **L3** | Stretch | **Progressive Web App**: static HTML/JS + `onnxruntime-web` (WASM) or `m2cgen`-generated JavaScript; service worker caches everything; installable on Android | Phone browser | Airplane mode on the phone |

L1 alone is enough to pass; L2 gives you the measured edge results for RQ4; L3 is the "wow" demo for interviews.

**Explanation in L3:** TreeSHAP is hard to run in a browser. Use the glass-box model (calibrated LR or EBM on top-k features) for in-browser contributions, and report its agreement with the main model's SHAP (methodology §6).

## 4. Technology stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.11 | Standard ML stack |
| Data | pandas, Polars (optional), PyArrow, pyreadr | Read RData, fast Parquet |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, interpret (EBM, optional) | Model families |
| Tuning / tracking | Optuna, MLflow (or CSV logs) | Reproducibility |
| Uncertainty | NumPy (own implementation) or MAPIE | Conformal prediction |
| Explainability | SHAP | TreeSHAP |
| Export | joblib, onnxmltools / skl2onnx, onnxruntime, m2cgen | Edge artifacts |
| App (L1/L2) | Streamlit | Fastest route to a working UI |
| App (L3) | HTML + vanilla JS / React, onnxruntime-web, Workbox service worker | Installable offline PWA |
| Storage | SQLite (desktop), IndexedDB (browser) | Local only |
| Charts | Plotly / Matplotlib | Report and dashboard |
| Packaging | Docker | Resource-limited, network-free benchmark |
| Benchmarking | `time.perf_counter`, `tracemalloc`/`psutil`, `tcpdump` or browser DevTools | RQ4 measurements |

FastAPI is **not needed** for the core plan: Streamlit calls the model directly in-process, which is simpler and fully offline. Add FastAPI only if you later want a separate local API.

## 5. Application screens

### Screen 1 — Patient input (Tier 1 by default)
- Age, sex
- Arrival mode (walk-in / ambulance / other), arrival time (auto-filled)
- Vitals: HR, SBP, DBP, RR, SpO₂ (+ on oxygen?), temperature — each with "not measured"
- Chief complaint: searchable dropdown of the 200 coded complaints
- Optional toggles: "ESI available" (adds field → T2 model), "History available" (adds key history fields → T3-lite)
- Input validation with plausible ranges and warnings

### Screen 2 — Result
- Calibrated admission probability, e.g. **"Estimated admission risk: 68% (High)"**
- Risk band colour (Low / Medium / High)
- Conformal output: "Likely admission" / "Likely discharge" / **"Uncertain — clinician review"**
- Model used (T1/T2/T3-lite), model version, inference time in ms
- Banner: *"Decision support only. Not a diagnosis. Clinical judgement decides."*

### Screen 3 — Explanation
- Top 5 factors raising and lowering the estimate (waterfall/bar chart)
- Plain-language lines: "SpO₂ 91% — associated with higher estimated risk"
- Note if inputs were missing and how that affects reliability

### Screen 4 — Research dashboard
- Test-set metrics per tier (AUC, PR-AUC, Brier, ECE) with CIs
- Calibration curve, decision curve, risk–coverage curve
- Edge benchmark: model size, p50/p95 latency, peak RAM
- Live status: "Network: OFFLINE · 0 bytes sent this session"

## 6. Data flow for one prediction

1. Nurse enters form → client-side validation.
2. Feature pipeline builds the exact feature vector used in training (same column order, same encoding — saved as `feature_spec.json`).
3. Compact model → raw score → calibrator → probability.
4. Conformal thresholds → label set → band.
5. Explanation computed locally.
6. Result rendered; optional anonymised log row saved locally (no name, no ID).

## 7. Reproducible model bundle

Every deployable model is saved as a folder:

```
models/t1_lgbm_v2/
├── model.onnx            (or model.joblib / model.js)
├── calibrator.json       (Platt a, b  or isotonic breakpoints)
├── conformal.json        (alpha, per-class q-hat)
├── thresholds.json       (t_low, t_high)
├── feature_spec.json     (ordered features, types, valid ranges, category lists)
├── metrics.json          (test metrics with CIs)
└── model_card.md         (intended use, data, limits, version, date)
```

## 8. Live demo script (5 minutes)

1. Show the laptop/phone with **Wi-Fi off** (or Docker `--network none`).
2. Open EdgeTriage. Enter a high-risk example (older patient, low SpO₂, ambulance arrival, shortness of breath) → High risk + explanation.
3. Enter a low-risk example (young, normal vitals, minor complaint) → Low risk.
4. Enter an ambiguous example → **"Uncertain — clinician review"**; explain conformal coverage in one sentence.
5. Mark SpO₂ "not measured" → show the estimate still works and the reliability note.
6. Open the dashboard → latency in milliseconds, model size in KB, 0 bytes network.
7. Close with: "Same model, same accuracy, no internet, runs on a ₹X device."

Use **synthetic example patients** you write yourself for the demo — never show real rows from the dataset in screenshots or slides.
