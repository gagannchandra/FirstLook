# 08 · Getting Started (Day 1 → Week 2)

## 1. Repository structure

```
edgetriage/
├── README.md
├── requirements.txt
├── .gitignore                 # must include data/ and *.RData, *.parquet
├── configs/
│   └── tiers.yaml             # feature lists per tier
├── data/                      # NEVER committed
│   ├── raw/                   # 5v_cleandf.RData
│   ├── interim/               # full.parquet
│   └── processed/             # t1.parquet, t2.parquet, t3.parquet, splits.parquet
├── notebooks/
│   ├── 01_load_and_dictionary.ipynb
│   ├── 02_eda.ipynb
│   └── 03_results_figures.ipynb
├── src/
│   ├── data.py                # load, clean, build tiers, splits
│   ├── train.py               # model training + Optuna
│   ├── calibrate.py           # Platt / isotonic
│   ├── conformal.py           # split conformal (own code)
│   ├── explain.py             # SHAP
│   ├── compress.py            # feature reduction, export
│   ├── evaluate.py            # metrics, bootstrap CIs, DeLong
│   └── benchmark.py           # size, latency, RAM, network
├── models/                    # exported bundles (see 04_system_design.md §7)
├── app/
│   ├── streamlit_app.py       # L1/L2
│   └── pwa/                   # L3 (stretch)
├── results/
│   ├── tables/
│   └── figures/
├── docker/
│   └── Dockerfile
└── docs/                      # this project pack
```

## 2. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`

```
pandas
numpy
pyarrow
pyreadr
polars
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
matplotlib
plotly
mlflow
onnx
onnxruntime
onnxmltools
skl2onnx
m2cgen
psutil
streamlit
interpret
```

`.gitignore` (minimum)

```
data/
*.RData
*.parquet
*.csv
.venv/
mlruns/
```

## 3. Download the data

1. Open the Zenodo record **1308993** (link in `10_references.md`) and download the ZIP.
2. Unzip into `data/raw/`. You should get `5v_cleandf.RData`.
3. Clone the original code for reference: `git clone https://github.com/yaleemmlc/admissionprediction` into a folder **outside** your repo.

## 4. Week-1 code: load and inspect

```python
import pyreadr
import pandas as pd

result = pyreadr.read_r("data/raw/5v_cleandf.RData")
print(result.keys())
df = result[list(result.keys())[0]]
print(df.shape)

df.to_parquet("data/interim/full.parquet")
```

```python
df = pd.read_parquet("data/interim/full.parquet")

print(df.columns[:60].tolist())
print(df.dtypes.value_counts())

for col in ["disposition", "esi", "dep_name", "gender", "arrivalmode"]:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).head(10), sep="\n")

id_like = [c for c in df.columns if "id" in c.lower() or "mrn" in c.lower()]
print("possible id columns:", id_like)
```

```python
miss = df.isna().mean().sort_values(ascending=False)
dictionary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "pct_missing": df.isna().mean().values.round(4),
    "n_unique": df.nunique().values,
})
dictionary["group"] = ""
dictionary["available_at_triage"] = ""
dictionary["tier"] = ""
dictionary.to_csv("results/tables/data_dictionary.csv", index=False)
```

Fill `group`, `available_at_triage` and `tier` by hand (use prefixes like `triage_vital_`, `cc_`, `meds_`, and the original R scripts). This file becomes Appendix A of the dissertation.

## 5. Week-2 code: first baseline

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

y = (df["disposition"] == "Admit").astype(int)
X = df[t2_columns]

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                          tree_method="hist", enable_categorical=True, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

p = model.predict_proba(X_val)[:, 1]
print(f"val AUC   {roc_auc_score(y_val, p):.4f}")
print(f"val PRAUC {average_precision_score(y_val, p):.4f}")
print(f"val Brier {brier_score_loss(y_val, p):.4f}")
```

`t2_columns` is the T2 feature list from `configs/tiers.yaml`, built from the data dictionary. Confirm the exact target label values first (`df["disposition"].unique()`). Only validation metrics in Weeks 2–10; the test set stays closed.

## 6. Minimal split conformal (for Week 7)

```python
import numpy as np

def fit_conformal(p_cal, y_cal, alpha=0.10):
    qhat = {}
    for k in [0, 1]:
        pk = np.where(y_cal == 1, p_cal, 1 - p_cal)[y_cal == k]
        scores = 1 - pk
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhat[k] = np.quantile(scores, level, method="higher")
    return qhat

def predict_set(p, qhat):
    include_admit = (1 - p) <= qhat[1]
    include_discharge = p <= qhat[0]
    return include_admit, include_discharge
```

A row where both are `True` → "Uncertain — clinician review".

## 7. Edge benchmark snippet (for Week 10)

```python
import time, os, psutil
import numpy as np

def benchmark(predict_fn, X_rows, n=1000):
    times = []
    for i in range(n):
        row = X_rows[i % len(X_rows):i % len(X_rows) + 1]
        t0 = time.perf_counter()
        predict_fn(row)
        times.append((time.perf_counter() - t0) * 1000)
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    return np.percentile(times, 50), np.percentile(times, 95), rss
```

Run inside Docker with no network and limited resources:

```bash
docker build -t edgetriage -f docker/Dockerfile .
docker run --rm --network none --cpus 2 --memory 2g edgetriage python src/benchmark.py
```

`--network none` gives the container no network interface at all, so the benchmark doubles as proof of offline inference. (Port mapping does not work in this mode, so for the UI demo just switch Wi-Fi off on the laptop.)

## 8. First-week checklist

- [ ] Share `01_proposal.md` with supervisor and get sign-off on title and RQs
- [ ] Repo created, `.gitignore` in place before any data is copied
- [ ] Dataset downloaded and loaded; shape printed
- [ ] Original paper read fully, including S1 Table (variable list)
- [ ] Literature notes for at least 8 papers
- [ ] Ask department about ethics requirement for public de-identified data
- [ ] Decide whether to apply for MIMIC-IV-ED access (optional)
