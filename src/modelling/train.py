"""Phase 2: train and tune 5 model families per tier (E2).

Optuna TPE, objective = validation log-loss. Boosting models use early stopping
on the validation set. T3 is tuned on a 20% training subsample, then the best
config is refit on the full training split (guide 03 §2).

Only validation and calibration (S3) rows are scored here. The S1 test rows are
never passed to a model in this script.

Usage:
  python -m src.modelling.train --tier t1 --model lgbm --trials 30
  python -m src.modelling.train --tier t2 --model all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import s1

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
PRED = ROOT / "data" / "preds"
MODELS = ROOT / "models" / "raw"
LOGS = ROOT / "results" / "logs"
SEED = 42
N_JOBS = 10
MODELS_ALL = ["lr", "lgbm", "xgb", "cat", "rf"]
DEFAULT_TRIALS = {"lr": 15, "rf": 10, "xgb": 30, "lgbm": 30, "cat": 15}
TIMEOUT_S = {"lr": 1200, "rf": 2400, "xgb": 2400, "lgbm": 2400, "cat": 3000}  # per-study cap (D24)


def load_tier(tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_parquet(PROC / "meta.parquet", columns=["y", "site", "s1"])
    X = pd.read_parquet(PROC / f"{tier}.parquet")
    return X, meta


# ---------------- model factory ----------------

def make_lr(params: dict, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=params["C"], class_weight=params.get("class_weight"), max_iter=2000,
                                  solver="lbfgs", random_state=seed)),
    ])


def fit_model(name: str, params: dict, Xtr, ytr, Xva, yva, seed: int = SEED):
    """Fit one model; returns (model, info)."""
    info: dict = {}
    spw = params.get("scale_pos_weight", 1.0)
    if name == "lr":
        m = make_lr(params, seed).fit(Xtr, ytr)
    elif name == "rf":
        m = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
            max_samples=params.get("max_samples"), class_weight=params.get("class_weight"),
            n_jobs=N_JOBS, random_state=seed).fit(Xtr, ytr)
    elif name == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 3000), learning_rate=params["learning_rate"],
            max_depth=params["max_depth"], subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"], min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"], scale_pos_weight=spw, tree_method="hist",
            eval_metric="logloss", early_stopping_rounds=50 if Xva is not None else None,
            n_jobs=N_JOBS, random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)] if Xva is not None else None, verbose=False)
        info["best_iteration"] = int(getattr(m, "best_iteration", params.get("n_estimators", 0)) or 0)
    elif name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 3000), learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"], max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"], subsample=params["subsample"], subsample_freq=1,
            colsample_bytree=params["colsample_bytree"], reg_lambda=params["reg_lambda"],
            scale_pos_weight=spw, n_jobs=N_JOBS, random_state=seed, verbose=-1)
        cb = [lgb.early_stopping(50, verbose=False)] if Xva is not None else None
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)] if Xva is not None else None, eval_metric="binary_logloss",
              callbacks=cb)
        info["best_iteration"] = int(m.best_iteration_ or params.get("n_estimators", 0))
    elif name == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(
            iterations=params.get("iterations", 3000), learning_rate=params["learning_rate"],
            depth=params["depth"], l2_leaf_reg=params["l2_leaf_reg"], scale_pos_weight=spw,
            loss_function="Logloss", thread_count=N_JOBS, random_seed=seed, verbose=False,
            od_type="Iter" if Xva is not None else None, od_wait=50 if Xva is not None else None,
            allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=(Xva, yva) if Xva is not None else None, use_best_model=Xva is not None)
        info["best_iteration"] = int(m.get_best_iteration() or params.get("iterations", 0))
    else:
        raise ValueError(name)
    return m, info


def suggest(trial: optuna.Trial, name: str, pos_ratio: float) -> dict:
    if name == "lr":
        return {"C": trial.suggest_float("C", 1e-4, 10, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    spw = trial.suggest_categorical("scale_pos_weight", [1.0, round(pos_ratio, 3)]) if name != "rf" else 1.0
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 8, 32),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.1, 0.25]),
                "max_samples": trial.suggest_categorical("max_samples", [0.3, 0.6, None]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"])}
    if name == "xgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True), "scale_pos_weight": spw}
    if name == "lgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 12]),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True), "scale_pos_weight": spw}
    if name == "cat":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True), "scale_pos_weight": spw}
    raise ValueError(name)


def predict(m, X) -> np.ndarray:
    return m.predict_proba(X)[:, 1]


def run(tier: str, name: str, n_trials: int | None, X: pd.DataFrame, meta: pd.DataFrame) -> dict:
    idx = s1(meta)
    y = meta["y"].to_numpy()
    tr, va, ca = idx["train"], idx["val"], idx["cal"]
    tune_tr = tr
    if tier == "t3":  # guide: tune T3 on a 20% training subsample
        rng = np.random.default_rng(SEED)
        tune_tr = np.sort(rng.choice(tr, size=int(0.2 * len(tr)), replace=False))
    Xtr, ytr = X.iloc[tune_tr], y[tune_tr]
    Xva, yva = X.iloc[va], y[va]
    pos_ratio = (ytr == 0).sum() / (ytr == 1).sum()
    n_trials = n_trials or DEFAULT_TRIALS[name]

    log_rows = []

    def objective(trial):
        params = suggest(trial, name, pos_ratio)
        t0 = time.time()
        m, info = fit_model(name, params, Xtr, ytr, Xva, yva)
        p = predict(m, Xva)
        ll = log_loss(yva, np.clip(p, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(yva, p)
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("info", info)
        log_rows.append({"trial": trial.number, "val_logloss": ll, "val_auc": auc, "sec": time.time() - t0,
                         **{k: str(v) for k, v in params.items()}, **info})
        print(f"  [{tier}/{name}] trial {trial.number}: logloss={ll:.4f} auc={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return ll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=TIMEOUT_S[name])
    tune_sec = time.time() - t0
    best = suggest(optuna.trial.FixedTrial(study.best_params), name, pos_ratio)
    LOGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(LOGS / f"optuna_{tier}_{name}.csv", index=False)

    # Refit on the full training split (only differs from tuning for T3)
    t0 = time.time()
    model, info = fit_model(name, best, X.iloc[tr], y[tr], Xva, yva)
    fit_sec = time.time() - t0
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / f"{tier}_{name}.joblib")

    PRED.mkdir(parents=True, exist_ok=True)
    rows = np.concatenate([va, ca])
    p = predict(model, X.iloc[rows])
    pd.DataFrame({"row": rows, "split": ["val"] * len(va) + ["cal"] * len(ca), "p": p.astype(np.float32)}) \
        .to_parquet(PRED / f"{tier}_{name}.parquet", index=False)
    pv = p[: len(va)]
    res = {"tier": tier, "model": name, "n_trials": len(study.trials), "tune_rows": int(len(tune_tr)),
           "tune_sec": round(tune_sec), "fit_sec": round(fit_sec), "best_params": best, **info,
           "val_logloss": float(log_loss(yva, np.clip(pv, 1e-7, 1 - 1e-7))), "val_auc": float(roc_auc_score(yva, pv))}
    (LOGS / f"best_{tier}_{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"DONE {tier}/{name}: val AUC={res['val_auc']:.4f} logloss={res['val_logloss']:.4f} "
          f"trials={res['n_trials']} tune={tune_sec:.0f}s fit={fit_sec:.0f}s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["t1", "t2", "t3"])
    ap.add_argument("--model", default="all")
    ap.add_argument("--trials", type=int, default=None)
    a = ap.parse_args()
    X, meta = load_tier(a.tier)
    names = MODELS_ALL if a.model == "all" else a.model.split(",")
    for n in names:
        if (LOGS / f"best_{a.tier}_{n}.json").exists():
            print(f"skip {a.tier}/{n} (already done)")
            continue
        run(a.tier, n, a.trials, X, meta)


if __name__ == "__main__":
    main()
