"""E8: compression variants, ONNX / m2cgen export, Pareto front (S1 validation).

Variants (guide 03 §7), built for T1 and for T3-lite:
  V0  full selected model (all features of the tier)
  V1  same family + tuned params retrained on top-k |SHAP| features, k in K_LIST
  V2  shallow: V1 features at max_depth 3-4 and 50-200 trees (LightGBM)
  V3  V2 exported to ONNX (onnxmltools) and m2cgen code
  V4  calibrated logistic regression on the same top-k features
T3-lite = T2 features + the top history features from the T3 SHAP ranking.

Pre-declared edge-model rule (DECISIONS D19): the smallest artifact whose val AUC is
within 0.01 of V0 of the same tier.

Usage: python -m src.compress
"""
from __future__ import annotations

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # D28: no ONNX Runtime telemetry

import io
import json
import os
import pickle
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd

from src import calibrate, viz
from src.common import PROC, ROOT, load_model, meta, rows_for, selection
from src.evaluate import fast_auc
from src.train import LOGS, fit_model, make_lr

TABLES = ROOT / "results" / "tables"
ART = ROOT / "models" / "variants"
K_LIST = [10, 15, 20, 30, 50]
SEED = 42


def model_bytes(m) -> int:
    """Serialized size of the model object (what would ship in the bundle)."""
    if hasattr(m, "booster_"):  # LightGBM -> native text model
        return len(m.booster_.model_to_string().encode())
    if hasattr(m, "get_booster"):  # XGBoost -> UBJSON
        return len(m.get_booster().save_raw("ubj"))
    if m.__class__.__name__ == "CatBoostClassifier":
        f = ART / "_tmp.cbm"; m.save_model(str(f)); n = f.stat().st_size; f.unlink(); return n
    buf = io.BytesIO(); joblib.dump(m, buf, compress=3); return buf.tell()


def latency(predict_fn, X: pd.DataFrame, n: int = 1000) -> tuple[float, float, float]:
    """p50/p95 ms of single-row predictions, plus peak traced RAM (MB) during the loop."""
    rows = [X.iloc[[i % len(X)]] for i in range(n)]
    for r in rows[:20]:
        predict_fn(r)
    tracemalloc.start()
    ts = []
    for r in rows:
        t0 = time.perf_counter(); predict_fn(r); ts.append((time.perf_counter() - t0) * 1000)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95)), peak / 1e6


def onnx_export(m, n_features: int):
    from onnxmltools import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    init = [("input", FloatTensorType([None, n_features]))]
    if hasattr(m, "booster_"):
        return convert_lightgbm(m, initial_types=init, zipmap=False, target_opset=15)
    if hasattr(m, "get_booster"):
        return convert_xgboost(m, initial_types=init, target_opset=15)
    if hasattr(m, "named_steps"):
        return lr_pipeline_to_onnx(m, n_features)
    raise TypeError(type(m))


def lr_pipeline_to_onnx(m, n_features: int):
    """Impute(median, +missing indicators) -> standardise -> logistic, as a hand-built ONNX graph.

    skl2onnx mis-sizes SimpleImputer(add_indicator=True), so the graph is written from
    primitive ops. Output layout matches the tree converters: [label, probabilities(N,2)].
    """
    from onnx import TensorProto, helper, numpy_helper
    imp, sc, lr = m.named_steps["impute"], m.named_steps["scale"], m.named_steps["lr"]
    med = imp.statistics_.astype(np.float32)
    ind = list(imp.indicator_.features_) if imp.indicator_ is not None else []
    mu, sd = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
    w = lr.coef_[0].astype(np.float32)
    # fold scaling into weights: z = sum w_j (x_j - mu_j)/sd_j + b
    w_main = w[:n_features] / sd[:n_features]
    w_ind = w[n_features:] / sd[n_features:]
    b = float(lr.intercept_[0] - np.sum(w[:n_features] * mu[:n_features] / sd[:n_features])
              - np.sum(w[n_features:] * mu[n_features:] / sd[n_features:]))
    sel = np.zeros((n_features, max(len(ind), 1)), np.float32)
    for j, f in enumerate(ind):
        sel[f, j] = 1.0
    w_ind_full = (w_ind if len(ind) else np.zeros(1, np.float32)).reshape(-1, 1)
    inits = [numpy_helper.from_array(med, "med"), numpy_helper.from_array(w_main.reshape(-1, 1), "w_main"),
             numpy_helper.from_array(sel, "sel"), numpy_helper.from_array(w_ind_full.astype(np.float32), "w_ind"),
             numpy_helper.from_array(np.array([b], np.float32), "b"),
             numpy_helper.from_array(np.array([1.0], np.float32), "one"),
             numpy_helper.from_array(np.array([0.5], np.float32), "half")]
    nodes = [
        helper.make_node("IsNaN", ["input"], ["isnan"]),
        helper.make_node("Where", ["isnan", "med", "input"], ["filled"]),
        helper.make_node("Cast", ["isnan"], ["nanf"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["filled", "w_main"], ["z1"]),
        helper.make_node("MatMul", ["nanf", "sel"], ["indv"]),
        helper.make_node("MatMul", ["indv", "w_ind"], ["z2"]),
        helper.make_node("Add", ["z1", "z2"], ["z12"]),
        helper.make_node("Add", ["z12", "b"], ["z"]),
        helper.make_node("Sigmoid", ["z"], ["p1"]),
        helper.make_node("Sub", ["one", "p1"], ["p0"]),
        helper.make_node("Concat", ["p0", "p1"], ["probabilities"], axis=1),
        helper.make_node("Greater", ["p1", "half"], ["labb"]),
        helper.make_node("Cast", ["labb"], ["label"], to=TensorProto.INT64),
    ]
    g = helper.make_graph(nodes, "firstlook_lr", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])],
                          [helper.make_tensor_value_info("label", TensorProto.INT64, [None, 1]),
                           helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 15)], producer_name="firstlook")


def onnx_session(onx):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    return ort.InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])


def onnx_proba(sess, X: np.ndarray) -> np.ndarray:
    out = sess.run(None, {"input": X.astype(np.float32)})
    prob = out[1]
    if isinstance(prob, list):  # zipmap dicts
        return np.array([d[1] for d in prob])
    return np.asarray(prob)[:, 1]


def shap_rank(tier: str) -> list[str]:
    f = TABLES / f"E7_shap_importance_{tier}.csv"
    return list(pd.read_csv(f, index_col=0).index)


def build(tier_name: str, base_tier: str, features_full: list[str], ranking: list[str], X: pd.DataFrame,
          v0_model, family: str, params: dict, k_list=None) -> list[dict]:
    md = meta(); y = md.y.to_numpy()
    tr, va, ca = rows_for("train"), rows_for("val"), rows_for("cal")
    out = []
    ART.mkdir(parents=True, exist_ok=True)

    def record(name, k, m, cols, size, predict_fn, extra=None):
        p = predict_fn(X.iloc[va][cols])
        auc = fast_auc(y[va], p)
        p50, p95, ram = latency(predict_fn, X.iloc[va[:1000]][cols])
        r = {"tier": tier_name, "variant": name, "k": k, "n_features": len(cols), "size_kb": size / 1024,
             "p50_ms": p50, "p95_ms": p95, "peak_ram_mb": ram, "val_auc": auc, **(extra or {})}
        out.append(r)
        print(f"  {tier_name} {name:28s} k={k:>4} size={size/1024:9.1f}KB auc={auc:.4f} p50={p50:.3f}ms", flush=True)
        return r

    # V0
    if v0_model is not None:
        record("V0 full", len(features_full), v0_model, features_full, model_bytes(v0_model),
               lambda Z: v0_model.predict_proba(Z)[:, 1])
    for k in (k_list or K_LIST):
        cols = ranking[:k]
        m, _ = fit_model(family, params, X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va])
        record(f"V1 top-{k}", k, m, cols, model_bytes(m), lambda Z, m=m: m.predict_proba(Z)[:, 1])
        joblib.dump((m, cols), ART / f"{tier_name}_V1_k{k}.joblib")
        lr = make_lr({"C": 1.0}).fit(X.iloc[tr][cols], y[tr])
        record(f"V4 LR top-{k}", k, lr, cols, model_bytes(lr), lambda Z, m=lr: m.predict_proba(Z)[:, 1])
        joblib.dump((lr, cols), ART / f"{tier_name}_V4_k{k}.joblib")
        for depth in (3, 4):
            for trees in (50, 100, 200):
                sp = {"learning_rate": 0.1, "num_leaves": 2 ** depth, "max_depth": depth, "min_child_samples": 100,
                      "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_estimators": trees}
                m2, _ = fit_model("lgbm", sp, X.iloc[tr][cols], y[tr], None, None)
                r = record(f"V2 shallow d{depth} t{trees} top-{k}", k, m2, cols, model_bytes(m2),
                           lambda Z, m=m2: m.predict_proba(Z)[:, 1], {"depth": depth, "trees": trees})
                joblib.dump((m2, cols), ART / f"{tier_name}_V2_k{k}_d{depth}_t{trees}.joblib")
    # V3: ONNX + m2cgen of every V2 (the export is what ships)
    for r in [r for r in out if r["variant"].startswith("V2")]:
        m2, cols = joblib.load(ART / f"{tier_name}_V2_k{r['k']}_d{r['depth']}_t{r['trees']}.joblib")
        onx = onnx_export(m2, len(cols))
        sess = onnx_session(onx)
        raw = onx.SerializeToString()
        Xv = X.iloc[va][cols].to_numpy(np.float32)
        parity = float(np.max(np.abs(onnx_proba(sess, Xv) - m2.predict_proba(X.iloc[va][cols])[:, 1])))
        rec = record(f"V3 ONNX d{r['depth']} t{r['trees']} top-{r['k']}", r["k"], None, cols, len(raw),
                     lambda Z, s=sess: onnx_proba(s, Z.to_numpy(np.float32)),
                     {"depth": r["depth"], "trees": r["trees"], "onnx_max_abs_diff": parity})
        (ART / f"{tier_name}_V3_k{r['k']}_d{r['depth']}_t{r['trees']}.onnx").write_bytes(raw)
        try:
            import m2cgen
            js = m2cgen.export_to_javascript(m2)
            rec["m2cgen_js_kb"] = len(js.encode()) / 1024
        except Exception as e:  # noqa: BLE001
            rec["m2cgen_js_kb"] = None; rec["m2cgen_error"] = str(e)[:80]
    return out


def main() -> None:
    import sys
    if "--t1-all" in sys.argv:
        return extend_t1_all()
    sel = selection()["tiers"]
    rows = []
    # ---- T1 ----
    X = pd.read_parquet(PROC / "t1.parquet")
    fam = sel["t1"]["model"]
    params = json.loads((LOGS / f"best_t1_{fam}.json").read_text())["best_params"]
    v0 = load_model("t1", fam)
    rows += build("t1", "t1", list(X.columns), shap_rank("t1"), X, v0, fam, params)
    del X
    # ---- T3-lite: T2 features + top history features from the T3 ranking ----
    X = pd.read_parquet(PROC / "t3.parquet")
    t2_cols = json.loads((ROOT / "configs" / "feature_spec.json").read_text())["tiers"]["t2"]
    hist_rank = [c for c in shap_rank("t3") if c not in t2_cols]
    t2_rank = [c for c in shap_rank("t2")]
    fam3 = sel["t3"]["model"]
    params3 = json.loads((LOGS / f"best_t3_{fam3}.json").read_text())["best_params"]
    # ranking for T3-lite = interleave by T3 importance over (T2 ∪ history)
    t3_rank = [c for c in shap_rank("t3")]
    v0_t3 = load_model("t3", fam3)
    rows += build("t3", "t3", list(X.columns), t3_rank, X, v0_t3, fam3, params3)
    del X
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "E8_compression_val.csv", index=False)
    pareto(df)


def extend_t1_all() -> None:
    """D27: no T1 variant with k <= 50 met the tolerance, so add k = all 222 features (append)."""
    sel = selection()["tiers"]
    X = pd.read_parquet(PROC / "t1.parquet")
    fam = sel["t1"]["model"]
    params = json.loads((LOGS / f"best_t1_{fam}.json").read_text())["best_params"]
    rows = build("t1", "t1", list(X.columns), list(X.columns), X, None, fam, params, k_list=[X.shape[1]])
    old = pd.read_csv(TABLES / "E8_compression_val.csv")
    old = old[~((old.tier == "t1") & (old.k == X.shape[1]) & (old.variant != "V0 full"))]
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(TABLES / "E8_compression_val.csv", index=False)
    pareto(df)


def pareto(df: pd.DataFrame) -> None:
    viz.setup()
    fig, axes = viz.plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, tier in zip(axes, ("t1", "t3")):
        d = df[df.tier == tier]
        groups = [("V0 full", viz.INK, "V0 full"), ("V1", viz.SERIES[0], "V1 top-k (tuned family)"),
                  ("V3", viz.SERIES[1], "V3 shallow LightGBM → ONNX"), ("V4", viz.SERIES[2], "V4 logistic regression")]
        for pre, col, lab in groups:
            dd = d[d.variant.str.startswith(pre)]
            ax.scatter(dd.size_kb, dd.val_auc, s=36, color=col, label=lab, edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
        v0 = d[d.variant == "V0 full"].val_auc.iloc[0]
        ax.axhline(v0 - 0.01, color=viz.AXIS, lw=1)
        ax.text(d.size_kb.min(), v0 - 0.01, " V0 − 0.01 AUC", va="bottom", fontsize=8, color=viz.INK2)
        ax.set_xscale("log"); ax.set_xlabel("Artifact size (KB, log scale)"); ax.set_title(viz.TIER_NAME[tier] if tier == "t1" else "T3 / T3-lite")
        ax.set_ylabel("Validation AUC")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("E8: AUC vs model size", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, "fig09_pareto_auc_size.png")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
