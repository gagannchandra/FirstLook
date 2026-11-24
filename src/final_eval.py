"""Phase 5: single, frozen evaluation on the S1 test set.

  FIRSTLOOK_OPEN_TEST=1 python -m src.final_eval

Fills Table A (all models x tiers, 95% bootstrap CIs), Table B (value of information,
DeLong + paired bootstrap), Table D (conformal on the random split), Table E test AUCs
of the compression variants named in configs/edge_selection.json, and the test figures.
Refuses to run unless configs/selection.json has date_frozen (src/common.py guard).
"""
from __future__ import annotations

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # D28: no ONNX Runtime telemetry

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src import calibrate, conformal, viz
from src.common import PROC, ROOT, assert_test_allowed, load_model, meta, rows_for, selection
from src.evaluate import (bootstrap_ci, core_metrics, delong_test, paired_bootstrap_delta_auc, reliability, ece,
                          threshold_metrics)

TABLES = ROOT / "results" / "tables"
PRED = ROOT / "data" / "preds"
NAMES = viz.MODEL_NAME
N_BOOT = 1000


def fmt(v):
    return f"{v[0]:.3f} ({v[1]:.3f}–{v[2]:.3f})"


def main():
    assert_test_allowed()
    viz.setup()
    sel = selection()
    y_all = meta().y.to_numpy()
    te = rows_for("test")
    ca = rows_for("cal")
    y = y_all[te]
    calk = pd.read_csv(TABLES / "E2_val_calibrated.csv").set_index(["tier", "model"]).calibrator
    preds = {}
    rows = []
    for t in ("t1", "t2", "t3"):
        X = pd.read_parquet(PROC / f"{t}.parquet")
        for m in ("lr", "rf", "xgb", "lgbm", "cat"):
            if (t, m) not in calk.index:
                continue
            cp = pd.read_parquet(PRED / f"{t}_{m}.parquet").query("split=='cal'").sort_values("row")
            assert (cp.row.to_numpy() == ca).all()
            cal = calibrate.make(calk[(t, m)]).fit(cp.p.to_numpy(), y_all[ca])
            raw = load_model(t, m).predict_proba(X.iloc[te])[:, 1]
            p = cal.predict(raw)
            preds[(t, m)] = p
            ci = bootstrap_ci(y, p, n_boot=N_BOOT)
            cm = core_metrics(y, p)
            rows.append({"tier": t.upper(), "model": NAMES[m], "selected": m == sel["tiers"][t]["model"],
                         "calibrator": calk[(t, m)], **{k: v for k, v in ci.items()},
                         "logloss": cm["logloss"], "cal_slope": cm["cal_slope"], "citl": cm["citl"],
                         "spec_at_90sens": cm["spec_at_90sens"]})
            print(t, m, fmt(ci["auc"]), flush=True)
        del X
    A = pd.DataFrame(rows)
    A.to_json(TABLES / "tableA_test.json", orient="records", indent=1)
    show = A.copy()
    for k in ("auc", "prauc", "brier", "ece", "sens_at_90spec"):
        show[k] = show[k].map(fmt)
    show = show.rename(columns={"auc": "ROC-AUC", "prauc": "PR-AUC", "brier": "Brier", "ece": "ECE",
                                "sens_at_90spec": "Sens@90Spec"})
    show = show[["tier", "model", "selected", "calibrator", "ROC-AUC", "PR-AUC", "Brier", "ECE", "Sens@90Spec"]]
    (TABLES / "tableA_test.md").write_text(show.to_markdown(index=False) +
                                           "\n| Ref | Hong et al. XGB triage / full | 0.87 / 0.92 | — | — | — | — |\n")
    show.to_csv(TABLES / "tableA_test.csv", index=False)

    # thresholds + clinical operating points of selected models
    ops = []
    for t in ("t1", "t2", "t3"):
        s = sel["tiers"][t]; p = preds[(t, s["model"])]
        for name, th in (("t_low (NPV≥0.95 on val)", s["t_low"]), ("t_high (PPV≥0.80 on val)", s["t_high"])):
            tm = threshold_metrics(y, p, th)
            ops.append({"tier": t.upper(), "operating point": name, **{k: round(v, 4) for k, v in tm.items()}})
        ops.append({"tier": t.upper(), "operating point": "share in Low / Medium / High band",
                    "threshold": None, "sens": float(np.mean(p < s["t_low"])),
                    "spec": float(np.mean((p >= s["t_low"]) & (p < s["t_high"]))), "ppv": float(np.mean(p >= s["t_high"]))})
    pd.DataFrame(ops).to_csv(TABLES / "operating_points_test.csv", index=False)
    (TABLES / "operating_points_test.md").write_text(pd.DataFrame(ops).round(4).to_markdown(index=False))

    # Table B
    B = []
    best = {t: preds[(t, sel["tiers"][t]["model"])] for t in ("t1", "t2", "t3")}
    for a, b in (("t1", "t2"), ("t2", "t3"), ("t1", "t3")):
        auc_a, auc_b, pval = delong_test(y, best[a], best[b])
        d, lo, hi = paired_bootstrap_delta_auc(y, best[a], best[b], n_boot=N_BOOT)
        B.append({"comparison": f"{a.upper()} → {b.upper()}", "AUC from": auc_a, "AUC to": auc_b, "ΔAUC": d,
                  "95% CI": f"{lo:.4f} to {hi:.4f}", "DeLong p": f"{pval:.2e}"})
    B = pd.DataFrame(B)
    B.to_csv(TABLES / "tableB_test.csv", index=False)
    (TABLES / "tableB_test.md").write_text(B.round(4).to_markdown(index=False))
    print(B.to_string(index=False))

    # Table D: conformal on random split, selected T1 and T2 (fit on S3 cal, calibrated probs)
    D = []
    for t in ("t1", "t2"):
        m = sel["tiers"][t]["model"]
        cp = pd.read_parquet(PRED / f"{t}_{m}.parquet").query("split=='cal'").sort_values("row")
        cal = calibrate.from_dict(sel["tiers"][t]["calibrator_params"])
        p_cal = cal.predict(cp.p.to_numpy())
        for alpha in (0.05, 0.10, 0.20):
            cfg = conformal.fit(p_cal, y_all[ca], alpha)
            r = conformal.evaluate(best[t], y, cfg)
            D.append({"tier": t.upper(), "alpha": alpha, "target": 1 - alpha, "coverage": r["coverage"],
                      "coverage admitted": r["coverage_admit"], "coverage discharged": r["coverage_discharge"],
                      "abstention %": 100 * r["abstention"], "error (answered)": r["error_answered"],
                      "error (all, p≥0.5)": r["error_all"]})
    D = pd.DataFrame(D)
    D.to_csv(TABLES / "tableD_test.csv", index=False)
    (TABLES / "tableD_test.md").write_text(D.round(4).to_markdown(index=False))
    print(D.round(4).to_string(index=False))

    # Table E: test AUC of compression variants named in edge_selection.json
    es = json.loads((ROOT / "configs" / "edge_selection.json").read_text())
    E = []
    for v in es.get("table_E", []):
        X = pd.read_parquet(PROC / f"{v['base_tier']}.parquet")
        if v["file"].endswith(".onnx"):
            from src.compress import onnx_proba
            import onnxruntime as ort
            sess = ort.InferenceSession(str(ROOT / v["file"]), providers=["CPUExecutionProvider"])
            p = onnx_proba(sess, X.iloc[te][v["features"]].to_numpy(np.float32))
        elif v["file"] == "V0":
            p = load_model(v["base_tier"], sel["tiers"][v["base_tier"]]["model"]).predict_proba(X.iloc[te])[:, 1]
        else:
            mdl, cols = joblib.load(ROOT / v["file"])
            p = mdl.predict_proba(X.iloc[te][cols])[:, 1]
        ci = bootstrap_ci(y, p, metrics=("auc",), n_boot=N_BOOT)
        E.append({"tier": v["tier"], "variant": v["name"], "test AUC": fmt(ci["auc"]), "auc": ci["auc"][0]})
        del X
    E = pd.DataFrame(E)
    E.to_csv(TABLES / "tableE_test_auc.csv", index=False)

    # bundle test metrics (the shipped edge models, incl. their own calibrators + conformal)
    from src.runtime import load_all
    bundle_rows = []
    for slot, b in load_all().items():
        X = pd.read_parquet(PROC / f"{b.tier}.parquet", columns=b.features)
        p = b.cal.predict(b.raw_proba(X.iloc[te]))
        ci = bootstrap_ci(y, p, n_boot=N_BOOT)
        c = conformal.evaluate(p, y, b.conf)
        met = json.loads((b.path / "metrics.json").read_text())
        met["test"] = {"auc": ci["auc"][0], "auc_ci": ci["auc"][1:], "prauc": ci["prauc"][0], "brier": ci["brier"][0],
                       "ece": ci["ece"][0], "ece_ci": ci["ece"][1:], "conformal_alpha_0.10": c,
                       "band_share": {"Low": float(np.mean(p < b.thr["t_low"])), "High": float(np.mean(p >= b.thr["t_high"]))}}
        (b.path / "metrics.json").write_text(json.dumps(met, indent=1))
        from src.bundle import write_card
        write_card(b.path, b.spec, met, b.cal.kind, "lgbm_treeshap" if (b.path / "explain_lgbm.txt").exists() else "linear_exact", len(b.features))
        bundle_rows.append({"bundle": b.path.name, "AUC": fmt(ci["auc"]), "PR-AUC": fmt(ci["prauc"]), "Brier": fmt(ci["brier"]),
                            "ECE": fmt(ci["ece"]), "coverage (α=0.10)": round(c["coverage"], 4),
                            "abstention %": round(100 * c["abstention"], 1)})
        del X
    pd.DataFrame(bundle_rows).to_csv(TABLES / "bundles_test.csv", index=False)
    (TABLES / "bundles_test.md").write_text(pd.DataFrame(bundle_rows).to_markdown(index=False))
    print(pd.DataFrame(bundle_rows).to_string(index=False))

    # figures: ROC + PR per tier (selected), reliability test
    fig, axes = viz.plt.subplots(1, 2, figsize=(11, 4.4))
    for t in ("t1", "t2", "t3"):
        p = best[t]
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        a = A[(A.tier == t.upper()) & A.selected].iloc[0]
        axes[0].plot(fpr, tpr, color=viz.TIER_COLOR[t], label=f"{viz.TIER_NAME[t]} · AUC {a.auc[0]:.3f}")
        axes[1].plot(rc, pr, color=viz.TIER_COLOR[t], label=f"{viz.TIER_NAME[t]} · PR-AUC {a.prauc[0]:.3f}")
    axes[0].plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC (S1 test)")
    axes[1].axhline(y.mean(), color=viz.AXIS, lw=1)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision–recall (S1 test)")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=8)
    fig.suptitle("Each tier adds discrimination: ESI helps, EHR history helps most", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); viz.save(fig, "fig03_roc_pr_test.png")

    fig, ax = viz.plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
    for t in ("t1", "t2", "t3"):
        mp, fy, _ = reliability(y, best[t])
        ax.plot(mp, fy, marker="o", ms=4, color=viz.TIER_COLOR[t], label=f"{viz.TIER_NAME[t]} · ECE {ece(y, best[t]):.3f}")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed admission rate"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Calibrated models on the S1 test set"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    viz.save(fig, "fig04b_reliability_test.png")

    # risk-coverage
    fig, ax = viz.plt.subplots(figsize=(6.2, 4.2))
    for t in ("t1", "t2", "t3"):
        cov, err, aurc = conformal.risk_coverage(best[t], y)
        ax.plot(cov, err, color=viz.TIER_COLOR[t], label=f"{viz.TIER_NAME[t]} · AURC {aurc:.3f}")
    for _, r in D[D.alpha == 0.10].iterrows():
        ax.scatter([1 - r["abstention %"] / 100], [r["error (answered)"]], s=50, color=viz.TIER_COLOR[r["tier"].lower()],
                   edgecolor=viz.SURFACE, linewidth=2, zorder=4)
    ax.set_xlabel("Share of patients answered (most confident first)"); ax.set_ylabel("Error rate on answered")
    ax.set_title("Risk–coverage (S1 test); dots = conformal α=0.10 operating point"); ax.legend(loc="upper left", fontsize=8)
    viz.save(fig, "fig08b_risk_coverage_test.png")
    print("Phase 5 evaluation complete.")


if __name__ == "__main__":
    main()
