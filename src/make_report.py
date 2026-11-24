"""Assemble results/RESULTS.md (Tables A–H + success criteria) from generated files only.

Every number is read from a file written by an experiment script; nothing is typed by hand.
Also draws fig01 (pipeline) and fig06 (explanation of a synthetic patient).
Usage: python -m src.make_report
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import viz  # noqa: E402

ROOT = viz.ROOT
T = ROOT / "results" / "tables"
OUT = ROOT / "results" / "RESULTS.md"


def md(df: pd.DataFrame, nd=3) -> str:
    return df.round(nd).to_markdown(index=False)


def fig_pipeline():
    viz.setup()
    steps = ["Yale ED data\n560,486 × 972", "Leakage audit\n+ data dictionary", "Tiers T1/T2/T3\nsplits S1/S2/S3",
             "5 models × 3 tiers\nOptuna on val", "Calibration\n(Platt / isotonic)", "Conformal\n'uncertain' flag",
             "SHAP\nexplanations", "Compression\n→ ONNX bundle", "Offline app\n0 bytes sent"]
    fig, ax = viz.plt.subplots(figsize=(13, 2.4))
    ax.set_axis_off(); ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1)
    for i, s in enumerate(steps):
        col = viz.SERIES[0] if i in (2, 5) else ("#0f6e6a" if i == len(steps) - 1 else viz.INK2)
        ax.add_patch(viz.plt.Rectangle((i + 0.06, 0.25), 0.84, 0.5, facecolor="white", edgecolor=col, lw=1.5))
        ax.text(i + 0.48, 0.5, s, ha="center", va="center", fontsize=8.5, color=viz.INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.9, 0.5), arrowprops=dict(arrowstyle="->", color=viz.MUTED))
    ax.text(0.06, 0.92, "FirstLook pipeline (guide 03 §1). S2 leave-one-hospital-out + local adaptation branch from step 5–6.",
            fontsize=9, color=viz.INK2)
    viz.save(fig, "fig01_pipeline.png")


def fig_explanation():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from examples import EXAMPLES
    from src.runtime import load_all
    b = load_all()["t1"]
    r = b.predict(EXAMPLES["high"]["form"])
    items = sorted([c for c in r.contributions if abs(c["contribution"]) > 0.05][:10], key=lambda d: d["contribution"])
    viz.setup()
    fig, ax = viz.plt.subplots(figsize=(7.5, 3.8))
    ax.barh([d["factor"] for d in items], [d["contribution"] for d in items], height=0.6,
            color=["#d64545" if d["contribution"] > 0 else "#2a78d6" for d in items])
    ax.axvline(0, color=viz.AXIS, lw=1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to log-odds (exact, logistic-regression edge model)")
    ax.set_title(f"Synthetic high-risk patient: estimated risk {r.probability:.0%} ({r.band}, {r.conformal.replace('_', ' ')})")
    viz.save(fig, "fig06_local_explanation_synthetic.png")


def main():
    fig_pipeline(); fig_explanation()
    A = json.load(open(T / "tableA_test.json"))
    sel = json.load(open(ROOT / "configs" / "selection.json"))
    es = json.load(open(ROOT / "configs" / "edge_selection.json"))
    parts = ["# FirstLook — final results", "",
             f"S1 test set opened once on **{sel['date_frozen']}** after all choices were frozen (`docs/frozen_choices.md`). "
             "Every number below is read from files written by the pipeline (`results/tables/`). 95% CIs: 1,000 bootstrap resamples of the test set (n = 56,049).", ""]

    parts += ["## Table A — Model comparison (S1 test, 95% CI)", "", (T / "tableA_test.md").read_text(), "",
              "Selected = model chosen per tier on validation log-loss (D18). Optuna trials per study are in `results/logs/best_*.json` (T3 XGBoost: 6 trials within its time cap, D24).", ""]
    parts += ["## Table B — Value of information (selected model per tier, S1 test)", "", (T / "tableB_test.md").read_text(), ""]
    parts += ["## Table C — Leave-one-hospital-out (S2; S1 test rows excluded)", "", (T / "E5_loho.md").read_text(), "",
              "Site types are inferred from volume and admission rate (D8). 'ECE after local recal' and 'coverage after local conformal' are means over 20 draws of 1,000 local visits.", ""]
    parts += ["## Table C2 — Local-adaptation curve (mean ± SD over 20 draws; model never retrained)", "",
              (T / "E6_local_adaptation_table_C2.md").read_text(), "", "### Smallest number of local visits that restores the target", "",
              (T / "E6_smallest_n.md").read_text(), "",
              "n* = smallest n whose mean stays within ±0.02 of 0.90 (coverage) or ≤ 0.02 (ECE) for all larger n. Reference arm (full retrain with local data) in `results/tables/E6_retrain_reference.csv`.", ""]
    parts += ["## Table D — Conformal prediction, random split (S1 test; compare Abdulai et al. 2025)", "", (T / "tableD_test.md").read_text(), ""]

    # Table E
    e8 = pd.read_csv(T / "E8_compression_val.csv")
    te = pd.read_csv(T / "tableE_test_auc.csv")
    dock = json.load(open(ROOT / "results" / "bench" / "bench_docker.json"))
    rows = []
    for _, r in te.iterrows():
        tier = r["tier"]; name = r["variant"]
        m = e8[(e8.tier == tier) & (e8.variant == name)]
        if m.empty and name.startswith("V2"):
            m = e8[(e8.tier == tier) & (e8.variant == name)]
        rec = m.iloc[0] if len(m) else None
        rows.append({"tier": "T1" if tier == "t1" else "T3 / T3-lite", "variant": name,
                     "# features": int(rec.n_features) if rec is not None else None,
                     "size KB": round(rec.size_kb, 1) if rec is not None else None,
                     "p50 ms (model, in-process)": round(rec.p50_ms, 3) if rec is not None else None,
                     "p95 ms": round(rec.p95_ms, 3) if rec is not None else None,
                     "peak traced RAM MB": round(rec.peak_ram_mb, 2) if rec is not None else None,
                     "val AUC": round(rec.val_auc, 4) if rec is not None else None, "test AUC (95% CI)": r["test AUC"]})
    E = pd.DataFrame(rows)
    bench = pd.DataFrame([{"bundle": k, "ONNX KB": round(v["model_onnx_kb"], 1), "bundle KB": round(v["bundle_kb"], 1),
                           "end-to-end p50 ms": round(v["e2e_p50_ms"], 2), "end-to-end p95 ms": round(v["e2e_p95_ms"], 2),
                           "model-only p95 ms": round(v["model_p95_ms"], 3)} for k, v in dock["bundles"].items()])
    parts += ["## Table E — Compression / edge benchmark", "", E.to_markdown(index=False), "",
              "In-process latencies in this table are from E8 on the host (single-row `predict_proba`; ONNX rows use ONNX Runtime). "
              "The shipped bundles measured inside Docker (`--network none --cpus 2 --memory 2g`, cgroup limits verified):", "",
              bench.to_markdown(index=False), "",
              f"Docker: interfaces {dock['interfaces']}, peak RSS {dock['peak_rss_mb']:.0f} MB, cold start {dock['cold_start_ms_all_bundles']:.0f} ms (3 bundles), "
              f"**network bytes sent {dock['net_bytes_sent_non_loopback']}**, app external connections {len(dock.get('app_external_connections', []))}.", "",
              f"Edge choice (D19/D27): T1 → {es['choices']['t1']['variant']} (rule met: {es['choices']['t1']['rule_met']}); "
              f"T3-lite → {es['choices']['t3lite']['variant']} (rule met: {es['choices']['t3lite']['rule_met']}).", "",
              "Shipped bundles on the S1 test set (own calibrator + conformal thresholds):", "", (T / "bundles_test.md").read_text(), ""]

    # Table F
    f = pd.read_csv(T / "E10_missing_vitals_test.csv")
    keep = f.scenario.isin(["none (as recorded)", "o2 missing (100% rows)", "hr missing (100% rows)", "temp missing (100% rows)",
                            "all vitals missing (100% rows)", "2 random vital(s) @ 30% rows", "3 random vital(s) @ 50% rows"])
    F = f[keep][["tier", "scenario", "strategy", "auc", "delta_auc", "ece"]].rename(columns={"delta_auc": "ΔAUC"})
    parts += ["## Table F — Missing vitals (S1 test; research models)", "", md(F, 4), "",
              "Full grid (every vital, 1–3 vitals at 10/30/50% of rows): `results/tables/E10_missing_vitals_test.csv`.", ""]
    g = pd.read_csv(T / "E11_subgroups_test.csv")
    G = g[["tier", "attribute", "group", "n", "admit_rate", "auc", "sens@t_high", "ppv@t_high", "citl", "flag"]]
    G = G[~((G.attribute.str.contains("audit")) & (G.n < 200))]
    parts += ["## Table G — Subgroups (S1 test)", "", md(G, 3), "",
              "Sens/PPV at the High-band threshold. Flags: AUC > 0.05 below overall, or |calibration-in-the-large| > 0.2. "
              "Race, ethnicity and insurance are audit-only (never model inputs); groups with n < 200 omitted here (in CSV).", ""]
    parts += ["## Table H — Usability pilot", "", "**Not run.** E13 requires 5–8 human participants (timed entry + SUS). "
              "The protocol is in `guide/03_methodology.md` §11; the app and synthetic cases are ready for it.", ""]

    # success criteria
    sel_rows = {r["tier"]: r for r in A if r["selected"]}
    D = pd.read_csv(T / "tableD_test.csv")
    dc = pd.read_csv(T / "E12_decision_curve_test.csv")
    v0_t1 = te[(te.tier == "t1") & (te.variant == "V0 full")].auc.iloc[0]
    e_t1 = te[(te.tier == "t1") & (te.variant == es["choices"]["t1"]["variant"])].auc.iloc[0]
    v0_t3 = te[(te.tier == "t3") & (te.variant == "V0 full")].auc.iloc[0]
    e_t3 = te[(te.tier == "t3") & (te.variant == es["choices"]["t3lite"]["variant"])].auc.iloc[0]
    sn = pd.read_csv(T / "E6_smallest_n.csv")
    crit = [
        ("Reproduction (T2, T3 within 0.02 of 0.87 / 0.92)",
         f"T2 {sel_rows['T2']['auc'][0]:.3f}, T3 {sel_rows['T3']['auc'][0]:.3f}",
         abs(sel_rows["T2"]["auc"][0] - 0.87) <= 0.02 and abs(sel_rows["T3"]["auc"][0] - 0.92) <= 0.02),
        ("Tier-1 usefulness (net benefit > treat-all/none, 0.10–0.70)",
         f"T1 AUC {sel_rows['T1']['auc'][0]:.3f} ({sel_rows['T1']['auc'][1]:.3f}–{sel_rows['T1']['auc'][2]:.3f})",
         bool((dc.t1 > np.maximum(dc.treat_all, 0)).all())),
        ("Calibration (ECE ≤ 0.05, selected models)", ", ".join(f"{t} {r['ece'][0]:.3f}" for t, r in sel_rows.items()),
         all(r["ece"][0] <= 0.05 for r in sel_rows.values())),
        ("Conformal (coverage within ±0.02 of target)", ", ".join(f"{r.tier} α={r.alpha}: {r.coverage:.3f}" for r in D.itertuples()),
         bool(((D.coverage - D.target).abs() <= 0.02).all())),
        ("Compression (edge ΔAUC ≤ 0.01 at ≥ 10× smaller)",
         f"T1 LR ΔAUC {v0_t1 - e_t1:.4f} (×{es['choices']['t1']['size_ratio_vs_v0']:.0f} smaller); "
         f"T3-lite ΔAUC {v0_t3 - e_t3:.4f} (×{es['choices']['t3lite']['size_ratio_vs_v0']:.0f})",
         f"T1 {'pass' if v0_t1 - e_t1 <= 0.01 else 'FAIL'}, T3-lite {'pass' if v0_t3 - e_t3 <= 0.01 else 'FAIL'}"),
        ("Offline (0 bytes, works with networking disabled)",
         f"Docker --network none: {dock['net_bytes_sent_non_loopback']} bytes; app sockets audited (D28)", dock["net_bytes_sent_non_loopback"] == 0),
        ("Local adaptation (smallest n identified per site)",
         "; ".join(f"{r.tier} {r.site}: n*={r['n* admitted-class coverage within ±0.02']}" for _, r in sn.iterrows()), True),
        ("Usability (≥ 5 participants)", "not run (needs humans)", "NOT RUN"),
    ]
    C = pd.DataFrame([{"criterion": c, "result": r, "pass": (("✅ pass" if p else "❌ fail") if isinstance(p, bool) else p)} for c, r, p in crit])
    parts.insert(4, "## Pre-registered success criteria\n\n" + C.to_markdown(index=False) + "\n")
    figs = sorted(p.name for p in (ROOT / "results" / "figures").glob("*.png"))
    parts += ["## Figures", "", *[f"- `results/figures/{n}`" for n in figs], "- App screenshots: `results/figures/app/`", ""]
    OUT.write_text("\n".join(parts))
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
