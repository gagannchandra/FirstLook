"""Summarise E5/E6: Tables C and C2, signature local-adaptation figure, LOHO calibration plots.

Usage: python -m src.report_loho   (after python -m src.loho)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import viz
from src.common import ROOT, meta
from src.evaluate import ece, reliability
from src.loho import ALPHA

TABLES = ROOT / "results" / "tables"
PRED = ROOT / "data" / "preds"
SITE_TYPE = {"A": "A (academic, inferred)", "B": "B (community, inferred)", "C": "C (free-standing, inferred)"}


def smallest_n(g: pd.DataFrame, col: str, target: float, tol: float = 0.02):
    """Smallest n whose mean is within tol of target AND stays there for all larger n."""
    m = g.groupby("n")[col].mean().sort_index()
    ok = (m - target).abs() <= tol
    for n in m.index:
        if ok.loc[n:].all():
            return int(n)
    return None


def main():
    viz.setup()
    df = pd.read_csv(TABLES / "E6_local_adaptation_raw.csv")
    target = 1 - ALPHA

    # ---------- Table C ----------
    rows = []
    for (t, s), g in df.groupby(["tier", "site"]):
        base = g[(g.arm == "none") & (g.n == 0)].iloc[0]
        ref = g[g.arm.str.startswith("source_val")].iloc[0]
        r1000 = g[(g.arm == "recal") & (g.n == 1000)]
        rc1000 = g[(g.arm == "recal+conformal") & (g.n == 1000)]
        rows.append({"held-out site": SITE_TYPE[s], "tier": t.upper(), "prevalence": base.prevalence,
                     "in-dist AUC (source val)": ref.auc, "AUC": base.auc, "calib slope": base.cal_slope,
                     "CITL": base.citl, "ECE before": base.ece, "ECE after local recal (n=1000)": r1000.ece.mean(),
                     "coverage (source conformal)": base.coverage, "coverage admitted": base.coverage_admit,
                     "coverage discharged": base.coverage_discharge,
                     "coverage after local conformal (n=1000)": rc1000.coverage.mean(),
                     "coverage admitted after (n=1000)": rc1000.coverage_admit.mean()})
    c = pd.DataFrame(rows)
    c.to_csv(TABLES / "E5_loho.csv", index=False)
    (TABLES / "E5_loho.md").write_text(c.round(3).to_markdown(index=False))
    print(c.round(3).to_string(index=False))

    # ---------- Table C2 ----------
    c2 = []
    for (t, s), g in df.groupby(["tier", "site"]):
        n0 = g[(g.arm == "none") & (g.n == 0)].iloc[0]
        c2.append({"tier": t.upper(), "site": s, "n": 0, "ECE (recal only)": f"{n0.ece:.3f}",
                   "coverage (recal + local conformal)": f"{n0.coverage:.3f}",
                   "coverage admitted": f"{n0.coverage_admit:.3f}", "coverage discharged": f"{n0.coverage_discharge:.3f}",
                   "abstention %": f"{100*n0.abstention:.1f}", "AUC (unchanged model)": f"{n0.auc:.3f}"})
        for n, gg in g[g.n > 0].groupby("n"):
            rec = gg[gg.arm == "recal"]; rc = gg[gg.arm == "recal+conformal"]
            f = lambda x: f"{x.mean():.3f} ± {x.std():.3f}"  # noqa: E731
            c2.append({"tier": t.upper(), "site": s, "n": int(n), "ECE (recal only)": f(rec.ece),
                       "coverage (recal + local conformal)": f(rc.coverage), "coverage admitted": f(rc.coverage_admit),
                       "coverage discharged": f(rc.coverage_discharge),
                       "abstention %": f"{100*rc.abstention.mean():.1f} ± {100*rc.abstention.std():.1f}",
                       "AUC (unchanged model)": f(rc.auc)})
    c2 = pd.DataFrame(c2)
    c2.to_csv(TABLES / "E6_local_adaptation_table_C2.csv", index=False)
    (TABLES / "E6_local_adaptation_table_C2.md").write_text(c2.to_markdown(index=False))

    # smallest n summary
    sn = []
    for (t, s), g in df.groupby(["tier", "site"]):
        rc = g[g.arm == "recal+conformal"]
        sn.append({"tier": t.upper(), "site": s,
                   "n* marginal coverage within ±0.02": smallest_n(rc, "coverage", target),
                   "n* admitted-class coverage within ±0.02": smallest_n(rc, "coverage_admit", target),
                   "n* discharged-class coverage within ±0.02": smallest_n(rc, "coverage_discharge", target),
                   "n* ECE ≤ 0.02 (recal only)": next((int(n) for n, v in g[g.arm == "recal"].groupby("n").ece.mean().items()
                                                        if v <= 0.02), None),
                   "coverage at n=0": float(g[(g.arm == "none") & (g.n == 0)].coverage.iloc[0]),
                   "admitted coverage at n=0": float(g[(g.arm == "none") & (g.n == 0)].coverage_admit.iloc[0])})
    sn = pd.DataFrame(sn)
    sn.to_csv(TABLES / "E6_smallest_n.csv", index=False)
    (TABLES / "E6_smallest_n.md").write_text(sn.to_markdown(index=False))
    print(sn.to_string(index=False))

    # retrain reference
    rt = df[df.arm == "retrain (ref)"].groupby(["tier", "site", "n"])[["auc", "ece", "coverage", "coverage_admit"]].mean()
    rt.to_csv(TABLES / "E6_retrain_reference.csv")

    # ---------- signature figure ----------
    for t in sorted(df.tier.unique()):
        d = df[df.tier == t]
        fig, axes = viz.plt.subplots(1, 3, figsize=(13, 4.0))
        panels = [("coverage_admit", "Coverage — admitted patients", target),
                  ("coverage_discharge", "Coverage — discharged patients", target),
                  ("ece", "ECE (calibration error)", None)]
        for ax, (col, title, tgt) in zip(axes, panels):
            for s in ("A", "B", "C"):
                g = d[d.site == s]
                arm = "recal" if col == "ece" else "recal+conformal"
                n0 = g[(g.arm == "none") & (g.n == 0)][col].iloc[0]
                m = g[g.arm == arm].groupby("n")[col].agg(["mean", "std"])
                xs = np.array([50] + m.index.tolist()); ys = np.array([n0] + m["mean"].tolist())
                sd = np.array([0] + m["std"].fillna(0).tolist())
                ax.fill_between(xs, ys - sd, ys + sd, color=viz.SITE_COLOR[s], alpha=0.12, lw=0)
                ax.plot(xs, ys, color=viz.SITE_COLOR[s], marker="o", ms=4, label=f"held-out site {s}")
                ax.scatter([50], [n0], s=50, color=viz.SURFACE, edgecolor=viz.SITE_COLOR[s], zorder=4, linewidth=2)
            if tgt:
                ax.axhline(tgt, color=viz.INK2, lw=1)
                ax.axhspan(tgt - 0.02, tgt + 0.02, color=viz.GRID, alpha=0.6, lw=0)
                ax.text(5200, tgt + 0.022, "target 0.90 ± 0.02", fontsize=8, color=viz.INK2, ha="right")
            ax.set_xscale("log"); ax.set_xticks([50, 100, 250, 500, 1000, 2000, 5000])
            ax.set_xticklabels(["0", "100", "250", "500", "1k", "2k", "5k"])
            ax.set_xlabel("Local visits used (no retraining)"); ax.set_title(title)
        axes[0].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Local adaptation at an unseen hospital ({t.upper()}): open circle = no adaptation; "
                     "line = local Platt + local conformal, mean ± SD of 20 draws", x=0.01, ha="left", fontweight="bold", fontsize=11)
        fig.tight_layout()
        viz.save(fig, f"fig08_local_adaptation_{t}.png")

    # ---------- LOHO reliability ----------
    y_all = meta().y.to_numpy()
    tiers = sorted(df.tier.unique())
    fig, axes = viz.plt.subplots(1, len(tiers), figsize=(5.2 * len(tiers), 4.2), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), tiers):
        ax.plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
        for s in ("A", "B", "C"):
            f = PRED / f"loho_{t}_{s}.parquet"
            if not f.exists():
                continue
            p = pd.read_parquet(f)
            yy = y_all[p.row.to_numpy()]
            mp, fy, _ = reliability(yy, p.p_source.to_numpy())
            ax.plot(mp, fy, marker="o", ms=4, color=viz.SITE_COLOR[s],
                    label=f"held-out {s}: ECE {ece(yy, p.p_source.to_numpy()):.3f}, prev {yy.mean():.0%}")
        ax.set_title(f"{t.upper()} model trained on the other two sites"); ax.set_xlabel("Predicted probability")
        ax.legend(loc="upper left", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Observed admission rate")
    fig.suptitle("E5: calibration at an unseen hospital, before any local adaptation", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, "fig11_loho_calibration.png")


if __name__ == "__main__":
    main()
