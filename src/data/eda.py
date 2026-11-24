"""EDA figures/tables (guide 02 §7). Aggregates only; no row-level data leaves data/.

Usage: python -m src.data.eda
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import viz
from src.data.build import SITE_TYPE
from src.data.features import VITALS

ROOT = viz.ROOT
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def bars(ax, labels, values, color, fmt="{:.0%}"):
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.6, color=color)
    ax.set_xticks(x, labels)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.01, fmt.format(v), ha="center", va="bottom", fontsize=8, color=viz.INK2)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def main() -> None:
    viz.setup()
    meta = pd.read_parquet(PROC / "meta.parquet")
    t1 = pd.read_parquet(PROC / "t1.parquet")
    y = meta.y

    # Figure 2: admission rate by ESI / age band / site
    meta["age_band"] = pd.cut(meta.age, [17, 39, 64, 79, 200], labels=["18–39", "40–64", "65–79", "80+"])
    esi = meta.groupby("esi").y.agg(["mean", "size"])
    age = meta.groupby("age_band", observed=True).y.agg(["mean", "size"])
    site = meta.groupby("site").y.agg(["mean", "size"])
    arr = meta.groupby("arrivalmode").y.agg(["mean", "size"]).sort_values("size", ascending=False)
    fig, axes = viz.plt.subplots(1, 3, figsize=(11, 3.2))
    esi_l = [e if e != "nan" else "missing" for e in esi.index]
    bars(axes[0], esi_l, esi["mean"].values, viz.SERIES[0]); axes[0].set_title("By ESI level"); axes[0].set_ylabel("Admission rate")
    bars(axes[1], list(age.index.astype(str)), age["mean"].values, viz.SERIES[0]); axes[1].set_title("By age band")
    bars(axes[2], [f"Site {s}" for s in site.index], site["mean"].values, viz.SERIES[0]); axes[2].set_title("By site")
    fig.suptitle("Admission rate rises with acuity and age; Site C admits about half as often", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, "fig02_admission_rate_by_esi_age_site.png")

    rows = []
    for name, g in [("esi", esi), ("age_band", age), ("site", site), ("arrivalmode", arr)]:
        for k, r in g.iterrows():
            rows.append({"variable": name, "level": str(k), "n": int(r["size"]), "admit_rate": round(float(r["mean"]), 4)})
    pd.DataFrame(rows).to_csv(TABLES / "eda_admission_rates.csv", index=False)

    # Missingness of triage vitals by site
    miss = t1[VITALS].isna().groupby(meta.site).mean().T
    miss.index = [v.replace("triage_vital_", "") for v in miss.index]
    miss.to_csv(TABLES / "eda_vital_missingness_by_site.csv")
    fig, ax = viz.plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(miss.index)); w = 0.26
    for i, s in enumerate(miss.columns):
        ax.bar(x + (i - 1) * w, miss[s].values, width=w - 0.02, color=viz.SITE_COLOR[s], label=f"Site {s}")
    ax.set_xticks(x, miss.index); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share missing"); ax.grid(axis="x", visible=False); ax.legend(ncol=3, loc="upper left")
    ax.set_title("Triage vitals are missing in about 30% of visits; SpO₂ in about half")
    viz.save(fig, "fig_eda_vital_missingness.png")

    # Vital distributions admitted vs discharged (quantiles only)
    q = t1[VITALS].groupby(y).quantile([0.25, 0.5, 0.75]).T
    q.to_csv(TABLES / "eda_vital_quantiles_by_outcome.csv")

    # Top 20 chief complaints by frequency (aggregate counts only)
    cc = [c for c in t1.columns if c.startswith("cc_")]
    n = t1[cc].sum()
    rate = (t1[cc].mul(y, axis=0).sum() / n.replace(0, np.nan))
    top = pd.DataFrame({"n": n, "admit_rate": rate}).sort_values("n", ascending=False).head(20)
    top.index = [c[3:] for c in top.index]
    top.round(4).to_csv(TABLES / "eda_top20_chief_complaints.csv")
    print(top.round(3).to_string())
    print(miss.round(3).to_string())


if __name__ == "__main__":
    main()
