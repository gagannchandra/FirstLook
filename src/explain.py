"""E7: SHAP global/local explanations + stability (S1 validation rows only).

Usage: python -m src.explain
Outputs: fig05_shap_beeswarm_{t1,t3}.png, fig05b_shap_bar_{tier}.png,
         results/tables/E7_shap_top_features.csv, E7_shap_stability.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src import viz
from src.common import PROC, ROOT, load_model, meta, rows_for, selection
from src.train import LOGS, fit_model

TABLES = ROOT / "results" / "tables"
N_SHAP = 5000
SEED = 42


def pretty(name: str) -> str:
    n = name.replace("triage_vital_", "").replace("cc_", "complaint: ").replace("meds_", "meds: ")
    n = {"hr": "heart rate", "sbp": "systolic BP", "dbp": "diastolic BP", "rr": "resp. rate", "o2": "SpO₂",
         "o2_device": "on oxygen", "temp": "temperature", "sex_male": "male sex", "esi": "ESI level"}.get(n, n)
    return n.replace("arrival=", "arrival: ").replace("hour=", "arrival hour ").replace("prevdispo=", "previous dispo: ")


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "named_steps"):  # logistic-regression pipeline -> exact linear SHAP on the transformed space
        imp, sc, lr = model.named_steps["impute"], model.named_steps["scale"], model.named_steps["lr"]
        Z = sc.transform(imp.transform(X))
        contrib = Z * lr.coef_[0]  # standardised features have mean 0 on training data
        n = X.shape[1]
        out = contrib[:, :n].copy()
        ind = imp.indicator_.features_ if imp.indicator_ is not None else []
        for j, f in enumerate(ind):  # fold missing-indicator contributions into their feature
            out[:, f] += contrib[:, n + j]
        return out
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def mean_abs(sv: np.ndarray, cols) -> pd.Series:
    return pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)


def main() -> None:
    viz.setup()
    sel = selection()["tiers"]
    m_all = meta()
    rng = np.random.default_rng(SEED)
    va = rows_for("val")
    sample = np.sort(rng.choice(va, N_SHAP, replace=False))
    tops, stab = [], []
    for tier in ("t1", "t2", "t3"):
        if tier not in sel:
            continue
        X = pd.read_parquet(PROC / f"{tier}.parquet")
        model = load_model(tier, sel[tier]["model"])
        Xs = X.iloc[sample]
        sv = shap_values(model, Xs)
        imp = mean_abs(sv, X.columns)
        imp.to_frame("mean_abs_shap").assign(rank=np.arange(1, len(imp) + 1)).head(50) \
            .to_csv(TABLES / f"E7_shap_importance_{tier}.csv")
        for r, (f, v) in enumerate(imp.head(20).items(), 1):
            tops.append({"tier": tier, "rank": r, "feature": f, "mean_abs_shap": v})

        # global bar (top 15)
        top = imp.head(15)[::-1]
        fig, ax = viz.plt.subplots(figsize=(6.5, 4.6))
        ax.barh([pretty(c) for c in top.index], top.values, height=0.6, color=viz.TIER_COLOR[tier])
        ax.set_xlabel("Mean |SHAP| (log-odds)"); ax.grid(axis="y", visible=False)
        ax.set_title(f"{viz.TIER_NAME[tier]}: features most associated with model-estimated risk")
        viz.save(fig, f"fig05b_shap_bar_{tier}.png")

        # beeswarm (top 15)
        cols = list(imp.head(15).index)
        ci = [list(X.columns).index(c) for c in cols]
        exp = shap.Explanation(values=sv[:, ci], data=Xs[cols].to_numpy(), feature_names=[pretty(c) for c in cols])
        viz.plt.figure(figsize=(7, 5.2))
        shap.plots.beeswarm(exp, max_display=15, show=False, color_bar=True)
        f = viz.plt.gcf()
        f.axes[0].set_title(f"{viz.TIER_NAME[tier]} · SHAP on {N_SHAP:,} validation visits", loc="left", fontweight="bold")
        f.axes[0].set_xlabel("SHAP value (impact on log-odds of admission)")
        viz.save(f, f"fig05_shap_beeswarm_{tier}.png")

        # stability across sites: SHAP importance computed separately on each site's val rows
        site = m_all.site.to_numpy()
        per_site = {}
        for s in ("A", "B", "C"):
            rs = va[site[va] == s]
            rs = np.sort(rng.choice(rs, min(3000, len(rs)), replace=False))
            per_site[s] = mean_abs(shap_values(model, X.iloc[rs]), X.columns)
        for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
            top_union = list(dict.fromkeys(list(per_site[a].head(10).index) + list(per_site[b].head(10).index)))
            rho = spearmanr(per_site[a][top_union], per_site[b][top_union]).statistic
            overlap = len(set(per_site[a].head(10).index) & set(per_site[b].head(10).index))
            stab.append({"tier": tier, "comparison": f"site {a} vs {b}", "spearman_top10_union": rho, "top10_overlap": overlap})

        # stability across seeds (T1 and T2 only: retrain with 3 seeds; T3 too costly)
        if tier in ("t1", "t2"):
            params = json.loads((LOGS / f"best_{tier}_{sel[tier]['model']}.json").read_text())["best_params"]
            tr = rows_for("train"); y = m_all.y.to_numpy()
            seed_imp = {}
            for sd in (1, 2, 3):
                mm, _ = fit_model(sel[tier]["model"], params, X.iloc[tr], y[tr], X.iloc[va], y[va], seed=sd)
                seed_imp[sd] = mean_abs(shap_values(mm, Xs), X.columns)
            seed_imp[0] = imp
            for a, b in [(0, 1), (0, 2), (0, 3)]:
                top_union = list(dict.fromkeys(list(seed_imp[a].head(10).index) + list(seed_imp[b].head(10).index)))
                rho = spearmanr(seed_imp[a][top_union], seed_imp[b][top_union]).statistic
                overlap = len(set(seed_imp[a].head(10).index) & set(seed_imp[b].head(10).index))
                stab.append({"tier": tier, "comparison": f"seed 42 vs {b}", "spearman_top10_union": rho, "top10_overlap": overlap})
        print(tier, "top 10:", [pretty(c) for c in imp.head(10).index], flush=True)
        del X
    pd.DataFrame(tops).to_csv(TABLES / "E7_shap_top_features.csv", index=False)
    st = pd.DataFrame(stab)
    st.to_csv(TABLES / "E7_shap_stability.csv", index=False)
    print(st.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
