"""E10 missing-vitals robustness, E11 subgroups/fairness, E12 decision curves.

Usage:
  python -m src.secondary --split val            # Phase 3 (development)
  FIRSTLOOK_OPEN_TEST=1 python -m src.secondary --split test   # Phase 5 only
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import viz
from src.common import PROC, ROOT, calibrator, load_model, meta, rows_for, selection
from src.evaluate import ece, fast_auc, net_benefit, net_benefit_all, threshold_metrics, calibration_slope_intercept
from src.features import VITALS
from src.train import LOGS, fit_model

TABLES = ROOT / "results" / "tables"
SEED = 42
MAIN_VITALS = [v for v in VITALS if v != "triage_vital_o2_device"]


# ---------------- E10 ----------------

def train_indicator_model(tier, sel, X, y):
    """Variant with explicit missing-indicator columns for the 7 vitals."""
    params = json.loads((LOGS / f"best_{tier}_{sel['model']}.json").read_text())["best_params"]
    Xi = add_indicators(X)
    tr, va = rows_for("train"), rows_for("val")
    m, _ = fit_model(sel["model"], params, Xi.iloc[tr], y[tr], Xi.iloc[va], y[va])
    return m


def add_indicators(X):
    X = X.copy()
    for v in VITALS:
        X[f"miss_{v}"] = X[v].isna().astype(np.float32)
    return X


def e10(split: str) -> pd.DataFrame:
    sel_all = selection()["tiers"]
    y_all = meta().y.to_numpy()
    rows_out = []
    for tier in ("t1", "t2"):
        sel = sel_all[tier]
        X = pd.read_parquet(PROC / f"{tier}.parquet")
        model = load_model(tier, sel["model"])
        cal = calibrator(tier)
        tr = rows_for("train")
        med = X.iloc[tr][VITALS].median()
        ind_model = train_indicator_model(tier, sel, X, y_all)
        # indicator model gets its own Platt calibrator on S3 cal rows
        from src.calibrate import Platt
        ca = rows_for("cal")
        ind_cal = Platt().fit(ind_model.predict_proba(add_indicators(X.iloc[ca]))[:, 1], y_all[ca])
        rows = rows_for(split)
        Xe = X.iloc[rows].reset_index(drop=True)
        y = y_all[rows]
        rng = np.random.default_rng(SEED)

        def evaluate(Xm, scenario):
            out = []
            p_native = cal.predict(model.predict_proba(Xm)[:, 1])
            Xmed = Xm.copy(); Xmed[VITALS] = Xmed[VITALS].fillna(med)
            p_med = cal.predict(model.predict_proba(Xmed)[:, 1])
            p_ind = ind_cal.predict(ind_model.predict_proba(add_indicators(Xm))[:, 1])
            for strat, p in [("native NaN", p_native), ("median impute", p_med), ("missing indicator", p_ind)]:
                out.append({"tier": tier, "scenario": scenario, "strategy": strat, "auc": fast_auc(y, p), "ece": ece(y, p)})
            return out

        rows_out += evaluate(Xe, "none (as recorded)")
        for v in MAIN_VITALS + ["triage_vital_o2_device"]:
            Xm = Xe.copy(); Xm[v] = np.nan
            if v == "triage_vital_o2":
                Xm["triage_vital_o2_device"] = Xm["triage_vital_o2_device"]  # device kept
            rows_out += evaluate(Xm, f"{v.replace('triage_vital_', '')} missing (100% rows)")
        Xm = Xe.copy(); Xm[VITALS] = np.nan
        rows_out += evaluate(Xm, "all vitals missing (100% rows)")
        for k in (1, 2, 3):
            for frac in (0.1, 0.3, 0.5):
                Xm = Xe.copy()
                hit = np.flatnonzero(rng.random(len(Xm)) < frac)
                vals = Xm[MAIN_VITALS].to_numpy().copy()
                for r in hit:
                    vals[r, rng.choice(len(MAIN_VITALS), k, replace=False)] = np.nan
                Xm[MAIN_VITALS] = vals
                rows_out += evaluate(Xm, f"{k} random vital(s) @ {int(frac*100)}% rows")
        print(f"E10 {tier} done", flush=True)
        del X
    df = pd.DataFrame(rows_out)
    base = df[df.scenario == "none (as recorded)"].set_index(["tier", "strategy"]).auc
    df["delta_auc"] = [r.auc - base[(r.tier, r.strategy)] for r in df.itertuples()]
    df.to_csv(TABLES / f"E10_missing_vitals_{split}.csv", index=False)

    viz.setup()
    fig, axes = viz.plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tier in zip(axes, ("t1", "t2")):
        d = df[(df.tier == tier) & df.scenario.str.contains("random")]
        for i, strat in enumerate(["native NaN", "median impute", "missing indicator"]):
            for k, ls in zip((1, 2, 3), ("-", "-", "-")):
                dd = d[(d.strategy == strat) & d.scenario.str.startswith(f"{k} ")]
                fr = [int(s.split("@ ")[1].split("%")[0]) for s in dd.scenario]
                ax.plot(fr, dd.auc, color=viz.SERIES[i], alpha=[1, 0.7, 0.45][k - 1], marker="o", ms=4,
                        label=f"{strat}" if k == 1 else None)
                if strat == "native NaN":
                    ax.annotate(f"{k} vital{'s' if k > 1 else ''}", (fr[-1], dd.auc.iloc[-1]), xytext=(4, 0),
                                textcoords="offset points", fontsize=8, color=viz.INK2, va="center")
        ax.set_title(viz.TIER_NAME[tier]); ax.set_xlabel("% of visits with vitals removed"); ax.set_xticks([10, 30, 50])
        ax.set_xlim(5, 60)
    axes[0].set_ylabel(f"AUC ({split})"); axes[0].legend(loc="lower left")
    fig.suptitle("Removing vitals at random costs little AUC; strategies are close (shade = 1, 2, 3 vitals)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, f"fig10_missing_vitals_{split}.png")
    return df


# ---------------- E11 ----------------

def subgroup_table(y, p, groups: pd.Series, t_high: float, label: str) -> list[dict]:
    out = []
    for g in groups.dropna().unique():
        m = (groups == g).to_numpy()
        if m.sum() < 50 or y[m].min() == y[m].max():
            continue
        tm = threshold_metrics(y[m], p[m], t_high)
        _, citl = calibration_slope_intercept(y[m], p[m])
        out.append({"attribute": label, "group": str(g), "n": int(m.sum()), "admit_rate": float(y[m].mean()),
                    "auc": fast_auc(y[m], p[m]), "sens@t_high": tm["sens"], "ppv@t_high": tm["ppv"],
                    "citl": citl, "mean_p": float(p[m].mean())})
    return out


def e11(split: str) -> pd.DataFrame:
    sel_all = selection()["tiers"]
    md = meta()
    rows_out = []
    for tier in ("t1", "t2"):
        sel = sel_all[tier]
        X = pd.read_parquet(PROC / f"{tier}.parquet")
        rows = rows_for(split)
        p = calibrator(tier).predict(load_model(tier, sel["model"]).predict_proba(X.iloc[rows])[:, 1])
        y = md.y.to_numpy()[rows]
        sub = md.iloc[rows].reset_index(drop=True)
        overall_auc = fast_auc(y, p)
        age = pd.cut(sub.age, [17, 39, 64, 79, 200], labels=["18–39", "40–64", "65–79", "80+"]).astype(str)
        res = []
        res += subgroup_table(y, p, pd.Series(["all"] * len(y)), sel["t_high"], "overall")
        res += subgroup_table(y, p, age, sel["t_high"], "age band")
        res += subgroup_table(y, p, sub.gender, sel["t_high"], "sex")
        res += subgroup_table(y, p, sub.site, sel["t_high"], "site (S1)")
        res += subgroup_table(y, p, sub.race, sel["t_high"], "race (audit only)")
        res += subgroup_table(y, p, sub.insurance_status, sel["t_high"], "insurance (audit only)")
        res += subgroup_table(y, p, sub.ethnicity, sel["t_high"], "ethnicity (audit only)")
        for r in res:
            r["tier"] = tier; r["split"] = split
            r["flag"] = ("AUC gap > 0.05" if r["auc"] < overall_auc - 0.05 else "") + \
                        (" |CITL| > 0.2" if abs(r["citl"]) > 0.2 else "")
        rows_out += res
        del X
    df = pd.DataFrame(rows_out)
    df.to_csv(TABLES / f"E11_subgroups_{split}.csv", index=False)
    return df


# ---------------- E12 ----------------

def e12(split: str) -> pd.DataFrame:
    sel_all = selection()["tiers"]
    th = np.round(np.arange(0.10, 0.701, 0.01), 2)
    out = {"threshold": th}
    y = None
    for tier in ("t1", "t2", "t3"):
        if tier not in sel_all:
            continue
        X = pd.read_parquet(PROC / f"{tier}.parquet")
        rows = rows_for(split)
        p = calibrator(tier).predict(load_model(tier, sel_all[tier]["model"]).predict_proba(X.iloc[rows])[:, 1])
        y = meta().y.to_numpy()[rows]
        out[tier] = net_benefit(y, p, th)
        del X
    out["treat_all"] = net_benefit_all(y, th)
    out["treat_none"] = np.zeros_like(th)
    df = pd.DataFrame(out)
    df.to_csv(TABLES / f"E12_decision_curve_{split}.csv", index=False)
    viz.setup()
    fig, ax = viz.plt.subplots(figsize=(7.5, 4.4))
    for t in ("t1", "t2", "t3"):
        if t in df:
            ax.plot(df.threshold, df[t], color=viz.TIER_COLOR[t], label=viz.TIER_NAME[t])
    ax.plot(df.threshold, df.treat_all, color=viz.NEUTRAL, label="Treat all as admit")
    ax.plot(df.threshold, df.treat_none, color=viz.INK2, lw=1, label="Treat none")
    ax.set_ylim(-0.05, max(df.treat_all.max(), df[[c for c in ("t1", "t2", "t3") if c in df]].max().max()) * 1.1)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curves ({split}): every tier beats treat-all / treat-none from 0.10 to 0.70" if
                 all((df[t] > np.maximum(df.treat_all, 0)).all() for t in ("t1", "t2", "t3") if t in df)
                 else f"Decision curves ({split})")
    ax.legend(loc="upper right")
    viz.save(fig, f"fig07_decision_curves_{split}.png")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--only", default="e10,e11,e12")
    a = ap.parse_args()
    if "e12" in a.only:
        d = e12(a.split); print(d.iloc[::10].round(4).to_string(index=False))
    if "e11" in a.only:
        d = e11(a.split); print(d.round(3).to_string(index=False))
    if "e10" in a.only:
        d = e10(a.split); print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
