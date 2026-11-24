"""Metrics, bootstrap CIs and DeLong test (guide 05)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

EPS = 1e-7


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, 10 equal-width bins, weighted by bin size."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = np.bincount(bins, minlength=n_bins)
    sp = np.bincount(bins, weights=p, minlength=n_bins)
    sy = np.bincount(bins, weights=y, minlength=n_bins)
    m = tot > 0
    return float(np.sum(np.abs(sp[m] - sy[m])) / len(y))


def reliability(y, p, n_bins: int = 10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = np.bincount(bins, minlength=n_bins)
    sp = np.bincount(bins, weights=p, minlength=n_bins)
    sy = np.bincount(bins, weights=y, minlength=n_bins)
    m = tot > 0
    return sp[m] / tot[m], sy[m] / tot[m], tot[m]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _newton_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    off = 0 if offset is None else offset
    for _ in range(iters):
        z = X @ w + off
        mu = 1 / (1 + np.exp(-z))
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Slope: logistic recalibration slope on logit(p).
    Intercept: calibration-in-the-large (intercept with logit(p) as offset; 0 = perfect)."""
    y = np.asarray(y, float); lp = logit(p)
    slope = _newton_logistic(np.column_stack([np.ones_like(lp), lp]), y)[1]
    citl = _newton_logistic(np.ones((len(y), 1)), y, offset=lp)[0]
    return float(slope), float(citl)


def sens_at_spec(y, p, spec: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def spec_at_sens(y, p, sens: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= sens
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def threshold_metrics(y, p, t: float) -> dict:
    y = np.asarray(y).astype(bool); pred = np.asarray(p) >= t
    tp = np.sum(pred & y); fp = np.sum(pred & ~y); fn = np.sum(~pred & y); tn = np.sum(~pred & ~y)
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    return {"threshold": t, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1,
            "accuracy": (tp + tn) / len(y)}


def core_metrics(y, p) -> dict:
    y = np.asarray(y); p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    slope, citl = calibration_slope_intercept(y, p)
    return {
        "auc": roc_auc_score(y, p), "prauc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "logloss": log_loss(y, p), "ece": ece(y, p),
        "cal_slope": slope, "citl": citl,
        "sens_at_90spec": sens_at_spec(y, p), "spec_at_90sens": spec_at_sens(y, p),
    }


# ---------- fast AUC for bootstrap ----------

def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    r = stats.rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


BOOT_METRICS = {
    "auc": lambda y, p: fast_auc(y, p),
    "prauc": average_precision_score,
    "brier": brier_score_loss,
    "ece": ece,
    "sens_at_90spec": sens_at_spec,
}


def bootstrap_ci(y, p, metrics=("auc", "prauc", "brier", "ece", "sens_at_90spec"), n_boot: int = 1000,
                 seed: int = 42) -> dict:
    """Percentile 95% CIs from n_boot resamples of the evaluation set."""
    y = np.asarray(y); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, pp = y[i], p[i]
        for m in metrics:
            vals[m].append(BOOT_METRICS[m](yy, pp))
    out = {}
    for m in metrics:
        v = np.array(vals[m])
        out[m] = (float(BOOT_METRICS[m](y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def paired_bootstrap_delta_auc(y, p_a, p_b, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """AUC(b) - AUC(a) with percentile 95% CI over paired resamples."""
    y = np.asarray(y); rng = np.random.default_rng(seed); n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(fast_auc(y[i], p_b[i]) - fast_auc(y[i], p_a[i]))
    return float(fast_auc(y, p_b) - fast_auc(y, p_a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------- DeLong (fast, Sun & Xu 2014) ----------

def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x)


def delong_test(y, p_a, p_b) -> tuple[float, float, float]:
    """Paired DeLong test. Returns (auc_a, auc_b, two-sided p-value)."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p_a, p_b]).astype(float)
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in preds])
    tz_pos, tz_neg = tz[:, y], tz[:, ~y]
    aucs = (tz_pos.sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz_pos - tx) / n
    v10 = 1 - (tz_neg - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[1] - aucs[0]) / np.sqrt(var)
    pval = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(pval)


# ---------- decision curve ----------

def net_benefit(y, p, thresholds) -> np.ndarray:
    y = np.asarray(y).astype(bool); n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & y); fp = np.sum(pred & ~y)
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def net_benefit_all(y, thresholds) -> np.ndarray:
    prev = np.mean(y)
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])
