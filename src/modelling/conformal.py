"""Split conformal prediction for binary classification (own NumPy implementation).

Mondrian (class-conditional) variant: a separate threshold per true class, so the
coverage guarantee holds for admitted and discharged patients separately.

Non-conformity score of label k for a patient with P(admit)=p:
    s_k = 1 - p_k,  where p_1 = p and p_0 = 1 - p.
Calibration: for each class k, q_k = the ceil((n_k+1)(1-alpha))/n_k empirical
quantile of the scores of calibration patients whose true label is k.
Prediction set: {k : s_k <= q_k}.
Guarantee (exchangeability): P(true label in set | true label = k) >= 1 - alpha.
"""
from __future__ import annotations

import numpy as np

LABEL_ADMIT, LABEL_DISCHARGE = 1, 0


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:  # too few points for this alpha -> include the label always
        return 1.0
    return float(np.quantile(scores, level, method="higher"))


def fit(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10, mondrian: bool = True) -> dict:
    p_cal = np.asarray(p_cal, float); y_cal = np.asarray(y_cal).astype(int)
    p_true = np.where(y_cal == 1, p_cal, 1 - p_cal)
    scores = 1 - p_true
    if mondrian:
        q = {k: _quantile(scores[y_cal == k], alpha) for k in (0, 1)}
    else:
        qq = _quantile(scores, alpha)
        q = {0: qq, 1: qq}
    return {"alpha": alpha, "mondrian": mondrian, "qhat_discharge": q[0], "qhat_admit": q[1],
            "n_cal": int(len(y_cal))}


def predict_sets(p: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns boolean arrays (admit_in_set, discharge_in_set)."""
    p = np.asarray(p, float)
    admit = (1 - p) <= cfg["qhat_admit"]
    discharge = p <= cfg["qhat_discharge"]
    return admit, discharge


def label(p: float, cfg: dict) -> str:
    a, d = predict_sets(np.array([p]), cfg)
    a, d = bool(a[0]), bool(d[0])
    if a and not d:
        return "likely_admission"
    if d and not a:
        return "likely_discharge"
    return "uncertain"  # both labels, or the (rare) empty set


def evaluate(p: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    y = np.asarray(y).astype(int)
    a, d = predict_sets(p, cfg)
    covered = np.where(y == 1, a, d)
    both = a & d
    empty = ~a & ~d
    single = a ^ d
    pred_single = np.where(a, 1, 0)
    err_answered = float(np.mean(pred_single[single] != y[single])) if single.any() else float("nan")
    err_all = float(np.mean((np.asarray(p) >= 0.5).astype(int) != y))
    return {
        "coverage": float(covered.mean()),
        "coverage_admit": float(covered[y == 1].mean()) if (y == 1).any() else float("nan"),
        "coverage_discharge": float(covered[y == 0].mean()) if (y == 0).any() else float("nan"),
        "abstention": float((both | empty).mean()),
        "empty_rate": float(empty.mean()),
        "answered": float(single.mean()),
        "error_answered": err_answered,
        "error_all": err_all,
    }


def risk_coverage(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective-prediction curve: answer the most confident cases first.
    Returns (coverage fractions, error rates on answered cases, AURC)."""
    p = np.asarray(p, float); y = np.asarray(y).astype(int)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    return cov, cum, float(np.mean(cum))
