"""Conformal code: coverage on simulated data, edge cases."""
import numpy as np
import pytest

from src import conformal


def simulate(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * z + shift - 0.9)))).astype(int)
    p = 1 / (1 + np.exp(-(1.5 * z - 0.9)))  # model calibrated when shift = 0
    return p, y


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_and_class_coverage(alpha):
    covs, ca, cd = [], [], []
    for s in range(30):
        p_cal, y_cal = simulate(2000, s)
        p_te, y_te = simulate(5000, 1000 + s)
        cfg = conformal.fit(p_cal, y_cal, alpha)
        r = conformal.evaluate(p_te, y_te, cfg)
        covs.append(r["coverage"]); ca.append(r["coverage_admit"]); cd.append(r["coverage_discharge"])
    # guarantee: >= 1-alpha in expectation; upper bound 1-alpha+1/(n_k+1) (tiny here)
    assert abs(np.mean(covs) - (1 - alpha)) < 0.01
    assert np.mean(ca) >= 1 - alpha - 0.01 and np.mean(cd) >= 1 - alpha - 0.01


def test_coverage_breaks_under_shift_and_local_refit_restores():
    p_cal, y_cal = simulate(5000, 1)
    cfg = conformal.fit(p_cal, y_cal, 0.10)
    p_new, y_new = simulate(20000, 2, shift=-1.5)  # new site, far lower admission rate
    shifted = conformal.evaluate(p_new, y_new, cfg)
    assert abs(shifted["coverage_admit"] - 0.9) > 0.03  # class-conditional coverage breaks
    local = conformal.fit(p_new[:2000], y_new[:2000], 0.10)
    fixed = conformal.evaluate(p_new[2000:], y_new[2000:], local)
    assert abs(fixed["coverage_admit"] - 0.9) < 0.03 and abs(fixed["coverage_discharge"] - 0.9) < 0.03


def test_labels():
    cfg = {"alpha": 0.1, "qhat_admit": 0.6, "qhat_discharge": 0.6}
    assert conformal.label(0.9, cfg) == "likely_admission"
    assert conformal.label(0.1, cfg) == "likely_discharge"
    assert conformal.label(0.5, cfg) == "uncertain"
    cfg2 = {"alpha": 0.1, "qhat_admit": 0.1, "qhat_discharge": 0.1}
    assert conformal.label(0.5, cfg2) == "uncertain"  # empty set -> uncertain


def test_small_calibration_set_includes_label():
    cfg = conformal.fit(np.array([0.2, 0.8]), np.array([0, 1]), alpha=0.05)
    assert cfg["qhat_admit"] == 1.0 and cfg["qhat_discharge"] == 1.0


def test_risk_coverage_monotone_start():
    p, y = simulate(5000, 3)
    cov, err, aurc = conformal.risk_coverage(p, y)
    assert err[len(err) // 10] < err[-1]
    assert 0 < aurc < 0.5


def test_calibrators_roundtrip():
    from src import calibrate
    p, y = simulate(5000, 4)
    p_bad = np.clip(p ** 0.5, 1e-4, 1 - 1e-4)
    for kind in ("platt", "isotonic"):
        c = calibrate.make(kind).fit(p_bad, y)
        c2 = calibrate.from_dict(c.to_dict())
        assert np.allclose(c.predict(p_bad), c2.predict(p_bad), atol=1e-5)
        from src.evaluate import ece
        assert ece(y, c.predict(p_bad)) < ece(y, p_bad)
