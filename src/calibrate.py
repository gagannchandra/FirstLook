"""Platt and isotonic calibrators with JSON (de)serialisation for the model bundle."""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    kind = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, p, y):
        from src.evaluate import _newton_logistic  # training-time only
        lp = logit(p)
        w = _newton_logistic(np.column_stack([np.ones_like(lp), lp]), np.asarray(y, float))
        self.b, self.a = float(w[0]), float(w[1])
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-(self.a * logit(p) + self.b)))

    def to_dict(self):
        return {"kind": "platt", "a": self.a, "b": self.b, "formula": "1/(1+exp(-(a*logit(p)+b)))"}


class Isotonic:
    kind = "isotonic"

    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, float)
        self.y = None if y is None else np.asarray(y, float)

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression  # training-time only
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), np.asarray(y, float))
        self.x, self.y = ir.X_thresholds_.astype(float), ir.y_thresholds_.astype(float)
        return self

    def predict(self, p):
        # keep probabilities strictly inside (0,1) so conformal / logit stay finite
        return np.clip(np.interp(np.asarray(p, float), self.x, self.y), 1e-4, 1 - 1e-4)

    def to_dict(self):
        return {"kind": "isotonic", "x": self.x.tolist(), "y": self.y.tolist(),
                "formula": "linear interpolation of breakpoints, clipped to [1e-4, 1-1e-4]"}


class Identity:
    kind = "none"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, float)

    def to_dict(self):
        return {"kind": "none"}


def make(kind: str):
    return {"platt": Platt, "isotonic": Isotonic, "none": Identity}[kind]()


def from_dict(d: dict):
    if d["kind"] == "platt":
        return Platt(d["a"], d["b"])
    if d["kind"] == "isotonic":
        return Isotonic(d["x"], d["y"])
    return Identity()
