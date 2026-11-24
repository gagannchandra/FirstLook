"""Random synthetic patient for the "Random patient" autofill button.

Purely parametric: values are drawn from hand-set, clinically plausible ranges per acuity
level (never sampled from dataset rows — guide rule "demo patients are synthetic").
Every generated form passes app/validation.py without errors or soft-range warnings
(tests/test_app.py checks this over many seeds). The same generator exists in JavaScript
for the PWA (app/pwa/model.js: randomPatient) with the same ranges.

Usage:
    from random_patient import random_patient
    form, meta = random_patient(seed=123, acuity="high", allow_missing=True)
"""
from __future__ import annotations

import numpy as np

# Plausible ranges per acuity (uniform draws, then rounded). All inside validation SOFT ranges.
PROFILES: dict[str, dict] = {
    "low": {
        "age": (18, 48), "hr": (62, 94), "rr": (12, 18), "sbp": (106, 138), "o2": (97, 100), "temp": (97.4, 99.3),
        "oxygen_p": 0.0,
        "arrival": {"Walk-in": 0.45, "Car": 0.40, "Public Transportation": 0.10, "Other": 0.05},
        "complaints": ["cc_sorethroat", "cc_rash", "cc_earpain", "cc_dentalpain", "cc_anklepain", "cc_kneepain",
                       "cc_laceration", "cc_coldlikesymptoms", "cc_backpain", "cc_footpain", "cc_handpain", "cc_eyeproblem"],
        "esi": {4: 0.6, 5: 0.25, 3: 0.15}, "edvisits": 0.4, "admissions": 0.05, "surgeries": 0.3,
    },
    "medium": {
        "age": (35, 72), "hr": (80, 108), "rr": (16, 22), "sbp": (112, 168), "o2": (94, 99), "temp": (97.8, 100.6),
        "oxygen_p": 0.05,
        "arrival": {"Car": 0.55, "Walk-in": 0.25, "ambulance": 0.15, "Wheelchair": 0.05},
        "complaints": ["cc_abdominalpain", "cc_chestpain", "cc_headache-newonsetornewsymptoms", "cc_dizziness",
                       "cc_emesis", "cc_flankpain", "cc_cough", "cc_palpitations", "cc_fall", "cc_legswelling",
                       "cc_nausea", "cc_urinarytractinfection"],
        "esi": {3: 0.7, 2: 0.15, 4: 0.15}, "edvisits": 1.2, "admissions": 0.4, "surgeries": 1.0,
    },
    "high": {
        "age": (62, 94), "hr": (98, 132), "rr": (20, 32), "sbp": (82, 118), "o2": (84, 94), "temp": (98.6, 102.8),
        "oxygen_p": 0.5,
        "arrival": {"ambulance": 0.75, "Car": 0.15, "Wheelchair": 0.10},
        "complaints": ["cc_shortnessofbreath", "cc_alteredmentalstatus", "cc_weakness", "cc_syncope",
                       "cc_breathingdifficulty", "cc_fall>65", "cc_gibleeding", "cc_dyspnea", "cc_chestpain",
                       "cc_fever-9weeksto74years"],
        "esi": {2: 0.65, 1: 0.1, 3: 0.25}, "edvisits": 2.5, "admissions": 1.5, "surgeries": 2.5,
    },
}
ACUITIES = ("low", "medium", "high")
HOUR_BINS = ["23-02", "03-06", "07-10", "11-14", "15-18", "19-22"]
MISSABLE = ["triage_vital_o2", "triage_vital_temp", "triage_vital_rr"]  # vitals most often skipped at triage


def _pick(rng: np.random.Generator, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def random_patient(seed: int, acuity: str | None = None, allow_missing: bool = False,
                   valid_complaints: list[str] | None = None) -> tuple[dict, dict]:
    """Returns (form, meta). form uses the runtime schema; meta = {seed, acuity, esi, history, missing}."""
    rng = np.random.default_rng(seed)
    acuity = acuity if acuity in PROFILES else ACUITIES[int(rng.integers(3))]
    P = PROFILES[acuity]
    u = lambda lo, hi: rng.uniform(lo, hi)  # noqa: E731
    sbp = round(u(*P["sbp"]))
    dbp = int(np.clip(round(sbp * u(0.55, 0.68)), 40, sbp - 15))
    pool = [c for c in P["complaints"] if valid_complaints is None or c in valid_complaints]
    n_cc = 1 if rng.random() < 0.8 else 2
    form = {
        "age": int(round(u(*P["age"]))),
        "gender": "Female" if rng.random() < 0.5 else "Male",
        "arrivalmode": _pick(rng, P["arrival"]),
        "arrivalhour_bin": HOUR_BINS[int(rng.integers(len(HOUR_BINS)))],
        "triage_vital_hr": float(round(u(*P["hr"]))),
        "triage_vital_rr": float(round(u(*P["rr"]))),
        "triage_vital_sbp": float(sbp),
        "triage_vital_dbp": float(dbp),
        "triage_vital_o2": float(round(u(*P["o2"]))),
        "triage_vital_o2_device": 1 if rng.random() < P["oxygen_p"] else 0,
        "triage_vital_temp": float(round(u(*P["temp"]), 1)),
        "complaints": [str(c) for c in rng.choice(pool, size=min(n_cc, len(pool)), replace=False)],
    }
    missing = []
    if allow_missing and rng.random() < 0.6:
        v = MISSABLE[int(rng.integers(len(MISSABLE)))]
        form[v] = None
        if v == "triage_vital_o2":
            form["triage_vital_o2_device"] = None
        missing.append(v)
    meta = {
        "seed": int(seed), "acuity": acuity, "missing": missing,
        "esi": int(_pick(rng, P["esi"])),
        "history": {"n_edvisits": int(rng.poisson(P["edvisits"])), "n_admissions": int(rng.poisson(P["admissions"])),
                    "n_surgeries": int(rng.poisson(P["surgeries"]))},
    }
    return form, meta
