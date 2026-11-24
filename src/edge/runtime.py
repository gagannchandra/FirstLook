"""Offline inference runtime used by the app (no training code, no network).

A bundle folder (guide 04 §7) contains:
  model.onnx            inference model (ONNX Runtime, CPU, 1 thread)
  explain.json | explain_lgbm.txt   data for local explanations
  calibrator.json  conformal.json  thresholds.json  feature_spec.json
  metrics.json  model_card.md
"""
from __future__ import annotations

import os

# ONNX Runtime >= 1.2x sends 1DS telemetry to mobile.events.data.microsoft.com on Linux by
# default; disable_telemetry_events() does not stop the connection, this variable does.
# It must be set before onnxruntime is imported (DECISIONS D28).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling import calibrate, conformal
from src.data.features import VITALS, build_features

ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ROOT / "models" / "bundles"

VITAL_LABEL = {
    "triage_vital_hr": ("Heart rate", "bpm"), "triage_vital_sbp": ("Systolic BP", "mmHg"),
    "triage_vital_dbp": ("Diastolic BP", "mmHg"), "triage_vital_rr": ("Respiratory rate", "/min"),
    "triage_vital_o2": ("SpO₂", "%"), "triage_vital_o2_device": ("Supplemental oxygen", ""),
    "triage_vital_temp": ("Temperature", "°F"),
}


# The dataset codes complaints as run-together words ("sorethroat"). Readable names for the app; single words
# that are already correct (e.g. "cellulitis") are left alone.
CC_LABEL = {
    "abdominalcramping": "Abdominal cramping", "abdominaldistention": "Abdominal distension", "abdominalpainpregnant": "Abdominal pain (pregnant)",
    "abnormallab": "Abnormal lab result", "addictionproblem": "Addiction problem", "alcoholintoxication": "Alcohol intoxication",
    "alcoholproblem": "Alcohol problem", "allergicreaction": "Allergic reaction", "animalbite": "Animal bite", "ankleinjury": "Ankle injury",
    "armswelling": "Arm swelling", "assaultvictim": "Assault victim", "blurredvision": "Blurred vision", "bodyfluidexposure": "Body fluid exposure",
    "breastpain": "Breast pain", "breathingdifficulty": "Breathing difficulty", "breathingproblem": "Breathing problem", "cardiacarrest": "Cardiac arrest",
    "chesttightness": "Chest tightness", "coldlikesymptoms": "Cold-like symptoms", "dentalpain": "Dental pain", "detoxevaluation": "Detox evaluation",
    "drugproblem": "Drug problem", "earproblem": "Ear problem", "epigastricpain": "Epigastric pain", "exposuretostd": "Exposure to STD",
    "extremitylaceration": "Limb laceration", "extremityweakness": "Limb weakness", "eyeproblem": "Eye problem", "eyeredness": "Eye redness",
    "facialinjury": "Facial injury", "faciallaceration": "Facial laceration", "facialpain": "Facial pain", "facialswelling": "Facial swelling",
    "femaleguproblem": "Female genitourinary problem", "feverimmunocompromised": "Fever (immunocompromised)", "fingerinjury": "Finger injury",
    "fingerpain": "Finger pain", "fingerswelling": "Finger swelling", "footinjury": "Foot injury", "footswelling": "Foot swelling",
    "foreignbodyineye": "Foreign body in eye", "fulltrauma": "Full trauma", "generalizedbodyaches": "Generalised body aches",
    "gibleeding": "GI bleeding", "handinjury": "Hand injury", "headinjury": "Head injury", "headlaceration": "Head laceration",
    "insectbite": "Insect bite", "irregularheartbeat": "Irregular heartbeat", "jointswelling": "Joint swelling", "kneeinjury": "Knee injury",
    "legswelling": "Leg swelling", "lossofconsciousness": "Loss of consciousness", "maleguproblem": "Male genitourinary problem",
    "medicalproblem": "Medical problem", "medicalscreening": "Medical screening", "medicationproblem": "Medication problem",
    "medicationrefill": "Medication refill", "modifiedtrauma": "Modified trauma", "motorcyclecrash": "Motorcycle crash",
    "multiplefalls": "Multiple falls", "nasalcongestion": "Nasal congestion", "nearsyncope": "Near fainting (near syncope)",
    "neurologicproblem": "Neurological problem", "oralswelling": "Mouth swelling", "panicattack": "Panic attack", "pelvicpain": "Pelvic pain",
    "psychiatricevaluation": "Psychiatric evaluation", "psychoticsymptoms": "Psychotic symptoms", "rapidheartrate": "Rapid heart rate",
    "rectalbleeding": "Rectal bleeding", "rectalpain": "Rectal pain", "respiratorydistress": "Respiratory distress", "shoulderinjury": "Shoulder injury",
    "shoulderpain": "Shoulder pain", "sicklecellpain": "Sickle cell pain", "sinusproblem": "Sinus problem", "skinirritation": "Skin irritation",
    "skinproblem": "Skin problem", "sorethroat": "Sore throat", "strokealert": "Stroke alert", "swallowedforeignbody": "Swallowed foreign body",
    "testiclepain": "Testicle pain", "thumbinjury": "Thumb injury", "tickremoval": "Tick removal", "urinaryfrequency": "Urinary frequency",
    "urinaryretention": "Urinary retention", "urinarytractinfection": "Urinary tract infection", "vaginalbleeding": "Vaginal bleeding",
    "vaginaldischarge": "Vaginal discharge", "vaginalpain": "Vaginal pain", "woundcheck": "Wound check", "woundinfection": "Wound infection",
    "wristinjury": "Wrist injury",
    "decreasedbloodsugar-symptomatic": "Low blood sugar (with symptoms)", "drug/alcoholassessment": "Drug or alcohol assessment",
    "elevatedbloodsugar-nosymptoms": "High blood sugar (no symptoms)", "elevatedbloodsugar-symptomatic": "High blood sugar (with symptoms)",
    "fever-75yearsorolder": "Fever (75 or older)", "fever-9weeksto74years": "Fever (age 9 weeks to 74)",
    "follow-upcellulitis": "Cellulitis follow-up", "headache-recurrentorknowndxmigraines": "Headache (recurrent or known migraine)",
    "headachere-evaluation": "Headache re-evaluation", "overdose-accidental": "Overdose (accidental)", "overdose-intentional": "Overdose (intentional)",
    "post-opproblem": "Post-op problem", "seizure-newonset": "Seizure (new onset)", "seizure-priorhxof": "Seizure (known history)",
    "suture/stapleremoval": "Suture or staple removal", "withdrawal-alcohol": "Alcohol withdrawal", "woundre-evaluation": "Wound re-evaluation",
    "bleeding/bruising": "Bleeding or bruising",
}


def complaint_label(col: str) -> str:
    s = col[3:] if col.startswith("cc_") else col
    special = {"shortnessofbreath": "Shortness of breath", "abdominalpain": "Abdominal pain", "chestpain": "Chest pain",
               "fall>65": "Fall, age > 65", "alteredmentalstatus": "Altered mental status", "other": "Other (not listed)",
               "motorvehiclecrash": "Motor vehicle crash", "headache-newonsetornewsymptoms": "Headache (new onset)"}
    if s in CC_LABEL:
        return CC_LABEL[s]
    if s in special:
        return special[s]
    return s.replace("-", " – ").replace("/", " / ").capitalize()


@dataclass
class Result:
    probability: float
    band: str
    conformal: str
    tier: str
    model_version: str
    latency_ms: float
    contributions: list = field(default_factory=list)
    missing_vitals: list = field(default_factory=list)
    feature_count: int = 0


class Bundle:
    def __init__(self, path: Path):
        import onnxruntime as ort
        self.path = Path(path)
        rd = lambda n: json.loads((self.path / n).read_text())  # noqa: E731
        self.spec = rd("feature_spec.json")
        self.cal = calibrate.from_dict(rd("calibrator.json"))
        self.conf = rd("conformal.json")
        self.thr = rd("thresholds.json")
        self.metrics = rd("metrics.json") if (self.path / "metrics.json").exists() else {}
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(self.path / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.features = self.spec["model_features"]
        self.tier = self.spec["base_tier"]
        self.version = self.spec["version"]
        self._explainer = None
        if (self.path / "explain_lgbm.txt").exists():
            import lightgbm as lgb
            self._explainer = ("lgbm", lgb.Booster(model_file=str(self.path / "explain_lgbm.txt")))
        elif (self.path / "explain.json").exists():
            self._explainer = ("linear", rd("explain.json"))

    @property
    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())

    @property
    def model_bytes(self) -> int:
        return (self.path / "model.onnx").stat().st_size

    def vectorize(self, raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, self.tier, self.spec)[self.features]

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        out = self.sess.run(None, {"input": X.to_numpy(np.float32)})
        prob = out[1]
        if isinstance(prob, list):
            return np.array([d[1] for d in prob])
        return np.asarray(prob)[:, 1]

    def band(self, p: float) -> str:
        if p < self.thr["t_low"]:
            return "Low"
        if p >= self.thr["t_high"]:
            return "High"
        return "Medium"

    def contributions(self, X: pd.DataFrame, raw_row: dict) -> list[dict]:
        if self._explainer is None:
            return []
        kind, obj = self._explainer
        x = X.to_numpy(np.float64)
        if kind == "lgbm":
            c = obj.predict(x, pred_contrib=True)[0][:-1]
        else:
            mu, sd, coef, med = (np.array(obj[k]) for k in ("mean", "scale", "coef", "median"))
            xi = np.where(np.isnan(x[0]), med, x[0])
            c = coef[: len(xi)] * (xi - mu[: len(xi)]) / sd[: len(xi)]
            ind = obj.get("indicator_features", [])
            for j, f in enumerate(ind):  # missing-indicator terms fold into their feature
                v = 1.0 if np.isnan(x[0][f]) else 0.0
                c[f] += coef[len(xi) + j] * (v - mu[len(xi) + j]) / sd[len(xi) + j]
        groups: dict[str, float] = {}
        labels: dict[str, str] = {}
        for f, v in zip(self.features, c):
            key, lab = self._group(f, raw_row)
            groups[key] = groups.get(key, 0.0) + float(v)
            labels[key] = lab
        out = [{"factor": labels[k], "key": k, "contribution": v} for k, v in groups.items() if abs(v) > 1e-6]
        return sorted(out, key=lambda d: -abs(d["contribution"]))

    @staticmethod
    def _group(f: str, raw: dict) -> tuple[str, str]:
        if f == "age":
            return "age", f"Age {int(raw['age'])}" if raw.get("age") is not None else "Age"
        if f == "sex_male":
            return "sex", f"Sex: {raw.get('gender', '—')}"
        if f.startswith("arrival="):
            m = raw.get("arrivalmode") or "not recorded"
            return "arrival", f"Arrival: {'ambulance' if m == 'ambulance' else m.lower()}"
        if f.startswith("hour="):
            return "hour", f"Arrival time {raw.get('arrivalhour_bin', '')}h"
        if f in VITALS:
            name, unit = VITAL_LABEL[f]
            v = raw.get(f)
            if f == "triage_vital_o2_device":
                return f, "On supplemental oxygen" if v == 1 else ("Room air" if v == 0 else "Oxygen use not recorded")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f, f"{name} not measured"
            return f, f"{name} {v:g}{(' ' + unit) if unit and unit != '%' else unit}"
        if f.startswith("cc_"):
            sel = [complaint_label(c) for c in raw.get("complaints", [])]
            return "cc", "Chief complaint: " + (", ".join(sel) if sel else "none selected")
        if f == "esi":
            return "esi", f"ESI level {raw.get('esi')}" if raw.get("esi") else "ESI not recorded"
        if f.startswith("prevdispo="):
            return "prevdispo", f"Previous ED visit outcome: {raw.get('previousdispo', 'none')}"
        return f, f"{f.replace('_', ' ')}: {raw.get(f, 0):g}"

    def predict(self, form: dict) -> Result:
        """form: raw-schema dict (age, gender, arrivalmode, arrivalhour_bin, vitals, complaints[, esi, history])."""
        t0 = time.perf_counter()
        raw = form_to_raw(form, self.spec)
        X = self.vectorize(raw)
        p_raw = float(self.raw_proba(X)[0])
        p = float(self.cal.predict(np.array([p_raw]))[0])
        cset = conformal.label(p, self.conf)
        band = self.band(p)
        latency = (time.perf_counter() - t0) * 1000
        contrib = self.contributions(X, form)
        missing = [VITAL_LABEL[v][0] for v in VITALS if v != "triage_vital_o2_device" and form.get(v) is None]
        return Result(p, band, cset, self.tier, self.version, latency, contrib, missing, len(self.features))


def form_to_raw(form: dict, spec: dict) -> pd.DataFrame:
    row = {"age": form.get("age"), "gender": form.get("gender"), "arrivalmode": form.get("arrivalmode"),
           "arrivalhour_bin": form.get("arrivalhour_bin"), "esi": form.get("esi"),
           "previousdispo": form.get("previousdispo") or "No previous dispo"}
    for v in VITALS:
        val = form.get(v)
        row[v] = np.nan if val is None else float(val)
    for c in spec["cc_columns"]:
        row[c] = 1.0 if c in form.get("complaints", []) else 0.0
    for h in spec.get("history_columns", []):
        row[h] = form.get(h, 0.0) if h in form else (0.0 if spec.get("history_default_zero", {}).get(h) else np.nan)
    df = pd.DataFrame([row])
    df["esi"] = df["esi"].astype("object")
    return df


def load_all() -> dict[str, Bundle]:
    out = {}
    if BUNDLES.exists():
        for d in sorted(BUNDLES.iterdir()):
            if (d / "model.onnx").exists():
                b = Bundle(d)
                out[b.spec["app_slot"]] = b
    return out
