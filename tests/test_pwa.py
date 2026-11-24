"""Offline web app (frontend/, L3): the JavaScript model must match the Python/ONNX runtime (probability, band, conformal)."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PWA = ROOT / "frontend"
MODEL_JS = PWA / "js" / "model.js"
MODELS_JSON = PWA / "data" / "models.json"

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not MODELS_JSON.exists(),
                                reason="node or frontend/data/models.json missing (run: python -m backend.export)")


def synthetic_forms(n=300, seed=0):
    from src.data.features import ARRIVAL_LEVELS, HOUR_BINS
    spec = json.loads((ROOT / "models/bundles/t1_v1/feature_spec.json").read_text())
    r = random.Random(seed)
    out = []
    for _ in range(n):
        def v(lo, hi, p_missing=0.1, dec=0):
            return None if r.random() < p_missing else round(r.uniform(lo, hi), dec)
        out.append({"age": r.randint(18, 100), "gender": r.choice(["Female", "Male"]),
                    "arrivalmode": r.choice(ARRIVAL_LEVELS), "arrivalhour_bin": r.choice(HOUR_BINS),
                    "triage_vital_hr": v(35, 180), "triage_vital_sbp": v(70, 220), "triage_vital_dbp": v(40, 120),
                    "triage_vital_rr": v(10, 40), "triage_vital_o2": v(80, 100), "triage_vital_o2_device": r.choice([0, 1, None]),
                    "triage_vital_temp": v(95, 104, dec=1), "esi": r.choice([None, 1, 2, 3, 4, 5]),
                    "complaints": r.sample(spec["complaint_order"], r.randint(0, 3))})
    return out


def test_js_matches_python_runtime(tmp_path):
    from src.edge.runtime import load_all
    B = load_all()
    forms = synthetic_forms()
    (tmp_path / "forms.json").write_text(json.dumps(forms))
    js = f"""
    const M = require({json.dumps(str(MODEL_JS))});
    const meta = require({json.dumps(str(MODELS_JSON))});
    const forms = require({json.dumps(str(tmp_path / 'forms.json'))});
    const out = {{}};
    for (const slot of ["t1", "t2"]) out[slot] = forms.map(f => M.predict(f, meta.models[slot], meta));
    console.log(JSON.stringify(out));
    """
    res = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    for slot in ("t1", "t2"):
        for f, j in zip(forms, res[slot]):
            py = B[slot].predict(f)
            assert abs(py.probability - j["probability"]) < 1e-4, (slot, f, py.probability, j["probability"])
            assert py.band == j["band"]
            assert py.conformal == j["conformal"], (slot, py.probability)


def test_pwa_assets_are_self_contained():
    sw = (PWA / "sw.js").read_text()
    files = [PWA / "index.html", PWA / "sw.js", PWA / "css" / "app.css", *sorted((PWA / "js").glob("*.js"))]
    for f in files:
        s = f.read_text()
        assert "http://" not in s.replace("http://www.w3.org", "") and "https://" not in s, f"external URL in {f.name}"
    # every file the app loads must be pre-cached by the service worker, or it breaks offline
    for asset in ("index.html", "css/app.css", "js/model.js", "js/charts.js", "js/app.js", "data/models.json", "data/dashboard.json",
                  "manifest.webmanifest", "icon.svg"):
        assert f'"{asset}"' in sw, asset
        assert (PWA / asset).exists(), asset
    assert "/api/" in sw  # live backend calls are never served from cache


def test_js_random_patients_valid_reproducible_and_ordered():
    """PWA random autofill: valid against the shared ranges, reproducible from the seed, risk ordered by acuity."""
    js = f"""
    const M = require({json.dumps(str(MODEL_JS))});
    const meta = require({json.dumps(str(MODELS_JSON))});
    const out = {{bad: [], mean: {{}}, repro: true}};
    for (const a of ["low", "medium", "high"]) {{
      let s = 0;
      for (let seed = 0; seed < 300; seed++) {{
        const {{ form }} = M.randomPatient(seed, a, seed % 2 === 0, meta);
        if (JSON.stringify(M.randomPatient(seed, a, seed % 2 === 0, meta).form) !== JSON.stringify(form)) out.repro = false;
        for (const [k, [lo, hi]] of Object.entries(meta.ranges)) {{
          const v = form[k]; if (v === null || v === undefined) continue;
          const soft = meta.soft[k];
          if (v < lo || v > hi || (soft && (v < soft[0] || v > soft[1]))) out.bad.push([a, seed, k, v]);
        }}
        if (form.triage_vital_dbp >= form.triage_vital_sbp) out.bad.push([a, seed, "dbp>=sbp"]);
        if (!form.complaints.length || !form.complaints.every(c => meta.complaint_order.includes(c))) out.bad.push([a, seed, "cc"]);
        s += M.predict(form, meta.models.t1, meta).probability;
      }}
      out.mean[a] = s / 300;
    }}
    console.log(JSON.stringify(out));
    """
    res = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert res["bad"] == [], res["bad"][:5]
    assert res["repro"]
    assert res["mean"]["low"] < res["mean"]["medium"] < res["mean"]["high"], res["mean"]


def test_js_validation_matches_python():
    """Both apps must accept/reject the same patients with the same messages."""
    from app.validation import validate_fields
    r = random.Random(1)
    edge = {"age": [None, 12, 18, 54, 110, 140], "triage_vital_hr": [None, 20, 30, 39, 41, 181, 299, 300],
            "triage_vital_sbp": [None, 60, 120, 250, 400], "triage_vital_dbp": [None, 20, 80, 130],
            "triage_vital_rr": [None, 3, 9, 16, 45, 70], "triage_vital_o2": [None, 40, 79, 97, 100],
            "triage_vital_temp": [None, 37.0, 94.5, 98.6, 104.2, 109.9, 110.5]}
    forms = []
    for _ in range(600):
        f = {k: r.choice(v) for k, v in edge.items()}
        f["gender"] = r.choice([None, "", "Female", "Male"])
        f["complaints"] = r.sample(["cc_fever", "cc_fall", "cc_other", "cc_cough", "cc_rash"], r.choice([0, 1, 2, 3, 4]))
        f["use_esi"] = r.random() < 0.3
        f["esi"] = r.choice([None, 3])
        forms.append(f)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(forms, fh)
    js = f"""
    const M = require({json.dumps(str(MODEL_JS))});
    const meta = require({json.dumps(str(MODELS_JSON))});
    const forms = require({json.dumps(fh.name)});
    console.log(JSON.stringify(forms.map(f => M.validateFields(f, meta))));
    """
    res = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    for f, j in zip(forms, res):
        assert validate_fields(f) == j, (f, validate_fields(f), j)
