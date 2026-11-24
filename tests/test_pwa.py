"""L3 PWA: the JavaScript model must match the Python/ONNX runtime (probability, band, conformal)."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PWA = ROOT / "app" / "pwa"

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (PWA / "models.json").exists(),
                                reason="node or app/pwa/models.json missing")


def synthetic_forms(n=300, seed=0):
    from src.features import ARRIVAL_LEVELS, HOUR_BINS
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
    from src.runtime import load_all
    B = load_all()
    forms = synthetic_forms()
    (tmp_path / "forms.json").write_text(json.dumps(forms))
    js = f"""
    const M = require({json.dumps(str(PWA / 'model.js'))});
    const meta = require({json.dumps(str(PWA / 'models.json'))});
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
    html = (PWA / "index.html").read_text()
    sw = (PWA / "sw.js").read_text()
    for s in (html, sw, (PWA / "model.js").read_text()):
        assert "http://" not in s.replace("http://www.w3.org", "") and "https://" not in s, "no external URLs"
    for asset in ("model.js", "models.json", "manifest.webmanifest", "icon.svg"):
        assert asset in sw
