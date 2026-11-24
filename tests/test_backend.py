"""Local backend (backend/server.py): API contract, validation, errors, and that it serves the frontend."""
import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from backend.server import make_server

pytest.importorskip("onnxruntime")
HIGH = {"age": 82, "gender": "Female", "arrivalmode": "ambulance", "arrivalhour_bin": "11-14", "triage_vital_hr": 118,
        "triage_vital_sbp": 96, "triage_vital_dbp": 58, "triage_vital_rr": 28, "triage_vital_o2": 88, "triage_vital_o2_device": 1,
        "triage_vital_temp": 100.8, "complaints": ["cc_shortnessofbreath"]}


def _start(static_only):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    srv = make_server("127.0.0.1", port, static_only=static_only)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def url():
    srv, u = _start(False)
    yield u
    srv.shutdown()


def call(url, path, body=None, ctype="application/json", raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_health_lists_three_models(url):
    st, _, b = call(url, "/api/health")
    j = json.loads(b)
    assert st == 200 and j["ok"] and set(j["models"]) == {"t1", "t2", "t3lite"}


def test_predict_matches_runtime_and_picks_model(url):
    from src.edge.runtime import load_all
    B = load_all()
    for form, slot in ((HIGH, "t1"), ({**HIGH, "esi": 2}, "t2"), ({**HIGH, "esi": 2, "n_edvisits": 3}, "t3lite")):
        st, _, b = call(url, "/api/predict", {"form": form})
        j = json.loads(b)
        assert st == 200 and j["slot"] == slot, j
        assert abs(j["probability"] - B[slot].predict(form).probability) < 1e-9
        assert j["band"] in ("Low", "Medium", "High") and j["conformal"] in ("likely_admission", "likely_discharge", "uncertain")
        assert j["contributions"] and {"label", "value"} <= set(j["contributions"][0])


def test_blank_labs_are_missing_not_zero(url):
    st, _, b = call(url, "/api/predict", {"form": {**HIGH, "rdw_last": None, "neutrophils_median": None}, "slot": "t3lite"})
    assert st == 200 and json.loads(b)["slot"] == "t3lite"


def test_validation_and_bad_requests(url):
    st, _, b = call(url, "/api/predict", {"form": {**HIGH, "age": 12}})
    assert st == 422 and json.loads(b)["fields"][0]["field"] == "age"
    assert call(url, "/api/predict", {"form": HIGH, "slot": "nope"})[0] == 400
    assert call(url, "/api/predict", raw=b"{not json")[0] == 400
    assert call(url, "/api/predict", {"form": HIGH}, ctype="text/plain")[0] == 415
    assert call(url, "/api/predict", raw=b"x" * 70000)[0] == 413
    assert call(url, "/api/predict", [1, 2])[0] == 400
    assert call(url, "/api/nope")[0] == 404


def test_serves_frontend_with_safe_headers(url):
    st, h, b = call(url, "/index.html")
    assert st == 200 and b"FirstLook" in b and h["X-Content-Type-Options"] == "nosniff"
    st, h, _ = call(url, "/manifest.webmanifest")
    assert st == 200 and h["Content-Type"].startswith("application/manifest+json")
    assert call(url, "/../README.md")[0] == 404


def test_static_only_mode_has_no_api():
    srv, u = _start(True)
    try:
        assert call(u, "/api/health")[0] == 404 and call(u, "/index.html")[0] == 200
    finally:
        srv.shutdown()
