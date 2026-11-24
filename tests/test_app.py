"""Feature-pipeline parity (training vs app), input validation, examples, bundles."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from validation import c_to_f, validate  # noqa: E402

from src.features import VITALS, build_features, load_spec  # noqa: E402
from src.runtime import form_to_raw  # noqa: E402

PROC = ROOT / "data" / "processed"
HAS_DATA = (PROC / "t2.parquet").exists() and (ROOT / "data" / "interim" / "full.parquet").exists()
BUNDLES = ROOT / "models" / "bundles"


def row_to_form(r: pd.Series, cc_cols) -> dict:
    """What the app's form collects, built from one raw-schema row."""
    f = {"age": float(r["age"]), "gender": str(r["gender"]),
         "arrivalmode": None if pd.isna(r["arrivalmode"]) else str(r["arrivalmode"]),
         "arrivalhour_bin": str(r["arrivalhour_bin"]),
         "esi": None if pd.isna(r["esi"]) else str(r["esi"]),
         "complaints": [c for c in cc_cols if r[c] == 1]}
    for v in VITALS:
        f[v] = None if pd.isna(r[v]) else float(r[v])
    return f


@pytest.mark.skipif(not HAS_DATA, reason="data not built")
@pytest.mark.parametrize("tier", ["t1", "t2"])
def test_app_pipeline_matches_training(tier):
    spec = load_spec()
    cols = ["age", "gender", "arrivalmode", "arrivalhour_bin", "esi"] + VITALS + spec["cc_columns"]
    full = pd.read_parquet(ROOT / "data" / "interim" / "full.parquet", columns=cols)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(full), 300, replace=False)
    train_X = pd.read_parquet(PROC / f"{tier}.parquet").iloc[idx].reset_index(drop=True)
    app_rows = []
    for i in idx:
        raw = form_to_raw(row_to_form(full.iloc[i], spec["cc_columns"]), spec)
        app_rows.append(build_features(raw, tier, spec))
    app_X = pd.concat(app_rows, ignore_index=True)
    assert list(app_X.columns) == list(train_X.columns)
    np.testing.assert_array_equal(app_X.to_numpy(np.float32), train_X.to_numpy(np.float32))


def test_spo2_100_maps_to_99():
    spec = load_spec()
    form = {"age": 40, "gender": "Male", "arrivalmode": "Car", "arrivalhour_bin": "07-10", "complaints": ["cc_fever"],
            "triage_vital_o2": 100.0}
    X = build_features(form_to_raw(form, spec), "t1", spec)
    assert X["triage_vital_o2"].iloc[0] == 99


def good_form(**kw):
    f = {"age": 50, "gender": "Female", "arrivalmode": "Car", "arrivalhour_bin": "07-10", "complaints": ["cc_fever"],
         "triage_vital_hr": 80.0, "triage_vital_sbp": 120.0, "triage_vital_dbp": 80.0, "triage_vital_rr": 16.0,
         "triage_vital_o2": 98.0, "triage_vital_temp": 98.6}
    f.update(kw)
    return f


def test_validation_accepts_good_form():
    e, w = validate(good_form())
    assert e == [] and w == []


@pytest.mark.parametrize("field,value", [("age", 12), ("age", 140), ("triage_vital_hr", 400), ("triage_vital_o2", 40),
                                         ("triage_vital_temp", 37.0), ("triage_vital_rr", 3)])
def test_validation_rejects_out_of_range(field, value):
    e, _ = validate(good_form(**{field: value}))
    assert e and any("range" in m or "between" in m for m in e)


def test_validation_messages_and_rules():
    assert validate(good_form(age=None))[0]
    assert any("Diastolic" in m for m in validate(good_form(triage_vital_dbp=130.0))[0])
    assert validate(good_form(complaints=[]))[0]
    assert validate(good_form(complaints=["a", "b", "c", "d"]))[0]
    assert validate(good_form(gender=None))[0]
    assert validate(good_form(use_esi=True, esi=None))[0]
    assert validate(good_form(triage_vital_hr=190.0))[1]  # unusual -> warning only
    assert validate(good_form(triage_vital_o2=None, triage_vital_hr=None))[0] == []  # not measured is fine


def test_celsius_conversion():
    assert c_to_f(37.0) == 98.6
    assert validate(good_form(triage_vital_temp=c_to_f(38.5)))[0] == []


@pytest.mark.skipif(not (BUNDLES).exists() or not any(BUNDLES.glob("t1_*")), reason="no bundle built")
def test_examples_land_where_promised():
    from examples import EXAMPLES
    from src.runtime import load_all
    b = load_all()["t1"]
    res = {k: b.predict(v["form"]) for k, v in EXAMPLES.items()}
    assert res["high"].band == "High" and res["high"].conformal == "likely_admission"
    assert res["low"].band == "Low" and res["low"].conformal == "likely_discharge"
    assert res["uncertain"].conformal == "uncertain"
    assert res["missing_spo2"].missing_vitals == ["SpO₂"]
    for k, v in EXAMPLES.items():
        assert validate(v["form"])[0] == [], k


@pytest.mark.skipif(not HAS_DATA or not (BUNDLES / "t3lite_v1").exists(), reason="data or bundle missing")
def test_t3lite_bundle_vector_matches_training():
    from src.runtime import Bundle
    from src.features import safe_name
    b = Bundle(BUNDLES / "t3lite_v1")
    spec = load_spec()
    hist = b.spec["history_inputs"]
    cols = ["age", "gender", "arrivalmode", "arrivalhour_bin", "esi", "previousdispo"] + VITALS + spec["cc_columns"] + hist
    full = pd.read_parquet(ROOT / "data" / "interim" / "full.parquet", columns=cols)
    idx = np.random.default_rng(1).choice(len(full), 200, replace=False)
    train_X = pd.read_parquet(PROC / "t3.parquet", columns=b.features).iloc[idx].reset_index(drop=True)
    rows = []
    for i in idx:
        r = full.iloc[i]
        f = row_to_form(r, spec["cc_columns"])
        f["previousdispo"] = str(r["previousdispo"])
        for h in hist:
            f[h] = float(r[h]) if not pd.isna(r[h]) else np.nan
        rows.append(b.vectorize(form_to_raw(f, b.spec)))
    app_X = pd.concat(rows, ignore_index=True)
    assert list(app_X.columns) == [safe_name(c) if c in hist else c for c in b.features] or list(app_X.columns) == b.features
    np.testing.assert_array_equal(app_X.to_numpy(np.float32), train_X.to_numpy(np.float32))


def test_external_socket_counter_ignores_inbound_but_counts_outbound():
    """Inbound (to our listening port) is not 'external'; an outbound non-loopback connection is."""
    import socket
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    import netstatus
    base = netstatus.external_sockets()
    # inbound: a server in this process accepting a connection from a non-loopback local address
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "127.0.0.1"
    if ip.startswith("127."):
        import pytest
        pytest.skip("no non-loopback address on this machine")
    srv = socket.socket(); srv.bind((ip, 0)); srv.listen(1)
    cli = socket.create_connection((ip, srv.getsockname()[1])); conn, _ = srv.accept()
    try:
        # server side is inbound (ignored); client side is outbound to a non-loopback IP (counted)
        assert netstatus.external_sockets() == base + 1
    finally:
        conn.close(); cli.close(); srv.close()
