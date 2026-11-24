"""FirstLook — offline admission-risk estimate at ED triage (research prototype).

Run:  streamlit run app/streamlit_app.py
Four screens: Patient → Result → Explanation → Research dashboard.
No network calls: the model runs in-process via ONNX Runtime.
"""
from __future__ import annotations

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # D28: no ONNX Runtime telemetry

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import psutil  # noqa: E402
import streamlit as st  # noqa: E402

import netstatus  # noqa: E402
from examples import EXAMPLES  # noqa: E402
from validation import RANGES, c_to_f, validate  # noqa: E402

from src.runtime import VITAL_LABEL, load_all  # noqa: E402

st.set_page_config(page_title="FirstLook · triage admission risk", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

ACCENT = "#0f6e6a"
INK, INK2, MUTED, LINE = "#14201f", "#4d5a58", "#7c8784", "#e2e6e3"
BAND = {"Low": ("#0a7d38", "#e8f5ec", "●"), "Medium": ("#a76500", "#fff4de", "◆"), "High": ("#c23232", "#fdecec", "▲")}
UNCERTAIN = ("#5b45c9", "#f1eefc")
RAISE, LOWER = "#d64545", "#2a78d6"
PAGES = ["Patient", "Result", "Explanation", "Dashboard"]
DASH_FILE = ROOT / "results" / "dashboard" / "dashboard.json"

# ------------------------------------------------------------------ style
st.markdown(f"""
<style>
  #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {{ display: none !important; }}
  .block-container {{ max-width: 1180px; padding-top: 1.1rem; padding-bottom: 4.5rem; }}
  html, body, [class*="css"] {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  .fl-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom: .35rem; }}
  .fl-brand {{ display:flex; align-items:baseline; gap:.6rem; }}
  .fl-brand .name {{ font-size: 1.45rem; font-weight: 700; color: {INK}; }}
  .fl-brand .name span {{ color: {ACCENT}; }}
  .fl-brand .sub {{ color: {MUTED}; font-size: .9rem; }}
  .fl-pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem; border-radius:999px;
             font-size:.8rem; font-weight:600; border:1px solid {LINE}; background:#fff; color:{INK2}; }}
  .fl-dot {{ width:.5rem; height:.5rem; border-radius:50%; display:inline-block; }}
  .fl-card {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.3rem; margin-bottom: .9rem; }}
  .fl-card h4 {{ margin: 0 0 .6rem 0; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 700; }}
  .fl-risk {{ border-radius:16px; padding:1.4rem 1.6rem; border:1px solid {LINE}; background:#fff; }}
  .fl-risk .lab {{ color:{INK2}; font-size:.95rem; font-weight:600; }}
  .fl-risk .val {{ font-size: 4.2rem; font-weight: 750; line-height: 1.05; margin:.25rem 0 .5rem 0; }}
  .fl-chip {{ display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .85rem; border-radius:999px; font-weight:700; font-size:.95rem; }}
  .fl-scale {{ display:flex; height:8px; border-radius:6px; overflow:hidden; margin-top:1.1rem; gap:2px; }}
  .fl-scale div {{ height:100%; }}
  .fl-scale-labels {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:.75rem; margin-top:.3rem; }}
  .fl-marker {{ position:relative; height:0; }}
  .fl-marker span {{ position:absolute; top:-15px; width:3px; height:14px; background:{INK}; border-radius:2px; }}
  .fl-conf {{ border-radius:14px; padding:1rem 1.2rem; border:1px solid {LINE}; background:#fff; display:flex; gap:.9rem; align-items:flex-start; }}
  .fl-conf .ico {{ font-size:1.5rem; line-height:1; }}
  .fl-conf .t {{ font-weight:700; font-size:1.1rem; color:{INK}; }}
  .fl-conf .d {{ color:{INK2}; font-size:.9rem; margin-top:.2rem; }}
  .fl-conf.unc {{ border:2px dashed {UNCERTAIN[0]}; background:{UNCERTAIN[1]}; }}
  .fl-conf.unc .t {{ color:{UNCERTAIN[0]}; }}
  .fl-meta {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:.6rem; }}
  .fl-meta div {{ background:#fafbfa; border:1px solid {LINE}; border-radius:10px; padding:.55rem .75rem; }}
  .fl-meta b {{ display:block; font-size:1.02rem; color:{INK}; }}
  .fl-meta span {{ font-size:.74rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.05em; }}
  .fl-note {{ background:#fff8e6; border:1px solid #f1dca6; color:#6b4e00; border-radius:10px; padding:.6rem .85rem; font-size:.9rem; }}
  .fl-factor {{ display:flex; gap:.6rem; align-items:baseline; padding:.45rem 0; border-bottom:1px solid #f0f2f0; font-size:.95rem; }}
  .fl-factor:last-child {{ border-bottom:none; }}
  .fl-factor .arrow {{ font-weight:800; width:1.1rem; }}
  .fl-disclaimer {{ position:fixed; left:0; right:0; bottom:0; z-index:999; background:{INK}; color:#f3f5f4;
                   text-align:center; font-size:.86rem; padding:.55rem 1rem; letter-spacing:.01em; }}
  .fl-disclaimer b {{ color:#fff; }}
  .fl-stat {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:.8rem 1rem; height:100%; }}
  .fl-stat .k {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; font-weight:700; }}
  .fl-stat .v {{ font-size:1.55rem; font-weight:700; color:{INK}; margin-top:.15rem; }}
  .fl-stat .s {{ font-size:.8rem; color:{INK2}; }}
  .fl-section {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.07em; color:{MUTED}; font-weight:700; margin:.6rem 0 .35rem 0; }}
  div[data-testid="stSegmentedControl"] button p {{ font-weight:600; }}
  .stButton>button {{ border-radius:10px; font-weight:600; }}
  .stButton>button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}
  [data-testid="stExpander"] details {{ border-radius:12px; border-color:{LINE}; background:#fff; }}
  table.fl-tab {{ width:100%; border-collapse:collapse; font-size:.88rem; font-variant-numeric: tabular-nums; }}
  table.fl-tab th {{ text-align:left; color:{MUTED}; font-weight:700; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid {LINE}; padding:.4rem .5rem; }}
  table.fl-tab td {{ border-bottom:1px solid #f0f2f0; padding:.42rem .5rem; color:{INK}; }}
  table.fl-tab td.ci {{ color:{MUTED}; font-size:.8rem; }}
</style>
<div class="fl-disclaimer"><b>Decision support only.</b> Not a diagnosis. Clinical judgement decides. · Research prototype — not for use with real patients.</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state
@st.cache_resource
def bundles():
    return load_all()


@st.cache_resource
def net_baseline():
    return {"t0": time.time(), "base": netstatus.baseline()}


@st.cache_data
def dashboard_data():
    return json.loads(DASH_FILE.read_text()) if DASH_FILE.exists() else None


B = bundles()
NET = net_baseline()
if "page" not in st.session_state:
    st.session_state.page = "Patient"
st.session_state.setdefault("latencies", [])


def goto(page: str):
    st.session_state.page = page


HOUR_BINS = [("23-02", "23:00–02:59"), ("03-06", "03:00–06:59"), ("07-10", "07:00–10:59"),
             ("11-14", "11:00–14:59"), ("15-18", "15:00–18:59"), ("19-22", "19:00–22:59")]
ARRIVALS = [("Walk-in", "Walk-in"), ("Car", "Private car / taxi"), ("ambulance", "Ambulance / EMS"),
            ("Wheelchair", "Wheelchair"), ("Public Transportation", "Public transport"), ("Police", "Police"), ("Other", "Other")]
O2_OPTS = ["Room air", "On supplemental oxygen", "Not recorded"]
VITAL_KEYS = ["triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr", "triage_vital_o2", "triage_vital_temp"]


def current_bin() -> str:
    h = dt.datetime.now().hour
    for b, _ in HOUR_BINS:
        a, z = (int(x) for x in b.split("-"))
        if (a <= z and a <= h <= z) or (a > z and (h >= a or h <= z)):
            return b
    return "11-14"


def init_form():
    ss = st.session_state
    ss.setdefault("f_age", None); ss.setdefault("f_gender", None); ss.setdefault("f_arrival", "Walk-in")
    ss.setdefault("f_hour", current_bin()); ss.setdefault("f_cc", []); ss.setdefault("f_o2dev", "Room air")
    ss.setdefault("f_tunit", "°F"); ss.setdefault("f_use_esi", False); ss.setdefault("f_esi", None)
    ss.setdefault("f_use_hist", False)
    for v in VITAL_KEYS:
        ss.setdefault(f"f_{v}", None); ss.setdefault(f"nm_{v}", False)


def load_example(name: str):
    ex = EXAMPLES[name]["form"]
    ss = st.session_state
    ss.f_age = ex["age"]; ss.f_gender = ex["gender"]; ss.f_arrival = ex["arrivalmode"]; ss.f_hour = ex["arrivalhour_bin"]
    ss.f_cc = list(ex["complaints"]); ss.f_tunit = "°F"; ss.f_use_esi = False; ss.f_esi = None; ss.f_use_hist = False
    dev = ex.get("triage_vital_o2_device")
    ss.f_o2dev = "Not recorded" if dev is None else ("On supplemental oxygen" if dev == 1 else "Room air")
    for v in VITAL_KEYS:
        val = ex.get(v)
        ss[f"nm_{v}"] = val is None
        ss[f"f_{v}"] = None if val is None else float(val)
    ss.example_name = EXAMPLES[name]["label"]


def clear_form():
    ss = st.session_state
    for k in [k for k in ss.keys() if k.startswith(("f_", "nm_"))]:
        del ss[k]
    ss.pop("result", None); ss.pop("form", None); ss.pop("example_name", None)


def collect_form() -> dict:
    ss = st.session_state
    f = {"age": ss.f_age, "gender": ss.f_gender, "arrivalmode": ss.f_arrival, "arrivalhour_bin": ss.f_hour,
         "complaints": list(ss.f_cc), "use_esi": ss.f_use_esi, "esi": ss.f_esi if ss.f_use_esi else None}
    for v in VITAL_KEYS:
        val = None if ss[f"nm_{v}"] else ss[f"f_{v}"]
        if v == "triage_vital_temp" and val is not None and ss.f_tunit == "°C":
            val = c_to_f(val)
        f[v] = None if val is None else float(val)
    f["triage_vital_o2_device"] = {"Room air": 0, "On supplemental oxygen": 1}.get(ss.f_o2dev)
    if ss.f_use_hist:
        for h, val in ss.get("hist_vals", {}).items():
            f[h] = np.nan if val is None else val
        f["previousdispo"] = ss.get("f_prevdispo", "No previous dispo")
    return f


def choose_bundle(form: dict, use_hist: bool):
    if use_hist and "t3lite" in B:
        return B["t3lite"]
    if form.get("esi") and "t2" in B:
        return B["t2"]
    return B["t1"]


def run_prediction():
    form = collect_form()
    errors, warnings = validate(form)
    st.session_state.errors, st.session_state.warnings = errors, warnings
    if errors:
        return
    b = choose_bundle(form, st.session_state.f_use_hist)
    res = b.predict(form)
    st.session_state.latencies.append(res.latency_ms)
    st.session_state.result = res
    st.session_state.form = form
    st.session_state.bundle_slot = b.spec["app_slot"]
    st.session_state.page = "Result"


# ------------------------------------------------------------------ header
def header():
    s = netstatus.status(NET["base"])
    on = s["online"]
    net_html = (f'<span class="fl-pill"><span class="fl-dot" style="background:{"#c23232" if on else "#0a7d38"}"></span>'
                f'Network: {"ONLINE" if on else "OFFLINE"} · app external connections: {max(s["external_sockets"], 0)}'
                + ("" if on else f' · {s["bytes_sent"]:,} bytes sent') + '</span>')
    st.markdown(f'<div class="fl-top"><div class="fl-brand"><div class="name">First<span>Look</span></div>'
                f'<div class="sub">Admission-risk estimate at ED triage · runs fully on this device</div></div>{net_html}</div>',
                unsafe_allow_html=True)
    st.segmented_control("Screen", PAGES, key="page", label_visibility="collapsed")


# ------------------------------------------------------------------ screen 1
def screen_patient():
    init_form()
    ss = st.session_state
    st.markdown('<div class="fl-section">Load a synthetic example patient</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for c, key in zip(cols, ["high", "low", "uncertain", "missing_spo2"]):
        c.button(EXAMPLES[key]["label"], key=f"ex_{key}", on_click=load_example, args=(key,), use_container_width=True,
                 help=EXAMPLES[key]["hint"])
    cols[4].button("Clear form", on_click=clear_form, use_container_width=True)
    if ss.get("example_name"):
        st.caption(f"Loaded synthetic example: **{ss.example_name}** — invented for the demo, not a real patient.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0;font-size:.78rem;letter-spacing:.06em;color:#7c8784'>PATIENT & ARRIVAL</h4>", unsafe_allow_html=True)
            a, b = st.columns(2)
            a.number_input("Age (years)", min_value=0, max_value=130, step=1, key="f_age", placeholder="e.g. 54")
            b.radio("Sex", ["Female", "Male"], key="f_gender", horizontal=True, index=None)
            a, b = st.columns(2)
            a.selectbox("Arrival mode", [k for k, _ in ARRIVALS], format_func=dict(ARRIVALS).get, key="f_arrival")
            b.selectbox("Arrival time (auto-filled)", [k for k, _ in HOUR_BINS], format_func=dict(HOUR_BINS).get, key="f_hour")
            spec = B["t1"].spec
            st.multiselect("Chief complaint (search, up to 3)", spec["complaint_order"],
                           format_func=lambda c: spec["complaint_labels"][c], key="f_cc", max_selections=3,
                           placeholder="Type to search — e.g. chest pain, fall, fever")
        with st.container(border=True):
            st.markdown("<h4 style='margin:0;font-size:.78rem;letter-spacing:.06em;color:#7c8784'>OPTIONAL INPUTS</h4>", unsafe_allow_html=True)
            st.toggle("ESI level available", key="f_use_esi", help="Adds the Emergency Severity Index (uses the T2 model).")
            if ss.f_use_esi:
                st.segmented_control("ESI level", [1, 2, 3, 4, 5], key="f_esi",
                                     format_func=lambda v: {1: "1 · resus", 2: "2 · emergent", 3: "3 · urgent",
                                                            4: "4 · less urgent", 5: "5 · non-urgent"}[v])
            has_hist = "t3lite" in B
            st.toggle("EHR history available", key="f_use_hist", disabled=not has_hist,
                      help="Adds key history fields (uses the T3-lite model)." if has_hist else "No history model bundled.")
            if ss.f_use_hist and has_hist:
                history_inputs(B["t3lite"].spec)
    with right:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0;font-size:.78rem;letter-spacing:.06em;color:#7c8784'>TRIAGE VITALS — tick “not measured” if a reading is unavailable</h4>", unsafe_allow_html=True)
            grid = [("triage_vital_hr", "Heart rate (bpm)"), ("triage_vital_rr", "Respiratory rate (/min)"),
                    ("triage_vital_sbp", "Systolic BP (mmHg)"), ("triage_vital_dbp", "Diastolic BP (mmHg)"),
                    ("triage_vital_o2", "SpO₂ (%)")]
            for v, lab in grid:
                c1, c2 = st.columns([3, 1.25])
                c1.number_input(lab, key=f"f_{v}", step=1.0, format="%.0f", disabled=ss[f"nm_{v}"], placeholder="—")
                c2.checkbox("Not measured", key=f"nm_{v}")
                if v == "triage_vital_o2":
                    st.radio("Oxygen at triage", O2_OPTS, key="f_o2dev", horizontal=True)
            c1, c2, c3 = st.columns([2.1, 0.9, 1.25])
            c1.number_input(f"Temperature ({ss.f_tunit})", key="f_triage_vital_temp", step=0.1, format="%.1f",
                            disabled=ss["nm_triage_vital_temp"], placeholder="—")
            c2.radio("Unit", ["°F", "°C"], key="f_tunit", label_visibility="hidden")
            c3.checkbox("Not measured", key="nm_triage_vital_temp")

    for e in ss.get("errors", []):
        st.error(e, icon="✋")
    for w in ss.get("warnings", []):
        st.warning(w, icon="⚠️")
    st.button("Estimate admission risk  →", type="primary", on_click=run_prediction, use_container_width=True)


HIST_LABEL = {"n_edvisits": "ED visits in the past year", "n_admissions": "Hospital admissions in the past year",
              "n_surgeries": "Past surgeries / procedures on record", "asthma": "Asthma", "hyperlipidem": "Hyperlipidaemia",
              "alcoholrelateddisorders": "Alcohol-related disorder", "neutrophils_median": "Neutrophils — median of past-year labs",
              "rdw_last": "RDW — most recent past-year lab", "meds_elect/caloric/h2o": "electrolytes / caloric / water"}


def hist_label(h: str) -> str:
    if h in HIST_LABEL and not h.startswith("meds_"):
        return HIST_LABEL[h]
    if h.startswith("meds_"):
        name = HIST_LABEL.get(h, h[5:].replace("drugs", " drugs").replace("/", " / "))
        return f"Outpatient meds: {name}"
    return h.replace("_", " ").capitalize()


def history_inputs(spec: dict):
    """History fields used by the T3-lite model. Counts default to 0 (= none on record, as in the data);
    lab values default to blank (= no prior lab), never to an invented value."""
    vals = {}
    st.caption("Counts cover the past year; leave at 0 if none on record. Leave lab values blank if unavailable.")
    if spec.get("uses_previousdispo"):
        st.selectbox("Outcome of previous ED visit", ["No previous dispo", "Discharge", "Admit", "Transfer to Another Facility",
                                                      "LWBS after Triage", "AMA", "Eloped", "LWBS before Triage", "Observation",
                                                      "Send to L&D"],
                     format_func=lambda v: "No previous ED visit" if v == "No previous dispo" else v, key="f_prevdispo")
    kinds = spec.get("history_input_kind", {})
    key_fields = [h for h in ("n_edvisits", "n_admissions", "n_surgeries") if h in spec["history_inputs"]]
    rest = [h for h in spec["history_inputs"] if h not in key_fields]

    def field(col, h):
        if kinds.get(h) == "binary":
            return 1.0 if col.checkbox(hist_label(h), key=f"h_{h}") else 0.0
        if kinds.get(h) == "lab":
            return col.number_input(hist_label(h), value=None, step=0.1, key=f"h_{h}", placeholder="not available")
        return float(col.number_input(hist_label(h), value=0, min_value=0, step=1, key=f"h_{h}"))

    cols = st.columns(len(key_fields) or 1)
    for c, h in zip(cols, key_fields):
        vals[h] = field(c, h)
    with st.expander(f"More history — outpatient medicines, conditions, past labs ({len(rest)} fields, optional)"):
        for group, title in (("binary", "Conditions on record"), ("count", "Outpatient medicine counts (by class)"),
                             ("lab", "Past-year lab values")):
            hs = [h for h in rest if kinds.get(h, "count") == group]
            if not hs:
                continue
            st.markdown(f"<div class='fl-section'>{title}</div>", unsafe_allow_html=True)
            gc = st.columns(2)
            for i, h in enumerate(hs):
                vals[h] = field(gc[i % 2], h)
    st.session_state.hist_vals = vals


# ------------------------------------------------------------------ screen 2
CONF_TEXT = {
    "likely_admission": ("🛏️", "Likely admission", "The conformal check gives a single answer at the 90% level: admission."),
    "likely_discharge": ("🏠", "Likely discharge", "The conformal check gives a single answer at the 90% level: discharge."),
    "uncertain": ("⚠️", "Uncertain — clinician review",
                  "Both outcomes remain plausible at the 90% confidence level. The model should not be relied on for this patient."),
}


def risk_scale_html(p: float, b) -> str:
    tl, th = b.thr["t_low"], b.thr["t_high"]
    seg = [(tl, BAND["Low"][0]), (th - tl, BAND["Medium"][0]), (1 - th, BAND["High"][0])]
    bars = "".join(f'<div style="width:{w*100:.2f}%;background:{c};opacity:.85"></div>' for w, c in seg)
    return (f'<div class="fl-scale">{bars}</div><div class="fl-marker"><span style="left:calc({p*100:.1f}% - 1px)"></span></div>'
            f'<div class="fl-scale-labels"><span>0%</span><span>Low &lt; {tl:.0%}</span><span>High ≥ {th:.0%}</span><span>100%</span></div>')


def screen_result():
    r = st.session_state.get("result")
    if r is None:
        st.info("No estimate yet — enter a patient on the **Patient** screen.")
        st.button("Go to patient form", on_click=goto, args=("Patient",))
        return
    b = B[st.session_state.bundle_slot]
    col, bg, ico = BAND[r.band]
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        st.markdown(f"""<div class="fl-risk"><div class="lab">Estimated admission risk</div>
            <div class="val" style="color:{col}">{r.probability*100:.0f}%</div>
            <span class="fl-chip" style="background:{bg};color:{col}">{ico} {r.band} risk</span>
            {risk_scale_html(r.probability, b)}</div>""", unsafe_allow_html=True)
    with c2:
        icon, title, desc = CONF_TEXT[r.conformal]
        cls = "fl-conf unc" if r.conformal == "uncertain" else "fl-conf"
        st.markdown(f'<div class="{cls}"><div class="ico">{icon}</div><div><div class="t">{title}</div>'
                    f'<div class="d">{desc}</div></div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="fl-meta">
            <div><span>Model</span><b>{b.spec['slot_name'].split('(')[-1].rstrip(')')}</b></div>
            <div><span>Version</span><b>{r.model_version}</b></div>
            <div><span>Inference</span><b>{r.latency_ms:.1f} ms</b></div>
            <div><span>Inputs used</span><b>{r.feature_count}</b></div></div>""", unsafe_allow_html=True)
        if r.missing_vitals:
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            st.markdown(f'<div class="fl-note">Not measured: <b>{", ".join(r.missing_vitals)}</b>. The estimate still works, '
                        f'but it is less reliable — the model treats these as unknown.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    a, bcol, c = st.columns([1, 1, 2])
    a.button("Why this estimate?  →", type="primary", on_click=goto, args=("Explanation",), use_container_width=True)
    bcol.button("New patient", on_click=lambda: (clear_form(), goto("Patient")), use_container_width=True)
    with st.expander("How to read this"):
        st.markdown(f"""
- **Risk %** is a calibrated probability: among past visits given about this estimate, about that share were admitted (in the US training hospitals).
- **Bands** were set on validation data: *Low* (< {b.thr['t_low']:.0%}) keeps the negative predictive value ≥ 95%; *High* (≥ {b.thr['t_high']:.0%}) keeps the positive predictive value ≥ 80%.
- **Uncertain — clinician review** comes from split conformal prediction (α = 0.10): when both outcomes stay plausible, the tool says so instead of guessing.
- The tool estimates *likelihood of admission in the training health system*; it does not assign triage level, diagnose, or recommend treatment.
""")


# ------------------------------------------------------------------ screen 3
def screen_explanation():
    r = st.session_state.get("result")
    if r is None:
        st.info("No estimate yet — enter a patient on the **Patient** screen.")
        return
    contrib = r.contributions
    min_c = 0.05  # hide near-zero contributions (|log-odds| < 0.05) to keep the list readable
    up = [c for c in contrib if c["contribution"] > min_c][:5]
    down = [c for c in contrib if c["contribution"] < -min_c][:5]
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown(f"**What moved this estimate** · {r.probability*100:.0f}% ({r.band})")
            items = (up + down)[::-1]
            items = sorted(up + down, key=lambda d: d["contribution"])
            fig = go.Figure(go.Bar(
                x=[d["contribution"] for d in items], y=[d["factor"] for d in items], orientation="h",
                marker=dict(color=[RAISE if d["contribution"] > 0 else LOWER for d in items], cornerradius=4),
                hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>", width=0.62))
            fig.add_vline(x=0, line_color="#c3c2b7", line_width=1)
            fig.update_layout(height=max(260, 44 * len(items) + 70), margin=dict(l=10, r=20, t=10, b=40),
                              paper_bgcolor="white", plot_bgcolor="white", font=dict(family="system-ui", color=INK, size=13),
                              xaxis=dict(title="← lowers estimated risk      raises estimated risk →", zeroline=False,
                                         gridcolor="#eef0ee", showticklabels=False, title_font=dict(size=12, color=MUTED)),
                              yaxis=dict(automargin=True), showlegend=False, bargap=0.35)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Bar length = contribution to the model's log-odds (exact TreeSHAP or linear contribution). "
                       "Hover a bar for the value. Associations, not causes.")
    with c2:
        with st.container(border=True):
            st.markdown("**In plain language**")
            rows = "".join(f'<div class="fl-factor"><span class="arrow" style="color:{RAISE}">↑</span>'
                           f'<span><b>{d["factor"]}</b> — associated with higher estimated risk</span></div>' for d in up)
            rows += "".join(f'<div class="fl-factor"><span class="arrow" style="color:{LOWER}">↓</span>'
                            f'<span><b>{d["factor"]}</b> — associated with lower estimated risk</span></div>' for d in down)
            st.markdown(rows or "No factors to show.", unsafe_allow_html=True)
        if r.missing_vitals:
            st.markdown(f'<div class="fl-note"><b>Incomplete inputs.</b> {", ".join(r.missing_vitals)} not measured. '
                        f'The model handles missing values as it learned from the data (where about 30% of visits lacked '
                        f'vitals), so the estimate is less precise. Measuring them may change it.</div>', unsafe_allow_html=True)
        st.caption("These factors explain the model, not the patient. They are statistical associations learned from "
                   "past visits and must not be read as causes.")


# ------------------------------------------------------------------ screen 4
def stat(col, k, v, s=""):
    col.markdown(f'<div class="fl-stat"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def base_layout(fig, h=320, xt="", yt=""):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="system-ui", color=INK, size=12),
                      legend=dict(orientation="h", yanchor="top", y=-0.28, x=0, font=dict(size=11)),
                      xaxis=dict(title=dict(text=xt, standoff=6, font=dict(size=11, color=MUTED)), gridcolor="#eef0ee",
                                 zeroline=False, linecolor="#c3c2b7"),
                      yaxis=dict(title=yt, gridcolor="#eef0ee", zeroline=False, linecolor="#c3c2b7"),
                      hoverlabel=dict(bgcolor="white"))
    return fig


TIER_COL = {"t1": "#2a78d6", "t2": "#eb6834", "t3": "#1baf7a"}
TIER_NAME = {"t1": "T1 nurse-only", "t2": "T2 + ESI", "t3": "T3 + history"}


def screen_dashboard():
    s = netstatus.status(NET["base"])
    b = B["t1"]
    proc = psutil.Process(os.getpid())
    lat = st.session_state.latencies
    c = st.columns(5)
    on = s["online"]
    stat(c[0], "Network", f'<span style="color:{"#c23232" if on else "#0a7d38"}">{"ONLINE" if on else "OFFLINE"}</span>',
         f"{s['bytes_sent']:,} bytes sent since app start")
    stat(c[1], "App external sockets", f"{s['external_sockets'] if s['external_sockets'] >= 0 else 'n/a'}",
         "connections to non-local hosts")
    stat(c[2], "T1 model size", f"{b.model_bytes/1024:.0f} KB", f"bundle {b.size_bytes/1024:.0f} KB")
    stat(c[3], "Latency (this session)", f"{np.median(lat):.1f} ms" if lat else "—",
         f"median of {len(lat)} prediction{'s' if len(lat) != 1 else ''}, end-to-end" if lat else "make a prediction")
    stat(c[4], "App memory (RSS)", f"{proc.memory_info().rss/1e6:.0f} MB", "this process")
    st.caption("Network figures are read from the operating system: bytes on all non-loopback interfaces since the app "
               "started (machine-wide, so other programs count too) and this process's open connections to non-local hosts.")

    d = dashboard_data()
    if d is None:
        st.info("Research results not built yet — run `python -m src.dashboard_data`.")
        return
    st.markdown(f'<div class="fl-section">Research models (best per tier) · {d["eval_label"]}</div>', unsafe_allow_html=True)
    rows = ""
    for r in d["metrics"]:
        ci = lambda k: f'<td class="ci">{r[k][1]:.3f}–{r[k][2]:.3f}</td>' if r.get(k) and r[k][1] is not None else '<td class="ci"></td>'  # noqa: E731
        rows += (f"<tr><td><b>{TIER_NAME[r['tier']]}</b></td><td>{r['model']}</td>"
                 f"<td>{r['auc'][0]:.3f}</td>{ci('auc')}<td>{r['prauc'][0]:.3f}</td>{ci('prauc')}"
                 f"<td>{r['brier'][0]:.3f}</td><td>{r['ece'][0]:.3f}</td></tr>")
    st.markdown(f'<div class="fl-card"><table class="fl-tab"><tr><th>Tier</th><th>Selected model</th><th>ROC-AUC</th><th>95% CI</th>'
                f'<th>PR-AUC</th><th>95% CI</th><th>Brier</th><th>ECE</th></tr>{rows}</table></div>', unsafe_allow_html=True)

    if d.get("edge"):
        st.markdown(f'<div class="fl-section">Shipped edge models (what this app runs) · {d["eval_label"]}</div>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td><b>{e['slot']}</b></td><td>{e['features']}</td><td>{e['size_kb']:.0f}</td>"
                       f"<td>{e['auc'][0]:.3f}</td><td class='ci'>{e['auc'][1]:.3f}–{e['auc'][2]:.3f}</td>"
                       f"<td>{e['ece'][0]:.3f}</td><td>{e['coverage']:.3f}</td><td>{100*e['abstention']:.0f}%</td></tr>" for e in d["edge"])
        st.markdown(f'<div class="fl-card"><table class="fl-tab"><tr><th>Model</th><th>Inputs</th><th>Size KB</th><th>ROC-AUC</th>'
                    f'<th>95% CI</th><th>ECE</th><th>Conformal coverage (α=0.10)</th><th>Uncertain</th></tr>{rows}</table></div>',
                    unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with st.container(border=True):
            st.markdown("**Calibration** · observed vs predicted admission rate (research models)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c2b7", width=1), name="Perfect", hoverinfo="skip"))
            for t, cv in d["calibration"].items():
                fig.add_trace(go.Scatter(x=cv["pred"], y=cv["obs"], mode="lines+markers", name=TIER_NAME[t],
                                         line=dict(color=TIER_COL[t], width=2), marker=dict(size=7, line=dict(color="white", width=2)),
                                         hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra>" + TIER_NAME[t] + "</extra>"))
            st.plotly_chart(base_layout(fig, 300, "Predicted", "Observed"), use_container_width=True, config={"displayModeBar": False})
    with g2:
        with st.container(border=True):
            st.markdown("**Decision curve** · net benefit across thresholds")
            dc = d["decision_curve"]
            fig = go.Figure()
            for t in ("t1", "t2", "t3"):
                if t in dc:
                    fig.add_trace(go.Scatter(x=dc["threshold"], y=dc[t], mode="lines", name=TIER_NAME[t], line=dict(color=TIER_COL[t], width=2)))
            fig.add_trace(go.Scatter(x=dc["threshold"], y=dc["treat_all"], mode="lines", name="Treat all", line=dict(color="#b9b8b1", width=2)))
            fig.add_trace(go.Scatter(x=dc["threshold"], y=[0] * len(dc["threshold"]), mode="lines", name="Treat none", line=dict(color="#4d5a58", width=1)))
            fig.update_yaxes(range=[-0.05, max(max(dc.get("t3", dc["t1"])), 0.3) * 1.1])
            st.plotly_chart(base_layout(fig, 300, "Threshold probability", "Net benefit"), use_container_width=True, config={"displayModeBar": False})
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with st.container(border=True):
            st.markdown("**Risk–coverage** · error rate when answering only the most confident cases")
            fig = go.Figure()
            for t, rc in d["risk_coverage"].items():
                fig.add_trace(go.Scatter(x=rc["coverage"], y=rc["error"], mode="lines", name=f"{TIER_NAME[t]} (AURC {rc['aurc']:.3f})",
                                         line=dict(color=TIER_COL[t], width=2)))
            st.plotly_chart(base_layout(fig, 300, "Share of patients answered", "Error on answered"), use_container_width=True, config={"displayModeBar": False})
    with g2:
        with st.container(border=True):
            st.markdown("**Local adaptation** · T1 conformal coverage at an unseen hospital vs local visits used")
            la = d.get("local_adaptation")
            if la:
                fig = go.Figure()
                cols = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
                for site, v in la["sites"].items():
                    fig.add_trace(go.Scatter(x=v["n"], y=v["coverage_admit"], mode="lines+markers", name=f"Site {site} · admitted",
                                             line=dict(color=cols[site], width=2), marker=dict(size=7, line=dict(color="white", width=2)),
                                             error_y=dict(type="data", array=v["coverage_admit_sd"], thickness=1, width=3, color=cols[site])))
                fig.add_hline(y=0.9, line_color="#4d5a58", line_width=1, annotation_text="target 0.90", annotation_font_size=10,
                              annotation_position="bottom right")
                fig.update_xaxes(type="log", tickvals=[50, 100, 250, 500, 1000, 2000, 5000],
                                 ticktext=["0", "100", "250", "500", "1k", "2k", "5k"])
                st.plotly_chart(base_layout(fig, 320, "Local visits used (no retraining)", "Coverage, admitted patients"),
                                use_container_width=True, config={"displayModeBar": False})
                st.caption(la["caption"])
    if d.get("benchmark"):
        st.markdown('<div class="fl-section">Edge benchmark (measured)</div>', unsafe_allow_html=True)
        def net_cell(r):
            if r.get("interfaces") == ["lo"]:
                return f"<b>{r['net_bytes']} bytes</b> (no network interface)"
            return f"{r.get('app_conns', 0)} app connections · {r['net_bytes']/1024:.0f} KB machine-wide"
        rows = "".join(f"<tr><td>{r['setting']}</td><td>{r['model']}</td><td>{r['size_kb']:.0f}</td><td>{r['p50_ms']:.1f}</td>"
                       f"<td>{r['p95_ms']:.1f}</td><td>{r['peak_rss_mb']:.0f}</td><td>{net_cell(r)}</td></tr>" for r in d["benchmark"])
        st.markdown(f'<div class="fl-card"><table class="fl-tab"><tr><th>Setting</th><th>Model</th><th>Size KB</th><th>p50 ms</th>'
                    f'<th>p95 ms</th><th>Peak RSS MB</th><th>Network during benchmark</th></tr>{rows}</table></div>', unsafe_allow_html=True)
        st.caption("Host: machine-wide interface bytes include other programs on a Wi-Fi-connected laptop; the app-level "
                   "measure is this process's connections to non-local hosts. Docker --network none has only a loopback interface.")
    st.caption(d.get("footnote", ""))


# ------------------------------------------------------------------ main
if "t1" not in B:
    st.error("No model bundle found in models/bundles/. Build it with `python -m src.bundle`.")
    st.stop()
header()
{"Patient": screen_patient, "Result": screen_result, "Explanation": screen_explanation,
 "Dashboard": screen_dashboard}.get(st.session_state.page or "Patient", screen_patient)()
