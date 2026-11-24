"""FirstLook — offline admission-risk estimate at ED triage (research prototype).

Run:  streamlit run app/streamlit_app.py
Four screens: Patient → Result → Explanation → Research dashboard.
No network calls: the model runs in-process via ONNX Runtime.
"""
from __future__ import annotations

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # D28: no ONNX Runtime telemetry

import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import psutil  # noqa: E402
import streamlit as st  # noqa: E402

import netstatus  # noqa: E402
from examples import EXAMPLES  # noqa: E402
from random_patient import random_patient  # noqa: E402
from validation import c_to_f, validate  # noqa: E402

from src.runtime import load_all  # noqa: E402

st.set_page_config(page_title="FirstLook · triage admission risk", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ design tokens
# Status colours from the dataviz reference palette (always paired with an icon + label, never colour alone);
# categorical series slots 1-3 validated with scripts/validate_palette.js (light surface).
ACCENT, ACCENT_SOFT = "#0f6e6a", "#e7f2f1"
INK, INK2, MUTED, LINE, SURFACE, BG = "#111a19", "#47534f", "#76817e", "#e3e7e5", "#ffffff", "#f3f5f4"
BAND = {"Low": ("#0ca30c", "#e9f6e9", "●"), "Medium": ("#fab219", "#fff5dc", "◆"), "High": ("#d03b3b", "#fcebeb", "▲")}
UNCERTAIN = ("#6250d6", "#f1effc")
RAISE, LOWER = "#e34948", "#2a78d6"  # diverging pair (red ↔ blue), neutral midpoint grey
SERIES = {"t1": "#2a78d6", "t2": "#eb6834", "t3": "#1baf7a"}
TIER_NAME = {"t1": "T1 nurse-only", "t2": "T2 + ESI", "t3": "T3 + history"}
PAGES = ["Patient", "Result", "Explanation", "Dashboard"]
PAGE_ICON = {"Patient": "①  Patient", "Result": "②  Result", "Explanation": "③  Explanation", "Dashboard": "Dashboard"}
DASH_FILE = ROOT / "results" / "dashboard" / "dashboard.json"

LOGO = ("<svg width='30' height='30' viewBox='0 0 64 64' aria-hidden='true'><rect width='64' height='64' rx='16' fill='#0f6e6a'/>"
        "<path d='M12 34h10l5-12 7 22 5-10h13' fill='none' stroke='#fff' stroke-width='4.5' stroke-linecap='round' "
        "stroke-linejoin='round'/></svg>")

# ------------------------------------------------------------------ style
st.markdown(f"""
<style>
  :root {{ --accent:{ACCENT}; --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --line:{LINE};
          --shadow: 0 1px 2px rgba(17,26,25,.04), 0 6px 20px -8px rgba(17,26,25,.10); }}
  #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {{ display: none !important; }}
  .stApp {{ background: {BG}; }}
  .block-container {{ max-width: 1200px; padding-top: 0.6rem; padding-bottom: 5rem; }}
  html, body, [class*="css"], .stApp {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
                                        -webkit-font-smoothing: antialiased; }}
  .fl-num {{ font-variant-numeric: tabular-nums; }}

  /* app bar */
  .fl-bar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.75rem 1rem; margin: 0 0 .7rem 0;
             background: rgba(255,255,255,.85); backdrop-filter: blur(8px); border:1px solid {LINE}; border-radius:16px; box-shadow: var(--shadow); }}
  .fl-brand {{ display:flex; align-items:center; gap:.7rem; }}
  .fl-brand .name {{ font-size: 1.3rem; font-weight: 750; color: {INK}; letter-spacing:-.02em; line-height:1; }}
  .fl-brand .name span {{ color: {ACCENT}; }}
  .fl-brand .sub {{ color: {MUTED}; font-size: .82rem; margin-top:.2rem; }}
  .fl-pill {{ display:inline-flex; align-items:center; gap:.5rem; padding:.4rem .8rem; border-radius:999px; font-size:.8rem;
             font-weight:600; border:1px solid {LINE}; background:#fff; color:{INK2}; white-space:nowrap; }}
  .fl-dot {{ width:.55rem; height:.55rem; border-radius:50%; display:inline-block; position:relative; }}
  .fl-dot::after {{ content:""; position:absolute; inset:-4px; border-radius:50%; background: inherit; opacity:.35; animation: flpulse 2s infinite; }}
  @keyframes flpulse {{ 0% {{ transform:scale(.6); opacity:.45 }} 100% {{ transform:scale(1.6); opacity:0 }} }}

  /* nav */
  .st-key-nav [data-testid="stButtonGroup"] {{ background:#e9eeec; padding:4px; border-radius:12px; display:inline-flex; }}
  .st-key-nav button {{ border:none !important; background:transparent !important; border-radius:9px !important; padding:.35rem 1rem !important; }}
  .st-key-nav button p {{ font-weight:600; color:{INK2}; }}
  .st-key-nav button[aria-checked="true"], .st-key-nav button[kind$="Active"] {{ background:#fff !important; box-shadow: 0 1px 3px rgba(17,26,25,.12); }}
  .st-key-nav button[aria-checked="true"] p, .st-key-nav button[kind$="Active"] p {{ color:{ACCENT}; }}

  /* cards (keyed containers) */
  div[class*="st-key-card"] {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:16px; padding:1.1rem 1.25rem 1.2rem;
                               box-shadow: var(--shadow); }}
  .fl-h {{ display:flex; align-items:center; gap:.55rem; margin:0 0 .3rem 0; }}
  .fl-h .n {{ width:1.5rem; height:1.5rem; border-radius:8px; background:{ACCENT_SOFT}; color:{ACCENT}; font-weight:750;
             font-size:.8rem; display:inline-flex; align-items:center; justify-content:center; }}
  .fl-h .t {{ font-weight:700; color:{INK}; font-size:1rem; }}
  .fl-h .s {{ color:{MUTED}; font-size:.82rem; margin-left:auto; }}
  .fl-eyebrow {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:{MUTED}; font-weight:700; margin:.2rem 0 .45rem 0; }}

  /* quick fill */
  .st-key-card-quick {{ background: linear-gradient(135deg, #ffffff 0%, #f2f8f7 100%) !important; }}
  .fl-tag {{ display:inline-flex; align-items:center; gap:.4rem; font-size:.82rem; color:{INK2}; background:{ACCENT_SOFT};
            border-radius:999px; padding:.25rem .7rem; margin-top:.2rem; }}

  /* buttons / inputs */
  .stButton>button {{ border-radius:10px; font-weight:600; border-color:{LINE}; transition: all .15s ease; }}
  .stButton>button:hover {{ border-color:{ACCENT}; color:{ACCENT}; transform: translateY(-1px); }}
  .stButton>button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; box-shadow: 0 6px 16px -6px rgba(15,110,106,.55); }}
  .stButton>button[kind="primary"]:hover {{ color:#fff; filter: brightness(1.07); }}
  .st-key-go button {{ padding:.8rem 1rem; font-size:1.02rem; }}
  .st-key-random button {{ background:{INK} !important; color:#fff !important; border-color:{INK} !important; }}
  .st-key-random button p, .st-key-random button span {{ color:#fff !important; }}
  .st-key-random button:hover {{ filter:brightness(1.35); }}
  div[data-baseweb="input"], div[data-baseweb="select"] > div {{ border-radius:10px !important; }}
  [data-testid="stExpander"] details {{ border-radius:12px; border-color:{LINE}; background:#fff; }}

  /* progress */
  .fl-progress {{ display:flex; align-items:center; gap:.8rem; color:{INK2}; font-size:.85rem; margin:.4rem 0 .5rem; }}
  .fl-progress .track {{ flex:1; height:6px; background:#e7ebe9; border-radius:4px; overflow:hidden; }}
  .fl-progress .fill {{ height:100%; background:{ACCENT}; border-radius:4px; transition: width .3s; }}

  /* result */
  .fl-hero {{ background:#fff; border:1px solid {LINE}; border-radius:20px; padding:1.4rem 1.6rem 1.3rem; box-shadow: var(--shadow); }}
  .fl-hero .lab {{ color:{INK2}; font-size:.92rem; font-weight:600; display:flex; justify-content:space-between; }}
  .fl-hero .val {{ font-size: 4.6rem; font-weight: 780; line-height: 1; margin:.45rem 0 .7rem 0; color:{INK}; letter-spacing:-.03em; }}
  .fl-hero .val small {{ font-size: 2rem; font-weight:700; color:{INK2}; margin-left:.1rem; }}
  .fl-chip {{ display:inline-flex; align-items:center; gap:.5rem; padding:.38rem .9rem; border-radius:999px; font-weight:700; font-size:.95rem; color:{INK}; }}
  .fl-chip i {{ font-style:normal; font-size:.95rem; }}
  .fl-meter {{ position:relative; margin-top:2.1rem; }}
  .fl-meter .segs {{ display:flex; gap:2px; height:10px; }}
  .fl-meter .segs div {{ height:100%; }}
  .fl-meter .segs div:first-child {{ border-radius:5px 0 0 5px; }} .fl-meter .segs div:last-child {{ border-radius:0 5px 5px 0; }}
  .fl-meter .mark {{ position:absolute; top:-6px; width:4px; height:22px; background:{INK}; border-radius:2px; border:2px solid #fff; box-sizing:content-box; margin-left:-4px; }}
  .fl-meter .bubble {{ position:absolute; top:-30px; transform:translateX(-50%); font-size:.75rem; font-weight:700; color:#fff; background:{INK};
                      padding:.1rem .45rem; border-radius:6px; }}
  .fl-meter .labels {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:.75rem; margin-top:.45rem; }}
  .fl-conf {{ border-radius:16px; padding:1.05rem 1.2rem; border:1px solid {LINE}; background:#fff; display:flex; gap:.9rem; align-items:flex-start; box-shadow: var(--shadow); }}
  .fl-conf .ico {{ width:2.4rem; height:2.4rem; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.25rem; flex:0 0 auto; background:{ACCENT_SOFT}; }}
  .fl-conf .t {{ font-weight:750; font-size:1.1rem; color:{INK}; }}
  .fl-conf .d {{ color:{INK2}; font-size:.9rem; margin-top:.2rem; line-height:1.45; }}
  .fl-conf.unc {{ border:2px dashed {UNCERTAIN[0]}; background:{UNCERTAIN[1]}; box-shadow:none; }}
  .fl-conf.unc .ico {{ background:#fff; }}
  .fl-conf.unc .t {{ color:{UNCERTAIN[0]}; }}
  .fl-meta {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:.6rem; margin-top:.8rem; }}
  .fl-meta div {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:.6rem .8rem; }}
  .fl-meta b {{ display:block; font-size:1.02rem; color:{INK}; margin-top:.1rem; }}
  .fl-meta span {{ font-size:.7rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
  .fl-summary {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.1rem 0 .9rem; }}
  .fl-summary span {{ background:#fff; border:1px solid {LINE}; border-radius:999px; padding:.22rem .7rem; font-size:.82rem; color:{INK2}; }}
  .fl-summary span.miss {{ border-style:dashed; color:{MUTED}; }}
  .fl-note {{ background:#fff8e6; border:1px solid #f3dfaa; color:#5e4500; border-radius:12px; padding:.65rem .9rem; font-size:.88rem; margin-top:.8rem; }}
  .fl-factor {{ display:flex; gap:.7rem; align-items:center; padding:.5rem 0; border-bottom:1px solid #f0f2f1; font-size:.93rem; color:{INK}; }}
  .fl-factor:last-child {{ border-bottom:none; }}
  .fl-factor .arrow {{ width:1.6rem; height:1.6rem; border-radius:8px; display:inline-flex; align-items:center; justify-content:center; font-weight:800; flex:0 0 auto; }}
  .fl-factor .v {{ margin-left:auto; color:{MUTED}; font-size:.82rem; }}

  /* dashboard */
  .fl-stat {{ background:#fff; border:1px solid {LINE}; border-radius:16px; padding:.9rem 1.05rem; height:100%; box-shadow: var(--shadow); }}
  .fl-stat .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:{MUTED}; font-weight:700; }}
  .fl-stat .v {{ font-size:1.65rem; font-weight:750; color:{INK}; margin-top:.2rem; letter-spacing:-.02em; }}
  .fl-stat .s {{ font-size:.8rem; color:{INK2}; }}
  .fl-section {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; color:{MUTED}; font-weight:700; margin:1.2rem 0 .45rem 0; }}
  .fl-card {{ background:#fff; border:1px solid {LINE}; border-radius:16px; padding:.6rem 1rem; box-shadow: var(--shadow); overflow-x:auto; }}
  table.fl-tab {{ width:100%; border-collapse:collapse; font-size:.88rem; font-variant-numeric: tabular-nums; }}
  table.fl-tab th {{ text-align:left; color:{MUTED}; font-weight:700; font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid {LINE}; padding:.55rem .5rem; }}
  table.fl-tab td {{ border-bottom:1px solid #f0f2f1; padding:.55rem .5rem; color:{INK}; }}
  table.fl-tab tr:last-child td {{ border-bottom:none; }}
  table.fl-tab td.ci {{ color:{MUTED}; font-size:.8rem; }}
  .fl-sw {{ display:inline-block; width:.7rem; height:.7rem; border-radius:3px; margin-right:.45rem; vertical-align:-1px; }}

  .fl-disclaimer {{ position:fixed; left:0; right:0; bottom:0; z-index:999; background:{INK}; color:#eef2f1;
                   text-align:center; font-size:.84rem; padding:.6rem 1rem; letter-spacing:.01em; }}
  .fl-disclaimer b {{ color:#fff; }}
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
    st.session_state.last_page = page


def _nav_changed():
    """Segmented controls toggle off when the active option is clicked again; keep the current screen instead."""
    ss = st.session_state
    if ss.page is None:
        ss.page = ss.get("last_page", "Patient")
    ss.last_page = ss.page


HOUR_BINS = [("23-02", "23:00–02:59"), ("03-06", "03:00–06:59"), ("07-10", "07:00–10:59"),
             ("11-14", "11:00–14:59"), ("15-18", "15:00–18:59"), ("19-22", "19:00–22:59")]
ARRIVALS = [("Walk-in", "Walk-in"), ("Car", "Private car / taxi"), ("ambulance", "Ambulance / EMS"),
            ("Wheelchair", "Wheelchair"), ("Public Transportation", "Public transport"), ("Police", "Police"), ("Other", "Other")]
O2_OPTS = ["Room air", "On supplemental oxygen", "Not recorded"]
VITAL_KEYS = ["triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr", "triage_vital_o2", "triage_vital_temp"]
VITAL_UI = [("triage_vital_hr", "Heart rate", "bpm", "typical 60–100"), ("triage_vital_rr", "Respiratory rate", "/min", "typical 12–20"),
            ("triage_vital_sbp", "Systolic BP", "mmHg", "typical 100–140"), ("triage_vital_dbp", "Diastolic BP", "mmHg", "typical 60–90"),
            ("triage_vital_o2", "SpO₂", "%", "typical 95–100")]
ESI_LABEL = {1: "1 · resus", 2: "2 · emergent", 3: "3 · urgent", 4: "4 · less urgent", 5: "5 · non-urgent"}
ACUITY_OPTS = ["Any", "Low", "Medium", "High"]


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
    ss.setdefault("f_use_hist", False); ss.setdefault("rand_acuity", "Any"); ss.setdefault("rand_missing", False)
    for v in VITAL_KEYS:
        ss.setdefault(f"f_{v}", None); ss.setdefault(f"nm_{v}", False)


def _fill(form: dict):
    ss = st.session_state
    ss.f_age = form["age"]; ss.f_gender = form["gender"]; ss.f_arrival = form["arrivalmode"]; ss.f_hour = form["arrivalhour_bin"]
    ss.f_cc = list(form["complaints"]); ss.f_tunit = "°F"
    dev = form.get("triage_vital_o2_device")
    ss.f_o2dev = "Not recorded" if dev is None else ("On supplemental oxygen" if dev == 1 else "Room air")
    for v in VITAL_KEYS:
        val = form.get(v)
        ss[f"nm_{v}"] = val is None
        ss[f"f_{v}"] = None if val is None else float(val)
    ss.pop("errors", None); ss.pop("warnings", None)


def load_example(name: str):
    ss = st.session_state
    _fill(EXAMPLES[name]["form"])
    ss.f_use_esi = False; ss.f_esi = None; ss.f_use_hist = False
    ss.fill_note = ("example", EXAMPLES[name]["label"])


def load_random():
    """Fill the form with a random synthetic patient (app/random_patient.py). Keeps the user's choice of
    optional inputs: if ESI / history are switched on they are filled consistently with the acuity."""
    ss = st.session_state
    seed = int(np.random.default_rng().integers(1, 100_000))
    acuity = None if ss.get("rand_acuity", "Any") in (None, "Any") else ss.rand_acuity.lower()
    form, meta = random_patient(seed, acuity, bool(ss.get("rand_missing")), B["t1"].spec["complaint_order"])
    _fill(form)
    if ss.get("f_use_esi"):
        ss.f_esi = meta["esi"]
    if ss.get("f_use_hist") and "t3lite" in B:
        for h, v in meta["history"].items():
            if h in B["t3lite"].spec["history_inputs"]:
                ss[f"h_{h}"] = v
    ss.fill_note = ("random", f"{meta['acuity']} acuity · seed {seed}")


def clear_form():
    ss = st.session_state
    for k in [k for k in ss.keys() if k.startswith(("f_", "nm_", "h_"))]:
        del ss[k]
    for k in ("result", "form", "fill_note", "errors", "warnings"):
        ss.pop(k, None)


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
    st.session_state.last_page = "Result"


def heading(n: str, title: str, sub: str = ""):
    st.markdown(f'<div class="fl-h"><span class="n">{n}</span><span class="t">{title}</span>'
                f'<span class="s">{sub}</span></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ header
def header():
    s = netstatus.status(NET["base"])
    on = s["online"]
    net_html = (f'<span class="fl-pill"><span class="fl-dot" style="background:{"#d03b3b" if on else "#0ca30c"}"></span>'
                f'{"Online" if on else "Offline"} · app external connections: {max(s["external_sockets"], 0)}'
                + ("" if on else f' · {s["bytes_sent"]:,} bytes sent') + '</span>')
    st.markdown(f'<div class="fl-bar"><div class="fl-brand">{LOGO}<div><div class="name">First<span>Look</span></div>'
                f'<div class="sub">Admission-risk estimate at ED triage · runs fully on this device</div></div></div>{net_html}</div>',
                unsafe_allow_html=True)
    with st.container(key="nav"):
        st.segmented_control("Screen", PAGES, key="page", label_visibility="collapsed", format_func=PAGE_ICON.get,
                            on_change=_nav_changed)


# ------------------------------------------------------------------ screen 1
def screen_patient():
    init_form()
    ss = st.session_state
    with st.container(key="card-quick"):
        q1, q2 = st.columns([1, 1], gap="large")
        with q1:
            st.markdown('<div class="fl-eyebrow">Quick fill · synthetic example patients</div>', unsafe_allow_html=True)
            r1 = st.columns(3)
            r2 = st.columns(3)
            for c, key in zip(r1 + r2[:1], ["high", "low", "uncertain", "missing_spo2"]):
                c.button(EXAMPLES[key]["label"], key=f"ex_{key}", on_click=load_example, args=(key,), use_container_width=True,
                         help=EXAMPLES[key]["hint"])
            r2[1].button("Clear form", on_click=clear_form, use_container_width=True, icon=":material/restart_alt:")
        with q2:
            st.markdown('<div class="fl-eyebrow">Random synthetic patient · choose acuity</div>', unsafe_allow_html=True)
            st.pills("Acuity", ACUITY_OPTS, key="rand_acuity", label_visibility="collapsed")
            r1, r2 = st.columns([1.1, 1], vertical_alignment="center")
            r1.checkbox("Include a missing vital", key="rand_missing")
            r2.button("Random patient", key="random", on_click=load_random, use_container_width=True,
                      icon=":material/casino:", help="Fills every field with plausible, invented values for the chosen acuity.")
        note = ss.get("fill_note")
        if note:
            kind, text = note
            label = "Random synthetic patient" if kind == "random" else "Synthetic example"
            st.markdown(f'<span class="fl-tag">{"🎲" if kind == "random" else "🧪"} <b>{label}:</b> {text} — invented for the demo, '
                        f'not a real patient</span>', unsafe_allow_html=True)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        with st.container(key="card-patient"):
            heading("1", "Patient & arrival")
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
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        with st.container(key="card-optional"):
            heading("3", "Optional inputs", "improve accuracy when available")
            st.toggle("ESI level available", key="f_use_esi", help="Adds the Emergency Severity Index (uses the T2 model).")
            if ss.f_use_esi:
                st.segmented_control("ESI level", [1, 2, 3, 4, 5], key="f_esi", format_func=ESI_LABEL.get)
            has_hist = "t3lite" in B
            st.toggle("EHR history available", key="f_use_hist", disabled=not has_hist,
                      help="Adds key history fields (uses the T3-lite model)." if has_hist else "No history model bundled.")
            if ss.f_use_hist and has_hist:
                history_inputs(B["t3lite"].spec)
    with right:
        with st.container(key="card-vitals"):
            n_done = sum(1 for v in VITAL_KEYS if ss[f"nm_{v}"] or ss[f"f_{v}"] is not None)
            heading("2", "Triage vitals", f"{n_done}/{len(VITAL_KEYS)} entered or marked")
            for v, lab, unit, typ in VITAL_UI:
                c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
                c1.number_input(f"{lab} ({unit})", key=f"f_{v}", step=1.0, format="%.0f", disabled=ss[f"nm_{v}"],
                                placeholder="not measured" if ss[f"nm_{v}"] else typ)
                c2.checkbox("Not measured", key=f"nm_{v}")
                if v == "triage_vital_o2":
                    st.radio("Oxygen at triage", O2_OPTS, key="f_o2dev", horizontal=True)
            c1, c2, c3 = st.columns([2.1, 0.9, 1.3], vertical_alignment="bottom")
            c1.number_input(f"Temperature ({ss.f_tunit})", key="f_triage_vital_temp", step=0.1, format="%.1f",
                            disabled=ss["nm_triage_vital_temp"],
                            placeholder="not measured" if ss["nm_triage_vital_temp"] else
                            ("typical 97–99.5" if ss.f_tunit == "°F" else "typical 36.1–37.5"))
            c2.radio("Unit", ["°F", "°C"], key="f_tunit", label_visibility="collapsed")
            c3.checkbox("Not measured", key="nm_triage_vital_temp")

    req = [ss.f_age is not None, ss.f_gender is not None, bool(ss.f_cc)]
    done = sum(req) + sum(1 for v in VITAL_KEYS if ss[f"nm_{v}"] or ss[f"f_{v}"] is not None)
    total = len(req) + len(VITAL_KEYS)
    st.markdown(f'<div class="fl-progress"><span><b>{done}</b> of {total} core fields complete</span>'
                f'<div class="track"><div class="fill" style="width:{100*done/total:.0f}%"></div></div></div>', unsafe_allow_html=True)
    for e in ss.get("errors", []):
        st.error(e, icon=":material/front_hand:")
    for w in ss.get("warnings", []):
        st.warning(w, icon=":material/warning:")
    with st.container(key="go"):
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
    ss = st.session_state
    vals = {}
    kinds = spec.get("history_input_kind", {})
    for h in spec["history_inputs"]:  # defaults via session state (so autofill can set them without widget warnings)
        ss.setdefault(f"h_{h}", {"binary": False, "lab": None}.get(kinds.get(h), 0))
    st.caption("Counts cover the past year; leave at 0 if none on record. Leave lab values blank if unavailable.")
    if spec.get("uses_previousdispo"):
        st.selectbox("Outcome of previous ED visit", ["No previous dispo", "Discharge", "Admit", "Transfer to Another Facility",
                                                      "LWBS after Triage", "AMA", "Eloped", "LWBS before Triage", "Observation",
                                                      "Send to L&D"],
                     format_func=lambda v: "No previous ED visit" if v == "No previous dispo" else v, key="f_prevdispo")
    key_fields = [h for h in ("n_edvisits", "n_admissions", "n_surgeries") if h in spec["history_inputs"]]
    rest = [h for h in spec["history_inputs"] if h not in key_fields]

    def field(col, h):
        if kinds.get(h) == "binary":
            return 1.0 if col.checkbox(hist_label(h), key=f"h_{h}") else 0.0
        if kinds.get(h) == "lab":
            return col.number_input(hist_label(h), step=0.1, key=f"h_{h}", placeholder="not available")
        return float(col.number_input(hist_label(h), min_value=0, step=1, key=f"h_{h}"))

    cols = st.columns(len(key_fields) or 1)
    for c, h in zip(cols, key_fields):
        vals[h] = field(c, h)
    with st.expander(f"More history — outpatient medicines, conditions, past labs ({len(rest)} fields, optional)"):
        for group, title in (("binary", "Conditions on record"), ("count", "Outpatient medicine counts (by class)"),
                             ("lab", "Past-year lab values")):
            hs = [h for h in rest if kinds.get(h, "count") == group]
            if not hs:
                continue
            st.markdown(f"<div class='fl-eyebrow'>{title}</div>", unsafe_allow_html=True)
            gc = st.columns(2)
            for i, h in enumerate(hs):
                vals[h] = field(gc[i % 2], h)
    ss.hist_vals = vals


# ------------------------------------------------------------------ screen 2
CONF_TEXT = {
    "likely_admission": ("🛏️", "Likely admission", "The conformal check gives a single answer at the 90% level: admission."),
    "likely_discharge": ("🏠", "Likely discharge", "The conformal check gives a single answer at the 90% level: discharge."),
    "uncertain": ("⚠️", "Uncertain — clinician review",
                  "Both outcomes remain plausible at the 90% confidence level. The model should not be relied on for this patient."),
}


def meter_html(p: float, b) -> str:
    """Hero meter: three band segments (2px gaps, rounded ends) + a marker with the value."""
    tl, th = b.thr["t_low"], b.thr["t_high"]
    seg = [(tl, BAND["Low"][0]), (th - tl, BAND["Medium"][0]), (1 - th, BAND["High"][0])]
    bars = "".join(f'<div style="width:{w*100:.2f}%;background:{c}"></div>' for w, c in seg)
    x = min(max(p * 100, 1), 99)
    return (f'<div class="fl-meter"><div class="bubble fl-num" style="left:{x:.1f}%">{p*100:.0f}%</div>'
            f'<div class="segs">{bars}</div><div class="mark" style="left:{x:.1f}%"></div>'
            f'<div class="labels"><span>0%</span><span>Low &lt; {tl:.0%}</span><span>High ≥ {th:.0%}</span><span>100%</span></div></div>')


def summary_html(form: dict, spec: dict) -> str:
    arr = dict(ARRIVALS).get(form.get("arrivalmode"), form.get("arrivalmode"))
    chips = [f"{form['age']} y · {'F' if form['gender'] == 'Female' else 'M'}", arr]
    chips += [spec["complaint_labels"].get(c, c) for c in form.get("complaints", [])]
    for v, lab, unit, _ in VITAL_UI + [("triage_vital_temp", "Temp", "°F", "")]:
        val = form.get(v)
        short = {"Heart rate": "HR", "Respiratory rate": "RR", "Systolic BP": "SBP", "Diastolic BP": "DBP"}.get(lab, lab)
        chips.append(f'<span class="miss">{short} not measured</span>' if val is None else f"{short} {val:g}{'' if unit == '%' else ' '}{unit}")
    if form.get("triage_vital_o2_device") == 1:
        chips.append("on O₂")
    if form.get("esi"):
        chips.append(f"ESI {form['esi']}")
    return '<div class="fl-summary">' + "".join(c if c.startswith("<span") else f"<span>{c}</span>" for c in chips) + "</div>"


def factor_rows(items: list[dict]) -> str:
    out = ""
    for d in items:
        up = d["contribution"] > 0
        col, bg = (RAISE, "#fdecec") if up else (LOWER, "#e8f1fc")
        out += (f'<div class="fl-factor"><span class="arrow" style="color:{col};background:{bg}">{"↑" if up else "↓"}</span>'
                f'<span><b>{d["factor"]}</b></span><span class="v fl-num">{d["contribution"]:+.2f}</span></div>')
    return out


def screen_result():
    r = st.session_state.get("result")
    if r is None:
        st.info("No estimate yet — enter a patient on the **Patient** screen.")
        st.button("Go to patient form", on_click=goto, args=("Patient",))
        return
    b = B[st.session_state.bundle_slot]
    col, bg, ico = BAND[r.band]
    st.markdown(summary_html(st.session_state.form, B["t1"].spec), unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        st.markdown(f"""<div class="fl-hero"><div class="lab"><span>Estimated admission risk</span>
            <span style="color:{MUTED};font-weight:500">{b.spec['slot_name']}</span></div>
            <div class="val fl-num">{r.probability*100:.0f}<small>%</small></div>
            <span class="fl-chip" style="background:{bg}"><i style="color:{col}">{ico}</i>{r.band} risk</span>
            {meter_html(r.probability, b)}</div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        a, bcol = st.columns(2)
        a.button("Why this estimate?  →", type="primary", on_click=goto, args=("Explanation",), use_container_width=True)
        bcol.button("New patient", on_click=lambda: (clear_form(), goto("Patient")), use_container_width=True,
                    icon=":material/person_add:")
    with c2:
        icon, title, desc = CONF_TEXT[r.conformal]
        cls = "fl-conf unc" if r.conformal == "uncertain" else "fl-conf"
        st.markdown(f'<div class="{cls}"><div class="ico">{icon}</div><div><div class="t">{title}</div>'
                    f'<div class="d">{desc}</div></div></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="fl-meta">
            <div><span>Model</span><b>{b.spec['slot_name'].split('(')[-1].rstrip(')')}</b></div>
            <div><span>Version</span><b>{r.model_version}</b></div>
            <div><span>Inference</span><b class="fl-num">{r.latency_ms:.1f} ms</b></div>
            <div><span>Inputs used</span><b class="fl-num">{r.feature_count}</b></div></div>""", unsafe_allow_html=True)
        top = [c for c in r.contributions if abs(c["contribution"]) > 0.05][:3]
        if top:
            st.markdown(f'<div class="fl-section" style="margin-top:1rem">Top factors</div>'
                        f'<div class="fl-card" style="padding:.3rem 1rem">{factor_rows(top)}</div>', unsafe_allow_html=True)
        if r.missing_vitals:
            st.markdown(f'<div class="fl-note">Not measured: <b>{", ".join(r.missing_vitals)}</b>. The estimate still works, '
                        f'but it is less reliable — the model treats these as unknown.</div>', unsafe_allow_html=True)
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
    st.markdown(summary_html(st.session_state.form, B["t1"].spec), unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        with st.container(key="card-explain"):
            heading("?", "What moved this estimate", f"{r.probability*100:.0f}% · {r.band} risk")
            items = sorted(up + down, key=lambda d: d["contribution"])
            span = max([abs(d["contribution"]) for d in items] + [0.5])
            fig = go.Figure(go.Bar(
                x=[d["contribution"] for d in items], y=[d["factor"] for d in items], orientation="h",
                marker=dict(color=[RAISE if d["contribution"] > 0 else LOWER for d in items], cornerradius=4),
                text=[f"{d['contribution']:+.2f}" for d in items], textposition="outside", cliponaxis=False,
                textfont=dict(color=INK2, size=11),
                hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>", width=0.58))
            fig.add_vline(x=0, line_color="#c3c2b7", line_width=1)
            fig.update_layout(height=max(260, 44 * len(items) + 70), margin=dict(l=10, r=30, t=6, b=40),
                              paper_bgcolor="white", plot_bgcolor="white", font=dict(family="system-ui", color=INK, size=13),
                              xaxis=dict(title="← lowers estimated risk      raises estimated risk →", zeroline=False,
                                         range=[-span * 1.25, span * 1.25], gridcolor="#f0f2f1", showticklabels=False,
                                         title_font=dict(size=12, color=MUTED)),
                              yaxis=dict(automargin=True), showlegend=False, bargap=0.35)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Bar length = contribution to the model's log-odds (exact TreeSHAP or linear contribution). "
                       "Associations, not causes.")
    with c2:
        with st.container(key="card-plain"):
            heading("Aa", "In plain language")
            rows = "".join(f'<div class="fl-factor"><span class="arrow" style="color:{RAISE};background:#fdecec">↑</span>'
                           f'<span><b>{d["factor"]}</b> — associated with higher estimated risk</span></div>' for d in up)
            rows += "".join(f'<div class="fl-factor"><span class="arrow" style="color:{LOWER};background:#e8f1fc">↓</span>'
                            f'<span><b>{d["factor"]}</b> — associated with lower estimated risk</span></div>' for d in down)
            st.markdown(rows or "No factors to show.", unsafe_allow_html=True)
        if r.missing_vitals:
            st.markdown(f'<div class="fl-note"><b>Incomplete inputs.</b> {", ".join(r.missing_vitals)} not measured. '
                        f'The model handles missing values as it learned from the data (where about 30% of visits lacked '
                        f'vitals), so the estimate is less precise. Measuring them may change it.</div>', unsafe_allow_html=True)
        st.caption("These factors explain the model, not the patient. They are statistical associations learned from "
                   "past visits and must not be read as causes.")
        st.button("← Back to result", on_click=goto, args=("Result",))


# ------------------------------------------------------------------ screen 4
def stat(col, k, v, s=""):
    col.markdown(f'<div class="fl-stat"><div class="k">{k}</div><div class="v fl-num">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def base_layout(fig, h=320, xt="", yt=""):
    fig.update_layout(height=h, margin=dict(l=10, r=70, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="system-ui", color=INK, size=12),
                      legend=dict(orientation="h", yanchor="top", y=-0.28, x=0, font=dict(size=11)),
                      xaxis=dict(title=dict(text=xt, standoff=6, font=dict(size=11, color=MUTED)), gridcolor="#f0f2f1",
                                 zeroline=False, linecolor="#c3c2b7"),
                      yaxis=dict(title=dict(text=yt, font=dict(size=11, color=MUTED)), gridcolor="#f0f2f1", zeroline=False,
                                 linecolor="#c3c2b7"),
                      hoverlabel=dict(bgcolor="white"), hovermode="x unified")
    return fig


def end_labels(fig, pts: list[tuple[float, float, str]], min_gap: float):
    """Selective direct labels at line ends (text in ink, not series colour), dodged vertically so
    labels of lines that finish close together never overlap (min_gap in data units)."""
    pts = sorted(pts, key=lambda p: p[1])
    ys: list[float] = []
    for _, y, _ in pts:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    shift = (ys[-1] - pts[-1][1]) / 2 if len(pts) > 1 else 0  # re-centre the stack on the true ends
    for (x, y, text), yl in zip(pts, ys):
        fig.add_annotation(x=x, y=yl - shift, text=text, showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=INK2))


def chart_card(key: str, title: str, sub: str):
    c = st.container(key=f"card-{key}")
    c.markdown(f"**{title}** <span style='color:{MUTED}'>· {sub}</span>", unsafe_allow_html=True)
    return c


def screen_dashboard():
    s = netstatus.status(NET["base"])
    b = B["t1"]
    proc = psutil.Process(os.getpid())
    lat = st.session_state.latencies
    c = st.columns(5)
    on = s["online"]
    stat(c[0], "Network", f'<span style="color:{"#d03b3b" if on else "#0ca30c"}">{"● ONLINE" if on else "● OFFLINE"}</span>',
         f"{s['bytes_sent']:,} bytes sent machine-wide since start")
    stat(c[1], "App external sockets", f"{s['external_sockets'] if s['external_sockets'] >= 0 else 'n/a'}",
         "outbound connections to non-local hosts")
    stat(c[2], "T1 model size", f"{b.model_bytes/1024:.0f} KB", f"bundle {b.size_bytes/1024:.0f} KB")
    stat(c[3], "Latency (this session)", f"{np.median(lat):.1f} ms" if lat else "—",
         f"median of {len(lat)} prediction{'s' if len(lat) != 1 else ''}, end-to-end" if lat else "make a prediction")
    stat(c[4], "App memory (RSS)", f"{proc.memory_info().rss/1e6:.0f} MB", "this process")
    st.caption("Network figures are read from the operating system: bytes on all non-loopback interfaces since the app "
               "started (machine-wide, so other programs count too) and this process's outbound connections to non-local hosts.")

    d = dashboard_data()
    if d is None:
        st.info("Research results not built yet — run `python -m src.dashboard_data`.")
        return
    st.markdown(f'<div class="fl-section">Research models (best per tier) · {d["eval_label"]}</div>', unsafe_allow_html=True)
    rows = ""
    for r in d["metrics"]:
        ci = lambda k: f'<td class="ci">{r[k][1]:.3f}–{r[k][2]:.3f}</td>' if r.get(k) and r[k][1] is not None else '<td class="ci"></td>'  # noqa: E731
        rows += (f"<tr><td><span class='fl-sw' style='background:{SERIES[r['tier']]}'></span><b>{TIER_NAME[r['tier']]}</b></td><td>{r['model']}</td>"
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
    st.markdown('<div class="fl-section">Charts</div>', unsafe_allow_html=True)
    cfg = {"displayModeBar": False}
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with chart_card("cal", "Calibration", "observed vs predicted admission rate"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c2b7", width=1, dash="dot"),
                                     name="Perfect", hoverinfo="skip"))
            for t, cv in d["calibration"].items():
                fig.add_trace(go.Scatter(x=cv["pred"], y=cv["obs"], mode="lines+markers", name=TIER_NAME[t],
                                         line=dict(color=SERIES[t], width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                                         hovertemplate="predicted %{x:.2f} · observed %{y:.2f}<extra>" + TIER_NAME[t] + "</extra>"))
            fig = base_layout(fig, 310, "Predicted", "Observed"); fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, use_container_width=True, config=cfg)
    with g2:
        with chart_card("dc", "Decision curve", "net benefit across thresholds"):
            dc = d["decision_curve"]
            fig = go.Figure()
            for t in ("t1", "t2", "t3"):
                if t in dc:
                    fig.add_trace(go.Scatter(x=dc["threshold"], y=dc[t], mode="lines", name=TIER_NAME[t],
                                             line=dict(color=SERIES[t], width=2), hovertemplate="%{y:.3f}"))
            fig.add_trace(go.Scatter(x=dc["threshold"], y=dc["treat_all"], mode="lines", name="Treat all",
                                     line=dict(color="#b9b8b1", width=2, dash="dash"), hovertemplate="%{y:.3f}"))
            fig.add_trace(go.Scatter(x=dc["threshold"], y=[0] * len(dc["threshold"]), mode="lines", name="Treat none",
                                     line=dict(color="#47534f", width=1), hoverinfo="skip"))
            end_labels(fig, [(dc["threshold"][-1], dc[t][-1], TIER_NAME[t].split(" ")[0]) for t in ("t1", "t2", "t3") if t in dc], 0.022)
            fig.update_yaxes(range=[-0.05, max(max(dc.get("t3", dc["t1"])), 0.3) * 1.1])
            st.plotly_chart(base_layout(fig, 310, "Threshold probability", "Net benefit"), use_container_width=True, config=cfg)
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with chart_card("rc", "Risk–coverage", "error when answering only the most confident cases"):
            fig = go.Figure()
            for t, rc in d["risk_coverage"].items():
                fig.add_trace(go.Scatter(x=rc["coverage"], y=rc["error"], mode="lines", name=f"{TIER_NAME[t]} (AURC {rc['aurc']:.3f})",
                                         line=dict(color=SERIES[t], width=2), hovertemplate="%{y:.3f}"))
            end_labels(fig, [(rc["coverage"][-1], rc["error"][-1], TIER_NAME[t].split(" ")[0]) for t, rc in d["risk_coverage"].items()], 0.014)
            st.plotly_chart(base_layout(fig, 310, "Share of patients answered", "Error on answered"), use_container_width=True, config=cfg)
    with g2:
        with chart_card("la", "Local adaptation", "T1 coverage at an unseen hospital vs local visits used"):
            la = d.get("local_adaptation")
            if la:
                fig = go.Figure()
                cols = {"A": SERIES["t1"], "B": SERIES["t2"], "C": SERIES["t3"]}
                for site, v in la["sites"].items():
                    fig.add_trace(go.Scatter(x=v["n"], y=v["coverage_admit"], mode="lines+markers", name=f"Site {site}",
                                             line=dict(color=cols[site], width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                                             error_y=dict(type="data", array=v["coverage_admit_sd"], thickness=1, width=3, color=cols[site]),
                                             hovertemplate="%{y:.3f}"))
                fig.add_hline(y=0.9, line_color="#47534f", line_width=1, line_dash="dot", annotation_text="target 0.90",
                              annotation_font=dict(size=10, color=INK2), annotation_position="top left")
                fig.update_xaxes(type="log", tickvals=[50, 100, 250, 500, 1000, 2000, 5000],
                                 ticktext=["0", "100", "250", "500", "1k", "2k", "5k"])
                st.plotly_chart(base_layout(fig, 310, "Local visits used (no retraining)", "Coverage, admitted patients"),
                                use_container_width=True, config=cfg)
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
