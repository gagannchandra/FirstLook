"""Drive the running Streamlit app with Playwright and save screenshots.

Usage: python scripts/screenshot_app.py [--url http://localhost:8501] [--out results/figures/app]
Uses the system Chrome (channel="chrome"). Logs every network request the page makes
to a non-localhost host (should be none).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def wait_idle(page, t=1.2):
    page.wait_for_timeout(300)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=15000)
    except Exception:  # noqa: BLE001
        pass
    page.wait_for_timeout(int(t * 1000))


def click_button(page, text):
    page.get_by_role("button", name=text, exact=False).first.click()
    wait_idle(page)


def nav(page, name):
    page.locator('[data-testid="stButtonGroup"]').first.get_by_role("radio", name=name, exact=True).click()
    wait_idle(page)


def shot(page, path, vw, vh):
    """Full-length capture: grow the viewport to the scroll height of Streamlit's main container."""
    h = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]'); return m ? m.scrollHeight : 0 }")
    page.set_viewport_size({"width": vw, "height": max(vh, int(h))})
    page.wait_for_timeout(700)
    page.screenshot(path=path)
    page.set_viewport_size({"width": vw, "height": vh})
    page.wait_for_timeout(300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "app"))
    ap.add_argument("--width", type=int, default=1366)
    ap.add_argument("--height", type=int, default=768)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    external = []
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        page = br.new_page(viewport={"width": a.width, "height": a.height}, device_scale_factor=1.5)
        page.on("request", lambda r: external.append(r.url) if urlparse(r.url).hostname not in ("localhost", "127.0.0.1", None) else None)
        page.goto(a.url); wait_idle(page, 2.5)
        shot(page, out / "01_patient_empty.png", a.width, a.height)
        # validation message
        click_button(page, "Estimate admission risk")
        shot(page, out / "01b_patient_validation.png", a.width, a.height)
        for key, label in [("high", "High risk"), ("low", "Low risk"), ("uncertain", "Uncertain"), ("missing", "SpO₂ not measured")]:
            nav(page, "Patient")
            click_button(page, label)
            if key == "high":
                shot(page, out / "01c_patient_filled_high.png", a.width, a.height)
            click_button(page, "Estimate admission risk")
            shot(page, out / f"02_result_{key}.png", a.width, a.height)
            if key in ("high", "missing"):
                click_button(page, "Why this estimate")
                shot(page, out / f"03_explanation_{key}.png", a.width, a.height)
        # optional inputs: ESI + EHR history (T3-lite model)
        nav(page, "Patient")
        click_button(page, "High risk")
        page.get_by_text("ESI level available", exact=True).click(); wait_idle(page)
        page.locator('[data-testid="stButtonGroup"]').nth(1).get_by_role("radio", name="2 · emergent").click(); wait_idle(page)
        page.get_by_text("EHR history available", exact=True).click(); wait_idle(page)
        shot(page, out / "01d_patient_esi_history.png", a.width, a.height)
        click_button(page, "Estimate admission risk")
        shot(page, out / "02_result_t3lite.png", a.width, a.height)
        nav(page, "Dashboard"); wait_idle(page, 2.5)
        shot(page, out / "04_dashboard.png", a.width, a.height)
        br.close()
    (out / "external_requests.json").write_text(json.dumps(external, indent=1))
    print("external requests:", len(external), external[:5])


if __name__ == "__main__":
    main()
