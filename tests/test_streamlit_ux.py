"""Streamlit app UX checks (Vercel Web Interface Guidelines subset that Streamlit allows), in a real browser.

Starts the app on a free localhost port. Skips when Playwright or Chrome is unavailable.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sync_api = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "bundles").exists(), reason="no bundles")


@pytest.fixture(scope="module")
def app_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    env = {**os.environ, "ORT_DISABLE_TELEMETRY": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(port),
                             "--server.address", "127.0.0.1", "--server.headless", "true"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=1).read() == b"ok":
                break
        except OSError:
            time.sleep(0.5)
    yield url
    proc.terminate(); proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chrome not available: {e}")
        yield b
        b.close()


def idle(pg, t=900):
    pg.wait_for_timeout(250)
    try:
        pg.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=15000)
    except Exception:  # noqa: BLE001
        pass
    pg.wait_for_timeout(t)


@pytest.fixture
def page(browser, app_url):
    ctx = browser.new_context(viewport={"width": 1366, "height": 900})
    pg = ctx.new_page(); pg.goto(app_url); pg.wait_for_selector("text=Quick fill"); idle(pg)
    yield pg
    ctx.close()


def click(pg, name):
    pg.get_by_role("button", name=name).first.click(); idle(pg)


def test_messages_next_to_fields_then_cleared_by_fill(page):
    click(page, "Estimate Admission Risk")
    assert page.locator(".fl-ferr").count() >= 3  # age, sex, complaint (+ blank vitals)
    assert page.get_by_text("need attention").count() == 1
    click(page, "Low risk")
    assert page.locator(".fl-ferr").count() == 0


def test_screen_in_url_and_title(page, app_url):
    click(page, "Uncertain"); click(page, "Estimate Admission Risk")
    assert "screen=result" in page.url and page.title().startswith("Result · 44%")
    page.goto(app_url + "/?screen=dashboard"); page.wait_for_selector("text=Research models"); idle(page)
    assert page.title().startswith("Dashboard")
    sel = page.locator('[data-testid="stButtonGroup"]').first.locator('[aria-checked="true"]').inner_text()
    assert "Dashboard" in sel


def test_clear_then_undo(page):
    click(page, "High risk")
    click(page, "Clear Form")
    assert page.locator(".st-key-undo button").count() == 1
    click(page, "Undo")
    assert page.locator('[data-testid="stNumberInputField"]').first.input_value() == "82"


def test_random_patient_runs_end_to_end(page):
    page.locator(".st-key-rand_acuity").get_by_text("Low", exact=True).click(); idle(page)
    click(page, "Random Patient")
    assert page.locator(".fl-tag").filter(has_text="low acuity").count() == 1
    click(page, "Estimate Admission Risk")
    assert page.locator(".fl-hero").count() == 1 and "Likely Discharge" in page.inner_text(".fl-conf")
