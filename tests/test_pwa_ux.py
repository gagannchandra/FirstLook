"""PWA UX checks against the Vercel Web Interface Guidelines, in a real browser (Playwright + system Chrome).

Skips when Playwright or Chrome is unavailable. Starts its own static server on a free port.
"""
import contextlib
import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest

PWA = Path(__file__).resolve().parents[1] / "app" / "pwa"
sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PWA))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/index.html"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chrome not available: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser, base_url):
    ctx = browser.new_context(viewport={"width": 1280, "height": 860}, service_workers="block")
    pg = ctx.new_page()
    pg.goto(base_url); pg.wait_for_selector("#examples button")
    yield pg
    ctx.close()


def test_empty_submit_shows_errors_next_to_fields_and_focuses_first(page):
    page.click("#go")
    assert page.is_visible("#err-age") and page.is_visible("#err-gender") and page.is_visible("#err-complaints")
    assert page.get_attribute("#age", "aria-invalid") == "true"
    assert page.evaluate("document.activeElement.id") == "age"
    assert page.is_visible("#errsum")
    assert "attention" in page.inner_text("#live")
    page.fill("#age", "54")  # editing clears that field's message only
    assert not page.is_visible("#err-age") and page.is_visible("#err-gender")


def test_enter_submits_and_routes_with_history_and_title(page):
    page.get_by_role("button", name="Uncertain").click()
    page.focus("#triage_vital_hr"); page.keyboard.press("Enter")
    page.wait_for_function("location.hash === '#/result'")
    assert "44%" in page.title() and page.title().startswith("Result")
    assert page.get_attribute("nav a[data-p=result]", "aria-current") == "page"
    page.go_back(); page.wait_for_function("location.hash !== '#/result'")
    assert page.is_visible("#patient") and not page.is_visible("#result")
    page.go_forward(); page.wait_for_function("location.hash === '#/result'")
    assert page.is_visible("#resbody")


def test_refresh_keeps_form_and_deep_link_without_result_shows_empty_state(page, base_url):
    page.get_by_role("button", name="High risk").click()
    page.reload(); page.wait_for_selector("#examples button")
    assert page.input_value("#age") == "82" and page.locator("#ccchips button").count() == 1
    page.goto(base_url + "#/explain"); page.wait_for_selector("#noexplain")
    assert page.is_visible("#noexplain") and page.locator("#noexplain a").count() == 1  # next step offered


def test_clear_offers_undo(page):
    page.get_by_role("button", name="Low risk").click()
    page.get_by_role("button", name="Clear Form").click()
    assert page.input_value("#age") == "" and page.is_visible("#toast")
    page.click("#undo")
    assert page.input_value("#age") == "24" and not page.is_visible("#toast")


def test_skip_link_does_not_change_screen(page):
    page.get_by_role("button", name="Uncertain").click(); page.click("#go")
    page.wait_for_function("location.hash === '#/result'")
    assert page.evaluate("document.activeElement.id") == "main"  # focus moved to the new screen
    page.reload(); page.wait_for_selector("#noresult")
    page.keyboard.press("Tab")  # first tab stop = skip link
    assert page.evaluate("document.activeElement.className") == "skip"
    page.keyboard.press("Enter")
    assert page.is_visible("#result")


def test_complaint_combobox_keyboard(page):
    page.focus("#ccsearch"); page.keyboard.type("chest")
    assert page.get_attribute("#ccsearch", "aria-expanded") == "true"
    page.keyboard.press("ArrowDown"); page.keyboard.press("Enter")
    assert page.locator("#ccchips button").count() == 1
    page.locator("#ccchips button").first.click()
    assert page.locator("#ccchips button").count() == 0


def test_controls_labelled_inputs_16px_and_targets(page):
    unlabeled = page.evaluate("""() => [...document.querySelectorAll('input, select')]
        .filter(e => e.type !== 'hidden' && !(e.labels?.length || e.getAttribute('aria-label') || e.getAttribute('aria-labelledby')))
        .map(e => e.id)""")
    assert unlabeled == [], unlabeled
    small = page.evaluate("""() => [...document.querySelectorAll('input[type=number], input[type=search], select')]
        .filter(e => parseFloat(getComputedStyle(e).fontSize) < 16).map(e => e.id)""")
    assert small == [], small
    tiny = page.evaluate("""() => [...document.querySelectorAll('button, a, label.check')]
        .filter(e => e.offsetParent && e.getBoundingClientRect().height < 24).map(e => e.textContent.trim().slice(0, 20))""")
    assert tiny == [], tiny


@pytest.mark.parametrize("width", [390, 768])
def test_no_horizontal_overflow_on_small_screens(browser, base_url, width):
    ctx = browser.new_context(viewport={"width": width, "height": 800}, service_workers="block")
    pg = ctx.new_page(); pg.goto(base_url); pg.wait_for_selector("#examples button")
    pg.get_by_role("button", name="High risk").click(); pg.click("#go"); pg.wait_for_function("location.hash === '#/result'")
    for route in ("#/result", "#/explain", "#/about", "#/patient"):
        pg.evaluate(f"location.hash = '{route}'"); pg.wait_for_timeout(100)
        assert pg.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), (width, route)
    ctx.close()
