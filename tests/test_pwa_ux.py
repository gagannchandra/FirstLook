"""Offline web app (frontend/) UX checks in a real browser (Playwright + system Chrome). Skips when Playwright or
Chrome is unavailable.

Most tests run against the static frontend only (browser models, no API): the offline case. The patient-record
test starts the real local backend (backend/server.py) on a free port.

The page's CSP has no 'unsafe-eval', so Playwright's string predicates (wait_for_function) are refused; waits use
selectors, URLs or the `until` poller below.
"""
import socket
import threading
import time

import pytest

from backend.server import make_server

sync_api = pytest.importorskip("playwright.sync_api")


def _serve(static_only: bool):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    srv = make_server("127.0.0.1", port, static_only=static_only)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}/index.html"


def until(fn, timeout=10.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not met in time")


@pytest.fixture(scope="module")
def base_url():
    srv, url = _serve(static_only=True)
    yield url
    srv.shutdown()


@pytest.fixture(scope="module")
def api_url():
    srv, url = _serve(static_only=False)
    yield url
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


def _open(browser, url, width=1280, height=860, **kw):
    ctx = browser.new_context(viewport={"width": width, "height": height}, service_workers="block", **kw)
    pg = ctx.new_page()
    pg.goto(url); pg.wait_for_selector("#examples button", state="attached")
    return ctx, pg


@pytest.fixture
def page(browser, base_url):
    ctx, pg = _open(browser, base_url + "#/assess")
    yield pg
    ctx.close()


def example(pg, name):
    pg.locator("#examples button", has_text=name).click()
    pg.wait_for_selector("#needs[data-state=ready]")


def show(pg):
    """Check and save: opens the separate result page."""
    pg.click("#go"); pg.wait_for_selector("#result:not([hidden])")


def pct(pg, sel="#pct"):
    return int(pg.inner_text(sel).split("%")[0])


# ---------- assessment form ----------

def test_empty_submit_shows_errors_next_to_fields_and_focuses_first(page):
    page.click("#go")
    assert page.is_visible("#err-age") and page.is_visible("#err-gender") and page.is_visible("#err-complaints")
    assert page.get_attribute("#age", "aria-invalid") == "true"
    assert page.evaluate("document.activeElement.id") == "age"
    assert page.is_visible("#errsum")
    until(lambda: "attention" in page.inner_text("#live"))
    page.fill("#age", "54")  # editing clears that field's message only
    assert not page.is_visible("#err-age") and page.is_visible("#err-gender")


def test_assess_and_result_are_separate_pages(page):
    example(page, "Uncertain")
    assert page.is_hidden("#result") and page.is_visible("#go")  # the form page shows a checklist, not the estimate
    page.click("#go"); page.wait_for_selector("#result:not([hidden])")
    assert page.url.endswith("#/result") and page.is_hidden("#assess")
    assert pct(page) == 44 and page.inner_text("#conftitle").startswith("Unsure")
    assert "44% chance of admission" in page.title()
    assert page.get_attribute("nav a[data-p=assess]", "aria-current") == "page"
    assert page.inner_text("#res-who").startswith("Priya Raman, Female, 58. Abdominal pain")
    page.click("#res-edit"); page.wait_for_selector("#assess:not([hidden])")
    page.fill("#age", "90"); show(page)
    assert pct(page) != 44
    page.go_back(); page.wait_for_selector("#assess:not([hidden])")
    assert page.input_value("#age") == "90"


def test_result_page_without_an_estimate_points_to_the_form(browser, base_url):
    ctx, pg = _open(browser, base_url + "#/result")
    assert pg.is_visible("#res-none") and pg.is_hidden("#res-body")
    pg.click("#res-none a"); pg.wait_for_selector("#assess:not([hidden])")
    ctx.close()


def test_next_patient_clears_with_undo(page):
    example(page, "High risk"); show(page)
    page.click("#res-new"); page.wait_for_selector("#assess:not([hidden])")
    assert page.input_value("#age") == "" and page.is_visible("#toast")
    page.click("#undo"); assert page.input_value("#age") == "82"


def test_refresh_keeps_form_and_old_links_still_work(page, base_url):
    example(page, "High risk")
    page.reload(); page.wait_for_selector("#needs[data-state=ready]")
    assert page.input_value("#age") == "82" and page.locator("#ccchips button").count() == 1
    page.goto(base_url + "#/patient"); page.wait_for_selector("#assess:not([hidden])")
    page.goto(base_url + "#/explain"); page.wait_for_selector("#result:not([hidden])")
    assert page.is_visible("#res-body")
    page.goto(base_url + "#/evidence"); page.wait_for_selector("#about:not([hidden])")


def test_clear_offers_undo(page):
    example(page, "Low risk")
    page.click("#clear")
    assert page.input_value("#age") == "" and page.is_visible("#toast") and page.get_attribute("#needs", "data-state") == "empty"
    page.click("#undo")
    assert page.input_value("#age") == "24" and not page.is_visible("#toast")


def test_skip_link_focuses_content_and_keeps_the_page(page):
    page.keyboard.press("Tab")  # first tab stop = skip link
    assert page.evaluate("document.activeElement.className") == "skip"
    page.keyboard.press("Enter")
    assert page.evaluate("document.activeElement.id") == "main"
    assert page.url.endswith("#/assess") and page.is_visible("#assess")


def test_complaint_combobox_keyboard(page):
    page.focus("#ccsearch"); page.keyboard.type("sore")
    assert page.get_attribute("#ccsearch", "aria-expanded") == "true"
    assert "Sore throat" in page.inner_text("#ccopts")  # readable names, not "Sorethroat"
    page.keyboard.press("ArrowDown"); page.keyboard.press("Enter")
    assert page.locator("#ccchips button").count() == 1
    page.locator("#ccchips button").first.click()
    assert page.locator("#ccchips button").count() == 0


def test_random_patient_is_marked_synthetic(page):
    page.select_option("#acuity", "high"); page.click("#random")
    page.wait_for_selector("#needs[data-state=ready]")
    assert page.is_visible("#exnote") and "not a real person" in page.inner_text("#exnote")


def test_controls_labelled_inputs_16px_and_targets(page):
    unlabeled = page.evaluate("""() => [...document.querySelectorAll('input, select')]
        .filter(e => e.type !== 'hidden' && !(e.labels?.length || e.getAttribute('aria-label') || e.getAttribute('aria-labelledby')))
        .map(e => e.id || e.name)""")
    assert unlabeled == [], unlabeled
    small = page.evaluate("""() => [...document.querySelectorAll('input[type=number], input[type=search], input[type=text], select')]
        .filter(e => e.offsetParent && parseFloat(getComputedStyle(e).fontSize) < 16).map(e => e.id)""")
    assert small == [], small
    tiny = page.evaluate("""() => [...document.querySelectorAll('button, a.btn, label.tick, .seg label')]
        .filter(e => e.offsetParent && e.getBoundingClientRect().height < 24).map(e => e.textContent.trim().slice(0, 20))""")
    assert tiny == [], tiny


@pytest.mark.parametrize("width", [390, 768])
def test_no_horizontal_overflow_on_small_screens(browser, base_url, width):
    ctx, pg = _open(browser, base_url + "#/assess", width=width, height=800)
    example(pg, "High risk"); show(pg)
    for route in ("#/assess", "#/result", "#/history", "#/about", "#/home"):
        pg.evaluate(f"location.hash = '{route}'"); pg.wait_for_selector(f"#{route[2:]}:not([hidden])")
        assert pg.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), (width, route)
    ctx.close()


# ---------- home ----------

def test_home_is_default_and_preview_runs_the_real_model(browser, base_url):
    ctx, pg = _open(browser, base_url)
    pg.wait_for_selector("#pv-tabs button")
    assert pg.is_visible("#home") and not pg.is_visible("#assess")
    assert pg.get_attribute("nav a[data-p=home]", "aria-current") == "page"
    pg.locator("#pv-tabs button", has_text="Low").click(); low = pct(pg, "#pv-pct")
    pg.locator("#pv-tabs button", has_text="High").click(); high = pct(pg, "#pv-pct")
    assert low < 20 < 60 < high, (low, high)
    assert pg.locator("#h-levels-rows tr").count() == 3
    assert pg.inner_text("#h-lat").strip()
    pg.click("#pv-open"); pg.wait_for_selector("#result:not([hidden])")  # opens the shown patient's estimate
    assert pct(pg) == high and pg.input_value("#age") == "82"
    ctx.close()


# ---------- about ----------

def test_about_charts_have_tooltip_and_table(page):
    page.click("nav a[data-p=about]"); page.wait_for_selector("#c-cal svg")
    page.evaluate("document.querySelectorAll('details.tech').forEach(d => d.open = true)")
    assert page.locator("#about svg[role=img]").count() == 3
    page.locator("#c-dca").scroll_into_view_if_needed()
    box = page.locator("#c-dca .hit").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] / 2)
    assert page.is_visible("#c-dca .tip")
    page.locator("#c-cal summary").click()
    assert page.locator("#c-cal table tbody tr").count() >= 10


def test_about_toc_link_scrolls_without_leaving_the_page(page):
    page.click("nav a[data-p=about]"); page.wait_for_selector("#about:not([hidden])")
    page.click(".toc a[href='#ab-private']")
    assert page.url.endswith("#/about") and page.is_visible("#about")
    until(lambda: page.evaluate("document.getElementById('ab-private').getBoundingClientRect().top < innerHeight"))


# ---------- patient-record level (needs the local helper) ----------

def test_without_server_record_switch_is_disabled_and_explained(page):
    until(lambda: page.inner_text("#nettext") not in ("", "…"))
    assert page.is_disabled("#usehist") and page.is_visible("#histoff")
    assert page.inner_text("#nettext").startswith("Works offline") and "records" not in page.inner_text("#nettext")


def test_with_server_record_model_runs_on_backend(browser, api_url):
    ctx, pg = _open(browser, api_url + "#/assess", height=900)
    pg.wait_for_selector("#usehist:not([disabled])", state="attached", timeout=30000)
    assert "patient records on" in pg.inner_text("#nettext") and not pg.is_visible("#histoff")
    example(pg, "High risk")
    pg.click("label[for=usehist]")
    pg.fill("#h_n_edvisits", "-1"); pg.click("#go")  # client-side check on record counts
    assert pg.is_visible("#err-h_n_edvisits")
    pg.fill("#h_n_edvisits", "4"); pg.fill("#h_n_admissions", "2")
    pg.locator("#esi label", has_text="2").click()
    pg.click("#go"); pg.wait_for_selector("#result:not([hidden])", timeout=30000)
    until(lambda: pg.inner_text("#mengine") == "This computer", timeout=30)
    assert pg.inner_text("#mtier") == "Triage data, triage level and patient record"
    ctx.close()


# ---------- history ----------

def _hist(pg):
    pg.click("nav a[data-p=history]"); pg.wait_for_selector("#history:not([hidden])")


def test_history_saves_lists_filters_and_reopens(page):
    assert page.is_hidden("#hcount")
    example(page, "High risk"); show(page)  # Check and save stores it
    until(lambda: page.inner_text("#hsave") == "Saved")
    assert page.is_disabled("#hsave") and page.inner_text("#hcount") == "1"
    page.fill("#hnote", "Bay 4"); assert page.inner_text("#hsave") == "Save note"
    page.press("#hnote", "Enter"); until(lambda: page.inner_text("#hsave") == "Saved")
    assert page.inner_text("#hcount") == "1"  # the note updated that entry, no duplicate
    page.click("#res-edit"); page.wait_for_selector("#assess:not([hidden])")
    example(page, "Low risk"); show(page)
    until(lambda: page.inner_text("#hcount") == "2")
    assert page.input_value("#hnote") == ""  # a note never carries over to the next patient

    _hist(page)
    items = page.locator("#hlist .hitem")
    assert items.count() == 2 and "Low" in items.nth(0).inner_text()  # newest first
    assert page.locator("#hlist .hday").first.inner_text() == "Today"
    page.locator("#hist-band label", has_text="High").click()
    assert items.count() == 1 and "Bay 4" in items.first.inner_text()
    page.locator("#hist-band label", has_text="All").click()
    page.fill("#hist-q", "bay 4"); assert items.count() == 1
    page.fill("#hist-q", "nobody"); assert items.count() == 0 and page.is_visible("#hlist .none")
    page.fill("#hist-q", "")

    page.locator("#hlist [data-open]").last.click()  # oldest = the high-risk patient
    page.wait_for_selector("#result:not([hidden])")
    assert page.input_value("#age") == "82" and page.input_value("#hnote") == "Bay 4"
    until(lambda: page.inner_text("#hsave") == "Saved")  # reopened, unchanged: no duplicate save


def test_history_delete_clear_undo_csv_and_persistence(page):
    for name in ("High risk", "Low risk"):
        page.goto(page.url.split("#")[0] + "#/assess"); page.wait_for_selector("#assess:not([hidden])")
        example(page, name); show(page); until(lambda: page.inner_text("#hsave") == "Saved")
    _hist(page)
    items = page.locator("#hlist .hitem")
    page.locator("#hlist [data-del]").first.click()
    assert items.count() == 1 and page.inner_text("#hcount") == "1"
    page.click("#undo"); assert items.count() == 2

    with page.expect_download() as dl:
        page.click("#hist-csv")
    rows = open(dl.value.path(), encoding="utf-8").read().splitlines()
    assert rows[0].startswith('"saved_at"') and len(rows) == 3

    page.reload(); page.wait_for_selector("#hlist .hitem")
    assert items.count() == 2  # stored in the browser, survives a reload

    page.click("#hist-clear")
    assert items.count() == 0 and page.is_visible("#hempty") and page.is_hidden("#hcount")
    page.click("#undo"); assert items.count() == 2


# ---------- patient name and contact ----------

def test_name_and_contact_are_optional_validated_and_saved(page):
    example(page, "High risk")  # sample patients come with an invented name and a 555-01xx number
    assert page.input_value("#pname") == "Margaret Hale" and page.input_value("#pphone") == "555-0142"
    page.fill("#pname", ""); page.fill("#pphone", "")
    page.wait_for_selector("#needs[data-state=ready]")  # neither is required
    page.fill("#pphone", "call me"); page.focus("#age")
    assert page.is_visible("#err-pphone") and page.get_attribute("#needs", "data-state") == "fix"
    page.fill("#pname", "=HYPERLINK(1)"); page.fill("#pphone", "+91 98450 00000"); show(page)
    assert "=HYPERLINK(1), Female, 82" in page.inner_text("#res-who")
    page.click("nav a[data-p=history]"); page.wait_for_selector("#hlist .hitem")
    page.fill("#hist-q", "98450"); assert page.locator("#hlist .hitem").count() == 1
    with page.expect_download() as dl:
        page.click("#hist-csv")
    rows = open(dl.value.path(), encoding="utf-8").read().splitlines()
    assert rows[0].startswith('"saved_at","name","contact"')
    assert '"\'=HYPERLINK(1)","+91 98450 00000"' in rows[1]  # a typed formula is neutralised; a phone number is not


def test_random_patient_gets_an_invented_identity_matching_sex(page):
    page.click("#random"); page.wait_for_selector("#needs[data-state=ready]")
    sex = page.evaluate("document.querySelector('input[name=gender]:checked').value")
    name, phone = page.input_value("#pname"), page.input_value("#pphone")
    assert name and phone.startswith("555-01")
    first = name.split()[0]
    assert first in (["Anna", "Rosa", "Leela", "Grace", "Nadia", "Helen", "Mei", "Sofia", "Joan", "Amara"] if sex == "Female"
                     else ["Tom", "Arjun", "Samuel", "Luis", "Omar", "Peter", "Kenji", "Victor", "Hugo", "Ben"])
