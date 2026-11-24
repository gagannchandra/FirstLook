"""Screenshot the offline web app (frontend/) and prove it works offline: load once (the service worker saves the app),
then cut the browser's network, reload and use every page. Any request that leaves localhost is recorded.

Usage: python scripts/screenshot_pwa.py [--url http://localhost:8600/index.html] [--out results/figures/app]
Writes pwa_01_home … pwa_08_result_mobile .png and pwa_offline_check.json to --out.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8600/index.html")
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "app"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = a.url.split("#")[0]
    ext, offline_failed, errors = [], [], []
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        # bypass_csp lets this harness use wait_for_function; the app itself never needs it
        ctx = br.new_context(viewport={"width": 1366, "height": 860}, device_scale_factor=1.5, bypass_csp=True,
                             reduced_motion="reduce")
        page = ctx.new_page()
        page.on("request", lambda r: ext.append(r.url) if urlparse(r.url).hostname not in ("localhost", "127.0.0.1") else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "#/about")
        page.wait_for_function("document.getElementById('swstate').textContent.startsWith('yes')", timeout=20000)

        ctx.set_offline(True)  # from here on, everything must come from the service-worker cache
        page.on("requestfailed", lambda r: offline_failed.append(r.url) if "/api/" not in r.url else None)
        page.reload(); page.wait_for_selector("#examples button", state="attached")
        def shot(name):  # grow the viewport to the page so sticky header/panel sit where a reader expects them
            page.evaluate("document.getElementById('toast').hidden = true; document.querySelector('main').style.minHeight = '0'")
            w = page.viewport_size["width"]
            h = page.evaluate("document.documentElement.scrollHeight")
            page.set_viewport_size({"width": w, "height": h}); page.wait_for_timeout(150)
            page.screenshot(path=out / f"pwa_{name}.png")
            page.set_viewport_size({"width": w, "height": 860 if w > 600 else 844}); page.evaluate("scrollTo(0, 0)")

        page.evaluate("location.hash = '#/home'"); page.wait_for_selector("#pv-tabs button")
        page.locator("#pv-tabs button", has_text="High").click()
        shot("01_home")

        def assess():
            page.evaluate("location.hash = '#/assess'"); page.wait_for_selector("#assess:not([hidden])")

        def check_and_save():
            page.click("#go"); page.wait_for_selector("#result:not([hidden])")
            page.wait_for_function("document.getElementById('hsave').textContent === 'Saved'")

        assess(); shot("02_assess_empty")
        page.locator("#examples button", has_text="Uncertain").click(); page.wait_for_selector("#needs[data-state=ready]")
        shot("03_assess_ready")
        check_and_save()
        page.fill("#hnote", "Bay 3"); page.click("#hsave")
        page.wait_for_function("document.getElementById('hsave').textContent === 'Saved'")
        page.evaluate("document.querySelector('details.entered').open = true")
        res = {"uncertain_pct": page.inner_text("#pct"), "uncertain_confidence": page.inner_text("#conftitle")}
        shot("04_result_uncertain")

        assess(); page.select_option("#acuity", "high"); page.click("#random"); page.wait_for_selector("#needs[data-state=ready]")
        check_and_save(); res["random_high_pct"] = page.inner_text("#pct")
        shot("05_result_random_high")

        assess(); page.locator("#examples button", has_text="Low risk").click(); page.wait_for_selector("#needs[data-state=ready]")
        check_and_save()
        page.click("nav a[data-p=history]"); page.wait_for_selector("#hlist .hitem")
        res["history_entries"] = page.locator("#hlist .hitem").count()
        shot("06_history")

        page.click("nav a[data-p=about]"); page.wait_for_selector("#c-cal svg")
        page.evaluate("document.querySelectorAll('details.tech').forEach(d => d.open = true)")
        shot("07_about")

        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("location.hash = '#/history'"); page.wait_for_selector("#hlist .hitem")
        page.locator("#hlist [data-open]").first.click(); page.wait_for_selector("#result:not([hidden])")
        shot("08_result_mobile")

        res.update({"network_pill": page.inner_text("#nettext"), "offline_failed_requests": offline_failed,
                    "external_requests": ext, "page_errors": errors})
        br.close()
    (out / "pwa_offline_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
