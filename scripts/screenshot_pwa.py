"""Screenshot the L3 PWA and prove it works offline: load once (service worker installs), then switch the
browser context offline, reload and run a prediction. Usage: python scripts/screenshot_pwa.py [--url ...]"""
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
    ext, offline_reqs = [], []
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        ctx = br.new_context(viewport={"width": 1366, "height": 768}, device_scale_factor=1.5, bypass_csp=True)  # test harness only
        page = ctx.new_page()
        page.on("request", lambda r: ext.append(r.url) if urlparse(r.url).hostname not in ("localhost", "127.0.0.1") else None)
        page.goto(a.url)
        page.wait_for_function("document.getElementById('swstate').textContent.startsWith('installed')", timeout=20000)
        ctx.set_offline(True)  # network cut: everything below must come from the service-worker cache
        page.on("requestfailed", lambda r: offline_reqs.append(r.url))
        page.reload(); page.wait_for_selector("#examples button")
        page.get_by_role("button", name="Uncertain").click()
        page.screenshot(path=out / "pwa_01_patient_offline.png", full_page=True)
        page.click("#go"); page.wait_for_timeout(300)
        page.screenshot(path=out / "pwa_02_result_offline.png", full_page=True)
        page.click("#why"); page.screenshot(path=out / "pwa_03_explanation_offline.png", full_page=True)
        page.click("nav a[data-p=about]"); page.screenshot(path=out / "pwa_04_about_offline.png", full_page=True)
        page.click("nav a[data-p=patient]")
        page.click("#acuity button[data-a=high]"); page.click("#random")
        page.screenshot(path=out / "pwa_05_random_offline.png", full_page=True)
        page.click("#go"); page.wait_for_timeout(300)
        random_pct = page.inner_text("#pct").replace("\n", "")
        page.screenshot(path=out / "pwa_06_random_result_offline.png", full_page=True)
        res = {"random_high_pct": random_pct, "pct": None, "offline_failed_requests": offline_reqs, "external_requests": ext,
               "network_pill": page.inner_text("#nettext")}
        page.click("nav a[data-p=patient]"); page.get_by_role("button", name="Uncertain").click(); page.click("#go")
        res["pct"] = page.inner_text("#pct").replace("\n", ""); res["conformal"] = page.inner_text("#conftitle")
        br.close()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
