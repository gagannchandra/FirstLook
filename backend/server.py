"""FirstLook local backend: serves the offline web app (frontend/) and a small JSON API on this machine.

Usage:  python -m backend.server [--port 8600] [--host 127.0.0.1] [--static-only]

Endpoints (same origin as the web app, so the page's CSP `connect-src 'self'` still holds):
  GET  /api/health    {"ok": true, "version": "v1", "models": {"t1": "...", "t2": "...", "t3lite": "..."}}
  POST /api/predict   body {"form": {...}, "slot": "t1"|"t2"|"t3lite" (optional)} -> estimate + explanation

The frontend works without this server: it runs the T1/T2 models in the browser. When the server is reachable it
also offers the EHR-history model (T3-lite, a tree ensemble run with ONNX Runtime). Nothing here talks to the
internet: it binds to localhost by default, and ONNX Runtime telemetry is disabled (DECISIONS D28).
Standard library only (plus the inference runtime's own dependencies), so it adds nothing to requirements.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

MAX_BODY = 64 * 1024
SLOTS = ("t1", "t2", "t3lite")

_lock = threading.Lock()
_bundles: dict | None = None


def bundles() -> dict:
    """Load the model bundles once, on first use (ONNX Runtime import takes ~1 s)."""
    global _bundles
    with _lock:
        if _bundles is None:
            from src.edge.runtime import load_all
            _bundles = load_all()
    return _bundles


def choose_slot(form: dict, requested: str | None, available: dict) -> str:
    if requested:
        if requested not in available:
            raise ValueError(f"unknown model '{requested}'")
        return requested
    if any(k.startswith(("n_", "meds_")) for k in form) and "t3lite" in available:
        return "t3lite"
    return "t2" if form.get("esi") and "t2" in available else "t1"


def clean_form(form: dict) -> dict:
    """JSON null -> None for vitals/labs; history numbers to float; NaN is never accepted from the wire."""
    out = dict(form)
    for k, v in form.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
    out["complaints"] = [c for c in form.get("complaints") or [] if isinstance(c, str)][:3]
    return out


def predict(payload: dict) -> tuple[int, dict]:
    from app.validation import validate_fields

    form = payload.get("form")
    if not isinstance(form, dict):
        return 400, {"error": "Body must be JSON with a 'form' object."}
    form = clean_form(form)
    errors = [d for d in validate_fields(form) if d["level"] == "error"]
    if errors:
        return 422, {"error": "Some fields need attention.", "fields": errors}
    B = bundles()
    try:
        slot = choose_slot(form, payload.get("slot"), B)
    except ValueError as e:
        return 400, {"error": str(e)}
    b = B[slot]
    if slot == "t3lite":  # lab values left blank = no prior lab (NaN), never an invented value
        for h in b.spec["history_inputs"]:
            if form.get(h) is None and b.spec.get("history_input_kind", {}).get(h) == "lab":
                form[h] = float("nan")
    r = b.predict(form)
    return 200, {
        "slot": slot, "name": b.spec["slot_name"], "version": r.model_version, "probability": r.probability,
        "band": r.band, "conformal": r.conformal, "latency_ms": r.latency_ms, "features": r.feature_count,
        "thresholds": {"t_low": b.thr["t_low"], "t_high": b.thr["t_high"]},
        "contributions": [{"label": d["factor"], "value": d["contribution"]} for d in r.contributions],
        "explain_kind": "shap" if slot == "t3lite" else "linear",
    }


class Handler(SimpleHTTPRequestHandler):
    static_only = False
    server_version = "FirstLook"
    sys_version = ""

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(FRONTEND), **k)

    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".webmanifest": "application/manifest+json",
                      ".js": "text/javascript", ".json": "application/json", ".svg": "image/svg+xml"}

    def log_message(self, fmt, *args):  # quiet by default; set FIRSTLOOK_LOG=1 to see requests
        if os.environ.get("FIRSTLOOK_LOG"):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.path.endswith(("sw.js", ".html", "/")) or self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/"):
            if self.static_only:
                return self._json(404, {"error": "API disabled (--static-only)"})
            if self.path.split("?")[0] == "/api/health":
                B = bundles()
                return self._json(200, {"ok": True, "version": next(iter(B.values())).version if B else None,
                                        "models": {s: B[s].spec["slot_name"] for s in SLOTS if s in B}})
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_POST(self):
        if self.static_only or self.path.split("?")[0] != "/api/predict":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._json(413 if n > MAX_BODY else 400, {"error": "Request body missing or too large."})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(415, {"error": "Send JSON (Content-Type: application/json)."})
        try:
            payload = json.loads(self.rfile.read(n), parse_constant=lambda c: None)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"error": "Body is not valid JSON."})
        if not isinstance(payload, dict):
            return self._json(400, {"error": "Body must be a JSON object."})
        try:
            status, body = predict(payload)
        except Exception as e:  # noqa: BLE001 - report, never crash the server
            status, body = 500, {"error": f"Prediction failed: {type(e).__name__}"}
        return self._json(status, body)

    def do_PUT(self):
        self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"})

    do_DELETE = do_PATCH = do_PUT


def make_server(host: str = "127.0.0.1", port: int = 8600, static_only: bool = False) -> ThreadingHTTPServer:
    handler = type("H", (Handler,), {"static_only": static_only})
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--static-only", action="store_true", help="serve the web app only (browser models, no API)")
    a = ap.parse_args()
    srv = make_server(a.host, a.port, a.static_only)
    if not a.static_only:
        threading.Thread(target=bundles, daemon=True).start()  # warm up while the browser loads
    print(f"FirstLook web app → http://localhost:{a.port}/  ({'static only' if a.static_only else 'API on /api/'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
