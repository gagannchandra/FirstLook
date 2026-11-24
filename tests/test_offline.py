"""Regression test: inference must not open any non-loopback connection.

ONNX Runtime 1.32 opened an HTTPS connection to a telemetry endpoint ~9 s after creating a
session (DECISIONS D28). This runs the real runtime path in a subprocess and watches its
sockets for 20 s.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import sys, time, os, ipaddress, psutil
from src.edge.runtime import load_all
from app.examples import EXAMPLES
B = load_all()
found = set(); t0 = time.time()
while time.time() - t0 < 20:
    for b in B.values():
        for v in EXAMPLES.values():
            b.predict(v["form"])
    for c in psutil.Process(os.getpid()).net_connections(kind="inet"):
        if c.raddr and not ipaddress.ip_address(c.raddr.ip).is_loopback:
            found.add((c.raddr.ip, c.raddr.port))
    time.sleep(0.2)
print("EXTERNAL", sorted(found))
'''


@pytest.mark.skipif(not (ROOT / "models" / "bundles").exists(), reason="no bundles")
def test_inference_opens_no_external_connections():
    env = {k: v for k, v in __import__("os").environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    out = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    line = [l for l in out.stdout.splitlines() if l.startswith("EXTERNAL")]
    assert line, out.stderr[-2000:]
    assert line[0] == "EXTERNAL []", line[0]
