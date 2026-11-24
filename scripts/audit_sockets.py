"""Run the Streamlit app with a Python audit hook that logs every outbound non-loopback
socket connection (with a stack trace). Used to verify the app makes no network calls.

Usage: python scripts/audit_sockets.py [port]   (log goes to stderr)
"""
import ipaddress
import sys
import traceback


def hook(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        try:
            addr = args[1]
            host = addr[0] if isinstance(addr, tuple) else str(addr)
            if event == "socket.getaddrinfo":
                host = args[0]
            if isinstance(host, bytes):
                host = host.decode()
            try:
                if ipaddress.ip_address(host).is_loopback:
                    return
            except ValueError:
                if host in ("localhost", None, ""):
                    return
            print(f"[AUDIT] {event} -> {args[:2]}", file=sys.stderr, flush=True)
            traceback.print_stack(limit=14, file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass


sys.addaudithook(hook)
port = sys.argv[1] if len(sys.argv) > 1 else "8598"
sys.argv = ["streamlit", "run", "app/streamlit_app.py", "--server.port", port]
from streamlit.web.cli import main  # noqa: E402

main()
