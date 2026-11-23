#!/usr/bin/env bash
# Start (or restart) the Streamlit app in the background on PORT (default 8599).
PORT="${1:-8599}"
cd "$(dirname "$0")/.."
for p in $(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill "$p"; done
sleep 1
nohup .venv/bin/streamlit run app/streamlit_app.py --server.port "$PORT" > "${TMPDIR:-/tmp}/firstlook_streamlit.log" 2>&1 &
for i in $(seq 1 30); do curl -s -o /dev/null "http://localhost:$PORT/" && break; sleep 0.5; done
echo "FirstLook on http://localhost:$PORT"
