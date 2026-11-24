#!/usr/bin/env bash
# Serve the offline web app (frontend/) with the local backend API on localhost (service workers need
# http://localhost or https). After the first load the app works with the network off.
# Usage: scripts/serve_pwa.sh [port] [--static-only]
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; [[ -x $PY ]] || PY=python3
exec "$PY" -m backend.server --port "${1:-8600}" "${@:2}"
