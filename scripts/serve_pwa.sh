#!/usr/bin/env bash
# Serve the L3 PWA on localhost (service workers need http://localhost or https). After the first load
# the app works with the network off. Usage: scripts/serve_pwa.sh [port]
cd "$(dirname "$0")/../app/pwa" && exec python3 -m http.server "${1:-8600}" --bind 127.0.0.1
