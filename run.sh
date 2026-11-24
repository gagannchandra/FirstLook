#!/usr/bin/env bash
# FirstLook — start / stop / check the three apps, plus the everyday developer tasks.
#
#   ./run.sh                   start everything (Streamlit app, offline web app, Docker if available)
#   ./run.sh start   [what]    what = all (default) | app | web | docker
#   ./run.sh stop    [what]    stop everything, or just one app
#   ./run.sh restart [what]    stop then start (use after pulling or editing backend code)
#   ./run.sh status            show what is running, and whether patient records are available in the web app
#   ./run.sh logs    [what]    follow a log: web (default) | app | docker
#   ./run.sh build             (re)build the Docker image
#   ./run.sh test              run the test suite (browser tests need Google Chrome)
#   ./run.sh export            re-export model bundles + research results as JSON for the web app (frontend/data/)
#   ./run.sh screenshots       offline proof + screenshots of the web app into results/figures/app/ (web app must run)
#
# URLs:  Streamlit app  http://localhost:8599   (all models + research dashboard)
#        Offline web app http://localhost:8600  (frontend/ + backend/ API; works with Wi-Fi off after the first load)
#        Docker          http://localhost:8501  (same app as 8599, limited to 2 CPUs / 2 GB)
#
# PIDs and logs go to .run/ (git-ignored). Override ports with APP_PORT / WEB_PORT / DOCKER_PORT.
# Needs: Linux (uses ss), the .venv from the README, curl; Docker/podman is optional.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$ROOT/.run"
APP_PORT="${APP_PORT:-8599}"
WEB_PORT="${WEB_PORT:-8600}"
DOCKER_PORT="${DOCKER_PORT:-8501}"
CONTAINER="firstlook_ui"
IMAGE="firstlook"
PY="$ROOT/.venv/bin/python"
mkdir -p "$RUN"

say()  { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
alive() { [[ -f "$RUN/$1.pid" ]] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
# PID of the process listening on a TCP port (Linux: ss). Recorded after start, so the PID file always
# names the real server, whatever wrapper launched it.
port_pid()  { ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true; }
record()    { local pid; pid="$(port_pid "$2")"; [[ -n "$pid" ]] && echo "$pid" >"$RUN/$1.pid"; }

wait_http() {  # wait_http URL EXPECT seconds
  local i
  for ((i = 0; i < $3; i++)); do
    if [[ "$(curl -s -m 2 "$1" 2>/dev/null || true)" == *"$2"* ]]; then return 0; fi
    sleep 1
  done
  return 1
}

start_app() {
  if alive app; then say "• Streamlit app already running → http://localhost:$APP_PORT"; return; fi
  [[ -x "$PY" ]] || { say "✗ No virtualenv at .venv — create it first (see README: 'uv venv' + 'uv pip install -r requirements.txt')."; return 1; }
  port_busy "$APP_PORT" && { say "✗ Port $APP_PORT is in use — stop that program or run: APP_PORT=8598 ./run.sh"; return 1; }
  (cd "$ROOT" && ORT_DISABLE_TELEMETRY=1 setsid nohup "$PY" -m streamlit run app/streamlit_app.py \
      --server.port "$APP_PORT" --server.address localhost --server.headless true \
      >"$RUN/app.log" 2>&1 </dev/null &)
  if wait_http "http://localhost:$APP_PORT/_stcore/health" ok 60; then
    record app "$APP_PORT"
    say "✓ Streamlit app      http://localhost:$APP_PORT"
  else
    say "✗ Streamlit app did not start — see .run/app.log"; return 1
  fi
}

start_web() {
  if alive web; then say "• Offline web app already running → http://localhost:$WEB_PORT"; return; fi
  port_busy "$WEB_PORT" && { say "✗ Port $WEB_PORT is in use — run: WEB_PORT=8601 ./run.sh"; return 1; }
  # frontend/ + local backend API (backend/server.py). Without the venv it still serves the browser models (T1/T2).
  local py="$PY" mode=()
  [[ -x "$py" ]] || { py=python3; mode=(--static-only); say "• No .venv — web app runs browser models only (no history model)"; }
  (cd "$ROOT" && ORT_DISABLE_TELEMETRY=1 setsid nohup "$py" -m backend.server --port "$WEB_PORT" "${mode[@]}" \
      >"$RUN/web.log" 2>&1 </dev/null &)
  if wait_http "http://localhost:$WEB_PORT/index.html" FirstLook 20; then
    record web "$WEB_PORT"
    say "✓ Offline web app    http://localhost:$WEB_PORT   (frontend + local API)"
    say "  Tip: after an update, refresh the page once — the browser keeps an offline copy of the previous version."
  else
    say "✗ Web app did not start — see .run/web.log"; return 1
  fi
}

start_docker() {
  if ! have docker; then say "• Docker not installed — skipped"; return; fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    say "• Docker image '$IMAGE' not built — skipped (build it with: ./run.sh build)"; return
  fi
  if [[ -n "$(docker ps -q --filter "name=^${CONTAINER}$" 2>/dev/null)" ]]; then
    say "• Docker app already running → http://localhost:$DOCKER_PORT"; return
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" -p "$DOCKER_PORT:8501" --cpus 2 --memory 2g "$IMAGE" >/dev/null 2>&1
  if wait_http "http://localhost:$DOCKER_PORT/_stcore/health" ok 60; then
    say "✓ Docker app         http://localhost:$DOCKER_PORT"
  else
    say "✗ Docker app did not answer — see: docker logs $CONTAINER"; return 1
  fi
}

stop_pid() {  # stop_pid NAME PORT LABEL PATTERN
  local name="$1" port="$2" label="$3" pattern="$4" pid=""
  alive "$name" && pid="$(cat "$RUN/$name.pid")"
  if [[ -z "$pid" ]]; then  # no/stale PID file: stop the port's listener, but only if it is our server
    local p; p="$(port_pid "$port")"
    if [[ -n "$p" ]] && grep -q -- "$pattern" "/proc/$p/cmdline" 2>/dev/null; then pid="$p"; fi
  fi
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
    say "✓ Stopped $label"
  fi
  rm -f "$RUN/$name.pid"
}

stop_app()    { stop_pid app "$APP_PORT" "Streamlit app" "streamlit_app.py"; }
stop_web()    { stop_pid web "$WEB_PORT" "offline web app" "backend.server"; }
stop_docker() {
  if have docker && [[ -n "$(docker ps -aq --filter "name=^${CONTAINER}$" 2>/dev/null)" ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && say "✓ Stopped Docker app"
  fi
}

stop_what() {  # stop_what all|app|web|docker
  case "$1" in
    all)    stop_app; stop_web; stop_docker; say "All stopped." ;;
    app)    stop_app ;;
    web)    stop_web ;;
    docker) stop_docker ;;
    *)      say "Unknown app '$1'. Use: all | app | web | docker"; exit 2 ;;
  esac
}

start_what() {  # start_what all|app|web|docker
  case "$1" in
    all)    start_app || true; start_web || true; start_docker || true
            say ""; say "Stop everything with: ./run.sh stop" ;;
    app)    start_app ;;
    web)    start_web ;;
    docker) start_docker ;;
    *)      say "Unknown app '$1'. Use: all | app | web | docker"; exit 2 ;;
  esac
}

need_venv() { [[ -x "$PY" ]] || { say "✗ No virtualenv at .venv — see README (uv venv + uv pip install -r requirements.txt)."; exit 1; }; }

status() {
  local s
  if alive app; then s="running → http://localhost:$APP_PORT"; else s="stopped"; fi; say "Streamlit app    $s"
  if alive web; then
    s="running → http://localhost:$WEB_PORT"
    if [[ "$(curl -s -m 2 "http://localhost:$WEB_PORT/api/health" 2>/dev/null || true)" == *t3lite* ]]; then
      s="$s   (patient records available)"
    else
      s="$s   (browser models only)"
    fi
  else s="stopped"; fi
  say "Offline web app  $s"
  if have docker && [[ -n "$(docker ps -q --filter "name=^${CONTAINER}$" 2>/dev/null)" ]]; then
    say "Docker app       running → http://localhost:$DOCKER_PORT"
  else
    say "Docker app       stopped"
  fi
}

case "${1:-start}" in
  start)   start_what "${2:-all}" ;;
  stop)    stop_what "${2:-all}" ;;
  restart) stop_what "${2:-all}"; start_what "${2:-all}" ;;
  status)  status ;;
  logs)    case "${2:-web}" in
             app|web) [[ -f "$RUN/${2:-web}.log" ]] || { say "No log yet: start it first."; exit 1; }
                      tail -n 50 -f "$RUN/${2:-web}.log" ;;
             docker)  docker logs -f "$CONTAINER" ;;
             *)       say "Unknown app '$2'. Use: app | web | docker"; exit 2 ;;
           esac ;;
  build)   have docker || { say "Docker not installed."; exit 1; }
           (cd "$ROOT" && docker build -t "$IMAGE" -f docker/Dockerfile .) ;;
  test)    need_venv; (cd "$ROOT" && ORT_DISABLE_TELEMETRY=1 "$PY" -m pytest -q "${@:2}") ;;
  export)  need_venv; (cd "$ROOT" && "$PY" -m backend.export) ;;
  screenshots)
           need_venv
           alive web || { say "✗ Start the web app first: ./run.sh start web"; exit 1; }
           (cd "$ROOT" && "$PY" scripts/screenshot_pwa.py --url "http://localhost:$WEB_PORT/index.html") ;;
  -h|--help|help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) say "Unknown command '$1'. Try: ./run.sh help"; exit 2 ;;
esac
