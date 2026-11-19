#!/usr/bin/env bash
# User-owned, loopback-only service. Never writes into input or archived runs.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OPC_ROOT="${SPACEQA_OPC_ROOT:-$(dirname -- "$REPO_DIR")}"
STATE_DIR="${SPACEQA_WEB_STATE:-$OPC_ROOT/web-service}"
PORT="${SPACEQA_WEB_PORT:-8765}"
PYTHON_BIN="${SPACEQA_PYTHON:-/usr/bin/python3}"
PID_FILE="$STATE_DIR/service.pid"
LOG_FILE="$STATE_DIR/service.log"
mkdir -p -- "$STATE_DIR"

managed_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  IFS= read -r SERVICE_PID < "$PID_FILE"
  [[ "$SERVICE_PID" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$SERVICE_PID" 2>/dev/null || return 1
  [[ -r "/proc/$SERVICE_PID/cmdline" ]] || return 1
  local arguments
  arguments="$(tr '\0' ' ' < "/proc/$SERVICE_PID/cmdline")"
  [[ "$arguments" == *"spaceqa_agent serve"* && "$arguments" == *"$STATE_DIR"* ]]
}

ACTION="${1:-start}"
case "$ACTION" in
  stop)
    if managed_pid; then
      kill -TERM "$SERVICE_PID"
      echo "Stop requested for SpaceQA PID $SERVICE_PID"
    else
      echo "No managed SpaceQA service is running"
    fi
    exit 0
    ;;
  status)
    if managed_pid; then
      echo "SpaceQA is running: PID=$SERVICE_PID URL=http://127.0.0.1:$PORT"
      "$PYTHON_BIN" -c 'import urllib.request,sys; print(urllib.request.urlopen(sys.argv[1],timeout=10).read().decode())' "http://127.0.0.1:$PORT/api/health"
    else
      echo "SpaceQA is not running"
      exit 1
    fi
    exit 0
    ;;
  start) ;;
  *) echo "Usage: $0 [start|status|stop]" >&2; exit 2 ;;
esac

if managed_pid; then
  echo "SpaceQA is already running: PID=$SERVICE_PID"
  exit 0
fi

ARGS=(--host 127.0.0.1 --port "$PORT" --state-dir "$STATE_DIR")
for DATA_ROOT in \
  "$OPC_ROOT/opc1018/newpublish" \
  "$OPC_ROOT/opc-agent/runs/roadshow-evaluation-20261101/extracted"; do
  [[ ! -d "$DATA_ROOT" ]] || ARGS+=(--data-root "$DATA_ROOT")
done
for RUN_ROOT in \
  "$OPC_ROOT/inference_prior_20261102/integrated_final_v1" \
  "$OPC_ROOT/inference_prior_20261102/integrated_review_v1"; do
  [[ ! -d "$RUN_ROOT" ]] || ARGS+=(--run-root "$RUN_ROOT")
done

nohup env PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" -m spaceqa_agent serve "${ARGS[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
SERVICE_PID=$!
printf '%s\n' "$SERVICE_PID" > "$PID_FILE"
echo "Starting SpaceQA: PID=$SERVICE_PID URL=http://127.0.0.1:$PORT"
echo "Server log: $LOG_FILE"
