#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_S="${PARLIAMENT_INTERVAL_S:-300}"
BACKEND="${PARLIAMENT_BACKEND:-auto}"
PANEL_SIZE="${PARLIAMENT_PANEL_SIZE:-2}"
TIMEOUT_S="${PARLIAMENT_TIMEOUT_S:-120}"
WALL_TIMEOUT_S="${PARLIAMENT_WALL_TIMEOUT_S:-270}"
PYTHON_BIN="${PARLIAMENT_PYTHON:-/opt/homebrew/bin/python3}"
PERSIST="${PARLIAMENT_PERSIST:-0}"
ARCHIVE="${PARLIAMENT_ARCHIVE:-0}"
EXECUTE_ACTIONS="${PARLIAMENT_EXECUTE_ACTIONS:-0}"
ACTION_TIMEOUT_S="${PARLIAMENT_ACTION_TIMEOUT_S:-420}"
WATCHDOG="${PARLIAMENT_WATCHDOG:-0}"
WATCHDOG_BACKEND="${PARLIAMENT_WATCHDOG_BACKEND:-auto}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p runs/parliament/scheduler

EXTRA_ARGS=()
if [[ "$PERSIST" == "1" ]]; then
  EXTRA_ARGS+=(--persist)
fi
if [[ "$ARCHIVE" == "1" ]]; then
  EXTRA_ARGS+=(--archive)
fi
if [[ "$EXECUTE_ACTIONS" == "1" ]]; then
  EXTRA_ARGS+=(--execute-actions)
fi
if [[ "$WATCHDOG" == "1" ]]; then
  EXTRA_ARGS+=(--watchdog --watchdog-backend "$WATCHDOG_BACKEND")
fi

echo "[parliament-loop] start interval=${INTERVAL_S}s backend=${BACKEND} panel=${PANEL_SIZE} persist=${PERSIST} archive=${ARCHIVE} execute_actions=${EXECUTE_ACTIONS} watchdog=${WATCHDOG}" >&2

while true; do
  date -u +"[parliament-loop] tick %Y-%m-%dT%H:%M:%SZ" >&2
  "$PYTHON_BIN" tools/parliament_tick.py \
    --backend "$BACKEND" \
    --panel-size "$PANEL_SIZE" \
    --timeout-s "$TIMEOUT_S" \
    --wall-timeout-s "$WALL_TIMEOUT_S" \
    --action-timeout-s "$ACTION_TIMEOUT_S" \
    "${EXTRA_ARGS[@]}" \
    || true
  sleep "$INTERVAL_S"
done
