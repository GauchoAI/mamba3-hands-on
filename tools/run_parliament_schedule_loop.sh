#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_S="${PARLIAMENT_INTERVAL_S:-300}"
BACKEND="${PARLIAMENT_BACKEND:-auto}"
PANEL_SIZE="${PARLIAMENT_PANEL_SIZE:-2}"
TIMEOUT_S="${PARLIAMENT_TIMEOUT_S:-120}"
WALL_TIMEOUT_S="${PARLIAMENT_WALL_TIMEOUT_S:-270}"
PYTHON_BIN="${PARLIAMENT_PYTHON:-/opt/homebrew/bin/python3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p runs/parliament/scheduler

echo "[parliament-loop] start interval=${INTERVAL_S}s backend=${BACKEND} panel=${PANEL_SIZE}" >&2

while true; do
  date -u +"[parliament-loop] tick %Y-%m-%dT%H:%M:%SZ" >&2
  "$PYTHON_BIN" tools/parliament_tick.py \
    --backend "$BACKEND" \
    --panel-size "$PANEL_SIZE" \
    --timeout-s "$TIMEOUT_S" \
    --wall-timeout-s "$WALL_TIMEOUT_S" \
    || true
  sleep "$INTERVAL_S"
done
