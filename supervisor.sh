#!/bin/bash
# Supervisor: keeps coordinator, renderer, and HTTP server alive.
# One script to start. Everything auto-restarts on crash.
#
# Usage: nohup bash supervisor.sh > supervisor.log 2>&1 &
#
# To stop everything: kill the supervisor PID (it cleans up children)

cd /root/mamba3-hands-on
git pull

echo "[$(date)] Supervisor starting..."
echo "[$(date)] PID: $$"

# Cleanup on exit
cleanup() {
    echo "[$(date)] Supervisor stopping..."
    kill $COORD_PID $RENDER_PID $HTTP_PID 2>/dev/null
    wait
    echo "[$(date)] All services stopped."
}
trap cleanup EXIT INT TERM

# ── Start services ──

start_coordinator() {
    .venv/bin/python -u coordinator.py \
        --generation-every 60 \
        --evolve-every 3 \
        --runs-dir runs \
        >> coordinator.log 2>&1 &
    COORD_PID=$!
    echo "[$(date)] Coordinator started (PID $COORD_PID)"
}

start_renderer() {
    .venv/bin/python -u render.py \
        --db metrics.db \
        --watch \
        --interval 15 \
        >> render.log 2>&1 &
    RENDER_PID=$!
    echo "[$(date)] Renderer started (PID $RENDER_PID)"
}

start_http() {
    .venv/bin/python serve.py >> /dev/null 2>&1 &
    HTTP_PID=$!
    echo "[$(date)] HTTP server started (PID $HTTP_PID, port 9090)"
}

# Initial start
mkdir -p runs logs
start_coordinator
start_renderer
start_http

echo "[$(date)] All services running."
echo "[$(date)] Coordinator: $COORD_PID"
echo "[$(date)] Renderer:    $RENDER_PID"
echo "[$(date)] HTTP:        $HTTP_PID"

# ── Monitor loop: restart anything that dies ──

while true; do
    sleep 10

    # Check coordinator
    if ! kill -0 $COORD_PID 2>/dev/null; then
        echo "[$(date)] ⚠ Coordinator died! Restarting..."
        start_coordinator
    fi

    # Check renderer
    if ! kill -0 $RENDER_PID 2>/dev/null; then
        echo "[$(date)] ⚠ Renderer died! Restarting..."
        start_renderer
    fi

    # Check HTTP server
    if ! kill -0 $HTTP_PID 2>/dev/null; then
        echo "[$(date)] ⚠ HTTP server died! Restarting..."
        start_http
    fi
done
