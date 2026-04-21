#!/bin/bash
# Local dashboard: pull metrics.db from H100, render, serve.
# No changes to remote. Pure read-only.
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=9090

echo "Local dashboard at http://localhost:$PORT"
echo "Pulling metrics.db from H100 every 30s, rendering locally."

# Start HTTP server
cd "$DIR"
python3 -m http.server $PORT &
SERVER_PID=$!

while true; do
    # Pull SQLite from H100
    scp -q -P 32783 root@ssh2.vast.ai:/root/mamba3-hands-on/metrics.db "$DIR/metrics.db" 2>/dev/null

    # Render if db exists
    if [ -f "$DIR/metrics.db" ]; then
        .venv/bin/python render.py --db "$DIR/metrics.db" 2>/dev/null
    fi

    sleep 30
done
