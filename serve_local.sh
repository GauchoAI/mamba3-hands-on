#!/bin/bash
# Local dashboard: safely pull metrics.db from H100, render, serve.
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=9090

echo "Dashboard at http://localhost:$PORT"

cd "$DIR"
python3 -m http.server $PORT &

while true; do
    # Safe copy: checkpoint WAL first, then copy snapshot
    ssh -p 32783 root@ssh2.vast.ai \
        'cd /root/mamba3-hands-on && sqlite3 metrics.db "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null; cp metrics.db /tmp/metrics_snap.db' 2>/dev/null
    scp -q -P 32783 root@ssh2.vast.ai:/tmp/metrics_snap.db "$DIR/metrics.db" 2>/dev/null

    if [ -f "$DIR/metrics.db" ]; then
        .venv/bin/python render.py --db "$DIR/metrics.db" 2>/dev/null
    fi

    sleep 30
done
