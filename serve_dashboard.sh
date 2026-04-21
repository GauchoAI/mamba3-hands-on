#!/bin/bash
# Poll H100 for dashboard files and serve locally
# Run this on the mini/air: bash serve_dashboard.sh
DIR="$(cd "$(dirname "$0")" && pwd)/dashboard_local"
mkdir -p "$DIR"
PORT=9090

echo "Dashboard server starting on http://localhost:$PORT"
echo "Polling H100 every 30 seconds for updates..."

# Start local HTTP server in background
cd "$DIR"
python3 -m http.server $PORT &
SERVER_PID=$!
echo "HTTP server PID: $SERVER_PID"

# Poll loop
while true; do
    scp -q -P 32783 root@ssh2.vast.ai:/root/mamba3-hands-on/dashboard.html "$DIR/index.html" 2>/dev/null
    scp -q -P 32783 root@ssh2.vast.ai:/root/mamba3-hands-on/dashboard.md "$DIR/dashboard.md" 2>/dev/null
    sleep 30
done
