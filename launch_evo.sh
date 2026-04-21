#!/bin/bash
# Launch evolutionary coordinator with auto-restart watchdog.
# Workers survive coordinator restarts.
cd /root/mamba3-hands-on
git pull

mkdir -p runs logs

echo "Starting evolutionary training with watchdog..."
echo "Workers will survive coordinator restarts."

while true; do
    echo "[$(date)] Starting coordinator..."
    .venv/bin/python -u coordinator.py \
        --generation-every 60 \
        --evolve-every 3 \
        --runs-dir runs \
        >> coordinator.log 2>&1

    EXIT_CODE=$?
    echo "[$(date)] Coordinator exited with code $EXIT_CODE" >> coordinator.log

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Clean exit. Stopping."
        break
    fi

    echo "[$(date)] Coordinator crashed. Restarting in 5s..." >> coordinator.log
    sleep 5
done
