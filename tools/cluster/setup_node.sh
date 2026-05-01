#!/bin/bash
# Setup a new training node — run this on any machine to join the cluster.
#
# Usage:
#   bash setup_node.sh                    # auto-detect everything
#   bash setup_node.sh --node-id my-gpu   # custom node ID
#   bash setup_node.sh --start            # register + start training immediately
#
# What it does:
#   1. Ensures Python dependencies are installed
#   2. Probes hardware capabilities
#   3. Registers this node to Firebase (so CLI and other nodes can discover it)
#   4. Starts heartbeat daemon (keeps node "online" in the cluster)
#   5. Optionally starts three_populations.py training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Mamba Platform — Node Setup ==="
echo "  Directory: $SCRIPT_DIR"

# Find Python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: No Python found. Install Python 3.10+ first."
    exit 1
fi

echo "  Python: $PYTHON ($($PYTHON --version 2>&1))"

# Check PyYAML
$PYTHON -c "import yaml" 2>/dev/null || {
    echo "  Installing PyYAML..."
    $PYTHON -m pip install pyyaml -q 2>/dev/null || true
}

# Probe capabilities
echo ""
echo "=== Hardware Probe ==="
$PYTHON server/node_agent.py --probe
echo ""

# Parse args
NODE_ID=""
START_TRAINING=false
for arg in "$@"; do
    case $arg in
        --node-id=*) NODE_ID="${arg#*=}" ;;
        --node-id) shift; NODE_ID="$1" ;;
        --start) START_TRAINING=true ;;
    esac
done

# Build node agent command
AGENT_CMD="$PYTHON server/node_agent.py --register"
if [ -n "$NODE_ID" ]; then
    AGENT_CMD="$AGENT_CMD --node-id $NODE_ID"
fi
AGENT_CMD="$AGENT_CMD --working-dir $SCRIPT_DIR"

if [ "$START_TRAINING" = true ]; then
    AGENT_CMD="$PYTHON server/node_agent.py --start"
    if [ -n "$NODE_ID" ]; then
        AGENT_CMD="$AGENT_CMD --node-id $NODE_ID"
    fi
    AGENT_CMD="$AGENT_CMD --working-dir $SCRIPT_DIR"
fi

# Start as background daemon
echo "=== Starting Node Agent ==="
nohup $AGENT_CMD > node_agent.log 2>&1 &
AGENT_PID=$!
echo "  Agent PID: $AGENT_PID"
echo "  Log: $SCRIPT_DIR/node_agent.log"

# Wait a moment and verify
sleep 3
if kill -0 $AGENT_PID 2>/dev/null; then
    echo ""
    echo "=== Node Online ==="
    echo "  Verify with: mamba nodes"
    echo "  View logs:   tail -f node_agent.log"
    if [ "$START_TRAINING" = true ]; then
        echo "  Training:    tail -f three_pop.log"
    fi
    echo ""
    echo "  To stop:     kill $AGENT_PID"
    echo "  To restart:  bash setup_node.sh"
else
    echo "ERROR: Agent failed to start. Check node_agent.log"
    tail -5 node_agent.log 2>/dev/null
    exit 1
fi
