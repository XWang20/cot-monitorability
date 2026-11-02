#!/bin/bash
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PID_FILE="$SCRIPT_DIR/CoT-Control-QA/logs/dashboard.pid"
LOG_FILE="$SCRIPT_DIR/CoT-Control-QA/logs/dashboard.log"
PORT=${2:-5011}

# NOTE: port 5010 is owned by `inspect view` (primary UI). This Flask dashboard is
# legacy and defaults to 5011 to avoid clashing with it.

mkdir -p "$SCRIPT_DIR/CoT-Control-QA/logs"

start() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "Dashboard is already running (PID: $OLD_PID) on port $PORT."
            exit 0
        fi
    fi
    nohup "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/monitor_platform.py" serve --port "$PORT" >> "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    sleep 1
    echo "✅ CoT Monitorability Dashboard (legacy) started (PID: $NEW_PID) at http://8.160.177.124:$PORT"
    echo "   ℹ️  Primary UI is Inspect View on port 5010: inspect view --log-dir $SCRIPT_DIR/eval_logs --port 5010"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        kill "$OLD_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "Stopped Dashboard (PID: $OLD_PID)."
    else
        echo "Dashboard is not running."
    fi
}

status() {
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "✅ Legacy dashboard is running (PID: $(cat "$PID_FILE")) at http://8.160.177.124:$PORT"
    else
        echo "⚪ Dashboard is stopped."
    fi
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    logs) tail -n 50 "$LOG_FILE" ;;
    *) start ;;
esac
