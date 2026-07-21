#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Starting LigQ 2 backend..."

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

nohup python -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info \
    > /tmp/ligq2_uvicorn.log 2>&1 < /dev/null &

PID=$!
echo "$PID" > /tmp/ligq2_uvicorn.pid

echo "Backend running (PID: $PID)"
echo "Logs: tail -f /tmp/ligq2_uvicorn.log"
