#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pkill -f "uvicorn main:app"
sleep 1
exec "$SCRIPT_DIR/start.sh"
