# gui/backend/stop.sh

#!/bin/bash
pkill -f "uvicorn main:app"
echo "Backend encerrado."