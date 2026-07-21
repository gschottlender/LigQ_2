# gui/backend/restart.sh

#!/bin/bash
pkill -f "uvicorn main:app"
sleep 1
cd /home/marcos/Documentos/Projetos/LigQ2/LigQ_2/gui/backend
nohup uvicorn main:app --port 8000 --log-level info > /tmp/ligq2_uvicorn.log 2>&1 &
echo "PID: $!" > /tmp/ligq2_uvicorn.pid
echo "Backend reiniciado (PID: $!)"