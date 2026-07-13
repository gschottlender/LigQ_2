# gui/backend/start.sh

#!/bin/bash
cd /home/marcos/Documentos/Projetos/LigQ2/LigQ_2/gui/backend

echo "Iniciando LigQ 2 backend..."
nohup uvicorn main:app --port 8000 --log-level info > /tmp/ligq2_uvicorn.log 2>&1 &
echo "PID: $!" > /tmp/ligq2_uvicorn.pid
echo "Backend rodando (PID: $!)"
echo "Logs: tail -f /tmp/ligq2_uvicorn.log"