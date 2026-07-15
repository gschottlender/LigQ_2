# LigQ 2 — Backend API

FastAPI backend that bridges the React frontend and the LigQ 2 pipeline scripts.

## Requirements

Runs inside the `ligq_2_env` conda environment (pandas, pyarrow, pydantic already present).
Install the web-framework extras:

```bash
conda activate ligq_2_env
pip install -r requirements.txt
```

## Start the server
```bash
cd gui/backend

# First time only — make the scripts executable
chmod +x start.sh stop.sh restart.sh

# Start (runs in the background)
./start.sh
```

The API starts at `http://127.0.0.1:8000`.

```bash
# Follow logs
tail -f /tmp/ligq2_uvicorn.log

# Interactive API docs
open http://127.0.0.1:8000/api/docs

# Stop
./stop.sh

# Restart after code changes
./restart.sh
```
or

```bash
conda activate ligq_2_env
cd gui/backend
uvicorn main:app --reload --port 8000
```

Interactive docs are served at `http://localhost:8000/api/docs`.

## Key endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/setup/status` | Inspect default-data readiness, required download size, and free disk space |
| POST | `/api/setup/download` | Start or reconnect to the default-data setup job |
| GET | `/api/databases` | List available compound databases |
| GET | `/api/databases/{name}/representations` | List representations for a database |
| GET | `/api/databases/{name}/columns` | Columns of an uploaded file (by temp ID) |
| POST | `/api/files/upload` | Upload a compound file and get its columns |
| POST | `/api/jobs/search` | Start a ligand search job |
| POST | `/api/jobs/build-database` | Build a new compound database |
| POST | `/api/jobs/add-representation` | Compute a new molecular representation |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{job_id}` | Poll job status |
| DELETE | `/api/jobs/{job_id}` | Cancel a running job |
| GET | `/api/jobs/{job_id}/summary` | Search results summary (all queries) |
| GET | `/api/jobs/{job_id}/queries/{query_id}/protein-ranking` | Protein ranking table |
| GET | `/api/jobs/{job_id}/queries/{query_id}/known-ligands` | Known ligands table |
| GET | `/api/jobs/{job_id}/queries/{query_id}/predicted-ligands` | Predicted ligands table |
| GET | `/api/jobs/{job_id}/download` | Download all TSVs as ZIP |
| GET | `/api/jobs/{job_id}/queries/{query_id}/download` | Download one query's TSVs as ZIP |
| GET | `/api/results` | List stored search result folders |
| DELETE | `/api/results` | Delete inactive search result folders after frontend confirmation |

## Polling pattern (frontend)

```
POST /api/jobs/search → { job_id }
↓
GET /api/jobs/{job_id}  every 3 s while status ∈ {queued, running, partial_results}
↓
GET /api/jobs/{job_id}/summary
GET /api/jobs/{job_id}/queries/{query_id}/protein-ranking?page=1&per_page=20
GET /api/jobs/{job_id}/queries/{query_id}/known-ligands?page=1&per_page=20
GET /api/jobs/{job_id}/queries/{query_id}/predicted-ligands?page=1&per_page=20
```

## CORS

Allowed origins by default: `http://localhost:5173`, `http://localhost:3000`.
Override with the `ALLOWED_ORIGINS` environment variable (comma-separated).

## Notes

- Jobs are stored in memory — state is lost on server restart.
- Initial setup is launched through `prepare_ligq_2_data.py`. It downloads only
  missing files from `gschottlender/LigQ_2` directly into `databases/`, includes
  the reusable default Morgan/Tanimoto ZINC predicted-ligand cache and the BSI
  models, and emits aggregate downloaded-byte and completed-file counters through
  the same structured progress events used by other GUI jobs.
- Uploaded files land in `gui/backend/uploads/` and can be deleted after the job finishes.
- Result endpoints are read-only except `DELETE /api/results`, which permanently
  removes inactive search output folders while preserving queued or running jobs.
