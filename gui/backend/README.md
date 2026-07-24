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
| POST | `/api/setup/download` | Start or reconnect to setup; JSON can select ECFP and FCFP cache packages |
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

- Heavy jobs run through a single FIFO queue. Job metadata is persisted in
  `gui/backend/state/jobs.sqlite3`; completed history survives restarts and
  unfinished work is marked `interrupted` when the backend starts again.
- Runtime paths can be overridden with the `LIGQ_PIPELINE_ROOT`,
  `LIGQ_DATABASES_DIR`, `LIGQ_RESULTS_DIR`, `LIGQ_UPLOADS_DIR`,
  `LIGQ_TEMP_RESULTS_DIR`, and `LIGQ_STATE_DIR` environment variables. Native
  execution keeps the repository paths as defaults.
- Initial setup is launched through `prepare_ligq_2_data.py`. It downloads only
  missing files from `gschottlender/LigQ_2` directly into `databases/`. The
  required package always includes the default databases and BSI models;
  `POST /api/setup/download` accepts `include_ecfp_cache` (default `true`) and
  `include_fcfp_cache` (default `false`) to select the two precomputed cache
  packages. Progress emits aggregate downloaded-byte and completed-file counters
  for the exact selection through the same structured events used by other GUI
  jobs.
- Uploaded files are streamed to `gui/backend/uploads/` with a default 20 GiB
  limit configurable through `LIGQ_MAX_UPLOAD_BYTES`.
- Result endpoints are read-only except `DELETE /api/results`, which permanently
  removes inactive search output folders while preserving queued or running jobs.
