# LigQ 2 — GUI

LigQ 2 is a ligand prioritization tool for protein sequences. Given a FASTA
file, it finds structurally similar proteins in PDB and ChEMBL, collects their
experimentally validated ligands, and expands the results by searching compound
databases (ZINC, LOTUS, or custom) using molecular fingerprint similarity.

This folder contains the web interface: a React frontend and a FastAPI backend
that wraps the LigQ 2 pipeline.

---

## Structure

```
gui/
├── frontend/   # React + TypeScript interface (Vite)
└── backend/    # FastAPI REST API (Python)
```

---

## Requirements

- `ligq_2_env` conda environment activated — update it before running the GUI:
```bash
  conda env update -f environment.yml
```
  The GUI backend requires the following additional packages (already included in `environment.yml`):
  `fastapi`, `uvicorn`, `python-multipart`, `aiofiles`
  
- Node.js 20.19+ or 22.12+ and npm (required by the current Vite version)

---

## Starting the backend

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

---

## Starting the frontend

```bash
cd gui/frontend

# First time only
npm install

npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Running with Docker

Docker provides the same frontend, API, and scientific pipeline in a CPU
`linux/amd64` deployment. It works with Docker Engine on Linux and Docker
Desktop on Windows (WSL2) or macOS. The native Conda/Node workflow above remains
available and uses the same default repository paths.

From the repository root:

```bash
# Build locally, or use `docker compose pull` for published GHCR images.
docker compose build
docker compose up -d
```

Open `http://localhost:8080`. The first visit detects missing databases and
offers the same **Initial setup required** download flow. Data and job history
are kept in Docker volumes; files placed in `work/` are shared with the host.

Useful helpers are available for both shells:

```bash
./docker/ligq.sh status
./docker/ligq.sh logs
./docker/ligq.sh stop
./docker/ligq.sh cli --input-fasta /work/query.fasta --output-dir /work/results
```

On Windows PowerShell, use `docker\ligq.ps1` with the same commands. Set
`LIGQ_WEB_PORT` in `.env` if port 8080 is already occupied, and set `HF_TOKEN`
only when Hugging Face authentication is required. The CPU image is intentionally
separate from `environment.yml`; no CUDA or NVIDIA runtime is needed.

---

## First-time setup

When the default LigQ 2 data is absent, the frontend displays **Initial setup
required** before exposing the search interface. It reads the required file
sizes from the official Hugging Face dataset, shows the missing download size
and available space on the database disk, and disables installation when space
is insufficient.

Click **Download and prepare data** to start a background setup job. The job
installs only missing files directly under `databases/`, reports progress in the
browser as both downloaded GB/total GB and completed files/total files, and can
resume after a failed or interrupted download. The complete GUI-ready dataset
currently contains 63 required files totaling approximately 6.95 GB (6.47 GiB),
including default ZINC/PDB-ChEMBL data, BLAST/Pfam resources, the reusable
Morgan/Tanimoto ZINC predicted-ligand cache with minimum coverage `0.4`, and the
supported BSI models. The live Hugging Face metadata remains the source of truth
if the repository changes.

Keep the backend running during this operation. When setup finishes, the
frontend automatically reloads the available databases and opens the normal
application.

---

## Basic workflow

**Run a search**
1. Open **Run Search** in the top navigation.
2. Upload a FASTA file, select a database and representation, optionally enable BSI, and choose search methods.
3. Click **Run Search** — the status panel shows the current pipeline step,
   processed items, ETA, and elapsed time while results appear per query.

**Inspect results**
- Click a query row to load its results below.
- Switch between **Protein Ranking**, **Known Bindings**, and **Predicted Ligands** tabs.
- Protein Ranking includes only proteins that contribute at least one retained
  known or predicted ligand after per-query deduplication.
- Click any compound to open the detail panel with 2D structure, Download SDF, and 3D Viewer.

**Add a compound database**
Go to **Manage Resources → Add new database**, upload a `.smi`, `.csv`, `.tsv`,
or `.parquet` file, and click **Process database**.

**Add a molecular representation**
Go to **Manage Resources → Add new representation**, select a database and preset,
and click **Process representation**. It becomes searchable only after its `.dat`
and `.meta.json` files exist in both the selected database and `pdb_chembl`.
Incomplete representations are hidden from Search and can be processed again.

**Restore a past search**
Click **History** (top right of Run Search) and **Load** next to any
previous run. Use **Clear history** at the bottom of the panel to permanently
delete stored result folders after confirming the action; active search outputs
are preserved.

---

## Notes

- The backend must be running before opening the frontend.
- Jobs (search, build database, add representation) run as background processes
  and survive browser refreshes.
- A failed job identifies the active step in a red status panel and includes the
  last error reported by the underlying script.
- During **Preparing predicted ligands**, the status panel reports processed
  candidate proteins as `X / total`; proteins already available in the compatible
  cache are included in the initial count.
- Search minimum cutoffs use the representation-specific pipeline defaults when
  available, rounded upward to two decimal places. Unknown representations start
  at `0.9`; the frontend enforces lower bounds of `0.2` for Tanimoto and `0.75`
  for Cosine representations. The maximum starts at `1.0`, and both controls use
  `0.01` increments.
- After FASTA validation, the sidebar displays the sequence count and the current
  frontend limit. The default maximum is `200`; larger files are blocked until
  the limit is increased under **Advanced options**. Large multi-sequence inputs
  can substantially increase total search time.
- The frontend restricts the Nearest K value to integers from `1` through `15`.
- Enabling BSI fixes the representation to `morgan_1024_r2`, displays `BSI Score`
  as the metric, and starts the minimum cutoff at `0.98`, with a lower bound of
  `0.97`. The maximum remains visible but fixed at `1.0`. BSI predictions are
  limited to protein families with a trained Pfam-specific model. In the GUI,
  BSI allows only Sequence and Nearest K search; Domain is cleared and disabled
  to avoid prohibitively slow domain-wide expansion.
- Results are stored on disk under `results/` and can be reloaded at any time
  via the History panel. Clearing history permanently removes inactive result
  folders to free disk space.

## Att gitignore

### Python
__pycache__/
*.py[cod]
*.egg-info/
.env
ligq_2_env/

### Node / frontend
gui/frontend/node_modules/
gui/frontend/dist/
gui/frontend/dist-ssr/
gui/frontend/.vite/
gui/frontend/.vscode/
gui/frontend/FRONTEND.md

### Vite/ESLint caches
gui/frontend/.eslintcache
gui/frontend/.cache/

### Backend
#### __pycache__/ and *.py[cod] above already cover gui/backend/**
### Uploaded files (temp storage, not part of the codebase)
gui/backend/uploads/
gui/backend/BACKEND.md

### Logs and temp
*.log
/tmp/ligq2_uvicorn.log
/tmp/ligq2_uvicorn.pid
nohup.out

### Pipeline outputs
results/
uploads/
databases/
temp_data/
temp_results/
