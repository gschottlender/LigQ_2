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

The default published images use the `main` tag and are built from the
repository's `main` branch. The legacy `gui` tag points to the same builds.

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
required** before exposing the search interface. It reads package file sizes
from the official Hugging Face dataset, shows each missing download size and
available space on the database disk, and disables installation when space is
insufficient for the selected packages.

The selector offers three packages:

- **Required databases** is always selected. It contains the default
  ZINC/PDB-ChEMBL data, BLAST/Pfam resources, and supported BSI models.
- **Morgan ECFP cache** covers Tanimoto scores from `0.4` upward and is selected
  by default.
- **Morgan Feature FCFP cache** covers scores from `0.5` upward and is optional.
  Its package also contains the compatible ZINC and PDB/ChEMBL FCFP
  representations required to use that cache.

Click **Download and prepare data** to start a background setup job. The job
installs only missing files from the selected packages directly under
`databases/`, reports progress in the browser as both downloaded GB/total GB and
completed files/total files, and can resume after a failed or interrupted
download. At the 2026-07-24 snapshot, the three package totals are approximately
5.93 GB, 0.68 GB, and 2.02 GB, respectively. Live Hugging Face metadata remains
the source of truth if the repository changes.

Keep the backend running during this operation. When setup finishes, the
frontend automatically reloads the available databases and opens the normal
application.

---

## Basic workflow

**Run a search**
1. Open **Run Search** in the top navigation.
2. Upload a FASTA file, select a database and representation, optionally enable BSI, and choose search methods.
3. Click **Run Search** — the status panel shows the current pipeline step while
   results appear per query. Structural-similarity searches also show processed
   items, ETA, and elapsed time. BSI searches show only the active step because
   processing time can vary substantially between proteins.

**Inspect results**
- Click a query row to load its results below.
- Switch between **Protein Ranking**, **Known Bindings**, and **Predicted Ligands** tabs.
- Protein Ranking includes only proteins that contribute at least one retained
  known or predicted ligand after per-query deduplication.
- Click any compound to open the detail panel with its official ZINC20, RCSB PDB,
  or ChEMBL page when available, plus 2D structure, Download SDF, and 3D Viewer.

**Add a compound database**
Go to **Manage Resources → Add new database**, upload a `.smi`, `.csv`, `.tsv`,
or `.parquet` file, and click **Process database**. While the job is queued or
running, **Cancel** stops its workers and removes the incomplete staging
database after confirmation.

**Add a molecular representation**
Go to **Manage Resources → Add new representation**, select a database and preset,
and click **Process representation**. It becomes searchable only after its `.dat`
and `.meta.json` files exist in both the selected database and `pdb_chembl`.
Incomplete representations are hidden from Search and can be processed again.
Long embedding or fingerprint jobs can be cancelled after confirmation.
Incomplete files from the active phase are removed, while a compatible copy
that already finished successfully is preserved for the next retry.
The graphical workflow enables ChemBERTa/HuggingFace generation only when its
backend detects a usable CUDA GPU. The CPU Docker image therefore disables these
presets. Native command-line generation is not restricted and may still use CPU.

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
- Database and representation jobs can be cancelled from Manage Resources. The
  API waits for worker termination and cleanup before reporting cancellation.
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
- After FASTA validation, the sidebar displays the sequence count without
  imposing a frontend maximum. Large multi-sequence inputs can substantially
  increase total search time and resource usage.
- The frontend restricts the Nearest K value to integers from `1` through `15`.
- Enabling BSI fixes the representation to `morgan_1024_r2`, displays `BSI Score`
  as the metric, and starts the minimum cutoff at `0.98`, with a lower bound of
  `0.97`. The maximum remains visible but fixed at `1.0`. BSI predictions are
  limited to protein families with a trained Pfam-specific model. The BSI control
  is enabled only when the backend verifies a usable CUDA GPU; CPU-only Docker
  deployments leave it disabled, and the API rejects GUI BSI submissions without
  CUDA. Command-line BSI remains available for administrative runs. In the GUI,
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
