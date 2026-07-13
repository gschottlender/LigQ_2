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
  
- Node.js 18+ and npm

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

## Basic workflow

**Run a search**
1. Open **Visualize Results** in the top navigation.
2. Upload a FASTA file, select a database and representation, choose search methods.
3. Click **Run Search** — results appear progressively as each query completes.

**Inspect results**
- Click a query row to load its results below.
- Switch between **Protein Ranking**, **Known Bindings**, and **Predicted Ligands** tabs.
- Click any compound to open the detail panel with 2D structure, Download SDF, and 3D Viewer.

**Add a compound database**
Go to **Configure Search → Add new database**, upload a `.smi`, `.csv`, `.tsv`,
or `.parquet` file, and click **Process database**.

**Add a molecular representation**
Go to **Configure Search → Add new representation**, select a database and preset,
and click **Process representation**.

**Restore a past search**
Click **History** (top right of Visualize Results) and **Load** next to any
previous run.

---

## Notes

- The backend must be running before opening the frontend.
- Jobs (search, build database, add representation) run as background processes
  and survive browser refreshes.
- Results are stored on disk under `results/` and can be reloaded at any time
  via the History panel.

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