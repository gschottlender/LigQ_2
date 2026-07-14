# LigQ 2 GUI — Architecture

This document describes how the GUI layers connect to each other and to the
root-level pipeline. It is intended for technical reviewers unfamiliar with
the codebase.

---

## 1. Overview

LigQ 2 has three layers:

```
┌─────────────────────────────────────────────────────────┐
│  Frontend  (gui/frontend/)                              │
│  React 19 + TypeScript + Vite · runs in the browser    │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP / JSON  (Vite proxy → port 8000)
┌────────────────────▼────────────────────────────────────┐
│  Backend  (gui/backend/)                                │
│  FastAPI + asyncio · orchestrates jobs and serves data  │
└────────────────────┬────────────────────────────────────┘
                     │  asyncio.create_subprocess_exec
┌────────────────────▼────────────────────────────────────┐
│  Pipeline  (repository root)                            │
│  run_ligq_2.py · build_compound_database.py             │
│  add_new_representation.py                              │
└─────────────────────────────────────────────────────────┘
```

The backend never imports pipeline code directly. It launches pipeline scripts
as child processes and parses their stdout in real time.

---

## 2. Frontend → Backend connection

### API base URL and proxy

`gui/frontend/src/lib/api.ts` creates a single shared Axios instance:

```ts
export const api = axios.create({ baseURL: '/api', timeout: 30_000 });
```

The base URL `/api` is relative, so in development the Vite dev server proxies
it to `http://localhost:8000` via `gui/frontend/vite.config.ts`:

```ts
server: {
  proxy: { '/api': 'http://localhost:8000' },
}
```

All components import `api` from `src/lib/api.ts` — there are no other HTTP
clients or hardcoded URLs.

### API calls by feature

#### Search

| Method | Endpoint | Caller |
|---|---|---|
| `POST` | `/api/jobs/search` | `Sidebar.tsx` |
| `GET` | `/api/jobs/{job_id}` | `VisualizeResults.tsx` (polling) |
| `GET` | `/api/jobs/{job_id}/summary` | `VisualizeResults.tsx` |

The search form is submitted as `multipart/form-data` (FASTA file + form
fields). The response contains a `job_id` that the frontend stores in state.

#### Result tables

| Method | Endpoint | Caller |
|---|---|---|
| `GET` | `/api/jobs/{job_id}/queries/{query_id}/protein-ranking` | `ProteinRankingTable.tsx` |
| `GET` | `/api/jobs/{job_id}/queries/{query_id}/known-ligands` | `KnownBindingsTable.tsx` |
| `GET` | `/api/jobs/{job_id}/queries/{query_id}/predicted-ligands` | `PredictedLigandsTable.tsx` |

All table endpoints accept `page`, `per_page`, `filters`, `sort_by`, and
`sort_dir` query parameters and return paginated results.

#### Downloads

| Method | Endpoint | Caller |
|---|---|---|
| `GET` | `/api/jobs/{job_id}/download` | `VisualizeResults.tsx` |
| `GET` | `/api/jobs/{job_id}/queries/{query_id}/download` | `ResultsPanel.tsx` |

Responses are `application/zip` streams.

#### Databases and representations

| Method | Endpoint | Caller |
|---|---|---|
| `GET` | `/api/databases` | `DatabaseContext.tsx` (on mount) |
| `GET` | `/api/databases/{name}/representations` | `DatabaseContext.tsx` |

#### File upload

| Method | Endpoint | Caller |
|---|---|---|
| `POST` | `/api/files/upload` | `AddNewDatabase.tsx` |

Returns `{ file_id, filename, columns }`. Used to preview column names before
building a database.

#### Job submission

| Method | Endpoint | Caller |
|---|---|---|
| `POST` | `/api/jobs/build-database` | `AddNewDatabase.tsx` |
| `POST` | `/api/jobs/add-representation` | `AddNewRepresentation.tsx` |

#### History

| Method | Endpoint | Caller |
|---|---|---|
| `GET` | `/api/results` | `VisualizeResults.tsx` (History panel) |

### Polling strategy

After submitting a search or a background job, the frontend polls
`GET /api/jobs/{job_id}` on a fixed `setInterval` of **3 000 ms**. The
interval clears itself when the job reaches a terminal status:
`completed`, `completed_with_warnings`, or `failed`.

During a search, `VisualizeResults.tsx` also polls
`GET /api/jobs/{job_id}/summary` on the same interval to show incremental
per-query results as they are written to disk.

---

## 3. Backend → Pipeline connection

### Subprocess invocation

`gui/backend/services/job_runner.py` launches all pipeline scripts via:

```python
asyncio.create_subprocess_exec(
    sys.executable,          # same Python interpreter as uvicorn
    *args,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=str(PIPELINE_ROOT),  # repository root as working directory
    limit=1024 * 1024 * 10,  # 10 MB line buffer
    env={
        **os.environ,
        "PYTHONUNBUFFERED": "1",   # prevent stdout buffering
        "FORCE_COLOR": "0",        # suppress ANSI escape codes in tqdm
    },
)
```

Using `sys.executable` ensures the subprocess inherits the same conda
environment that uvicorn was started in, without any activation step.

### Script called per job type

| Job type | Script |
|---|---|
| `search` | `run_ligq_2.py` |
| `build_database` | `build_compound_database.py` |
| `add_representation` | `add_new_representation.py` |

### Argument construction (`gui/backend/routers/jobs.py`)

#### `search`

```python
args = [
    "run_ligq_2.py",
    "--input-fasta",          str(fasta_path),       # saved to UPLOADS_DIR
    "--output-dir",           str(output_dir),        # results/{stem}_{timestamp}/
    "--ligand-provider",      ligand_provider,
    "--search-representation", search_representation,
    "--search-metric",        search_metric,
    "--progress-json",
]
# Optional flags
if search_threshold:    args += ["--search-threshold", str(search_threshold)]
if search_threshold_max: args += ["--search-threshold-max", str(search_threshold_max)]
if use_sequence:        args.append("--sequence")
if use_nearest_k:       args += ["--nearest_k", "--nearest-k", str(nearest_k)]
if use_domains:         args.append("--domains")
if known_only:          args.append("--known-only")
```

#### `build_database`

```python
args = [
    "build_compound_database.py",
    "--input-file",  str(upload_path),   # saved to UPLOADS_DIR with UUID name
    "--base-name",   base_name,
    "--progress-json",
]
# For CSV/TSV/Parquet only (not .smi):
if suffix != ".smi":
    args += ["--id-column", id_column, "--smiles-column", smiles_column]
```

#### `add_representation`

```python
args = [
    "add_new_representation.py",
    "--output-dir",          "databases",
    "--representation-type", body.representation_type,
    "--n-bits",              str(body.n_bits),
    "--rep-name",            body.rep_name,
    "--progress-json",
]
if rdkit:   args += ["--rdkit-fp-kind", body.rdkit_fp_kind]
if hf:      args += ["--model-id", body.model_id]
if batch:   args += ["--batch-size", str(body.batch_size)]
if n_jobs:  args += ["--n-jobs", str(body.n_jobs)]

# Built-in bases use --base; custom databases use --base-name + compatibility flag
if body.base_name in ("zinc", "local"):
    args += ["--base", body.base_name]
else:
    args += ["--base-name", body.base_name, "--ensure-local-compatible"]
```

### Stdout parsing (`job_runner.py`)

`_tail_stdout` reads the subprocess stdout line by line and updates the in-memory
job record. GUI-launched scripts receive `--progress-json` and emit newline
events prefixed with `LIGQ_PROGRESS `. Each JSON payload is validated as
`JobProgress` and includes:

```text
step, label, step_index, step_count, percent,
current, total, unit, context, eta_seconds
```

The frontend renders the current step, overall percentage, processed/total
count, ETA, and elapsed time. Structured percentages are monotonic. A parsed
`tqdm` line may enrich the current structured step with count and ETA, but it
does not replace the script's overall percentage.

The previous block, tqdm, and representation-build regexes remain as a legacy
fallback for scripts launched without structured events.

**`_WARNING_TOKENS`** — any line containing `"warning"`, `"no domains found"`,
`"no known ligands"`, or `"skipped"` is appended to the job's `warnings` list
and eventually results in `completed_with_warnings` status.

`_tail_stderr` runs concurrently and logs all stderr lines at INFO level
(no job state updates).

For search jobs, `_watch_fs` runs as a parallel asyncio task. It polls the
`search_results/` output directory every 2 seconds and updates
`completed_queries` as per-query directories appear on disk. It only estimates
`progress_percent` when the job has not emitted structured progress.

---

## 4. Backend → Filesystem connection

### Key paths (`gui/backend/core/config.py`)

```python
PIPELINE_ROOT    = Path(__file__).resolve().parents[3]
#                  gui/backend/core/config.py → 3 levels up → repository root

DATABASES_DIR    = PIPELINE_ROOT / "databases"
COMPOUND_DATA_DIR = DATABASES_DIR / "compound_data"
RESULTS_DIR      = PIPELINE_ROOT / "results"
UPLOADS_DIR      = PIPELINE_ROOT / "gui" / "backend" / "uploads"

ALLOWED_ORIGINS  = os.environ.get("ALLOWED_ORIGINS",
                       "http://localhost:5173,http://localhost:3000").split(",")
```

`PIPELINE_ROOT` resolves at import time from the location of `config.py` — no
environment variable is required.

### Result reading (`gui/backend/services/tsv_reader.py`)

**`read_tsv_paginated(path, page, per_page, filters, sort_by, sort_dir)`**  
Reads a TSV file with `pd.read_csv(path, sep="\t")`, applies column filters,
sorts, and returns a page slice. Columns named `binding_sites` and `pdb_ids`
are deserialized from Python-list, NumPy-style, comma-separated, or
semicolon-separated strings into Python lists.
`NaN` values are converted to `None` for JSON serialization.

**`read_summary(output_dir)`**  
Reads `search_results_summary.tsv` if it exists; otherwise builds an
incremental summary by scanning the `search_results/` subdirectories.

### Database and representation discovery (`gui/backend/services/fs_inspector.py`)

**`list_databases()`**  
Scans `COMPOUND_DATA_DIR`, excludes `pdb_chembl` (internal reference database
not exposed to users), and returns directories that contain a `ligands.parquet`
file.

**`list_representations(db_name)`**  
Lists only search-ready representations. A representation is search-ready when
both `{name}.dat` and `{name}.meta.json` exist under the selected database's
`reps/` directory and under `pdb_chembl/reps/`. Incomplete builds stay out of
the search selector and can be submitted again from Add new representation.
For each valid representation, calls `get_metric_from_manifest()` to determine
the similarity metric and loads its optional default cutoff from
`search_threshold_defaults.json`.
The search sidebar rounds this default upward to two decimal places and exposes
both cutoff controls in `0.01` increments; the shared pipeline value remains exact.

**`get_metric_from_manifest(rep_path)`**  
Checks the sidecar JSON in priority order:
1. `{name}.meta.json` alongside the `.dat` file → reads `search_metric` key
   (explicit), then falls back to `fingerprint_type` (known RDKit types →
   `tanimoto`, anything else → `cosine`).
2. `manifest.json` in the same directory (alternative layout).
3. Name-based keyword heuristic as a last resort (e.g. `chemberta` → `cosine`).

---

## 5. Job state management

All job state lives in `gui/backend/core/state.py` as two in-memory
dictionaries:

```python
jobs: dict[str, Job] = {}
processes: dict[str, asyncio.subprocess.Process] = {}
```

Access is serialized through an `asyncio.Lock`. The `Job` model (Pydantic,
defined in `gui/backend/models/job.py`) stores: `job_id`, `job_type`,
`status`, `progress_percent`, `progress_message`, structured `progress`, `warnings`,
`completed_queries`, `all_queries`, `output_dir`, timestamps, `error`, and a
structured `failure` containing the active step, step number, label, and final
stderr message. The frontend renders this failure in red on the relevant job form.

**State is not persisted.** Restarting uvicorn clears all in-memory job records.
Past search results remain accessible on disk via the History mechanism (see
section 6).

---

## 6. History and result restoration

`GET /api/results` (handled by `history_router` in
`gui/backend/routers/results.py`) scans `RESULTS_DIR` on disk and returns
folder metadata without loading any TSV content.

When the frontend loads a past result, it calls
`GET /api/jobs/{result_folder_name}/summary`. The helper
`_resolve_output_dir(job_id)` resolves the output path with the following
priority:

1. Checks `state.jobs` for an active in-memory record with a matching `job_id`.
2. Falls back to `RESULTS_DIR / job_id` if the directory exists on disk.
3. Raises HTTP 404 if neither is found.

This allows the same result-reading endpoints (`/summary`, `/protein-ranking`,
etc.) to serve both live and historical runs without any special branching in
the frontend.

---

## 7. Data flow diagrams

### Search flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Pipeline

    User->>Frontend: fill form, click Run Search
    Frontend->>Backend: POST /api/jobs/search (multipart: FASTA + params)
    Backend->>Backend: validate FASTA, save to uploads/{job_id}.fasta
    Backend->>Backend: create Job(queued) in state.jobs
    Backend-->>Frontend: { job_id }

    loop poll every 3s
        Frontend->>Backend: GET /api/jobs/{job_id}
        Backend-->>Frontend: { status, progress_percent, completed_queries }
        Frontend->>Backend: GET /api/jobs/{job_id}/summary
        Backend-->>Frontend: per-query counts
    end

    Backend->>Pipeline: asyncio.subprocess run_ligq_2.py --input-fasta ...
    Pipeline-->>Backend: stdout lines (block progress, tqdm, warnings)
    Backend->>Backend: _tail_stdout → update job state
    Backend->>Backend: _watch_fs → poll search_results/ every 2s

    Pipeline->>Backend: exit code 0
    Backend->>Backend: status = completed (or completed_with_warnings)

    Frontend->>Backend: GET /api/jobs/{job_id}/queries/{id}/protein-ranking
    Frontend->>Backend: GET /api/jobs/{job_id}/queries/{id}/known-ligands
    Frontend->>Backend: GET /api/jobs/{job_id}/queries/{id}/predicted-ligands
    Backend-->>Frontend: paginated TSV rows
```

### Database build flow

```mermaid
flowchart LR
    A[User uploads file\nAddNewDatabase.tsx] -->|POST /api/files/upload| B[Backend saves to uploads/\nreturns column names]
    B -->|User selects columns| C[POST /api/jobs/build-database]
    C --> D[build_compound_database.py\n--input-file --base-name]
    D --> E[(databases/compound_data/{name}/\nligands.parquet\nreps/morgan_1024_r2.dat)]
    E -->|GET /api/databases| F[DatabaseContext refetch\nNew DB appears in sidebar]
```

### Representation build flow

```mermaid
flowchart LR
    A[User picks preset\nAddNewRepresentation.tsx] -->|POST /api/jobs/add-representation| B[Backend builds args]
    B --> C{base_name in zinc/local?}
    C -->|yes| D[--base zinc/local]
    C -->|no| E[--base-name X\n--ensure-local-compatible]
    D & E --> F[add_new_representation.py]
    F --> G[(compound_data/{db}/reps/{name}.dat\n+ {name}.meta.json)]
    F -->|if custom DB| H[(compound_data/pdb_chembl/reps/{name}.dat)]
    G -->|GET /api/databases/{name}/representations| I[DatabaseContext refetch\nNew rep appears in sidebar]
```

---

## 8. Frontend component map

| Component | API calls | Notes |
|---|---|---|
| `VisualizeResults.tsx` | `GET /api/jobs/{id}`, `GET /api/jobs/{id}/summary`, `GET /api/results` | Manages search state, polling interval, history panel |
| `Sidebar.tsx` | `POST /api/jobs/search` | Submits `multipart/form-data`; reads database/representation lists from `DatabaseContext` |
| `QueryList.tsx` | — | Receives query data from `VisualizeResults` via props |
| `MetricCards.tsx` | — | Aggregates counts from the summary data passed via props |
| `ResultsPanel.tsx` | `GET /api/jobs/{id}/queries/{qid}/protein-ranking`, `/known-ligands`, `/predicted-ligands` | Fetches on tab change and pagination events |
| `AddNewDatabase.tsx` | `POST /api/files/upload`, `POST /api/jobs/build-database`, `GET /api/jobs/{id}` | Polls job until terminal; calls `refetchDatabases()` on completion |
| `AddNewRepresentation.tsx` | `POST /api/jobs/add-representation`, `GET /api/jobs/{id}` | Polls job; calls `refetchRepresentationsForDatabase()` on completion |
| `DatabaseContext.tsx` | `GET /api/databases`, `GET /api/databases/{name}/representations` | Loaded on mount; exposes `refetchDatabases` and `refetchRepresentationsForDatabase` |
| `SelectedResultPanel.tsx` | — (client-side only) | Uses `@rdkit/rdkit` WASM for 2D SVG rendering and SDF generation |
| `MoleculeViewerModal.tsx` | — (client-side only) | Uses `3dmol.js` for interactive 3D display; rendered via `createPortal` |

### Navigation and state preservation

`App.tsx` uses **CSS show/hide** (`block`/`hidden`) instead of React Router
`<Routes>` to keep all three page views mounted at all times:

```tsx
const isConfigure = pathname.startsWith('/configure');
const isHelp      = pathname.startsWith('/help');
// VisualizeResults is shown when neither flag is true
```

This preserves scroll positions, poll intervals, and form state across tab
navigation without any serialization.

---

## 9. External dependencies

### `@rdkit/rdkit` (WASM)

RDKit is distributed as a CommonJS module with no TypeScript `export default`.
It is loaded client-side and initialized once via a singleton in
`gui/frontend/src/lib/rdkit.ts`:

```ts
// Double-cast workaround for CJS under verbatimModuleSyntax
(import('@rdkit/rdkit') as unknown as Promise<{ default: InitFn }>)
  .then(mod => mod.default({ locateFile: () => '/RDKit_minimal.wasm' }))
```

The WASM binary is served from `gui/frontend/public/RDKit_minimal.wasm`.
All components that need molecular operations (`SelectedResultPanel`,
`MoleculeViewer`, `MoleculeViewerModal`) call `getRDKit()` which returns the
cached promise.

### `3dmol.js`

Used exclusively in `MoleculeViewerModal.tsx` for 3D structure display. Key
integration requirements:

- The container `<div>` must have `position: relative` and explicit pixel
  dimensions — `3dmol` positions its canvas absolutely inside.
- `createViewer(el, config)` must be deferred with `setTimeout(0)` to ensure
  the container has non-zero `clientWidth` after React rendering.
- `viewer.resize()` is called after `viewer.render()` to handle any layout
  discrepancy.
- The modal is rendered via `createPortal(content, document.body)` to escape
  the parent `z-30` stacking context.

### Subprocess Python environment

The backend uses `sys.executable` (the uvicorn interpreter) to launch pipeline
scripts. This means the conda environment must be activated when uvicorn starts.
`PYTHONUNBUFFERED=1` is injected into every subprocess environment to ensure
pipeline stdout is not buffered and reaches `_tail_stdout` in real time.
