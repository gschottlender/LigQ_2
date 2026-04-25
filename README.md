# LigQ_2 – Ligand Query Pipeline for Protein Targets

LigQ_2 is a modular bioinformatics and cheminformatics pipeline for **identifying known and predicted ligands for protein targets**, starting from protein sequences.

Given a FASTA file of query proteins, LigQ_2 integrates:
- Sequence-based search (BLAST)
- Domain-based search (Pfam / HMMER)
- Curated ligand knowledge from PDB and ChEMBL
- Similarity-based ligand prediction using ZINC or user-provided compound databases

The pipeline is designed to be:
- Reproducible
- Scalable (multi-core, large datasets)
- Usable out of the box, with automatic downloads from Hugging Face when required

It now uses an **on-demand predicted-ligand cache**:
- Candidate proteins are found from BLAST/Pfam for each query FASTA run.
- Predicted ligands are computed only for missing reference proteins.
- Cached predictions are reused in subsequent runs.
- Cache is namespaced by search method (**representation + metric**).
- Cache is automatically invalidated if the local provider database changes.

---

## Installation

LigQ_2 is distributed with a Conda environment file to ensure full reproducibility.

### 1. Clone the repository

```bash
git clone https://github.com/gschottlender/LigQ_2.git
cd LigQ_2
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml -n ligq_2_env
```

This command will create a Conda environment named **ligq_2_env**, containing:
- Python
- RDKit
- BLAST+
- HMMER
- All required Python dependencies

(The environment name is defined inside `environment.yml`.)

### 3. Activate the environment

```bash
conda activate ligq_2_env
```

Once activated, all LigQ_2 scripts can be run directly.

---

## Quick Start

### 1) Default run (recommended)

Run LigQ_2 with the default settings:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results
```

- By default, candidate-protein search uses:
  - strict `sequence` BLAST matches
  - `nearest_k` BLAST matches (`K=5`) excluding proteins already found by `sequence`
    and restricted to proteins sharing at least one Pfam domain with the query
- Domain-based search is optional (enable with `--domains`).
- Ligands are **collapsed per query** (one row per ligand ID).
- If a ligand is found by both sequence and domain searches, the
  sequence-based hit is prioritized.
- This mode produces cleaner and more compact result tables.

---

### 2) Keep repeated ligands (no collapse)

Run LigQ_2 while **keeping repeated ligands** coming from different
proteins or search types (sequence / domain):

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results \
  --keep-repeated-ligands
```

- Ligands are **not collapsed**.
- Multiple rows per ligand may appear if the same compound is associated
  with different proteins or search types.
- This mode preserves more contextual information, at the cost of larger
  output files.

---

### 3) Search method flags

Control candidate-protein methods explicitly:

- `--sequence`: enable strict sequence-based BLAST method.
- `--nearest_k`: enable nearest-K BLAST method.
- `--nearest-k`: set K for nearest-K method (default: `5`).
- `--domains`: enable Pfam/HMMER domain method.

Nearest-K candidates are ranked by BLAST proximity and then filtered to
proteins that share at least one Pfam domain detected in the query.
The final output returns up to `K` proteins per query after that domain
filter. If fewer than `K` proteins share query domains, only those
available proteins are returned.

If no method flags are provided, LigQ_2 defaults to:
- `--sequence` ON
- `--nearest_k` ON (`--nearest-k 5`)
- `--domains` OFF

---

### 4) Increase sequence diversity (lower identity threshold)

Run LigQ_2 with a **lower minimum sequence identity** to retrieve a more
diverse set of candidate proteins from sequence-based searches:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results \
  --min-identity 0.5
```

- A lower identity threshold increases the diversity of sequence hits.
- This may lead to more candidate proteins and a broader ligand space.
- Useful for exploratory analyses or distant homology searches.

---

### 5) Combine both options

Retrieve diverse sequence hits **and** keep all repeated ligands:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results \
  --min-identity 0.5 \
  --keep-repeated-ligands
```

- Maximizes coverage and diversity.
- Recommended only when detailed, redundant information is desired.

---

## Main Script: run_ligq_2.py

This is the primary user-facing entry point.

### Workflow

1. Ensure base data (sequences and results_databases)
2. Prepare complementary databases (Pfam, BLAST)
3. Run BLAST (strict sequence hits + ranked nearest-K pool)
4. Run HMMER when domain results and/or nearest-K domain gating are needed
5. Combine candidate proteins (sequence / nearest_k / domain)
6. Ensure/refresh known binding table (PDB + ChEMBL)
7. Run **on-demand** predicted-ligand search for missing candidate proteins only
8. Reuse local cache for proteins already processed with the same method
9. Write per-query results and a global summary

### On-demand cache layout

Method-specific predicted ligand cache is stored under:

```
<data-dir>/results_databases/predicted_bindings/<provider>/
  search_representation=<...>__search_metric=<...>__.../
    predicted_binding_data.parquet
    predicted_binding_progress.json
    manifest.json
    .cache.lock
```

The `manifest.json` captures method configuration and provider database
fingerprint to ensure cache consistency. Locking avoids concurrent write races. During on-demand computation, a tqdm progress bar reports completed vs pending requested proteins.

### New useful options in `run_ligq_2.py`

Default provider is `zinc`, but the pipeline now uses a provider interface (`--ligand-provider`) so additional sources can be added without changing the main workflow.

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results \
  --ligand-provider zinc \
  --search-representation morgan_1024_r2 \
  --search-metric tanimoto \
  --search-threshold 0.5 \
  --cluster-threshold 0.8
```

Legacy `--zinc-*` option names are still accepted as aliases, but the neutral
`--search-*` flags are the preferred interface going forward.

When `--ligand-provider` is set to a custom base name, LigQ_2 expects to find:

```text
<data-dir>/compound_data/<provider-name>/
  ligands.parquet
  reps/
    <search-representation>.dat
    <search-representation>.meta.json
```

Optional rebuild controls:

- `--force-rebuild-known-binding`
- `--force-rebuild-protein-domains`
- `--force-rebuild-predicted-cache`

---

## Utility Script: `add_new_representation.py`

This helper script builds an additional compound representation under:

```
<output-dir>/compound_data/<base>/reps/<rep-name>.dat
<output-dir>/compound_data/<base>/reps/<rep-name>.meta.json
```

It can target:
- the legacy built-in spaces (`zinc`, `pdb_chembl`)
- any custom base under `compound_data/<base-name>/`
- an explicit database root via `--target-root`

Compatibility with the local `pdb_chembl` base can still be requested when
needed with `--ensure-local-compatible`.

### Supported representation families

1. **HuggingFace embeddings** (`--representation-type huggingface`)
   - Example default: `seyonec/ChemBERTa-zinc-base-v1`
   - Output vectors stored as dense embeddings (float16 memmap).
   - `--n-bits` is optional; when omitted, the model `hidden_size` is used.
   - Some HuggingFace repositories ship **custom Python code** for the tokenizer,
     config, or model classes. In those cases you must add
     `--trust-remote-code` so `transformers` can load the repository correctly.
   - If you want reproducible loading of a remote-code model, you can also pin
     a specific repository revision with `--revision <commit-or-tag>`.

2. **RDKit fingerprints** (`--representation-type rdkit`)
   - `--n-bits` is required.
   - `--rdkit-fp-kind ap`: **Atom Pair** fingerprint (hashed bit vector).
   - `--rdkit-fp-kind topological_torsion`: **Topological Torsion** fingerprint (hashed bit vector).
   - `--rdkit-fp-kind rdkit`: **RDKit/Daylight-like** fingerprint (bit vector).
   - `--rdkit-fp-kind morgan_feature`: **Feature Morgan / FCFP-like** fingerprint (bit vector).
   - `--rdkit-fp-kind maccs`: **MACCS keys** fingerprint (fixed 167 bits).

3. **Morgan fingerprints**
   - Already available in the database build pipeline as `morgan_1024_r2`
     (used by default for ZINC similarity search in `run_ligq_2.py`).

### Multi-core CPU optimization

For RDKit bit fingerprints, generation is optimized for multi-core CPU in the same
style as Morgan fingerprints:
- batched processing,
- multiprocessing workers,
- packed bit storage in memmap (`uint8`),
- metadata with timing and failure statistics.

Progress display for **all representation builders** (Morgan, RDKit, HuggingFace):
- single-line tqdm progress bar,
- percentage complete,
- elapsed time and **estimated remaining time (ETA)**.

Parallelism controls:
- `--n-jobs` (number of workers; default: all CPUs)
- `--chunksize` (chunk size for worker scheduling)

### Example commands

Build an Atom Pair representation:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type rdkit \
  --rdkit-fp-kind ap \
  --n-bits 1024 \
  --rep-name atom_pair_1024 \
  --n-jobs 16 \
  --chunksize 500
```

Build a Topological Torsion representation:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type rdkit \
  --rdkit-fp-kind topological_torsion \
  --n-bits 1024 \
  --rep-name topological_torsion_1024
```

Build an RDKit/Daylight-like representation:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type rdkit \
  --rdkit-fp-kind rdkit \
  --n-bits 2048 \
  --rep-name rdkit_daylight_2048
```

Build a MACCS representation:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type rdkit \
  --rdkit-fp-kind maccs \
  --n-bits 167 \
  --rep-name maccs_167
```

Build a Feature Morgan representation in a custom base:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base-name vendor \
  --representation-type rdkit \
  --rdkit-fp-kind morgan_feature \
  --n-bits 1024
```

Build a HuggingFace representation:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type huggingface \
  --rep-name chemberta_zinc_base_768 \
  --model-id seyonec/ChemBERTa-zinc-base-v1 \
  --n-bits 768 \
  --batch-size 14
```

Build a HuggingFace representation from a repository that requires custom code
(e.g. MoLFormer):

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type huggingface \
  --rep-name ibm-MolFormer \
  --model-id ibm-research/MoLFormer-XL-both-10pct \
  --batch-size 14 \
  --trust-remote-code
```

If you see a HuggingFace error saying the repository contains custom code that
must be executed locally, rerun the command with `--trust-remote-code`. For
extra reproducibility, add `--revision <commit-or-tag>` to pin the exact model
revision being loaded.

---

## Outputs

### Global Summary

```
results/search_results_summary.tsv
```

Contains one row per query protein, summarizing the results obtained from
sequence-based and domain-based searches.

Columns report:
- the number of candidate proteins,
- the number of unique known ligands,
- and the number of predicted ligands from the selected provider,

separately for **sequence** and **domain** searches.

---

### Per-Query Results

```
results/search_results/<QUERY_ID>/
```

Each directory corresponds to a single query protein and contains
ligand-level results derived from two complementary search strategies:

---

#### `search_type: sequence` vs `search_type: domain`

For every ligand entry, the column `search_type` indicates **how the
protein–ligand association was obtained**:

- **`sequence`**

  The ligand is associated with a protein identified by **direct
  sequence similarity** to the query protein (full-length alignment).

  These hits typically reflect:
  - closer evolutionary relationships,
  - higher confidence functional similarity,
  - more conservative ligand transfer.

- **`domain`**

  The ligand is associated with a protein identified by **shared protein
  domains** (e.g. Pfam-based matches), even when full-length sequence
  similarity is low.

  These hits typically reflect:
  - conserved catalytic or binding domains,
  - broader functional similarity,
  - increased chemical diversity at the cost of lower specificity.

When the same ligand is retrieved by both strategies, **sequence-based
hits are always prioritized** over domain-based hits when ligand
collapsing is enabled.

---

### Known Ligands

```
known_ligands.tsv
```

Curated ligands retrieved from **PDB and ChEMBL** that are associated with
candidate proteins identified by sequence-based and/or domain-based
searches.

Each row represents a ligand–protein association and may include:
- structural identifiers,
- binding site information,
- experimental annotations.

---

### Predicted Ligands

```
predicted_ligands.tsv
```

Predicted ligands retrieved from the selected provider (for example **ZINC**
or a user-provided compound database), identified by similarity searches
starting from known ligands associated with candidate proteins.

Depending on the chosen options:
- ligands may be **collapsed to one row per compound**, prioritizing
  sequence-based hits, or
- multiple rows may be present for the same ligand when repeated ligands
  are kept.

---

## Updating Databases

### 1. Update PDB and ChEMBL (requires specification of ChEMBL version to download)

```bash
python update_databases.py --chembl-version 36
```

### 2. Update ZINC

```bash
python update_zinc_databases.py
```

By default, this command now backs up any existing files under:

```bash
databases/compound_data/zinc/reps/
```

into a timestamped folder under:

```bash
databases/compound_data/zinc/old_reps_backup/
```

before rebuilding the updated ZINC base. After the update finishes,
`reps/` contains only the newly generated representation associated with the
fresh ZINC database build (by default, `morgan_1024_r2`).

This avoids mixing old representations computed on a previous ZINC version
with the new updated base. Additional representations for the updated ZINC
database should then be regenerated with:

```bash
python add_new_representation.py --output-dir databases --base zinc ...
```

The ZINC predicted-binding cache is also moved by default from:

```bash
databases/results_databases/predicted_bindings/zinc/
```

into:

```bash
databases/results_databases/old_predicted_bindings_backup/zinc/
```

This keeps per-protein predicted results computed against an older ZINC build
separate from the cache that will be generated for the updated database.

If you explicitly want to keep the current contents of `reps/` in place during
the update, use:

```bash
python update_zinc_databases.py --keep-existing-reps
```

If you explicitly want to keep the existing ZINC predicted cache in place, use:

```bash
python update_zinc_databases.py --keep-existing-predicted-cache
```

### 3. Run queries directly (single-script operational mode)

```bash
python run_ligq_2.py --input-fasta queries.fasta --output-dir results
```

No mandatory global precomputation step is required anymore.
Predicted ligands are computed incrementally and cached on demand
via provider-specific cache namespaces.

### 4. Build a custom compound database

Import a user-provided compound table into the LigQ_2 internal format:

```bash
python build_compound_database.py \
  --input-file vendor.csv \
  --base-name vendor \
  --id-column compound \
  --smiles-column SMILES
```

Supported input formats:
- `csv`
- `tsv`
- `smi`
- `parquet`

If `--id-column` or `--smiles-column` are omitted, the importer tries common
column names automatically for table-like inputs (`csv`, `tsv`, `parquet`).

For `.smi`, the current supported convention is:

```text
SMILES compound_id
```

with no header row. The first column is interpreted as `SMILES` and the second
as the compound identifier. The command creates:

```text
databases/compound_data/vendor/
  ligands.parquet
  reps/
    morgan_1024_r2.dat
    morgan_1024_r2.meta.json
```

After that, the base can be queried directly with:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results \
  --ligand-provider vendor
```


---

## Performance Notes

- Parallel execution supported
- Resume-safe long computations
- Designed for HPC but works locally

---

## Dependencies

- Python ≥ 3.9
- BLAST+
- HMMER
- RDKit
- pandas, numpy, pyarrow
- huggingface_hub

---

## Data Sources

- PDB
- ChEMBL
- UniProt
- Pfam
- ZINC

---

## Status

Active development  
Databases synced December 2025
