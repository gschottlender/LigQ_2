# LigQ_2 – Ligand Query Pipeline for Protein Targets

LigQ_2 is a modular bioinformatics and cheminformatics pipeline for **identifying known and predicted ligands for protein targets**, starting from protein sequences.

Given a FASTA file of query proteins, LigQ_2 integrates:
- Sequence-based search (BLAST)
- Domain-based search (Pfam / HMMER)
- Curated ligand knowledge from PDB and ChEMBL
- Similarity-based ligand prediction using ZINC

The pipeline is designed to be:
- Reproducible
- Scalable (multi-core, large datasets)
- Usable out of the box, with automatic downloads from Hugging Face when required

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

### 3) Increase sequence diversity (lower identity threshold)

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

### 4) Combine both options

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

1. Ensure base data (sequences and results databases)
2. Prepare complementary databases (Pfam, BLAST)
3. Run BLAST (sequence-based search)
4. Run HMMER (domain-based search)
5. Combine candidate proteins
6. Retrieve known and predicted ligands
7. Write per-query results and a global summary

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
- and the number of predicted ZINC ligands,

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

### ZINC Ligands

```
zinc_ligands.tsv
```

Predicted ligands retrieved from **ZINC**, identified by similarity
searches starting from known ligands associated with candidate proteins.

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

### 3. Generate Result Tables (Long Step), requires generated PDB, ChEMBL and ZINC databases

```bash
python generate_results_tables.py --regenerate
# Optional custom output folder:
# python generate_results_tables.py --regenerate --output-dir chemberta_results
# Optional search thresholds:
# python generate_results_tables.py --regenerate --zinc-search-threshold 0.65 --cluster-threshold 0.85
```

This step can take ~3 days for a full run and is resume-safe.

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
