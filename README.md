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

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results \
```

If required databases are missing, they are automatically downloaded from Hugging Face (updated December 2025).

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

Contains one row per query protein summarizing the search.

### Per-Query Results

```
results/search_results/<QUERY_ID>/
```

Each directory contains:

#### known_ligands.tsv
Curated ligands from PDB and ChEMBL with structural and annotation data.

#### zinc_ligands.tsv
Predicted ligands from ZINC based on similarity search.

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
