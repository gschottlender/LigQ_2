# LigQ_2

LigQ_2 is a protein-to-ligand search pipeline. Given a FASTA file with protein
queries, it recovers related reference proteins and reports known and predicted
ligands for those targets.

The pipeline combines:

- BLAST sequence search.
- Pfam/HMMER domain search.
- Curated ligand associations from PDB and ChEMBL.
- Predicted ligand expansion against ZINC or user-provided compound databases.
- Optional BSI scoring for Pfam-supported protein families.

LigQ_2 is designed to run directly from the command line with automatic base
data downloads from Hugging Face when local data is missing.

This README is intended to be the operational source of truth for users,
testers, and reviewers. It focuses on what the current command-line tool does,
which files it expects, which outputs it writes, and which behaviors should be
checked before publishing or reviewing a release.

## Contents

- [Quick Start](#quick-start)
- [Input Requirements](#input-requirements)
- [Main Workflow](#main-workflow)
- [Candidate Protein Modes](#candidate-protein-modes)
- [Predicted Ligand Search](#predicted-ligand-search)
- [Cache Behavior](#cache-behavior)
- [Data Layout](#data-layout)
- [Output Files](#output-files)
- [Custom Compound Databases](#custom-compound-databases)
- [Additional Representations](#additional-representations)
- [Updating Databases](#updating-databases)
- [Performance Notes](#performance-notes)

## Quick Start

### Install

```bash
git clone https://github.com/gschottlender/LigQ_2.git
cd LigQ_2
conda env create -f environment.yml -n ligq_2_env
conda activate ligq_2_env
```

The environment includes Python, RDKit, BLAST+, HMMER, pandas, numpy, pyarrow,
and `huggingface_hub`.

### Run the default search

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results
```

Default behavior:

- Uses strict BLAST sequence candidates.
- Uses nearest-K BLAST candidates (`--nearest-k 5`) filtered to proteins sharing
  at least one Pfam domain with the query.
- Does not use full domain-only candidate expansion unless `--domains` is set.
- Uses `zinc` as the predicted-ligand provider.
- Uses `morgan_1024_r2` and `tanimoto` for structural similarity.
- Collapses duplicate ligand IDs per query unless `--keep-repeated-ligands` is set.
- Downloads missing default-ready base data from Hugging Face. For the default
  `zinc` provider this includes the required ZINC ligand table and
  `morgan_1024_r2` representation.
- Downloads the precomputed Morgan/Tanimoto ZINC predicted-ligand cache with
  minimum coverage `0.4` when base data has to be downloaded and
  `--skip-hf-predicted-cache` is not set. Compatible stricter searches reuse it;
  if no compatible cache exists, LigQ_2 computes missing entries on demand.

The automatic Hugging Face download path is default-ready for `zinc`. Custom
providers are loaded from local `databases/compound_data/<provider>/` directories
and are not created by `run_ligq_2.py`.

The web application provides a dedicated first-time setup before search. When
the default data is absent, it reports the live required download size and free
space, then runs `prepare_ligq_2_data.py` as a background job. This installs only
missing default files directly under `databases/`, including the reusable
Morgan/Tanimoto ZINC predicted-ligand cache and the BSI family models exposed by
the frontend. During installation the GUI reports downloaded bytes against the
complete data size and completed files against the 63-file manifest, then
refreshes the available GUI databases on completion.

### Known ligands only

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results_known_only \
  --known-only
```

`--known-only` still recovers candidate proteins, but skips provider setup,
predicted-ligand cache generation, and ZINC/custom-compound searches.
Only the core sequence, Pfam, BLAST, known-ligand, and protein-domain data are
required in this mode.

### BSI search

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results_bsi \
  --ligand-provider zinc \
  --bsi \
  --bsi-threshold 0.5
```

BSI uses `morgan_1024_r2`, models under `<data-dir>/bsi_models/mpg_1024`, and
reports `bsi_score` plus the selected `pfam_id`. Proteins without a supported
Pfam family produce no BSI predicted ligands.

Useful BSI-specific controls:

```bash
--bsi-threshold 0.5
--bsi-model-batch-size 65536
--bsi-max-known-ligands 10
```

## Input Requirements

Input is a protein FASTA file:

```text
>query_1
MSEQUENCE...
>query_2
MSEQUENCE...
```

LigQ_2 uses the first whitespace-delimited token after `>` as the query ID. That
ID becomes the output directory name under `search_results/`.

## Main Workflow

`run_ligq_2.py` performs these steps:

1. Parse query IDs and sequences from the input FASTA.
2. Ensure required base data exists under `--data-dir` (default `databases`).
3. Download missing default-ready data from Hugging Face when needed.
4. Prepare BLAST and Pfam/HMMER complementary databases.
5. Recover candidate proteins by sequence, nearest-K, and optionally domains.
6. Ensure runtime tables for known ligands and protein domains exist.
7. Build or reuse provider-specific predicted-ligand cache on demand.
8. Write per-query outputs and a global summary.

No mandatory global precomputation step is required for predicted ligands.
Predictions are computed only for candidate proteins needed by the current run
and cached for reuse. Structured search progress reports the number of requested
candidate proteins already cached and advances after each remaining protein is
processed during the predicted-ligand step.

## Candidate Protein Modes

If no candidate method flags are provided, LigQ_2 uses:

- `--sequence` ON.
- `--nearest_k` ON.
- `--nearest-k 5`.
- `--domains` OFF.

If any candidate method flag is provided explicitly, the implicit default is
disabled and only the requested candidate modes are enabled. For example,
`--domains` alone runs domain candidate expansion only; BLAST can still be run
internally to rank domain candidates, but sequence hits are not included unless
`--sequence` is also passed.

Available flags:

```bash
--sequence
--nearest_k
--nearest-k 5
--domains
--max-domain-candidates-per-domain 20
```

`--sequence` applies strict BLAST thresholds:

```bash
--min-identity 0.9
--min-query-coverage 0.9
--min-subject-coverage 0.7
--blast-evalue-max 1e-5
--max-hits 150
```

`--nearest_k` uses the broader ranked BLAST pool, excludes strict sequence hits,
filters to proteins sharing query Pfam domains, and caps the final output after
domain filtering.

`--domains` runs HMMER against Pfam and expands candidates by shared domains.
Within each query Pfam, candidate proteins are ranked by BLAST proximity when
available and capped by `--max-domain-candidates-per-domain`.

## Predicted Ligand Search

Default provider:

```bash
--ligand-provider zinc
```

Default representation and metric:

```bash
--search-representation morgan_1024_r2
--search-metric tanimoto
```

For each recovered candidate protein, LigQ_2 uses its curated PDB/ChEMBL
ligands as query compounds and searches the selected provider database for
similar or BSI-compatible ligands. Candidate proteins without curated known
ligands can still appear in `protein_ranking.tsv`, but they cannot seed a
predicted-ligand search.

Supported provider types:

- `zinc`: built-in default provider under `compound_data/zinc/`.
- custom provider: any local `compound_data/<provider>/` directory with
  `ligands.parquet` and compatible representations under `reps/`.

Additional provider controls:

```bash
--search-threshold <float>
--search-threshold-max <float>
--cluster-threshold 0.8
--search-device auto
--search-query-batch-size <int>
--search-target-chunk-size <int>
--search-per-iteration-topk 1000
--search-global-topk 10000
```

### Default search thresholds

If `--search-threshold` is omitted, non-BSI predicted searches use
representation-specific percentile-99.5 defaults:

```text
chemberta_zinc_base_768              0.936140
rdkit_1024                           0.930324
maccs                                0.831169
ap_rdkit                             0.767087
morgan_feature_1024_r2               0.509451
topological_torsion_rdkit_1024       0.502932
morgan_1024_r2                       0.415094
```

Rules:

- Explicit `--search-threshold` always overrides the representation default.
- Legacy `--zinc-search-threshold` is still accepted as an alias.
- Unknown representations require an explicit `--search-threshold`.
- `--known-only` does not use predicted-search thresholds.
- `--bsi` uses `--bsi-threshold`, not this structural-similarity threshold map.
- The representation name must match the files under both
  `compound_data/pdb_chembl/reps/` and the selected provider's `reps/`
  directory.

## Cache Behavior

Predicted-ligand caches live under:

```text
<data-dir>/results_databases/predicted_bindings/<provider>/
  search_representation=<...>__search_metric=<...>__cache_threshold_min=<...>/
    predicted_binding_data.parquet
    predicted_binding_progress.json
    cached_proteins.json
    predicted_binding_rowgroup_index.json
    manifest.json
    .cache.lock
```

The cache system is provider-generic:

- Built-in ZINC uses provider namespace `zinc`.
- Custom providers use their `--ligand-provider` name.
- BSI providers use `<provider>_bsi`, for example `zinc_bsi`.
- The same indexing and RAM-control machinery is used for ZINC and custom
  providers.

The cache is designed for large runs:

- `cached_proteins.json` tracks completed proteins without reading the full
  predicted-ligand parquet.
- `predicted_binding_progress.json` preserves resume information.
- `predicted_binding_rowgroup_index.json` maps proteins to parquet row groups so
  result generation can read only relevant blocks.
- `manifest.json` stores provider, method, threshold coverage, and database
  fingerprint information.
- `.cache.lock` avoids concurrent cache writers.

A cache generated with a lower minimum threshold can serve stricter later
queries if provider, method, and database fingerprint match. For example, the
Hugging Face `morgan_1024_r2` cache with `cache_threshold_min=0.4` can
serve the current default `morgan_1024_r2` threshold `0.415094`.

Use `--force-rebuild-predicted-cache` to discard and regenerate the compatible
cache namespace for the selected provider, method, and threshold coverage.

Use this flag to avoid downloading the optional precomputed ZINC cache while
still allowing required base-data downloads:

```bash
python run_ligq_2.py ... --skip-hf-predicted-cache
```

## Data Layout

Default data root:

```bash
--data-dir databases
```

Default-ready data layout:

```text
databases/
  db_metadata.json
  sequences/
    target_sequences.fasta
    target_sequences.pkl
  merged_databases/
    binding_data_merged.parquet
    ligs_smiles_merged.parquet
    uncurated_binding_data.parquet
  results_databases/
    known_binding_data.parquet
    protein_domains.parquet
    predicted_bindings/
  compound_data/
    pdb_chembl/
      ligands.parquet
      reps/
        morgan_1024_r2.dat
        morgan_1024_r2.meta.json
    zinc/
      ligands.parquet
      reps/
        morgan_1024_r2.dat
        morgan_1024_r2.meta.json
  complementary_databases/
    blast/
    pfam/
  bsi_models/
    mpg_1024/
```

`merged_databases/` is needed when runtime tables are rebuilt locally with
`--force-rebuild-known-binding` or after `update_databases.py` detects PDB or
ChEMBL changes. Regular runs use the runtime-ready files in
`results_databases/`.

The canonical Hugging Face dataset is:

```text
gschottlender/LigQ_2
https://huggingface.co/datasets/gschottlender/LigQ_2/tree/main
```

The current default-ready Hugging Face tree provides `morgan_1024_r2` for both
`pdb_chembl` and `zinc`. Extra representations must exist locally for both
`compound_data/pdb_chembl/reps/` and the selected target provider before they
can be used in predicted-ligand search.

## Output Files

Global summary:

```text
<output-dir>/search_results_summary.tsv
```

The global summary contains one row per FASTA query, including queries with no
recovered candidates.

Per-query directory, when the query has candidates or ligand results:

```text
<output-dir>/search_results/<QUERY_ID>/
  protein_ranking.tsv
  known_ligands.tsv
  predicted_ligands.tsv
```

Per-query files are written only when they contain information:

- `protein_ranking.tsv` is written when candidate proteins are recovered.
- `known_ligands.tsv` is written when known ligands are found.
- `predicted_ligands.tsv` is written when predicted ligands are found.
- `predicted_ligands.tsv` is always absent in `--known-only` mode.

If a query has no recovered candidates and no ligands, it remains represented in
`search_results_summary.tsv` but may not have a per-query directory.

### `protein_ranking.tsv`

Ranks candidate proteins recovered for the query.

- Sequence and nearest-K candidates are ranked by BLAST evidence.
- Domain candidates with BLAST evidence are ranked above domain-only candidates.
- Domain-only candidates are ranked by Pfam/HMMER evidence.
- `n_shared_domains` reports shared Pfam domains when available.

This table is informational; it does not change ligand retrieval or cache
behavior.

### `known_ligands.tsv`

Curated PDB/ChEMBL ligands associated with recovered proteins.
The table keeps the known-binding columns from
`results_databases/known_binding_data.parquet` plus `search_type`, so reviewers
can trace each ligand back to the candidate recovery mode.

### `predicted_ligands.tsv`

Predicted ligands from the selected provider.

Score columns:

- `tanimoto` for Tanimoto searches.
- `similarity` for cosine/embedding searches.
- `bsi_score` for BSI.

`search_type` indicates whether the ligand came through `sequence`, `nearest_k`,
or `domain` candidate proteins. When duplicate ligand IDs are collapsed,
evidence priority is `sequence`, then `nearest_k`, then `domain`.

### `search_results_summary.tsv`

The summary reports per-query counts split by candidate source:

- `n_proteins_sequence`, `n_proteins_nearest_k`, `n_proteins_domain`
- `n_known_ligands_sequence`, `n_known_ligands_nearest_k`,
  `n_known_ligands_domain`
- `n_predicted_ligands_sequence`, `n_predicted_ligands_nearest_k`,
  `n_predicted_ligands_domain`

## Custom Compound Databases

Build a custom provider:

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

For `.smi`, the expected format is:

```text
SMILES compound_id
```

The command creates:

```text
databases/compound_data/vendor/
  ligands.parquet
  reps/
    morgan_1024_r2.dat
    morgan_1024_r2.meta.json
```

`build_compound_database.py` builds the default `morgan_1024_r2`
representation for the custom provider. Additional representations must be
added separately with `add_new_representation.py`.

Run against the custom provider:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results_vendor \
  --ligand-provider vendor
```

For custom providers, `run_ligq_2.py` expects
`databases/compound_data/vendor/` to already exist. The automatic Hugging Face
download does not create arbitrary custom provider directories.

Run BSI against the custom provider:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir results_vendor_bsi \
  --ligand-provider vendor \
  --bsi
```

## Additional Representations

Use `add_new_representation.py` to build extra representations:

```bash
python add_new_representation.py \
  --output-dir databases \
  --base zinc \
  --representation-type rdkit \
  --rdkit-fp-kind ap \
  --n-bits 1024 \
  --rep-name ap_rdkit \
  --n-jobs 16 \
  --chunksize 500
```

For ZINC, the legacy default `--base zinc` also ensures the same representation
exists in `compound_data/pdb_chembl/`, which is required for search
compatibility. For custom providers, pass `--base-name <provider>` and
`--ensure-local-compatible` when the representation must also be built for
`pdb_chembl`.

Supported RDKit fingerprint kinds:

- `ap`
- `topological_torsion`
- `rdkit`
- `morgan_feature`
- `maccs`

Hugging Face embedding example:

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

For reproducible embedding builds, add `--revision <commit-or-tag>`.

Important: a representation used in search must exist for both `pdb_chembl` and
the selected target provider. If it is missing from either side, provider setup
will fail before the search runs.

The percentile-99.5 default thresholds in `run_ligq_2.py` are keyed by exact
representation names. Use `--rep-name` to match those names when building extra
representations:

| Search representation | Required build flags |
| --- | --- |
| `chemberta_zinc_base_768` | `--representation-type huggingface --model-id seyonec/ChemBERTa-zinc-base-v1 --n-bits 768 --rep-name chemberta_zinc_base_768` |
| `rdkit_1024` | `--representation-type rdkit --rdkit-fp-kind rdkit --n-bits 1024 --rep-name rdkit_1024` |
| `maccs` | `--representation-type rdkit --rdkit-fp-kind maccs --n-bits 167 --rep-name maccs` |
| `ap_rdkit` | `--representation-type rdkit --rdkit-fp-kind ap --n-bits 1024 --rep-name ap_rdkit` |
| `morgan_feature_1024_r2` | `--representation-type rdkit --rdkit-fp-kind morgan_feature --n-bits 1024 --rep-name morgan_feature_1024_r2` |
| `topological_torsion_rdkit_1024` | `--representation-type rdkit --rdkit-fp-kind topological_torsion --n-bits 1024 --rep-name topological_torsion_rdkit_1024` |
| `morgan_1024_r2` | Built by default for HF data and custom providers. |

If a representation name is not listed in the default-threshold table, run with
an explicit `--search-threshold`.

## Updating Databases

### Update PDB and ChEMBL

```bash
python update_databases.py --chembl-version 36
```

When PDB or ChEMBL changes are detected, this command:

- refreshes processed `pdb/` and/or `chembl/` data;
- rebuilds `merged_databases/`;
- regenerates runtime tables:

```text
results_databases/known_binding_data.parquet
results_databases/protein_domains.parquet
```

During regular search runs, the same runtime tables can be rebuilt from local
`merged_databases/` with:

```bash
python run_ligq_2.py \
  --input-fasta queries.fasta \
  --output-dir rebuild_runtime_results \
  --known-only \
  --force-rebuild-known-binding \
  --force-rebuild-protein-domains
```

### Update ZINC

```bash
python update_zinc_databases.py
```

Default behavior:

- backs up existing `databases/compound_data/zinc/reps/` into
  `databases/compound_data/zinc/old_reps_backup/<timestamp>/`;
- rebuilds the fresh ZINC base and default representation;
- moves existing ZINC predicted cache from
  `databases/results_databases/predicted_bindings/zinc/` to
  `databases/results_databases/old_predicted_bindings_backup/zinc/`.

Optional overrides:

```bash
python update_zinc_databases.py --keep-existing-reps
python update_zinc_databases.py --keep-existing-predicted-cache
```

After updating ZINC, rebuild any non-default representations needed for that
new ZINC base.

## Performance Notes

- BLAST uses `--n-workers`.
- Predicted ligand search is incremental by protein.
- Large predicted caches are not loaded wholesale during normal result writing.
- `--search-global-topk` caps per-protein predicted hit sets before large
  metadata joins.
- `--search-target-chunk-size` and `--search-device` control CPU/GPU memory use.
- `--block3-query-chunk-size` and `--block3-predicted-filter-batch-size`
  control memory use while writing query-level outputs.
- `--temp-results-dir` is recreated for each run and stores intermediate BLAST,
  HMMER, and candidate-mapping files useful for debugging.
- BSI supports CPU/GPU execution and chunked target evaluation.

BSI benchmark example:

```bash
python benchmark_bsi_search.py \
  --data-dir databases \
  --ligand-provider zinc \
  --device cpu \
  --target-limit 100000 \
  --seed-counts 1,5,10 \
  --target-chunk-sizes 25000,50000,100000 \
  --model-batch-sizes 32768,65536
```

Set `--target-limit 0` to benchmark the full provider database.

## Data Sources

- PDB
- ChEMBL
- UniProt
- Pfam
- ZINC

## Status

Active development. Current default public data is distributed through the
Hugging Face dataset `gschottlender/LigQ_2`.
