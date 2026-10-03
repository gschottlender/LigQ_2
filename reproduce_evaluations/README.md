# Reproduce the LigQ2 publication figures

This independent package reproduces the **13 result figures actually embedded**
in the manuscript and supplementary document inspected on 2 October 2026:
main Figures 2–3, supplementary Figures S1–S8, and three category-level appendix
panels. Figure 1 (workflow), computational performance, and experiments present
only in tables or text are excluded. The inspected document hashes and embedded
image identifiers are recorded in `figure_manifest.json`; the DOCX files are
not required or redistributed.

The original LigQ2 application, its GUI, and its operational databases are
not changed. Scientific kernels are vendored under `engine/`. File and figure
names describe their evaluation; there are no `reviewer_*` output names.

## Published historical inputs

The complete additional historical input bundle is published at
[gschottlender/LigQ_2_evaluations](https://huggingface.co/datasets/gschottlender/LigQ_2_evaluations/tree/1dff255d8ed67e8994c41015bd31ddb2f359915c),
revision **`1dff255d8ed67e8994c41015bd31ddb2f359915c`**. All 20 uploaded input
hashes were verified. `artifact_lock.json` pins this revision and its corresponding
scientific file inventory; `download` uses it automatically.

The original public LigQ2 snapshot lacks five representations and the exact
merged tables/BLAST rankings; the separate evaluation dataset supplies those
frozen resources without altering the operational databases. The downloader
refuses to substitute current databases or regenerate ChemBERTa embeddings.
Plot-only reproduction works from the local historical reference tables.
Publication and hash checks alone do not certify a full molecular recalculation;
the independent end-to-end verification is recorded separately.

The initially launched full run was stopped at the user's request; it is not
reported as a completed full recalculation. The local `VALIDATION_STATUS.md`
records the completed bounded checks and their limitations, when available.

### Code-only Git distribution

Git includes scripts, tests, environment definitions, documentation and pinned
resource manifests only. Downloaded inputs, calculation outputs, reference
tables/images and local validation reports are intentionally excluded.
`download` retrieves the pinned scientific inputs, **not** `reference_data/`.
Historical plotting and comparison against retained results require the local
`reference_data/` directory matching `reference_lock.json`; calculation
orchestration also uses its historical neighbor-cohort table. Preserve that
directory separately: a code-only checkout is not by itself a complete
historical-reference installation. The reference results are not currently
included in the published Hugging Face input bundle.

## Installation

From the LigQ2 repository root:

```bash
conda env create -f reproduce_evaluations/environments/calculation.yml
conda env create -f reproduce_evaluations/environments/chemistry-and-figures.yml
cd reproduce_evaluations
```

Two environments preserve the historical difference between the main searches
(RDKit 2025.09.2) and the later preprocessing/property analyses (RDKit 2025.03.3).
The latter environment also provides the Matplotlib 3.9.4 renderer used for the
final publication images. Conda environment files pin application versions;
they are not platform-specific solver/build locks. Review solver conflicts
rather than upgrading scientific libraries silently.

The main representation and neighbor computations use CPU as in the retained
recalculation. Preprocessing sensitivity used CUDA historically; its retained
script uses `auto`. On another device, exact equality must be checked against
the references, not assumed. No new transformer inference is needed: the
historical ChemBERTa embeddings are imported/downloaded byte-for-byte.

## Immediately redraw the publication figures

```bash
conda activate ligq2-publication-chemistry
python reproduce.py list
python reproduce.py plot --source historical
python reproduce.py validate --figures-only
```

Outputs are in `outputs/figures/`, with descriptive filenames, PNG/PDF/SVG
exports and full-precision plot CSVs. For a single figure:

```bash
python reproduce.py plot --figure Figure_2 --source historical
python reproduce.py plot --figure S5 --source historical
```

The appendix renderer produces all three seven-category panels and a combined
three-page PDF. Historical PNGs are validation references, **not** the source
of a copy-and-rename operation: plots are rendered again from their data.
PNG pixel comparisons ignore serialization metadata. Word's resizing and
recompression are not a scientific or graphical equality criterion.

## Download or import exact inputs

`resource_lock.json` pins the benchmark to:

- Dataset: `gschottlender/LigQ_2_benchmark`.
- Revision: `f566b163e4d0ab02f9d02ba4f9183561c32f4346`.
- 61 targets, five partitions per target, 305 target/partition cells.
- Random states, preserving execution order: **42, 10, 27, 3, 8**.

Download all required, pinned inputs (benchmark plus historical input bundle):

```bash
python reproduce.py download
```

Or download only the benchmark:

```bash
python reproduce.py download --only-benchmark
```

On the original workstation, `local_sources.json` records verified local input
locations. It is intentionally Git-ignored and is not portable documentation:

```bash
python reproduce.py import-local --inventory local_sources.json --include-benchmark
```

The importer copies and hashes inputs; it does not modify the source files.
To explicitly specify the published frozen bundle instead of using the defaults:

```bash
python reproduce.py download \
  --artifact-repository gschottlender/LigQ_2_evaluations \
  --artifact-revision 1dff255d8ed67e8994c41015bd31ddb2f359915c
```

Tags and `main` are rejected. `--offline` restricts access to the existing HF
cache. Every file is checked by size and SHA-256, including benchmark CSVs.
`--data-dir /path/to/data` changes the download/import location. The package
does not download ZINC, predicted-binding caches, or the large Pfam HMM: these
are unnecessary for the result figures in scope.

## Calculate the evaluations

Activate the main environment and record its interpreter before switching:

```bash
conda activate ligq2-publication-calculation
CALCULATION_PYTHON="$CONDA_PREFIX/bin/python"
conda activate ligq2-publication-chemistry
CHEMISTRY_PYTHON="$CONDA_PREFIX/bin/python"
python reproduce.py run --evaluation all --resume \
  --python "$CALCULATION_PYTHON" --chemistry-python "$CHEMISTRY_PYTHON"
```

This can take **hours or days**. Required frozen inputs occupy about 2.27 GB
and the benchmark about 3.5 GB; allow substantially more space for ranking
caches, intermediate tables and outputs. A 16 GB RAM machine may require
reducing engine batch sizes; computational performance is not evaluated here.

Inspect the planned commands without writing or calculating anything:

```bash
python reproduce.py run --evaluation all --dry-run \
  --python "$CALCULATION_PYTHON" --chemistry-python "$CHEMISTRY_PYTHON"
```

The `--evaluation` choices are `representations`, `neighbors`, `adaptive`,
`combinations`, `preprocessing`, `distributions`, and `all`. Required upstream
stages run automatically. Inputs are read-only; everything generated belongs
to `outputs/calculations/`. Resume checks code, input hashes, environments and
completed output hashes. Changed inputs require a new `--output-dir`; there
is no destructive force/reset command in the package interface.

After the full calculation, render and validate:

```bash
python reproduce.py plot --source recalculated
python reproduce.py validate
```

Validation checks exact known/evaluation/background IDs, historical retrieved
compound sets, scientific summary values (numerical tolerance 1e-8), and
publication PNG pixels. A failed check is reported, not silently accepted.
Figure-only validation does **not** certify a full molecular recalculation.

## Scientific protocol preserved

See `METHODS.md` for thresholds, seed construction, aggregation and figure
mapping. Highlights that must not be replaced by current platform defaults:

- The neighbor benchmark excludes every UniProt in the original benchmark
  mapping; it does not additionally exclude strict sequence hits or require
  shared Pfam for nearest-neighbor selection.
- Domain sources do require shared Pfam and exclude the complete benchmark
  mapping. MaxMin uses the same requested seed-size cap, permitting short pools.
- The original K values are calculated first and reused when completing K=2–15.
- Expanded union and fixed-budget mean-rank fusion preserve their different
  denominators and historical category aggregation; CAH2 is not silently
  reassigned in the historical union.
- Adaptive thresholds and the 12-target sensitivity panel are frozen to the
  published choices. They are not selected again by a new target census.
- Histogram groups are normalized separately before equal-weight averaging;
  compounds are never pooled across targets or families.

## Small tests; no full run

```bash
python reproduce.py smoke-test
python reproduce.py smoke-test --inventory local_sources.json \
  --calculation-python "$CALCULATION_PYTHON"
```

The first command uses synthetic fixtures. The second additionally recalculates
PGH1, seed 42, ECFP4 only, and checks EF, known-active IDs and retrieved sets
against the original results. Neither starts the full benchmark.

## Independent end-to-end verification

After setting the two historical interpreter variables shown above:

```bash
python test_publication.py --launch \
  --python "$CALCULATION_PYTHON" --chemistry-python "$CHEMISTRY_PYTHON"
```

This uses a separate initially empty HF cache and data directory, runs the small
tests, downloads both immutable datasets, redraws/checks the historical figures,
then recalculates every in-scope evaluation and renders/checks its figures.
No local resource import is used. The worker continues independently of the
interactive session and can take hours or days. Its state and log are:

```text
outputs/huggingface_end_to_end/end_to_end_status.json
outputs/huggingface_end_to_end/end_to_end.log
```

`status: passed` and `full_recalculation_certified: true` are required for a
successful complete verification. `running` is not a success claim; failures
stop the pipeline and preserve their logs. An interrupted/failed worker can be
restarted with `--launch --resume`, subject to the usual input/code checks.

To stop only this isolated worker and its children, preserving downloads:

```bash
python stop_publication_test.py
```

## Bounded verification when a full run is too expensive

```bash
python test_bounded_publication.py --data-dir data \
  --python "$CALCULATION_PYTHON" --chemistry-python "$CHEMISTRY_PYTHON" \
  --max-minutes 15
```

This downloads the 20 frozen resources and only AMPC/random-state-42 benchmark
files (plus shared metadata). It checks all seven molecular representations
against historical EF and retrieved IDs, mean-rank fixed-budget combinations,
the complete K=2–15 sweep with retained-K reuse, a full-domain cell, identity/
ligand-budget/rescue policy kernels, FCFP4/Butina and Bemis-Murcko sensitivity
against historical results, and the three-group distributions. All 13 images
are separately redrawn from historical tables and checked pixel-for-pixel.
This is a functional/regression sample, **not** a full 61-target reproduction.
The total time bound includes downloads; if it is reached, the current child
process group is stopped and the report identifies the unfinished step.

Outputs and per-step logs are in `outputs/bounded_publication/`; the explicit
scope, status and checks are in `bounded_validation_report.json`. The report
always retains `full_recalculation_certified: false`, even when all bounded
checks pass. Existing matching downloads are reused. `--data-dir` can point to
the interrupted full test's data directory, and `--cache-dir` to its `hf_cache`,
without deleting or replacing either. A later full calculation still requires
all benchmark files; `validate_benchmark` rejects incomplete input collections.

The normal `download` command also accepts `--targets ampc --seeds 42` for
bounded input preparation. With neither option, it still downloads the complete
historical benchmark. No molecular protocol or operational LigQ2 behavior changes.

## Maintainer publication commands

```bash
python reproduce.py export-resources --inventory local_sources.json
```

The export command prepares `publication_export/frozen/` and its resource lock;
it requires about 2.27 GB and never uploads. The separate maintainer command
`python publish_resources.py` explicitly creates/uploads to the authenticated
account's public `LigQ_2_evaluations` dataset and writes `artifact_lock.json`.
It allowlists the 20 inputs, inventory and public dataset card, refuses to replace
different remote files, and never publishes workstation paths or manuscript files.
Ordinary reproduction does not need an upload token. See
`PUBLICATION_RESOURCES.md` for the exact inventory and pinned publication.

`prepare_reference_inventory.py` is a maintainer-only capture utility; ordinary
users never need the original notebooks or evaluation workspace. Scientific
dependency attribution and original hashes are retained under
`engine/dependencies/SOURCES.json`; adapter hashes are in `engine/SOURCE_CAPTURE.json`.
