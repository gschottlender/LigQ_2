---
language:
  - en
license: other
license_name: upstream-data-licenses
license_link: https://www.ebi.ac.uk/chembl/faq
tags:
  - chemistry
  - bioinformatics
  - reproducibility
  - virtual-screening
---

# LigQ2 evaluation resources

Exact historical input files for the LigQ2 publication-result evaluations.
This dataset is independent of the operational LigQ2 databases and does not
replace them. The 20 frozen inputs occupy 2,267,375,884 bytes.

## Contents

`frozen/` contains the merged binding and SMILES tables, the original target
mapping, protein sequences and BLAST rankings, and the historical PDB/ChEMBL
compound store with seven precomputed molecular representations: ECFP4,
FCFP4, RDKit Path, MACCS, Atom Pair, Topological Torsion and ChemBERTa.
Every input's size and SHA-256 are recorded in `resource_lock.json`.
Do not change row order, rebuild embeddings, or replace these files with current
database snapshots when attempting an exact historical reproduction.

The separate 61-target, five-partition benchmark is available from
[LigQ_2_benchmark](https://huggingface.co/datasets/gschottlender/LigQ_2_benchmark/tree/f566b163e4d0ab02f9d02ba4f9183561c32f4346).
The required benchmark revision is `f566b163e4d0ab02f9d02ba4f9183561c32f4346`;
its 1,223 required files are also individually locked in the inventory.
Execution random states are 42, 10, 27, 3 and 8.

## Reproduction

The independent `reproduce_evaluations/` package belongs to the
[LigQ2 source repository](https://github.com/gschottlender/LigQ_2).
Consult its README for the two historically pinned environments and commands.
Always download this dataset at the full immutable commit recorded in the
package's `artifact_lock.json`, not at `main`.

The package reproduces main result Figures 2–3, supplementary Figures S1–S8
and three seven-category appendix panels. Computational performance and
table-only experiments are not included. Plotting archived summaries is
distinct from recalculating the molecular evaluations; a full run must pass
the package's scientific and image regression checks before exact independent
reproduction can be claimed.

## Provenance and licensing

These are derived public-resource inputs from PDB, ChEMBL and UniProt, with
ChemBERTa representations based on
[seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1).
The historical embedding metadata does not identify a model commit; the
published embedding array and its SHA-256 preserve the actual evaluated input.
Underlying records remain subject to their upstream licenses and attribution
requirements. The LigQ2 source-code license does not replace those data licenses.
Consult the respective providers for reuse terms and cite LigQ2 and the
underlying databases/model in derivative work.

No manuscript documents, workstation paths, credentials or operational caches
are included in this dataset.
