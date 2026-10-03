# Frozen methods and figure mapping

The methods below describe the retained experiments, not a redesign or the
current application's default evidence-transfer behavior. Figures use the
original plotting functions and parameters. `figure_manifest.json` maps
each figure to its source table, filename and inspected DOCX image relationship.

## Representations: Figure 2, S2 and A1

Five partitions use random states 42, 10, 27, 3 and 8. The 61-target benchmark
uses ECFP4/Butina redundancy reduction at Tanimoto 0.8, 10% known actives,
90% held-out actives, at least 50 known actives, and a sampled background at
200 compounds per evaluation active. The historical splitter and input row
order are preserved; regenerated split and pool IDs must match published CSVs.

Candidates are ranked by maximum similarity to any known-active seed. The
seven methods are ECFP4, FCFP4, Atom Pair, MACCS, RDKit Path, Topological Torsion
and ChemBERTa, with Tanimoto for fingerprints and cosine for ChemBERTa.
Historical precomputed arrays are authoritative. ECFP4/FCFP4 use radius 2 and
1024 bits. ChemBERTa uses attention-mask mean pooling of final hidden states
of `seyonec/ChemBERTa-zinc-base-v1`, stored as 768-dimensional float16 vectors.
Its old metadata did not record a model revision; the embedding-array hash,
not an assumed historical model commit, therefore guarantees the input identity.

Cumulative EF is calculated for eight percentiles; the representation plots
display 99.5, 99, 98.5, 98, 95 and 90. Within each category and partition, take
the median target EF; then take the median across five partitions for each
category. Global curves show the median and Q25–Q75 across seven categories,
transformed to log10 for display. Figure 2 selects four methods; S2 and A1
retain seven. A1 plots the individual category medians without inventing
between-category IQR bands within a category.

## Evidence transfer: Figure 3, S6–S8 and A2

The matched set contains 56 targets. BLAST eligibility uses E-value ≤1e-10,
bitscore ≥80, identity ≥30% and query coverage ≥60%. Ranking and the complete
benchmark UniProt exclusion universe are frozen in the archived hit table.
Nearest-neighbor selection applies neither the live shared-Pfam filter nor
the live strict-sequence-hit exclusion. K is a maximum, not a requirement that
K ligand-bearing proteins exist.

Calculate K=2,3,4,5,10,15 first, then reuse those results and calculate only
missing integer K values up to 15. Full domain pools eligible ligand-bearing
proteins sharing at least one query Pfam and sorts source UniProts as in the
historical code. Diversity cleanup and MaxMin selection are reused verbatim;
the requested ligand-seed cap equals the target's corresponding known-active
count. If the cleaned pool is short, use the available ligands.

Adaptive grids use identity thresholds 10–70% in steps of 5 and cleaned
candidate-ligand thresholds 50–500 in steps of 50, with minimum K=2 and maximum
K=15. Ligand counts are assessed before MaxMin, not on final selected seeds.
The combined displayed rule begins with identity ≥55% and adds ranked neighbors
until at least 50 eligible candidate ligands are available or K=15 is reached.
Independent identity/ligand curves retain the publication choices ≥55% and ≥50.
These are retrospective selections on this benchmark, not externally validated
universal optima. Validation catches any changed policy selection.

EF is summarized by median partitions within target, median targets within
category, and median/Q25/Q75 across categories. S6 compares target-specific
known-actives (the sequence-reference condition), K=5 and Full domain. S7
and A2 display K=2,3,5,10,15 and Full domain. Figure 3 displays K=5, K=15,
Full domain and the combined rule. S8 displays K=5 and the three adaptive rules.
The seed cap is matched, but realized seed counts can differ.

## Complementarity: S4–S5

S4 unions and deduplicates the compounds recovered by each component method
at its inclusive percentile-99.5 score cutoff. It does not recalculate the
cutoff after merging or impose a fixed compound budget. The four historical
configurations and original category assignment, including CAH2's original
omission, are retained. Aggregation is target median within category/partition,
category median per partition, then median across partitions.

S5 fixes N=ceil(0.005 × complete evaluation-pool size), excluding known seeds.
Each method assigns consecutive full-pool integer ranks by descending float32
score, resolving score ties by lexicographic compound ID. Eligible combination
candidates are the deduplicated union above component P99.5 cutoffs. Mean-rank
fusion averages full-pool ranks, then orders by mean rank, best individual rank,
and compound ID. Exactly N unique compounds are retained. Labels are never
used for selection. The top four mean-rank configurations are selected using
the original plotting rule, which includes single methods. Summaries use
partition medians within target, then category medians and category balancing.

## Active-set preprocessing sensitivity: S3

The fixed panel is CYP2C9, CYP3A4, AA2AR, DRD3, PLK1, MK01, ESR1, ANDR,
AMPC, CAH2, BACE1 and FA10, spanning six categories. Original ECFP4/Butina
results are reused; FCFP4/Butina and Bemis–Murcko alternatives are calculated
using the original backgrounds and the five partitions. ECFP4, FCFP4,
Topological Torsion and ChemBERTa are the evaluated retrieval methods.

FCFP4/Butina uses radius 2, 1024 feature bits and Tanimoto 0.8
(distance cutoff 0.2). Cluster representatives maximize mean within-cluster
similarity, with original index as tie-break. Bemis–Murcko grouping uses
canonical non-stereochemical scaffolds, lexicographically ordered groups,
and seeded random representatives with random state seed+104729. Acyclic
compounds remain grouped by InChIKey. The vendored code defines all remaining
split, cap, background and ranking details. S3 displays category-balanced
cumulative EF at the top 0.5%.

## Retrieved compound distributions: S1 and A3

For each of 61 targets and five partitions, score the complete evaluation pool
using maximum ECFP4/Tanimoto similarity to the corresponding known actives.
The inclusive P99 cutoff defines TP (retrieved held-out actives), putative FP
(retrieved background) and putative TN (background strictly below cutoff).
Non-retrieved actives are FN and are not included in the TN curve. Background
labels do not imply experimentally confirmed inactivity.

Use the stored SMILES with RDKit 2025.03.3 and the original molecular-property
function: molecular weight, calculated logP, HBA and HBD, with HBA−HBD plotted.
No new salt, charge or tautomer standardization is introduced. Similarity bins
have width 0.02; MW bins 25 Da; cLogP bins 0.25; HBA−HBD bins 1. Normalize each
group separately within each target/partition, then average five partitions,
average targets equally within category, and average seven categories equally
for the global panel. This uses means of normalized bins, not pooled counts
or medians. A3 stops at category-level averages and shows similarity only.

Display limits are TI 0–1, MW 0–1200, cLogP −5–10 and HBA−HBD −5.5–20.5;
observations outside the property windows remain in the distributions and
their tail fractions are annotated. No display-window exclusion changes
the denominators.
