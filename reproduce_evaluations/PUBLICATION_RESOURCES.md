# Published frozen evaluation resources

The benchmark is published at the revision in `resource_lock.json`.
The following additional historical bundle is now published at
[gschottlender/LigQ_2_evaluations](https://huggingface.co/datasets/gschottlender/LigQ_2_evaluations/tree/1dff255d8ed67e8994c41015bd31ddb2f359915c),
immutable revision `1dff255d8ed67e8994c41015bd31ddb2f359915c`.
All 20 remote input hashes and sizes were verified against the local inventory.
The original inventory's pre-publication status is a capture-time note;
`artifact_lock.json` records the current immutable publication and verification.

| Relative path | Size (MB) |
| --- | ---: |
| `frozen/binding_data.parquet` | 11.471 |
| `frozen/smiles_data.parquet` | 32.423 |
| `frozen/targets.csv` | 0.003 |
| `frozen/target_sequences.fasta` | 8.791 |
| `frozen/neighbor_blast_hits.csv` | 0.284 |
| `frozen/compound_data/pdb_chembl/ligands.parquet` | 55.410 |
| `frozen/compound_data/pdb_chembl/reps/ap_rdkit.dat` | 95.734 |
| `frozen/compound_data/pdb_chembl/reps/ap_rdkit.meta.json` | 0.000 |
| `frozen/compound_data/pdb_chembl/reps/chemberta_zinc_base_768.dat` | 1531.736 |
| `frozen/compound_data/pdb_chembl/reps/chemberta_zinc_base_768.meta.json` | 0.000 |
| `frozen/compound_data/pdb_chembl/reps/maccs.dat` | 20.942 |
| `frozen/compound_data/pdb_chembl/reps/maccs.meta.json` | 0.000 |
| `frozen/compound_data/pdb_chembl/reps/morgan_1024_r2.dat` | 127.645 |
| `frozen/compound_data/pdb_chembl/reps/morgan_1024_r2.meta.json` | 0.000 |
| `frozen/compound_data/pdb_chembl/reps/morgan_feature_1024_r2.dat` | 127.645 |
| `frozen/compound_data/pdb_chembl/reps/morgan_feature_1024_r2.meta.json` | 0.000 |
| `frozen/compound_data/pdb_chembl/reps/rdkit_1024.dat` | 127.645 |
| `frozen/compound_data/pdb_chembl/reps/rdkit_1024.meta.json` | 0.000 |
| `frozen/compound_data/pdb_chembl/reps/topological_torsion_rdkit_1024.dat` | 127.645 |
| `frozen/compound_data/pdb_chembl/reps/topological_torsion_rdkit_1024.meta.json` | 0.000 |

Total additional bundle: **2.267 GB**.
All SHA-256 values and benchmark file hashes are recorded in `resource_lock.json`.

## Publication and verification workflow

1. Run `python reproduce.py export-resources --inventory local_sources.json`.
2. Inspect `publication_export/frozen/` and the copied resource lock.
3. Run the explicitly separate maintainer upload command `python publish_resources.py`.
4. Its verified immutable commit is saved to `artifact_lock.json` and used by `download`.
5. Run `test_publication.py` for a fresh download and complete scientific reproduction
   before announcing full independent reproducibility; see the README for commands/status paths.

Do not rebuild ChemBERTa embeddings, reorder merged tables, change molecule
standardization, or substitute the live platform's BLAST neighbors.
The old embedding metadata does not record a model revision; its array hash
is authoritative. Generic BSI models, ZINC and predicted caches are not required.
Publishing scripts alone does not resolve missing frozen inputs.

Normal download/calculation/plot commands never upload. Only the explicitly
invoked maintainer `publish_resources.py` calls upload APIs. Local source locations
remain Git-ignored. Neither existing LigQ2 dataset was altered.
