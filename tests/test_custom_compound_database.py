import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_compound_database import build_compound_database
from query_processing.ligand_providers import build_provider
from query_processing.query_processing_functions import build_query_ligand_results_parallel


class TestCustomCompoundDatabase(unittest.TestCase):
    def test_build_compound_database_from_csv_with_flexible_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "vendor.csv"
            pd.DataFrame(
                {
                    "compound": ["EXT1", "EXT2"],
                    "SMILES": ["CCO", "CCC"],
                }
            ).to_csv(input_path, index=False)

            db_root = build_compound_database(
                input_file=input_path,
                output_dir=root,
                base_name="vendor",
                default_rep_batch_size=2,
                default_rep_n_jobs=1,
                default_rep_chunksize=1,
            )

            self.assertTrue((db_root / "ligands.parquet").exists())
            self.assertTrue((db_root / "reps" / "morgan_1024_r2.dat").exists())
            self.assertTrue((db_root / "reps" / "morgan_1024_r2.meta.json").exists())

    def test_custom_provider_uses_custom_base_without_zinc_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            pdb_input = root / "pdb.csv"
            pd.DataFrame(
                {
                    "compound": ["L1", "L2"],
                    "SMILES": ["CCO", "CCN"],
                }
            ).to_csv(pdb_input, index=False)
            build_compound_database(
                input_file=pdb_input,
                output_dir=root,
                base_name="pdb_chembl",
                default_rep_batch_size=2,
                default_rep_n_jobs=1,
                default_rep_chunksize=1,
            )

            vendor_input = root / "vendor.csv"
            pd.DataFrame(
                {
                    "compound": ["EXT1", "EXT2"],
                    "SMILES": ["CCO", "CCCC"],
                }
            ).to_csv(vendor_input, index=False)
            build_compound_database(
                input_file=vendor_input,
                output_dir=root,
                base_name="vendor",
                default_rep_batch_size=2,
                default_rep_n_jobs=1,
                default_rep_chunksize=1,
            )

            provider = build_provider(
                provider_name="vendor",
                data_dir=root,
                search_representation="morgan_1024_r2",
                search_metric="tanimoto",
                search_threshold=0.3,
                search_threshold_max=None,
                cluster_threshold=0.8,
                search_per_iteration_topk=100,
                search_global_topk=1000,
            )
            known_binding = pd.DataFrame(
                [
                    {"uniprot_id": "P1", "chem_comp_id": "L1", "pchembl": 7.0},
                ]
            )

            results = provider.compute_for_protein("P1", known_binding)

            self.assertEqual(provider.provider_name, "vendor")
            self.assertFalse(results.empty)
            self.assertIn("chem_comp_id", results.columns)
            self.assertTrue(results["chem_comp_id"].astype(str).str.startswith("ZINC").sum() == 0)

    def test_block3_writes_predicted_ligands_and_summary_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"

            df_queries = pd.DataFrame([{"qseqid": "Q1"}])
            df_candidates_all = pd.DataFrame(
                [{"qseqid": "Q1", "sseqid": "P1", "search_type": "sequence"}]
            )
            known_db = pd.DataFrame(
                [{"uniprot_id": "P1", "chem_comp_id": "L1", "smiles": "CCO"}]
            )
            predicted_db = pd.DataFrame(
                [
                    {
                        "uniprot_id": "P1",
                        "chem_comp_id": "EXT1",
                        "query_id": "L1",
                        "similarity": 0.9,
                        "smiles": "CCO",
                    }
                ]
            )

            summary = build_query_ligand_results_parallel(
                df_queries=df_queries,
                df_candidates_all=df_candidates_all,
                known_db=known_db,
                predicted_db=predicted_db,
                output_dir=out_dir,
                predicted_score_col="similarity",
                predicted_threshold_min=0.3,
                predicted_threshold_max=None,
                predicted_filter_batch_size=10,
                njobs=1,
                chunk_size_queries=1,
            )

            self.assertTrue((out_dir / "search_results" / "Q1" / "predicted_ligands.tsv").exists())
            self.assertIn("n_predicted_ligands_sequence", summary.columns)


if __name__ == "__main__":
    unittest.main()
