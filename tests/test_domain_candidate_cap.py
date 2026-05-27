import tempfile
import unittest
from pathlib import Path

import pandas as pd

from query_processing.query_processing_functions import map_pfam_hits_to_candidate_proteins


class TestDomainCandidateCap(unittest.TestCase):
    def test_caps_each_query_pfam_by_blast_rank_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data"
            results_dir = data_dir / "results_databases"
            results_dir.mkdir(parents=True)

            pd.DataFrame(
                {
                    "uniprot_id": [f"P{i:02d}" for i in range(1, 7)] + ["P03"],
                    "pfam_id": ["PF1"] * 6 + ["PF2"],
                }
            ).to_parquet(results_dir / "protein_domains.parquet", index=False)

            df_hmmer = pd.DataFrame(
                {
                    "qseqid": ["Q1", "Q1"],
                    "pfam_id": ["PF1", "PF2"],
                }
            )
            df_blast_ranked = pd.DataFrame(
                {
                    "qseqid": ["Q1"] * 6,
                    "sseqid": [f"P{i:02d}" for i in range(1, 7)],
                    "bitscore": [10, 60, 50, 40, 30, 20],
                    "evalue": [1e-4] * 6,
                    "pident": [90] * 6,
                    "qcov": [0.9] * 6,
                    "scov": [0.8] * 6,
                }
            )

            got = map_pfam_hits_to_candidate_proteins(
                df_hmmer=df_hmmer,
                data_dir=data_dir,
                temp_results_dir=root / "tmp",
                save_full_hits=False,
                save_candidates=False,
                max_candidates_per_domain=3,
                df_blast_ranked=df_blast_ranked,
            )

            self.assertEqual(got["search_type"].unique().tolist(), ["domain"])
            self.assertEqual(got["sseqid"].tolist(), ["P02", "P03", "P04"])


if __name__ == "__main__":
    unittest.main()
