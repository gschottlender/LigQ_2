import tempfile
import unittest
from pathlib import Path

import pandas as pd

from query_processing.query_processing_functions import _read_parquet_rows_for_uniprot_ids


class TestParquetEmptySchema(unittest.TestCase):
    def test_no_matches_preserves_parquet_columns(self):
        with tempfile.TemporaryDirectory() as td:
            parquet_path = Path(td) / "predicted.parquet"
            pd.DataFrame(
                {
                    "uniprot_id": ["P1"],
                    "chem_comp_id": ["Z1"],
                    "tanimoto": [0.7],
                    "smiles": ["CCO"],
                }
            ).to_parquet(parquet_path, index=False)

            got = _read_parquet_rows_for_uniprot_ids(parquet_path, ["P2"])

            self.assertTrue(got.empty)
            self.assertEqual(
                got.columns.tolist(),
                ["uniprot_id", "chem_comp_id", "tanimoto", "smiles"],
            )

    def test_empty_id_list_preserves_parquet_columns(self):
        with tempfile.TemporaryDirectory() as td:
            parquet_path = Path(td) / "predicted.parquet"
            pd.DataFrame(
                {
                    "uniprot_id": ["P1"],
                    "chem_comp_id": ["Z1"],
                }
            ).to_parquet(parquet_path, index=False)

            got = _read_parquet_rows_for_uniprot_ids(parquet_path, [])

            self.assertTrue(got.empty)
            self.assertEqual(got.columns.tolist(), ["uniprot_id", "chem_comp_id"])


if __name__ == "__main__":
    unittest.main()
