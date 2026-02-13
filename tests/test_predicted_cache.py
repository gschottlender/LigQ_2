import tempfile
import unittest
from pathlib import Path

import pandas as pd

from query_processing.predicted_cache import ensure_provider_cache


class DummyProvider:
    provider_name = "dummy"

    def method_signature(self):
        return {
            "provider": "dummy",
            "search_representation": "toy_rep",
            "search_metric": "toy_metric",
        }

    def database_fingerprint(self, data_dir: Path) -> str:
        return "db_v1"

    def compute_for_protein(self, prot: str, known_binding: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"chem_comp_id": f"DUMMY_{prot}", "query_id": "QX", "similarity": 0.9, "smiles": "C"}]
        )


class TestPredictedCache(unittest.TestCase):
    def test_incremental_cache_reuse(self):
        provider = DummyProvider()
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            known = pd.DataFrame(
                [
                    {"uniprot_id": "P1", "chem_comp_id": "L1", "smiles": "CC"},
                    {"uniprot_id": "P2", "chem_comp_id": "L2", "smiles": "CCC"},
                ]
            )

            df1 = ensure_provider_cache(
                data_dir=data_dir,
                provider=provider,
                known_binding=known,
                proteins_needed={"P1"},
                force_rebuild_cache=False,
            )
            self.assertEqual(set(df1["uniprot_id"].unique()), {"P1"})

            df2 = ensure_provider_cache(
                data_dir=data_dir,
                provider=provider,
                known_binding=known,
                proteins_needed={"P1", "P2"},
                force_rebuild_cache=False,
            )
            self.assertEqual(set(df2["uniprot_id"].unique()), {"P1", "P2"})


if __name__ == "__main__":
    unittest.main()
