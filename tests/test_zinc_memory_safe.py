import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from compound_processing.compound_helpers import Representation
from compound_processing.compound_database_search import search_similar_in_zinc


class TestZincMemorySafe(unittest.TestCase):
    def test_global_topk_is_applied_before_return(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            query_memmap = np.memmap(
                root / "query.dat",
                dtype=np.uint8,
                mode="w+",
                shape=(1, 1),
            )
            query_memmap[:] = np.array([[255]], dtype=np.uint8)
            query_memmap.flush()

            target_memmap = np.memmap(
                root / "target.dat",
                dtype=np.uint8,
                mode="w+",
                shape=(6, 1),
            )
            target_memmap[:] = np.array([[255], [254], [252], [248], [240], [224]], dtype=np.uint8)
            target_memmap.flush()

            meta = {
                "dim": 8,
                "dtype": "uint8",
                "packed_bits": True,
                "packed_dim": 1,
                "n_ligands": 1,
            }
            rep_ref = Representation(
                name="test",
                memmap=query_memmap,
                meta=meta,
                id_to_idx={"Q": 0},
            )
            rep_zinc = Representation(
                name="test",
                memmap=target_memmap,
                meta={**meta, "n_ligands": 6},
                id_to_idx={f"Z{i}": i for i in range(6)},
            )
            store_zinc = SimpleNamespace(
                ligands=pd.DataFrame(
                    {
                        "lig_idx": list(range(6)),
                        "chem_comp_id": [f"Z{i}" for i in range(6)],
                        "smiles": ["C"] * 6,
                    }
                )
            )

            got = search_similar_in_zinc(
                query_ids=["Q"],
                store_ref=SimpleNamespace(),
                rep_ref=rep_ref,
                store_zinc=store_zinc,
                rep_zinc=rep_zinc,
                tanimoto_threshold=0.0,
                q_batch_size=1,
                zinc_chunk_size=2,
                n_jobs=1,
                global_topk=3,
            )

            self.assertEqual(len(got), 3)
            self.assertEqual(got["chem_comp_id"].tolist(), ["Z0", "Z1", "Z2"])
            self.assertEqual(got["tanimoto"].tolist(), sorted(got["tanimoto"], reverse=True))


if __name__ == "__main__":
    unittest.main()
