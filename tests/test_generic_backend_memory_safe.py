import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compound_processing.backend import SearchRequest, search
from compound_processing.compound_helpers import Representation


class DummyStore:
    def __init__(self, ligands: pd.DataFrame):
        self.ligands = ligands


class TestGenericBackendMemorySafe(unittest.TestCase):
    def test_global_topk_applies_to_cosine_before_metadata_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            query_memmap = np.memmap(
                root / "query_float.dat",
                dtype=np.float32,
                mode="w+",
                shape=(1, 2),
            )
            query_memmap[:] = np.array([[1.0, 0.0]], dtype=np.float32)
            query_memmap.flush()

            target_memmap = np.memmap(
                root / "target_float.dat",
                dtype=np.float32,
                mode="w+",
                shape=(5, 2),
            )
            target_memmap[:] = np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.7, 0.3],
                    [0.6, 0.4],
                ],
                dtype=np.float32,
            )
            target_memmap.flush()

            q_rep = Representation(
                name="float",
                memmap=query_memmap,
                meta={"dim": 2, "dtype": "float32", "packed_bits": False, "n_ligands": 1},
                id_to_idx={"Q": 0},
            )
            target_rep = Representation(
                name="float",
                memmap=target_memmap,
                meta={"dim": 2, "dtype": "float32", "packed_bits": False, "n_ligands": 5},
                id_to_idx={f"T{i}": i for i in range(5)},
            )
            target_store = DummyStore(
                pd.DataFrame(
                    {
                        "lig_idx": list(range(5)),
                        "chem_comp_id": [f"T{i}" for i in range(5)],
                        "smiles": ["C"] * 5,
                    }
                )
            )

            got = search(
                SearchRequest(
                    query_ids=["Q"],
                    store_ref=DummyStore(pd.DataFrame()),
                    rep_ref=q_rep,
                    store_target=target_store,
                    rep_target=target_rep,
                    metric="cosine",
                    mode="threshold",
                    threshold=0.0,
                    device="cpu",
                    q_batch_size=1,
                    target_chunk_size=2,
                    global_topk=3,
                )
            )

            self.assertEqual(len(got), 3)
            self.assertEqual(got["chem_comp_id"].tolist(), ["T0", "T1", "T2"])
            self.assertEqual(
                got["similarity"].tolist(),
                sorted(got["similarity"].tolist(), reverse=True),
            )


if __name__ == "__main__":
    unittest.main()
