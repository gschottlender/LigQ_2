import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compound_processing.compound_helpers import build_rdkit_representation, rdkit_fp_bits


class TestRDKitRepresentations(unittest.TestCase):
    def test_rdkit_fp_bits_supported_kinds(self):
        smiles = "CCO"

        for fp_kind in ("ap", "topological_torsion", "rdkit"):
            arr = rdkit_fp_bits(smiles, fp_kind=fp_kind, n_bits=128)
            self.assertIsNotNone(arr)
            self.assertEqual(arr.shape, (128,))
            self.assertEqual(arr.dtype, np.uint8)

        maccs = rdkit_fp_bits(smiles, fp_kind="maccs", n_bits=167)
        self.assertIsNotNone(maccs)
        self.assertEqual(maccs.shape, (167,))

    def test_build_rdkit_representation_maccs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pd.DataFrame(
                {
                    "chem_comp_id": ["A", "B"],
                    "smiles": ["CCO", "invalid_smiles"],
                    "inchikey": [None, None],
                    "lig_idx": [0, 1],
                }
            ).to_parquet(root / "ligands.parquet", index=False)

            build_rdkit_representation(
                root=root,
                fp_kind="maccs",
                n_bits=1024,
                batch_size=2,
                name="maccs_test",
                n_jobs=1,
                chunksize=1,
            )

            data_path = root / "reps" / "maccs_test.dat"
            meta_path = root / "reps" / "maccs_test.meta.json"
            self.assertTrue(data_path.exists())
            self.assertTrue(meta_path.exists())

            with meta_path.open() as f:
                meta = json.load(f)

            self.assertEqual(meta["dim"], 167)
            self.assertEqual(meta["packed_dim"], 21)
            self.assertEqual(meta["fingerprint_type"], "maccs")
            self.assertEqual(meta["failed_smiles"], 1)


if __name__ == "__main__":
    unittest.main()
