from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "analyze_identity_ligand_rescue_policy.py"
SPEC = importlib.util.spec_from_file_location("identity_ligand_rescue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IdentityLigandRescueTests(unittest.TestCase):
    def test_identity_prefix_stops_before_first_neighbor_below_floor(self):
        selected = MODULE.identity_prefix_k(
            [90, 80, 61, 57, 52, 49], floor=55, min_k=2, max_k=15
        )
        self.assertEqual(selected, 4)

    def test_sparse_ligand_evidence_extends_identity_prefix(self):
        identity_k = np.array([2, 7, 10])
        budget_k = np.array([5, 4, 15])
        selected = MODULE.combine_identity_and_budget_k(identity_k, budget_k)
        np.testing.assert_array_equal(selected, [5, 7, 15])

    def test_unreached_budget_falls_back_to_k15(self):
        candidates = np.array([[10, 20, 30, 40]])
        counts = np.array([2, 3, 4, 5])
        selected, reached = MODULE.ligand_budget_k(
            candidates, counts, budget=100, min_k=2, max_k=5
        )
        np.testing.assert_array_equal(selected, [5])
        np.testing.assert_array_equal(reached, [False])


if __name__ == "__main__":
    unittest.main()
