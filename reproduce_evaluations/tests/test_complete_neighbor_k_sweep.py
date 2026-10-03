from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "14_complete_neighbor_k_sweep.py"
SPEC = importlib.util.spec_from_file_location("complete_neighbor_k_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompleteNeighborKSweepTests(unittest.TestCase):
    def test_parse_counts_supports_ranges_and_deduplicates(self):
        self.assertEqual(MODULE.parse_counts("2-5,5,10,11-15"), [2, 3, 4, 5, 10, 11, 12, 13, 14, 15])

    def test_filter_component_keeps_only_requested_k_and_targets(self):
        frame = pd.DataFrame({
            "target": ["aa2ar", "akt1", "aa2ar"],
            "neighbor_count_sweep": [5, 5, 6],
            "value": [1, 2, 3],
        })
        observed = MODULE.filter_component(
            frame, count=5, selected_targets={"aa2ar"}
        )
        self.assertEqual(observed[["target", "neighbor_count_sweep", "value"]].to_dict("records"), [
            {"target": "aa2ar", "neighbor_count_sweep": 5, "value": 1}
        ])

    def test_validate_seed_grid_requires_every_requested_k(self):
        metrics = pd.DataFrame({
            "target": ["aa2ar"] * 3,
            "neighbor_count_sweep": [2, 3, 4],
            "percentile": [99.5, 99.5, 99.5],
        })
        metadata = metrics.drop(columns="percentile")
        self.assertEqual(MODULE.validate_seed_grid(metrics, metadata, [2, 3, 4]), {"aa2ar"})
        with self.assertRaisesRegex(ValueError, "No target contains"):
            MODULE.validate_seed_grid(metrics.iloc[:2], metadata.iloc[:2], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
