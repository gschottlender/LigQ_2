from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "plot_fixed_budget_recovery_vs_purity.py"
SPEC = importlib.util.spec_from_file_location("plot_fixed_budget_recovery_vs_purity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_summary() -> pd.DataFrame:
    rows = []
    for policy in ("best", "mean", "balanced"):
        for position, (combination, n_methods, recall, precision) in enumerate((
            ("morgan_1024_r2__target_seeds", 1, .60, .59),
            ("morgan_1024_r2__target_seeds + morgan_feature_1024_r2__target_seeds", 2, .63, .61),
            ("morgan_1024_r2__target_seeds + topological_torsion_rdkit_1024__target_seeds", 2, .62, .60),
            ("morgan_feature_1024_r2__target_seeds", 1, .58, .57),
            ("maccs__target_seeds", 1, .20, .19),
        )):
            rows.append({
                "policy": policy,
                "method_combination": combination,
                "n_methods": n_methods,
                "recall_at_n": recall,
                "precision_at_n": precision,
                "ef_at_n": 100.0 - position,
                "n_targets": 5,
            })
    return pd.DataFrame(rows)


class FixedBudgetRecoveryPurityPlotTests(unittest.TestCase):
    def test_selection_is_recovery_first_and_includes_single_methods(self):
        selected = MODULE.select_top_combinations(synthetic_summary(), "best", 4)
        self.assertEqual(selected["plot_rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(selected.iloc[0]["n_methods"], 2)
        self.assertIn(1, selected["n_methods"].tolist())
        self.assertEqual(selected.iloc[0]["recall_at_n"], .63)

    def test_validation_rejects_duplicate_strategy(self):
        frame = synthetic_summary()
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            MODULE.validate_summary(duplicate)

    def test_plot_writes_all_publication_formats(self):
        rows = MODULE.select_top_combinations(synthetic_summary(), "mean", 4)
        with tempfile.TemporaryDirectory() as directory:
            paths = MODULE.plot_policy(rows, "mean", Path(directory), dpi=80)
            self.assertEqual({path.suffix for path in paths}, {".png", ".pdf", ".svg"})
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))

    def test_pretty_combination_uses_compact_publication_names(self):
        value = "morgan_1024_r2__target_seeds + topological_torsion_rdkit_1024__target_seeds"
        self.assertEqual(MODULE.pretty_combination(value), "ECFP4 + TT")


if __name__ == "__main__":
    unittest.main()
