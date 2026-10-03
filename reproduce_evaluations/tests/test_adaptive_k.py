from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "15_analyze_adaptive_k.py"
SPEC = importlib.util.spec_from_file_location("analyze_adaptive_k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_rows() -> pd.DataFrame:
    rows = []
    clean = {5: 30, 6: 45, 7: 55, 8: 80, 9: 90, 10: 110,
             11: 115, 12: 120, 13: 125, 14: 130, 15: 135}
    for count in range(2, 16):
        rows.append({
            "target": "akt1",
            "seed": 42,
            "neighbor_count_sweep": count,
            "actual_neighbor_count": count,
            "actual_boundary_pident": 90 - 5 * count,
            "nominal_k_boundary_pident": 90 - 5 * count,
            "n_neighbor_candidates_clean": clean.get(count, 10 * count),
            "k_requested": 75,
            "k_effective": min(75, clean.get(count, 10 * count)),
        })
    return pd.DataFrame(rows)


class AdaptiveKTests(unittest.TestCase):
    def test_identity_policy_stops_before_first_neighbor_below_floor(self):
        identities = [90, 85, 80, 75, 70, 60, 55, 49, 45]
        count, reason, excluded = MODULE.choose_identity_k(
            identities, floor=50, min_k=5, max_k=15
        )
        self.assertEqual(count, 7)
        self.assertEqual(reason, "next_neighbor_below_identity_floor")
        self.assertEqual(excluded, 49)

    def test_identity_policy_uses_all_available_when_list_is_short(self):
        count, reason, excluded = MODULE.choose_identity_k(
            [90, 85, 80, 75, 70, 65, 60], floor=50, min_k=5, max_k=15
        )
        self.assertEqual(count, 7)
        self.assertEqual(reason, "all_available_neighbors_used")
        self.assertTrue(pd.isna(excluded))

    def test_ligand_budget_uses_first_k_or_k15_fallback(self):
        rows = synthetic_rows()
        self.assertEqual(
            MODULE.choose_ligand_budget_k(rows, budget=75, min_k=5, max_k=15),
            (8, "ligand_budget_reached", True),
        )
        self.assertEqual(
            MODULE.choose_ligand_budget_k(rows, budget=200, min_k=5, max_k=15),
            (15, "maximum_k_fallback_budget_not_reached", False),
        )

    def test_policy_choice_does_not_receive_or_use_ef_values(self):
        metadata = synthetic_rows()
        blast = pd.DataFrame({
            "target": ["akt1"] * 15,
            "neighbor_rank": range(1, 16),
            "pident": [90, 85, 80, 75, 70, 60, 55, 49, 45, 44, 43, 42, 41, 40, 39],
        })
        choices = MODULE.build_policy_choices(
            metadata, blast, min_k=5, max_k=15,
            identity_floors=[50], ligand_budgets=[50, 75, 100],
            include_combined=False,
        ).set_index("policy")
        self.assertEqual(choices.loc["identity_floor_50", "selected_k"], 7)
        self.assertEqual(choices.loc["ligand_budget_50", "selected_k"], 7)
        self.assertEqual(choices.loc["ligand_budget_75", "selected_k"], 8)
        self.assertEqual(choices.loc["ligand_budget_100", "selected_k"], 10)
        self.assertEqual(choices.loc["matched_target_seed_budget", "selected_k"], 8)

    def test_multiple_policies_may_select_the_same_k(self):
        choices = pd.DataFrame({
            "target": ["akt1", "akt1"],
            "seed": [42, 42],
            "policy": ["identity_floor_50", "ligand_budget_50"],
            "selected_k": [7, 7],
        })
        metrics = pd.DataFrame({
            "target": ["akt1", "akt1"],
            "seed": [42, 42],
            "neighbor_count_sweep": [7, 7],
            "percentile": [99.5, 99.0],
            "EF_band": [2.0, 1.5],
            "EF_cumulative": [2.0, 1.8],
        })
        selected = MODULE.attach_selected_results(choices, metrics)
        self.assertEqual(len(selected), 4)
        self.assertFalse(
            selected.duplicated(["target", "seed", "policy", "percentile"]).any()
        )


if __name__ == "__main__":
    unittest.main()
