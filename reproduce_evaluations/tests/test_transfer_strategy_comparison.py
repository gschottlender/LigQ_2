from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "plot_ecfp4_transfer_strategy_comparison.py"
SPEC = importlib.util.spec_from_file_location("transfer_strategy_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TransferStrategyComparisonTests(unittest.TestCase):
    def _rows(self) -> pd.DataFrame:
        rows = []
        values = {
            "cdk2": {MODULE.SEQUENCE: 30.0, MODULE.NEAREST_K5: 20.0, MODULE.FULL_DOMAIN: 10.0},
            "thrb": {MODULE.SEQUENCE: 50.0, MODULE.NEAREST_K5: 40.0, MODULE.FULL_DOMAIN: 20.0},
        }
        for target, strategies in values.items():
            for strategy, base in strategies.items():
                for seed_index, seed in enumerate(MODULE.SEEDS):
                    for percentile in MODULE.RAW_PERCENTILES:
                        rows.append(
                            {
                                "target": target,
                                "strategy": strategy,
                                "seed": seed,
                                "percentile": percentile,
                                "EF_cumulative": base + seed_index,
                                "n_pool_eval": 1000,
                                "n_pos_eval": 100,
                            }
                        )
        return pd.DataFrame(rows)

    def test_alignment_requires_common_complete_targets_and_equal_pools(self):
        rows = self._rows()
        sequence = rows[rows.strategy.eq(MODULE.SEQUENCE)]
        transferred = rows[~rows.strategy.eq(MODULE.SEQUENCE)]
        combined, targets = MODULE.align_and_validate_rows(sequence, transferred)
        self.assertEqual(targets, ["cdk2", "thrb"])
        self.assertEqual(len(combined), 2 * 3 * 5 * len(MODULE.RAW_PERCENTILES))

        broken = transferred.copy()
        index = broken.index[0]
        broken.loc[index, "n_pool_eval"] = 999
        with self.assertRaisesRegex(ValueError, "pool"):
            MODULE.align_and_validate_rows(sequence, broken)

    def test_summary_uses_partition_target_family_order(self):
        target, family, spread = MODULE.summarize(self._rows())
        cdk2 = target[
            target.target.eq("cdk2")
            & target.strategy.eq(MODULE.SEQUENCE)
            & target.percentile.eq(98.0)
        ].iloc[0]
        self.assertEqual(cdk2["median_partition_EF_cumulative"], 32.0)
        self.assertEqual(cdk2["n_partitions"], 5)

        sequence = spread[
            spread.strategy.eq(MODULE.SEQUENCE) & spread.percentile.eq(98.0)
        ].iloc[0]
        self.assertEqual(sequence["category_balanced_median"], 42.0)
        self.assertEqual(sequence["category_q25"], 37.0)
        self.assertEqual(sequence["category_q75"], 47.0)
        self.assertEqual(sequence["n_categories"], 2)
        self.assertEqual(family[family.strategy.eq(MODULE.SEQUENCE)]["n_targets"].max(), 1)


if __name__ == "__main__":
    unittest.main()
