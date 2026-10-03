from __future__ import annotations

import unittest

import pandas as pd

from ligq2_evaluation.category_spread import summarize_category_spread


class CategorySpreadTests(unittest.TestCase):
    def test_summary_uses_one_median_per_category(self):
        frame = pd.DataFrame(
            {
                "method": ["A", "A", "A", "A", "A", "B", "B"],
                "percentile": [99.5] * 7,
                "familia": ["c1", "c1", "c2", "c3", "c4", "c1", "c2"],
                "ef": [1.0, 3.0, 4.0, 8.0, 100.0, 10.0, 20.0],
            }
        )

        category_values, spread = summarize_category_spread(
            frame,
            group_columns=["method", "percentile"],
            value_column="ef",
        )

        method_a_categories = category_values[category_values["method"] == "A"]
        self.assertEqual(method_a_categories["category_value"].tolist(), [2.0, 4.0, 8.0, 100.0])

        method_a = spread[spread["method"] == "A"].iloc[0]
        self.assertEqual(method_a["category_balanced_median"], 6.0)
        self.assertEqual(method_a["category_q25"], 3.5)
        self.assertEqual(method_a["category_q75"], 31.0)
        self.assertEqual(method_a["category_min"], 2.0)
        self.assertEqual(method_a["category_max"], 100.0)
        self.assertEqual(method_a["n_categories"], 4)

    def test_missing_columns_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            summarize_category_spread(
                pd.DataFrame({"method": ["A"]}),
                group_columns=["method", "percentile"],
                value_column="ef",
            )


if __name__ == "__main__":
    unittest.main()
