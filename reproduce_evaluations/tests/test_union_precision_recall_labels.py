"""Historical-union relabeling must preserve points and protocol differences."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from plot_historical_union_precision_recall import notebook_family_mapping
from notebook_analysis import plot_union_recovery_vs_purity_publication
from plot_fixed_budget_recovery_vs_purity import plot_policy


class UnionLabelTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame({
            "total_evaluation_actives_recovered_percent": [58.1, 61.1],
            "active_fraction_among_retrieved_percent": [57.8, 41.9],
            "label": ["ECFP4 (1024 bits)", "ECFP4 (1024 bits) + FCFP4 (1024 bits) + Topological Torsion"],
        })

    def test_relabeling_does_not_change_coordinates(self):
        rows = self.rows()
        before = rows.copy(deep=True)
        fig, ax = plot_union_recovery_vs_purity_publication(rows)
        self.assertEqual(ax.get_xlabel(), "Category-balanced median recall (%)")
        self.assertEqual(ax.get_ylabel(), "Category-balanced median precision (%)")
        actual = np.concatenate([collection.get_offsets() for collection in ax.collections])
        expected = rows[["total_evaluation_actives_recovered_percent", "active_fraction_among_retrieved_percent"]].to_numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["ECFP4", "ECFP4 + TT + FCFP4"])
        pd.testing.assert_frame_equal(rows, before)
        plt.close(fig)

    def test_typography_matches_fixed_budget_but_labels_keep_budget_difference(self):
        fixed = pd.DataFrame(dict(method_combination=["morgan_1024_r2"], recall_at_n=[0.581], precision_at_n=[0.578]))
        with patch.object(Figure, "savefig", autospec=True) as save:
            plot_policy(fixed, "mean", Path("unused"), 600)
            fixed_ax = save.call_args.args[0].axes[0]
        fig, union_ax = plot_union_recovery_vs_purity_publication(self.rows())
        self.assertEqual(union_ax.xaxis.label.get_fontsize(), fixed_ax.xaxis.label.get_fontsize())
        self.assertEqual(union_ax.get_legend().get_texts()[0].get_fontsize(), fixed_ax.get_legend().get_texts()[0].get_fontsize())
        self.assertEqual(fixed_ax.get_xlabel().replace("@N", ""), union_ax.get_xlabel())
        self.assertEqual(fixed_ax.get_ylabel().replace("@N", ""), union_ax.get_ylabel())
        plt.close(fig)

    def test_archived_mapping_is_read_without_executing_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.ipynb"
            path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [
                "raise RuntimeError('must not execute')\n",
                "agrupacion_subniveles = {'Category': {'Subset': ['target']}}\n",
            ]}]}))
            self.assertEqual(notebook_family_mapping(path), {"Category": {"Subset": ["target"]}})


if __name__ == "__main__":
    unittest.main()
