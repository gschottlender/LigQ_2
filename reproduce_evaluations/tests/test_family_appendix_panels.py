"""Plot-only supplementary panels must preserve values and counting units."""
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_family_appendices as appendix


def curves():
    return pd.DataFrame([
        {"protein_category": category, "strategy": strategy, "percentile": percentile,
         "category_median_EF_cumulative": 2.+index, "n_targets": 2}
        for category in appendix.CATEGORY_ORDER
        for strategy in appendix.NEIGHBOR_STRATEGIES
        for index, percentile in enumerate(appendix.NEIGHBOR_PERCENTILES)
    ])


def histograms():
    return pd.DataFrame([
        {"protein_family": category, "property": "similarity", "group": group,
         "bin": index, "bin_left": index/50, "bin_right": (index+1)/50,
         "fraction": .02, "n_targets": 2}
        for category in appendix.CATEGORY_ORDER for group in appendix.GROUPS
        for index in range(50)
    ])


class FamilyAppendixTests(unittest.TestCase):
    def test_one_figure_contains_all_seven_categories(self):
        frame = curves()
        original = frame.copy(deep=True)
        fig = appendix.plot_ef(
            frame, series="strategy", value="category_median_EF_cumulative",
            order=appendix.NEIGHBOR_STRATEGIES, percentiles=appendix.NEIGHBOR_PERCENTILES,
            colors={s:"blue" for s in appendix.NEIGHBOR_STRATEGIES},
            labels={s:s for s in appendix.NEIGHBOR_STRATEGIES},
            counts={c:2 for c in appendix.CATEGORY_ORDER}, title="Test", note="Test",
        )
        try:
            self.assertEqual(len(fig.axes), 9)
            self.assertEqual(sum(ax.axison for ax in fig.axes), 7)
            self.assertEqual(len(fig.legends), 1)
            for ax in fig.axes[:7]:
                self.assertEqual(len(ax.collections), 0)
                for line in ax.lines[:6]:
                    np.testing.assert_array_equal(line.get_ydata(), np.arange(2., 7.))
            pd.testing.assert_frame_equal(frame, original)
        finally:
            plt.close(fig)

    def test_duplicate_or_incomplete_curves_fail(self):
        args = ("strategy", "category_median_EF_cumulative",
                appendix.NEIGHBOR_STRATEGIES, appendix.NEIGHBOR_PERCENTILES)
        frame = curves()
        with self.assertRaises(ValueError):
            appendix.validate_curves(pd.concat([frame, frame.iloc[:1]]), *args)
        with self.assertRaises(ValueError):
            appendix.validate_curves(frame.iloc[1:], *args)

    def test_all_three_histogram_groups_are_normalized(self):
        frame = histograms()
        original = frame.copy(deep=True)
        result = appendix.validate_histograms(frame)
        self.assertEqual(len(result), 7*3*50)
        pd.testing.assert_frame_equal(frame, original)
        fig = appendix.plot_similarity(result)
        try:
            self.assertEqual(sum(ax.axison for ax in fig.axes), 7)
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual(len(fig.legends[0].get_texts()), 3)
        finally:
            plt.close(fig)

    def test_invalid_normalization_or_missing_tn_fails(self):
        frame = histograms()
        with self.assertRaises(ValueError):
            appendix.validate_histograms(frame[frame.group.ne("putative_TN")])
        frame.loc[0, "fraction"] = .5
        with self.assertRaises(ValueError):
            appendix.validate_histograms(frame)


if __name__ == "__main__":
    unittest.main()
