"""Check that the main figure is only a subset of the existing summaries."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import preview_category_balanced_spread as plotting
from plot_main_representation_comparison import MAIN_METHODS, select_main_methods


def synthetic_summary():
    rows = []
    for method in plotting.METHOD_ORDER:
        for percentile in plotting.PERCENTILES:
            row = {"method_label": method, "percentile": percentile, "n_categories": 7}
            for name, value in (
                ("median_EF", 10), ("q25_EF", 5), ("q75_EF", 15),
                ("min_EF", 2), ("max_EF", 20),
            ):
                row[name] = value
                row[f"log10_{name}"] = np.log10(value)
            rows.append(row)
    return pd.DataFrame(rows)


class MainRepresentationTests(unittest.TestCase):
    def test_statistics_are_unchanged(self):
        source = synthetic_summary()
        expected = source.loc[source.method_label.isin(MAIN_METHODS)]
        pd.testing.assert_frame_equal(select_main_methods(source), expected)

    def test_missing_method_rejected(self):
        source = synthetic_summary()
        with self.assertRaises(ValueError):
            select_main_methods(source.loc[source.method_label.ne(MAIN_METHODS[0])])

    def test_missing_percentile_rejected(self):
        source = synthetic_summary()
        missing = source.method_label.eq(MAIN_METHODS[0]) & source.percentile.eq(99.5)
        with self.assertRaises(ValueError):
            select_main_methods(source.loc[~missing])

    def test_duplicate_rows_rejected(self):
        source = synthetic_summary()
        selected_row = source.loc[source.method_label.eq(MAIN_METHODS[0])].iloc[:1]
        with self.assertRaises(ValueError):
            select_main_methods(pd.concat([source, selected_row]))

    def test_overlay_uses_four_original_colors_and_iqr_bands(self):
        with patch.object(plotting, "save_figure") as save:
            plotting.plot_overlay(synthetic_summary(), Path("unused"), methods=MAIN_METHODS)
            ax = save.call_args.args[0].axes[0]
            curves = ax.lines[:4]
            self.assertEqual(len(curves), 4)
            self.assertEqual(len(ax.collections), 4)
            for curve, method in zip(curves, MAIN_METHODS):
                self.assertEqual(curve.get_label(), plotting.METHOD_LABELS[method])
                self.assertEqual(curve.get_color(), plotting.METHOD_COLORS[method])

    def test_four_facets_include_iqr_and_full_range(self):
        with patch.object(plotting, "save_figure") as save:
            plotting.plot_facets(synthetic_summary(), Path("unused"), methods=MAIN_METHODS)
            axes = save.call_args.args[0].axes
            self.assertEqual(len(axes), 4)
            self.assertTrue(all(len(ax.collections) == 2 for ax in axes))

    def test_default_overlay_keeps_all_seven_methods(self):
        with patch.object(plotting, "save_figure") as save:
            plotting.plot_overlay(synthetic_summary(), Path("unused"))
            self.assertEqual(len(save.call_args.args[0].axes[0].collections), 7)

    def test_large_font_facets_preserve_values_and_enlarge_titles(self):
        with patch.object(plotting, "save_figure") as save:
            plotting.plot_facets(synthetic_summary(), Path("unused"), methods=MAIN_METHODS, font_size=16)
            fig = save.call_args.args[0]
            self.assertEqual(tuple(fig.get_size_inches()), (10, 8.8))
            for ax in fig.axes:
                self.assertEqual(ax.title.get_fontsize(), 16)
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), np.ones(6))


if __name__ == "__main__":
    unittest.main()
