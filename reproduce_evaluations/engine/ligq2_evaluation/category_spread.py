"""Category-balanced summaries and between-category spread statistics."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def summarize_category_spread(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    category_column: str = "familia",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one value per category plus category-balanced spread statistics.

    Values are first collapsed to one median per category and comparison group.
    The spread table then reports the median, interquartile range, full range,
    and category count across those category-level medians.
    """

    group_columns = list(group_columns)
    required = set(group_columns) | {value_column, category_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for category summary: {missing}")

    work = frame[[*group_columns, category_column, value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[category_column, value_column])
    if work.empty:
        raise ValueError("No finite category values are available for summarization")

    category_values = (
        work.groupby([*group_columns, category_column], as_index=False, observed=True)[
            value_column
        ]
        .median()
        .rename(columns={value_column: "category_value"})
    )
    spread = (
        category_values.groupby(group_columns, as_index=False, observed=True)
        .agg(
            category_balanced_median=("category_value", "median"),
            category_q25=("category_value", lambda values: values.quantile(0.25)),
            category_q75=("category_value", lambda values: values.quantile(0.75)),
            category_min=("category_value", "min"),
            category_max=("category_value", "max"),
            n_categories=("category_value", "count"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return category_values, spread
