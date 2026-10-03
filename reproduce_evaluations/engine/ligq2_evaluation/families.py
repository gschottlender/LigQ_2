from __future__ import annotations

import pandas as pd


def build_target_family_map(groups: dict) -> pd.DataFrame:
    rows = []
    for family, subgroups in groups.items():
        for subfamily, targets in subgroups.items():
            rows.extend({"target": target, "familia": family, "subfamilia": subfamily} for target in targets)
    return pd.DataFrame(rows)


def add_family_annotations(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    return df.merge(build_target_family_map(groups), on="target", how="left")


def compute_family_balanced_ef(df, groups, ef_col="EF_band", group_level="familia"):
    annotated = add_family_annotations(df, groups)
    annotated = annotated[annotated[group_level].notna() & annotated[ef_col].notna()].copy()
    by_group = (annotated.groupby(["method", "percentile", group_level], as_index=False)
                .agg(EF_group=(ef_col, "median"), n_targets=("target", "nunique")))
    final = (by_group.groupby(["method", "percentile"], as_index=False)
             .agg(EF_balanced=("EF_group", "median"), n_groups=(group_level, "nunique")))
    return final, by_group, annotated


def compute_family_balanced_ef_both(df, groups):
    band, band_groups, annotated = compute_family_balanced_ef(df, groups, "EF_band")
    cumulative, cumulative_groups, _ = compute_family_balanced_ef(df, groups, "EF_cumulative")
    band = band.rename(columns={"EF_balanced": "EF_balanced_band", "n_groups": "n_groups_band"})
    cumulative = cumulative.rename(columns={"EF_balanced": "EF_balanced_cumulative", "n_groups": "n_groups_cumulative"})
    band_groups = band_groups.rename(columns={"EF_group": "EF_group_band", "n_targets": "n_targets_band"})
    cumulative_groups = cumulative_groups.rename(columns={"EF_group": "EF_group_cumulative", "n_targets": "n_targets_cumulative"})
    return (band.merge(cumulative, on=["method", "percentile"], how="outer"),
            band_groups.merge(cumulative_groups, on=["method", "percentile", "familia"], how="outer"), annotated)
