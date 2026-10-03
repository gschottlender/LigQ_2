from __future__ import annotations
import ast
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import re
import textwrap


# Source: analizar_EF.ipynb cell 1




def prepare_family_balanced_plot_df(
    df,
    group_col=None,
    across_group_agg="median",
):
    """
    Normalize a DataFrame for plotting family-balanced EF.

    Accept two input formats:

    1) An already summarized DataFrame (for example, df_family_summary) with columns such as:
       - method
       - percentile
       - method_label
       - metric_eval
       - EF_balanced_band
       - EF_balanced_cumulative
       - n_groups_band / n_groups_cumulative

    2) A detailed group/family DataFrame (for example, df_family_groups) with columns such as:
       - method
       - percentile
       - familia or subfamilia (legacy schema fields)
       - EF_group_band
       - EF_group_cumulative
       - method_label
       - metric_eval

    Always return a standardized DataFrame with:
       - method
       - percentile
       - method_label
       - metric_eval
       - EF_balanced_band
       - EF_balanced_cumulative
       - n_groups_band
       - n_groups_cumulative
    """
    df = df.copy()

    # -----------------------------
    # Case 1: already summarized
    # -----------------------------
    balanced_cols = {"EF_balanced_band", "EF_balanced_cumulative"}
    if balanced_cols.issubset(df.columns):
        out = df.copy()

        if "n_groups_band" not in out.columns:
            out["n_groups_band"] = np.nan
        if "n_groups_cumulative" not in out.columns:
            out["n_groups_cumulative"] = np.nan

        out["percentile"] = pd.to_numeric(out["percentile"], errors="coerce")
        out["EF_balanced_band"] = pd.to_numeric(out["EF_balanced_band"], errors="coerce")
        out["EF_balanced_cumulative"] = pd.to_numeric(out["EF_balanced_cumulative"], errors="coerce")
        out["n_groups_band"] = pd.to_numeric(out["n_groups_band"], errors="coerce")
        out["n_groups_cumulative"] = pd.to_numeric(out["n_groups_cumulative"], errors="coerce")

        return out

    # -----------------------------
    # Case 2: detailed by family/subfamily
    # -----------------------------
    group_cols_available = [c for c in ["familia", "subfamilia"] if c in df.columns]
    if group_col is None:
        if len(group_cols_available) == 0:
            raise ValueError(
                "Could not infer the group column. Expected the legacy field "
                "'familia' or 'subfamilia', or already summarized balanced columns."
            )
        group_col = group_cols_available[0]

    required_group_cols = {
        "method",
        "percentile",
        "method_label",
        "metric_eval",
        group_col,
        "EF_group_band",
        "EF_group_cumulative",
    }
    missing = required_group_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columns required to summarize the DataFrame by group are missing: {sorted(missing)}")

    def _aggfunc(series, agg_name):
        if agg_name == "median":
            return series.median()
        elif agg_name == "mean":
            return series.mean()
        else:
            raise ValueError("across_group_agg must be 'median' or 'mean'")

    rows = []
    group_keys = ["method", "percentile", "method_label", "metric_eval"]

    for keys, g in df.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys))

        band_vals = pd.to_numeric(g["EF_group_band"], errors="coerce").dropna()
        cum_vals = pd.to_numeric(g["EF_group_cumulative"], errors="coerce").dropna()

        row["EF_balanced_band"] = _aggfunc(band_vals, across_group_agg) if len(band_vals) > 0 else np.nan
        row["EF_balanced_cumulative"] = _aggfunc(cum_vals, across_group_agg) if len(cum_vals) > 0 else np.nan

        row["n_groups_band"] = g.loc[g["EF_group_band"].notna(), group_col].nunique()
        row["n_groups_cumulative"] = g.loc[g["EF_group_cumulative"].notna(), group_col].nunique()

        rows.append(row)

    out = pd.DataFrame(rows)

    out["percentile"] = pd.to_numeric(out["percentile"], errors="coerce")
    out["EF_balanced_band"] = pd.to_numeric(out["EF_balanced_band"], errors="coerce")
    out["EF_balanced_cumulative"] = pd.to_numeric(out["EF_balanced_cumulative"], errors="coerce")
    out["n_groups_band"] = pd.to_numeric(out["n_groups_band"], errors="coerce")
    out["n_groups_cumulative"] = pd.to_numeric(out["n_groups_cumulative"], errors="coerce")

    return out


def plot_family_balanced_ef_clean(
    df,
    ef_mode="band",  # "band" or "cumulative"
    selected_methods=None,
    percentile_order=(99.5, 99.0, 98.5, 98.0, 95.0, 90.0, 80.0, 50.0),
    pretty_labels=None,
    use_log10=False,
    across_group_agg="median",

    # Customizable labels
    x_label=None,
    y_label=None,
    x_tick_labels=None,

    title=None,
    figsize=(8.2, 5.2),
    dpi=300,
    save_dpi=600,
    linewidth=2.4,
    markersize=6,
    legend_ncol=2,
    save_prefix=None,
):
    """
    Plot family-balanced EF using the new format.

    Parameters
    ----------
    df : pd.DataFrame
        May be:
        - df_family_summary, already summarized, or
        - df_family_groups / ef_por_familia, with group-level details.

    ef_mode : str
        - "band": use EF_balanced_band and show percentile bands.
        - "cumulative": use EF_balanced_cumulative and show cumulative
          percentile thresholds.

    selected_methods : list or None
        Methods to include.

    percentile_order : tuple or list
        Percentile order on the X axis.

    pretty_labels : dict or None
        Mapping used to customize method names.

    use_log10 : bool
        If True, transform EF with log10.

    across_group_agg : str
        Aggregation used when the DataFrame contains detailed results by
        family, group, or category.

    x_label : str or None
        Custom X-axis label. If None, use the automatic label for ef_mode.

    y_label : str or None
        Custom Y-axis label. If None, use an automatic label based on use_log10.

    x_tick_labels : list, tuple, dict or None
        Custom X-axis tick labels.

        May be:

        - A list with the same length as percentile_order:
          ["99.5", "99", "98.5", ...]

        - A mapping from percentile to label:
          {
              99.5: "Top 0.5%",
              99.0: "Top 1%",
              ...
          }

        If None, labels are generated automatically.

    title : str or None
        Plot title. If None, generate it automatically. Use title="" to omit it.

    save_prefix : str or None
        Prefix for saving PNG and PDF versions.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    if ef_mode not in {"band", "cumulative"}:
        raise ValueError("ef_mode must be 'band' or 'cumulative'")

    plot_df = prepare_family_balanced_plot_df(
        df,
        across_group_agg=across_group_agg
    ).copy()

    if selected_methods is not None:
        plot_df = plot_df[
            plot_df["method_label"].isin(selected_methods)
        ].copy()

    # ---------------------------------------------------------
    # Automatic columns and labels based on the EF type
    # ---------------------------------------------------------
    if ef_mode == "band":
        y_col = "EF_balanced_band"
        default_x_label = "Percentile band"

        if title is None:
            title = (
                "Family-balanced enrichment factor "
                "across percentile bands"
            )

    else:
        y_col = "EF_balanced_cumulative"
        default_x_label = "Percentile threshold"

        if title is None:
            title = (
                "Family-balanced enrichment factor "
                "across cumulative percentile thresholds"
            )

    plot_df = plot_df[plot_df[y_col].notna()].copy()

    plot_df["percentile"] = pd.Categorical(
        plot_df["percentile"],
        categories=list(percentile_order),
        ordered=True
    )

    plot_df = plot_df.sort_values(
        ["method_label", "percentile"]
    )

    # ---------------------------------------------------------
    # X-axis label helper functions
    # ---------------------------------------------------------
    def _fmt_percentile(p):
        p = float(p)

        if p.is_integer():
            return str(int(p))

        return str(p)

    def _make_band_labels(percentiles):
        """
        Example:
        (99.5, 99.0, 98.5, 98.0, 95.0)
        ->
        ['≥99.5', '99.5–99', '99–98.5', '98.5–98', '98–95']
        """
        labels = []

        for i, p in enumerate(percentiles):
            if i == 0:
                labels.append(f"≥{_fmt_percentile(p)}")
            else:
                previous_p = percentiles[i - 1]
                labels.append(
                    f"{_fmt_percentile(previous_p)}–"
                    f"{_fmt_percentile(p)}"
                )

        return labels

    def _make_cumulative_labels(percentiles):
        """
        For cumulative mode, display the percentile thresholds directly.
        """
        return [
            _fmt_percentile(p)
            for p in percentiles
        ]

    def _resolve_x_tick_labels(
        custom_labels,
        percentiles,
        default_labels
    ):
        """
        Resolve custom X-axis labels.

        custom_labels may be:
        - None
        - list/tuple
        - dict
        """
        if custom_labels is None:
            return default_labels

        if isinstance(custom_labels, dict):
            return [
                str(
                    custom_labels.get(
                        p,
                        custom_labels.get(
                            float(p),
                            default_label
                        )
                    )
                )
                for p, default_label in zip(
                    percentiles,
                    default_labels
                )
            ]

        if isinstance(custom_labels, (list, tuple)):
            if len(custom_labels) != len(percentiles):
                raise ValueError(
                    "x_tick_labels must have the same length as "
                    "percentile_order."
                )

            return [
                str(label)
                for label in custom_labels
            ]

        raise TypeError(
            "x_tick_labels must be None, a list, a tuple, or a dictionary."
        )

    # ---------------------------------------------------------
    # Y-axis transformation
    # ---------------------------------------------------------
    eps = 1e-8

    if use_log10:
        plot_df["y"] = np.log10(
            pd.to_numeric(
                plot_df[y_col],
                errors="coerce"
            ).clip(lower=eps)
        )

        default_y_label = (
            r"Family-balanced median "
            r"$\log_{10}(\mathrm{EF})$"
        )

    else:
        plot_df["y"] = pd.to_numeric(
            plot_df[y_col],
            errors="coerce"
        )

        default_y_label = (
            "Family-balanced enrichment factor"
        )

    # Use automatic labels when custom labels are not provided
    final_x_label = (
        default_x_label
        if x_label is None
        else x_label
    )

    final_y_label = (
        default_y_label
        if y_label is None
        else y_label
    )

    # ---------------------------------------------------------
    # Visual configuration
    # ---------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    x = np.arange(len(percentile_order))

    if ef_mode == "band":
        default_x_tick_labels = _make_band_labels(
            percentile_order
        )
    else:
        default_x_tick_labels = _make_cumulative_labels(
            percentile_order
        )

    final_x_tick_labels = _resolve_x_tick_labels(
        custom_labels=x_tick_labels,
        percentiles=percentile_order,
        default_labels=default_x_tick_labels
    )

    cmap = plt.get_cmap("tab10")

    methods = (
        plot_df["method_label"]
        .drop_duplicates()
        .tolist()
    )

    # ---------------------------------------------------------
    # Plot each method
    # ---------------------------------------------------------
    for i, method in enumerate(methods):
        sub = plot_df[
            plot_df["method_label"] == method
        ].copy()

        sub = (
            sub.groupby(
                "percentile",
                as_index=False,
                observed=False
            )["y"]
            .median()
            .set_index("percentile")
            .reindex(percentile_order)
            .reset_index()
        )

        label = (
            pretty_labels.get(method, method)
            if pretty_labels
            else method
        )

        ax.plot(
            x,
            sub["y"].values,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            label=label,
            color=cmap(i % 10),
            zorder=3
        )

    # ---------------------------------------------------------
    # Reference line: EF = 1
    # ---------------------------------------------------------
    if use_log10:
        ax.axhline(
            0,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="gray"
        )
    else:
        ax.axhline(
            1,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="gray"
        )

    # ---------------------------------------------------------
    # Axes, title, and legend
    # ---------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        final_x_tick_labels,
        rotation=25,
        ha="right"
    )

    ax.set_xlabel(final_x_label)
    ax.set_ylabel(final_y_label)

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25
    )

    ax.grid(
        axis="x",
        visible=False
    )

    ax.legend(
        frameon=False,
        ncol=legend_ncol,
        loc="best",
        handlelength=2.2
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------
    if save_prefix is not None:
        suffix = f"_{ef_mode}"
    
        # High-resolution PNG
        fig.savefig(
            f"{save_prefix}{suffix}.png",
            dpi=save_dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
        )
    
        # Vector PDF: lines, markers, and text remain vector elements
        fig.savefig(
            f"{save_prefix}{suffix}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
            metadata={
                "Title": "LigQ2 molecular representation benchmark",
                "Creator": "Matplotlib",
            },
        )

    return fig, ax


# Source: analizar_EF.ipynb cell 2

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_ef_supplementary_by_family(
    df_family_summary,
    families=None,
    family_order=None,
    family_label_map=None,
    ef_mode="cumulative",
    selected_methods=None,
    percentile_order=(99.5, 99.0, 98.5, 98.0, 95.0, 90.0, 80.0, 50.0),
    pretty_labels=None,
    use_log10=True,

    # Plot labels
    x_label=None,
    y_label=None,
    x_tick_labels=None,
    title_template="{family}",

    # Visual configuration
    figsize=(8.2, 5.2),
    dpi=300,
    linewidth=2.4,
    markersize=6,
    legend_ncol=2,

    # Shared Y scale
    shared_y_limits=True,
    y_limits=None,
    y_padding_fraction=0.06,

    # Saving
    save_dir=None,
    filename_prefix="supplementary_EF",
    save_png=True,
    save_pdf=True,
    close_after_save=False,
):
    """
    Generate a separate EF plot for each protein family by reusing
    plot_family_balanced_ef_clean.

    df_family_summary must contain results already combined across repetitions,
    including the legacy 'familia' column.

    Returns
    -------
    plots : dict
        Dictionary:
            {
                familia: {
                    "fig": fig,
                    "ax": ax,
                    "data": dataframe_filtrado
                }
            }
    """

    if "familia" not in df_family_summary.columns:
        raise ValueError(
            "df_family_summary must contain a column named 'familia'."
        )

    df = df_family_summary.copy()

    # ---------------------------------------------------------
    # Families to include
    # ---------------------------------------------------------
    available_families = df["familia"].dropna().drop_duplicates().tolist()

    if families is None:
        families = available_families
    else:
        missing = [f for f in families if f not in available_families]

        if missing:
            raise ValueError(
                f"These families are not present in df_family_summary: {missing}"
            )

    if family_order is not None:
        ordered = [f for f in family_order if f in families]
        remaining = [f for f in families if f not in ordered]
        families = ordered + remaining

    # ---------------------------------------------------------
    # Family labels
    # ---------------------------------------------------------
    if family_label_map is None:
        family_label_map = {}

    def pretty_family(family):
        return family_label_map.get(family, family)

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def safe_filename(text):
        text = str(text).strip()
        text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
        return text.strip("_")

    # ---------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------
    plots = {}
    all_y_values = []

    for family in families:
        family_df = df[df["familia"] == family].copy()

        if family_df.empty:
            continue

        family_display = pretty_family(family)

        if title_template is None:
            title = ""
        else:
            title = title_template.format(
                family=family_display,
                family_raw=family
            )

        fig, ax = plot_family_balanced_ef_clean(
            df=family_df,
            ef_mode=ef_mode,
            selected_methods=selected_methods,
            percentile_order=percentile_order,
            pretty_labels=pretty_labels,
            use_log10=use_log10,
            across_group_agg="median",
            x_label=x_label,
            y_label=y_label,
            x_tick_labels=x_tick_labels,
            title=title,
            figsize=figsize,
            dpi=dpi,
            linewidth=linewidth,
            markersize=markersize,
            legend_ncol=legend_ncol,
            save_prefix=None,
        )

        # Retrieve method lines only. The EF=1 line has no "o" marker.
        for line in ax.get_lines():
            if line.get_marker() == "o":
                values = np.asarray(line.get_ydata(), dtype=float)
                all_y_values.extend(values[np.isfinite(values)])

        plots[family] = {
            "fig": fig,
            "ax": ax,
            "data": family_df,
        }

    # ---------------------------------------------------------
    # Shared Y limits
    # ---------------------------------------------------------
    if y_limits is not None:
        final_y_limits = y_limits

    elif shared_y_limits and len(all_y_values) > 0:
        y_min = float(np.min(all_y_values))
        y_max = float(np.max(all_y_values))

        y_range = y_max - y_min

        if np.isclose(y_range, 0):
            padding = 0.1 if np.isclose(y_max, 0) else abs(y_max) * 0.1
        else:
            padding = y_range * y_padding_fraction

        final_y_limits = (
            y_min - padding,
            y_max + padding
        )

    else:
        final_y_limits = None

    if final_y_limits is not None:
        for result in plots.values():
            result["ax"].set_ylim(final_y_limits)

    # ---------------------------------------------------------
    # Save after setting the Y scale
    # ---------------------------------------------------------
    if save_dir is not None:
        mode_suffix = "band" if ef_mode == "band" else "cumulative"

        for family, result in plots.items():
            family_name = safe_filename(pretty_family(family))

            output_prefix = save_dir / (
                f"{filename_prefix}_{family_name}_{mode_suffix}"
            )

            if save_png:
                result["fig"].savefig(
                    f"{output_prefix}.png",
                    dpi=dpi,
                    bbox_inches="tight"
                )

            if save_pdf:
                result["fig"].savefig(
                    f"{output_prefix}.pdf",
                    bbox_inches="tight"
                )

            if close_after_save:
                plt.close(result["fig"])

    return plots

from pathlib import Path
from string import ascii_uppercase
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_supplementary_plots_to_pdf(
    supplementary_plots,
    output_pdf,
    family_order=None,
    family_label_map=None,

    # First-page header
    si_file_label="Supporting Information File S2",
    si_title="Protein-category-specific performance of molecular representations",
    caption=None,

    # Caption formatting
    caption_wrap_width=110,
    header_fontsize=12,
    title_fontsize=10,
    caption_fontsize=9,

    # Additional height reserved on the first page
    first_page_extra_height=2.0,

    close_after_save=False,
):
    """
    Save all figures in supplementary_plots to a single multipage PDF.

    Add the file identifier and overall caption only to the first page. All
    remaining pages retain the plots' original format.

    Parameters
    ----------
    supplementary_plots : dict
        Dictionary returned by plot_ef_supplementary_by_family:

        {
            familia: {
                "fig": fig,
                "ax": ax,
                "data": family_df
            }
        }

    output_pdf : str or Path
        Path to the final PDF.

    family_order : list or None
        Order in which families appear. If None, use dictionary order.

    family_label_map : dict or None
        Translation or customization of family names.

    si_file_label : str
        Supplementary-file identifier shown on the first page.

    si_title : str
        Overall title for the PDF contents.

    caption : str or None
        Overall caption shown only on the first page. If None, use a default caption.

    caption_wrap_width : int
        Approximate number of characters per caption line.

    first_page_extra_height : float
        Additional height, in inches, added only to the first page to
        accommodate the header without shrinking the plot excessively.

    close_after_save : bool
        If True, close figures after saving them.
    """

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if family_label_map is None:
        family_label_map = {}

    if caption is None:
        caption = (
            "Protein-category-specific enrichment performance of the evaluated "
            "molecular representations. Curves show the median log10(EF) across "
            "five known/evaluation active partitions at cumulative ranking "
            "percentile thresholds. The dashed horizontal line indicates random "
            "expectation, log10(EF) = 0."
        )

    wrapped_caption = textwrap.fill(
        caption,
        width=caption_wrap_width,
    )

    available_families = list(supplementary_plots.keys())

    if family_order is None:
        ordered_families = available_families
    else:
        ordered_families = [
            family
            for family in family_order
            if family in supplementary_plots
        ]

        remaining = [
            family
            for family in available_families
            if family not in ordered_families
        ]

        ordered_families += remaining

    with PdfPages(
        output_pdf,
        metadata={
            "Title": "LigQ2 Supporting Information File S2",
            "Subject": (
                "Protein-category-specific molecular representation benchmark"
            ),
            "Creator": "Matplotlib",
        },
    ) as pdf:

        for i, family in enumerate(ordered_families):
            result = supplementary_plots[family]

            fig = result["fig"]
            ax = result["ax"]

            family_label = family_label_map.get(family, family)

            # Family title
            ax.set_title(
                family_label,
                fontsize=12,
                pad=10,
            )

            if i == 0:
                # Increase only the first page's height
                original_width, original_height = fig.get_size_inches()

                fig.set_size_inches(
                    original_width,
                    original_height + first_page_extra_height,
                    forward=True,
                )

                # Supplementary-file identifier
                fig.text(
                    0.5,
                    0.975,
                    si_file_label,
                    ha="center",
                    va="top",
                    fontsize=header_fontsize,
                    fontweight="bold",
                )

                # Overall title
                fig.text(
                    0.06,
                    0.925,
                    si_title,
                    ha="left",
                    va="top",
                    fontsize=title_fontsize,
                    fontweight="bold",
                )

                # Overall caption
                fig.text(
                    0.06,
                    0.875,
                    wrapped_caption,
                    ha="left",
                    va="top",
                    fontsize=caption_fontsize,
                    linespacing=1.25,
                )

                # Reserve the upper area for the header
                fig.tight_layout(
                    rect=(0.02, 0.02, 0.98, 0.72)
                )

            else:
                # Subsequent pages retain the original format
                fig.tight_layout()

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            if close_after_save:
                plt.close(fig)

    print(f"Supplementary PDF saved to: {output_pdf}")

# Source: analizar_EF.ipynb cell 16

def ids_to_set(x):
    """
    Convert a cell containing a list of IDs to a set.
    Accept a list object or a string such as "['CHEMBL1', 'CHEMBL2']".
    """
    if x is None:
        return set()

    if isinstance(x, float) and np.isnan(x):
        return set()

    if isinstance(x, set):
        return {str(i) for i in x}

    if isinstance(x, (list, tuple, np.ndarray)):
        return {str(i) for i in x if pd.notna(i)}

    if isinstance(x, str):
        x = x.strip()

        if x == "" or x.lower() in {"nan", "none"}:
            return set()

        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple, set)):
                return {str(i) for i in parsed}
            elif parsed is None:
                return set()
            else:
                return {str(parsed)}
        except Exception:
            # Fallback for comma-separated text without list syntax
            return {i.strip().strip("'").strip('"') for i in x.split(",") if i.strip()}

    return {str(x)}


def union_recovery_stats(
    df,
    percentile=99.5,
    max_combination_size=3,
    target_col="target_id",
    method_col="method_label",
    percentile_col="percentile",
    active_col="retrieved_active_ids",
    inactive_col="retrieved_inactive_ids",
):
    """
    Compute recovery statistics for individual methods and combinations of
    two or three methods for each target.

    Return a DataFrame with:
    - target_id
    - method_combination
    - n_methods
    - n_actives
    - n_inactives
    - total_recovered
    - active_fraction
    - active_inactive_ratio
    """

    df = df.copy()

    # Ensure numeric percentile comparison
    df[percentile_col] = pd.to_numeric(df[percentile_col], errors="coerce")

    df_99 = df[df[percentile_col] == percentile].copy()

    # Convert lists to sets
    df_99["active_set"] = df_99[active_col].apply(ids_to_set)
    df_99["inactive_set"] = df_99[inactive_col].apply(ids_to_set)

    # Combine all data if more than one target/method row exists
    per_method_rows = []

    for (target, method), subdf in df_99.groupby([target_col, method_col]):
        active_set = set().union(*subdf["active_set"].tolist())
        inactive_set = set().union(*subdf["inactive_set"].tolist())

        # If a compound appears as both active and inactive, prioritize the
        # active label and remove it from the inactive set.
        overlap = active_set & inactive_set
        inactive_set = inactive_set - active_set

        per_method_rows.append({
            target_col: target,
            method_col: method,
            "active_set": active_set,
            "inactive_set": inactive_set,
            "n_overlap_active_inactive": len(overlap),
        })

    per_method_df = pd.DataFrame(per_method_rows)

    results = []

    for target, target_df in per_method_df.groupby(target_col):
        method_records = target_df.to_dict("records")
        n_available_methods = len(method_records)

        max_size = min(max_combination_size, n_available_methods)

        for comb_size in range(1, max_size + 1):
            for combo in itertools.combinations(method_records, comb_size):

                methods = [x[method_col] for x in combo]

                active_union = set().union(*(x["active_set"] for x in combo))
                inactive_union = set().union(*(x["inactive_set"] for x in combo))

                # Apply the same safeguard to combinations
                overlap = active_union & inactive_union
                inactive_union = inactive_union - active_union

                n_actives = len(active_union)
                n_inactives = len(inactive_union)
                total_recovered = n_actives + n_inactives

                active_fraction = (
                    n_actives / total_recovered
                    if total_recovered > 0
                    else np.nan
                )

                active_inactive_ratio = (
                    n_actives / n_inactives
                    if n_inactives > 0
                    else np.inf if n_actives > 0 else np.nan
                )

                results.append({
                    target_col: target,
                    "method_combination": " + ".join(methods),
                    "methods_tuple": tuple(methods),
                    "n_methods": comb_size,
                    "n_actives": n_actives,
                    "n_inactives": n_inactives,
                    "total_recovered": total_recovered,
                    "active_fraction": active_fraction,
                    "active_inactive_ratio": active_inactive_ratio,
                    "n_overlap_active_inactive": len(overlap),
                })

    stats_df = pd.DataFrame(results)

    return stats_df

# Source: analizar_EF.ipynb cell 17

def build_target_group_table(agrupacion_subniveles):
    rows = []

    for grupo_grande, subgrupos in agrupacion_subniveles.items():
        for subgrupo, targets in subgrupos.items():
            for target in targets:
                rows.append({
                    "target_id": target.lower(),
                    "protein_group": grupo_grande,
                    "protein_subgroup": subgrupo
                })

    return pd.DataFrame(rows)


def normalize_method_combination(row):
    if "methods_tuple" in row and isinstance(row["methods_tuple"], (tuple, list)):
        methods = list(row["methods_tuple"])
    else:
        methods = str(row["method_combination"]).split(" + ")

    methods = sorted(methods)
    return " + ".join(methods)


def median_by_protein_group_with_recovery(combo_stats_df, agrupacion_subniveles):
    target_group_df = build_target_group_table(agrupacion_subniveles)

    df = combo_stats_df.copy()

    if "active_recovery_fraction" not in df.columns:
        raise ValueError("The active_recovery_fraction column is missing from combo_stats_df")

    if "active_recovery_percent" not in df.columns:
        df["active_recovery_percent"] = 100 * df["active_recovery_fraction"]

    df["target_id_norm"] = df["target_id"].astype(str).str.lower()

    df = df.merge(
        target_group_df,
        left_on="target_id_norm",
        right_on="target_id",
        how="left",
        suffixes=("", "_groupmap")
    )

    unmapped_targets = sorted(
        df.loc[df["protein_group"].isna(), "target_id_norm"].unique()
    )

    if len(unmapped_targets) > 0:
        print("Targets without an assigned group:")
        print(unmapped_targets)

    df = df[df["protein_group"].notna()].copy()

    df["method_combination_norm"] = df.apply(
        normalize_method_combination,
        axis=1
    )

    df["n_methods"] = df["method_combination_norm"].apply(
        lambda x: len(x.split(" + "))
    )

    group_median_df = (
        df
        .groupby(
            ["protein_group", "n_methods", "method_combination_norm"],
            as_index=False
        )
        .agg(
            n_targets=("target_id_norm", "nunique"),

            median_n_actives=("n_actives", "median"),
            median_n_inactives=("n_inactives", "median"),
            median_total_recovered=("total_recovered", "median"),

            # Purity of the retrieved set
            median_active_fraction=("active_fraction", "median"),

            # Target-active coverage
            median_active_recovery_fraction=("active_recovery_fraction", "median"),
            median_active_recovery_percent=("active_recovery_percent", "median"),

            median_active_inactive_ratio=("active_inactive_ratio", "median"),
        )
    )

    group_median_df = group_median_df.sort_values(
        [
            "protein_group",
            "n_methods",
            "median_active_recovery_percent",
            "median_active_fraction"
        ],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)

    return group_median_df

def median_across_protein_groups_with_recovery(group_combo_median_df):
    df = group_combo_median_df.copy()

    summary_df = (
        df
        .groupby(
            ["n_methods", "method_combination_norm"],
            as_index=False
        )
        .agg(
            n_protein_groups=("protein_group", "nunique"),

            median_of_group_median_n_actives=(
                "median_n_actives", "median"
            ),

            median_of_group_median_n_inactives=(
                "median_n_inactives", "median"
            ),

            median_of_group_median_total_recovered=(
                "median_total_recovered", "median"
            ),

            # Purity
            median_of_group_median_active_fraction=(
                "median_active_fraction", "median"
            ),

            # Active coverage
            median_of_group_median_active_recovery_fraction=(
                "median_active_recovery_fraction", "median"
            ),

            median_of_group_median_active_recovery_percent=(
                "median_active_recovery_percent", "median"
            ),

            median_of_group_median_active_inactive_ratio=(
                "median_active_inactive_ratio", "median"
            ),
        )
    )

    summary_df = summary_df.sort_values(
        [
            "n_methods",
            "median_of_group_median_active_recovery_percent",
            "median_of_group_median_active_fraction"
        ],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return summary_df

def add_active_recovery_fraction(
    combo_stats_df,
    conteos_activos,
    denominator_col="n_pool_unknown_actives",
    seed_filter=None
):
    df = combo_stats_df.copy()
    counts = conteos_activos.copy()

    # Normalize target_id
    df["target_id_norm"] = df["target_id"].astype(str).str.lower()
    counts["target_id_norm"] = counts["target_id"].astype(str).str.lower()

    # Optionally filter by seed
    if seed_filter is not None:
        counts = counts[counts["seed"] == seed_filter].copy()

    # Retain non-skipped targets when the column is present
    if "skipped_by_eval" in counts.columns:
        counts = counts[counts["skipped_by_eval"] == False].copy()

    # Denominator table
    denom_df = counts[
        ["target_id_norm", denominator_col]
    ].drop_duplicates()

    # Check for more than one row per target
    duplicated = denom_df["target_id_norm"].duplicated().sum()
    if duplicated > 0:
        print("Warning: conteos_activos contains duplicate targets.")
        print("Check whether filtering by seed is required.")
        denom_df = denom_df.drop_duplicates("target_id_norm", keep="first")

    # Merge
    df = df.merge(
        denom_df,
        on="target_id_norm",
        how="left"
    )

    # Fraction of recovered actives
    df["active_recovery_fraction"] = (
        df["n_actives"] / df[denominator_col]
    )

    df["active_recovery_percent"] = (
        100 * df["active_recovery_fraction"]
    )

    return df

def run_union_analysis_for_seed(
    base_dir,
    seed,
    agrupacion_subniveles,
    percentile=99.5,
    denominator_col="n_pool_unknown_actives"
):
    seed_dir = Path(base_dir) / f"seed_{seed}"

    resultados_EF_inactivos = pd.read_csv(
        seed_dir / "retrieved_active_sets_all_methods.csv"
    )

    conteos_activos = pd.read_csv(
        seed_dir / "target_total_counts.csv"
    )

    combo_stats_df = union_recovery_stats(
        resultados_EF_inactivos,
        percentile=percentile
    )

    combo_stats_df = add_active_recovery_fraction(
        combo_stats_df,
        conteos_activos,
        denominator_col=denominator_col
    )

    group_combo_median_df = median_by_protein_group_with_recovery(
        combo_stats_df,
        agrupacion_subniveles
    )

    combo_summary_df = median_across_protein_groups_with_recovery(
        group_combo_median_df
    )

    combo_stats_df["seed"] = seed
    group_combo_median_df["seed"] = seed
    combo_summary_df["seed"] = seed

    return combo_stats_df, group_combo_median_df, combo_summary_df

# Source: analizar_EF.ipynb cell 37

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import textwrap


pretty_labels = {
    "morgan_1024_r2": "ECFP4 (1024 bits)",
    "morgan_feature_1024_r2": "FCFP4 (1024 bits)",
    "ap_rdkit": "Atom Pair",
    "chemberta_zinc_base_768": "ChemBERTa",
    "ibm-MolFormer": "MolFormer",
    "maccs": "MACCS",
    "rdkit_1024": "RDKit FP",
    "topological_torsion_rdkit_1024": "Topological Torsion",
}


def prettify_method_name(method_name, pretty_labels):
    clean = method_name.replace("__target_seeds", "")
    return pretty_labels.get(clean, clean)


def prettify_combination(combination, pretty_labels, multiline=False):
    methods = [m.strip() for m in combination.split(" + ")]
    pretty_methods = [prettify_method_name(m, pretty_labels) for m in methods]

    if multiline:
        return "\n+ ".join(pretty_methods)

    return " + ".join(pretty_methods)


def build_union_plot_df(
    top_15_purity,
    top_15_recovery,
    selected_combinations,
    pretty_labels
):
    purity_df = top_15_purity[
        [
            "method_combination_norm",
            "median_active_fraction_across_seeds",
        ]
    ].copy()

    recovery_df = top_15_recovery[
        [
            "method_combination_norm",
            "median_active_recovery_fraction_across_seeds",
        ]
    ].copy()

    plot_df = recovery_df.merge(
        purity_df,
        on="method_combination_norm",
        how="inner"
    )

    plot_df = plot_df[
        plot_df["method_combination_norm"].isin(selected_combinations)
    ].copy()

    missing = set(selected_combinations) - set(plot_df["method_combination_norm"])

    if missing:
        print("Warning: these combinations are not present in both tables:")
        for m in missing:
            print(f" - {m}")

    plot_df["total_evaluation_actives_recovered_percent"] = (
        plot_df["median_active_recovery_fraction_across_seeds"] * 100
    )

    plot_df["active_fraction_among_retrieved_percent"] = (
        plot_df["median_active_fraction_across_seeds"] * 100
    )

    plot_df["label"] = plot_df["method_combination_norm"].apply(
        lambda x: prettify_combination(x, pretty_labels, multiline=False)
    )

    plot_df["label_multiline"] = plot_df["method_combination_norm"].apply(
        lambda x: prettify_combination(x, pretty_labels, multiline=True)
    )

    order_map = {comb: i for i, comb in enumerate(selected_combinations)}
    plot_df["plot_order"] = plot_df["method_combination_norm"].map(order_map)

    plot_df = plot_df.sort_values("plot_order").reset_index(drop=True)

    return plot_df

# Source: analizar_EF.ipynb cell 38

def plot_union_recovery_vs_purity_publication(
    plot_df,
    title="",
    savepath=None,
    dpi=600,
    legend_multiline=False,
    show_point_labels=False,
    xlabel="Category-balanced median recall (%)",
    ylabel="Category-balanced median precision (%)",
):
    """
    Scatter plot of total active recovery versus active fraction.

    The figure uses a single plotting area and places a compact legend in the
    upper-right corner, avoiding the side panel used in earlier versions.
    """

    def _compact_legend_label(label):
        """
        Shorten names for a legend that fits within one column.
        """
        label = str(label).replace("\n", " ")
        label = re.sub(r"\s+", " ", label).strip()

        label = label.replace(
            "ECFP4 (1024 bits)",
            "ECFP4"
        )
        label = label.replace(
            "FCFP4 (1024 bits)",
            "FCFP4"
        )
        label = label.replace(
            "Topological Torsion",
            "TT"
        )
        # Match the displayed order of the same triple in mean-rank fusion.
        label = label.replace("ECFP4 + FCFP4 + TT", "ECFP4 + TT + FCFP4")

        return label

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.2,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):

        # A single panel. Its size preserves a useful aspect ratio and allows
        # the figure to be reduced to one column later.
        fig, ax = plt.subplots(
            figsize=(7.0, 5.0),
            dpi=dpi
        )

        colors = [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#D55E00",
            "#CC79A7",
            "#56B4E9",
            "#000000",
            "#F0E442",
        ]

        markers = [
            "o",
            "s",
            "^",
            "D",
            "P",
            "X",
            "v",
            "*",
        ]

        x_col = "total_evaluation_actives_recovered_percent"
        y_col = "active_fraction_among_retrieved_percent"

        legend_handles = []

        # Resetting the index avoids depending on an original index that is
        # consecutive or numeric.
        plot_data = plot_df.reset_index(drop=True)

        for position, row in plot_data.iterrows():
            color = colors[position % len(colors)]
            marker = markers[position % len(markers)]

            ax.scatter(
                row[x_col],
                row[y_col],
                s=105,
                marker=marker,
                facecolor=color,
                edgecolor="black",
                linewidth=0.85,
                alpha=0.96,
                zorder=4,
            )

            if legend_multiline:
                legend_label = row["label_multiline"]
            else:
                legend_label = _compact_legend_label(
                    row["label"]
                )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    label=legend_label,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.85,
                    markersize=8.5,
                )
            )

            if show_point_labels:
                ax.annotate(
                    str(position + 1),
                    xy=(
                        row[x_col],
                        row[y_col],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8.5,
                    zorder=5,
                )

        # The caption already describes the figure, so the internal title may
        # remain empty.
        if title:
            ax.set_title(
                title,
                loc="left",
                pad=10,
            )

        ax.set_xlabel(
            xlabel,
            labelpad=8,
        )

        ax.set_ylabel(
            ylabel,
            labelpad=8,
        )

        x = plot_data[x_col]
        y = plot_data[y_col]

        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()

        x_span = x_max - x_min
        y_span = y_max - y_min

        x_margin = max(
            x_span * 0.22,
            1.3,
        )

        y_margin = max(
            y_span * 0.18,
            1.8,
        )

        ax.set_xlim(
            x_min - x_margin,
            x_max + x_margin,
        )

        ax.set_ylim(
            y_min - y_margin,
            y_max + y_margin,
        )

        ax.xaxis.set_major_locator(
            MultipleLocator(2.5)
        )

        ax.yaxis.set_major_locator(
            MultipleLocator(2.5)
        )

        ax.grid(
            True,
            which="major",
            color="0.7",
            linestyle="-",
            linewidth=0.45,
            alpha=0.25,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4.2,
            width=0.9,
            pad=4,
        )

        # The upper-right corner is free of points.
        legend = ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            borderpad=0.7,
            labelspacing=0.65,
            handletextpad=0.65,
            borderaxespad=0.65,
            handlelength=1.2,
        )

        legend.get_frame().set_edgecolor("0.75")
        legend.get_frame().set_linewidth(0.85)
        legend.get_frame().set_alpha(0.96)
        legend.get_frame().set_facecolor("white")

        fig.tight_layout(
            pad=0.7
        )

        if savepath is not None:
            fig.savefig(
                savepath,
                bbox_inches="tight",
                pad_inches=0.05,
                dpi=dpi,
                facecolor="white",
                transparent=False,
            )

        return fig, ax

# Source: analizar_EF.ipynb cell 54

def parse_id_list(x):
    """
    Convert a list stored as a string into a set of IDs.
    Accept strings such as "['A', 'B']", lists, tuples, or sets.
    """
    if isinstance(x, set):
        return x

    if isinstance(x, (list, tuple)):
        return set(x)

    if pd.isna(x):
        return set()

    if isinstance(x, str):
        x = x.strip()

        if x in ["", "nan", "None"]:
            return set()

        return set(ast.literal_eval(x))

    raise ValueError(f"Cannot interpret this value: {x}")


def build_retrieved_sets(
    actives_methods,
    actives_bsi,
    targets=None,
    bsi_method_label="bsi_target_exclusion",
):
    """
    Combine actives_methods and actives_bsi into a common structure:

    target_id | method_label | retrieved_set | n_retrieved_inactives

    BSI is treated as another method.
    """

    dfs = []

    # -------------------------
    # Standard methods
    # -------------------------
    df_methods = actives_methods.copy()

    if targets is not None:
        df_methods = df_methods[df_methods["target_id"].isin(targets)]

    df_methods = df_methods[
        [
            "target_id",
            "method_label",
            "retrieved_active_ids",
            "n_retrieved_inactives",
        ]
    ].copy()

    dfs.append(df_methods)

    # -------------------------
    # BSI
    # -------------------------
    df_bsi = actives_bsi.copy()

    if targets is not None:
        df_bsi = df_bsi[df_bsi["target_id"].isin(targets)]

    if "method_label" not in df_bsi.columns:
        df_bsi["method_label"] = bsi_method_label

    df_bsi = df_bsi[
        [
            "target_id",
            "method_label",
            "retrieved_active_ids",
            "n_retrieved_inactives",
        ]
    ].copy()

    dfs.append(df_bsi)

    # -------------------------
    # Combine
    # -------------------------
    df_all = pd.concat(dfs, ignore_index=True)

    df_all["retrieved_set"] = df_all["retrieved_active_ids"].apply(parse_id_list)

    # If more than one row exists for a target/method pair, merge the active
    # sets and sum the inactive counts.
    df_sets = (
        df_all
        .groupby(["target_id", "method_label"], as_index=False)
        .agg(
            retrieved_set=(
                "retrieved_set",
                lambda sets: set().union(*sets),
            ),
            n_retrieved_inactives=(
                "n_retrieved_inactives",
                "sum",
            ),
        )
    )

    return df_sets

def compare_method_combinations(
    actives_methods,
    actives_bsi,
    targets,
    total_active_ids_by_target=None,
    bsi_method_label="bsi_target_exclusion",
    include_single_methods=True,
):
    """
    Compare all pairwise method combinations for each target.

    Compute:
    - actives retrieved by method A
    - actives retrieved by method B
    - overlap between active sets
    - actives unique to A
    - actives unique to B
    - actives retrieved by the A ∪ B combination
    - percentage of actives retrieved relative to the internal denominator
    - inactives retrieved by A
    - inactives retrieved by B
    - inactives retrieved by the combination, estimated as A + B
    - actives / (actives + inactives) for the combination
    """

    df_sets = build_retrieved_sets(
        actives_methods=actives_methods,
        actives_bsi=actives_bsi,
        targets=targets,
        bsi_method_label=bsi_method_label,
    )

    results = []

    for target in targets:
        df_target = df_sets[df_sets["target_id"] == target].copy()

        if df_target.empty:
            print(f"Target without data: {target}")
            continue

        method_to_set = dict(
            zip(df_target["method_label"], df_target["retrieved_set"])
        )

        method_to_inactives = dict(
            zip(df_target["method_label"], df_target["n_retrieved_inactives"])
        )

        methods = sorted(method_to_set.keys())

        # Denominator for the internal percentage: by default, use the union
        # of all actives retrieved by any method.
        if total_active_ids_by_target is not None:
            total_active_set = set(total_active_ids_by_target[target])
        else:
            total_active_set = set().union(*method_to_set.values())

        n_total_actives = len(total_active_set)

        if include_single_methods:
            method_pairs = itertools.combinations_with_replacement(methods, 2)
        else:
            method_pairs = itertools.combinations(methods, 2)

        for method_a, method_b in method_pairs:
            set_a = method_to_set[method_a]
            set_b = method_to_set[method_b]

            n_inactives_a = method_to_inactives[method_a]
            n_inactives_b = method_to_inactives[method_b]

            combined_set = set_a | set_b
            overlap_set = set_a & set_b

            recovered_combined = combined_set & total_active_set
            n_recovered_combined = len(recovered_combined)

            # Combined inactives. Because only counts, not IDs, are available,
            # addition is the most direct estimate. This may overestimate the
            # result when methods retrieve the same inactives.
            if method_a == method_b:
                n_retrieved_inactives_combined = n_inactives_a
            else:
                n_retrieved_inactives_combined = n_inactives_a + n_inactives_b

            n_retrieved_total_combined = (
                n_recovered_combined + n_retrieved_inactives_combined
            )

            if n_total_actives > 0:
                pct_recovered_combined = (
                    100 * n_recovered_combined / n_total_actives
                )
            else:
                pct_recovered_combined = np.nan

            if n_retrieved_total_combined > 0:
                active_fraction_retrieved_combined = (
                    n_recovered_combined / n_retrieved_total_combined
                )
            else:
                active_fraction_retrieved_combined = np.nan

            active_pct_retrieved_combined = (
                100 * active_fraction_retrieved_combined
            )

            results.append(
                {
                    "target_id": target,
                    "method_a": method_a,
                    "method_b": method_b,

                    "n_recovered_method_a": len(set_a),
                    "n_recovered_method_b": len(set_b),

                    "n_retrieved_inactives_method_a": n_inactives_a,
                    "n_retrieved_inactives_method_b": n_inactives_b,
                    "n_retrieved_inactives_combined": n_retrieved_inactives_combined,

                    "n_overlap": len(overlap_set),
                    "n_unique_method_a": len(set_a - set_b),
                    "n_unique_method_b": len(set_b - set_a),

                    "n_recovered_combined": n_recovered_combined,
                    "n_retrieved_total_combined": n_retrieved_total_combined,

                    "active_fraction_retrieved_combined": active_fraction_retrieved_combined,
                    "active_pct_retrieved_combined": active_pct_retrieved_combined,

                    "pct_recovered_combined": pct_recovered_combined,
                    "n_total_actives_denominator": n_total_actives,
                }
            )

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        df_results = df_results.sort_values(
            by=[
                "target_id",
                "active_pct_retrieved_combined",
                "pct_recovered_combined",
                "n_recovered_combined",
            ],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

    return df_results

def add_active_total_percentages(
    df_comparacion,
    conteos_activos,
    denominator_col="n_active_total_raw",
):
    """
    Add each target's total active count to df_comparacion and compute the
    percentage of actives retrieved by each combination.

    If both df_comparacion and conteos_activos have a 'seed' column, merge on
    ['seed', 'target_id'].

    If df_comparacion has no seed, merge on 'target_id' only.
    """

    df = df_comparacion.copy()

    # Define merge columns
    if "seed" in df.columns and "seed" in conteos_activos.columns:
        merge_cols = ["seed", "target_id"]

        conteos_use = (
            conteos_activos[merge_cols + [denominator_col]]
            .drop_duplicates()
            .rename(columns={denominator_col: "n_active_total_reference"})
        )

    else:
        merge_cols = ["target_id"]

        # With multiple seeds, n_active_total_raw should be invariant.
        # n_unknown_actives or n_pool_unknown_actives may vary by seed, so use
        # the first value for each target in that case.
        conteos_use = (
            conteos_activos
            .groupby("target_id", as_index=False)
            .agg(n_active_total_reference=(denominator_col, "first"))
        )

    df = df.merge(
        conteos_use,
        on=merge_cols,
        how="left",
    )

    df["pct_total_actives_recovered"] = np.where(
        df["n_active_total_reference"] > 0,
        100 * df["n_recovered_combined"] / df["n_active_total_reference"],
        np.nan,
    )

    return df

def add_pool_active_recovery_percentage(
    df_comparacion,
    conteos_activos,
    denominator_col="n_pool_unknown_actives",
):
    """
    Add n_pool_unknown_actives and compute:

    pct_pool_unknown_actives_recovered =
        100 * n_recovered_combined / n_pool_unknown_actives
    """

    df = df_comparacion.copy()

    if "seed" in df.columns and "seed" in conteos_activos.columns:
        merge_cols = ["seed", "target_id"]

        conteos_use = (
            conteos_activos[merge_cols + [denominator_col]]
            .drop_duplicates()
            .rename(columns={denominator_col: "n_pool_unknown_actives"})
        )

    else:
        merge_cols = ["target_id"]

        conteos_use = (
            conteos_activos
            .groupby("target_id", as_index=False)
            .agg(n_pool_unknown_actives=(denominator_col, "first"))
        )

    df = df.merge(
        conteos_use,
        on=merge_cols,
        how="left",
    )

    df["pct_pool_unknown_actives_recovered"] = np.where(
        df["n_pool_unknown_actives"] > 0,
        100 * df["n_recovered_combined"] / df["n_pool_unknown_actives"],
        np.nan,
    )

    return df

def add_retrieved_inactives_to_df_comparacion(
    df_comparacion_total,
    actives_methods,
    actives_bsi,
    bsi_method_label=None,
):
    """
    Add or recompute inactives retrieved by method_a and method_b.

    Compute:

        active_pct_retrieved_combined =
            100 * retrieved_actives / (retrieved_actives + retrieved_inactives)

    Important: if the DataFrame already contains inactive columns, remove and
    recompute them to avoid merge conflicts with _x / _y suffixes.
    """

    df = df_comparacion_total.copy()

    # --------------------------------------------------
    # Remove previous columns to avoid _x / _y suffixes
    # --------------------------------------------------
    cols_to_recalculate = [
        "n_retrieved_inactives_method_a",
        "n_retrieved_inactives_method_b",
        "n_retrieved_inactives_combined",
        "n_retrieved_total_combined",
        "active_fraction_retrieved_combined",
        "active_pct_retrieved_combined",
    ]

    existing_cols_to_drop = [
        col for col in cols_to_recalculate
        if col in df.columns
    ]

    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    # --------------------------------------------------
    # Minimum checks
    # --------------------------------------------------
    required_comparison_cols = [
        "target_id",
        "method_a",
        "method_b",
        "n_recovered_combined",
    ]

    for col in required_comparison_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from df_comparacion_total")

    for col in ["target_id", "method_label", "n_retrieved_inactives"]:
        if col not in actives_methods.columns:
            raise ValueError(f"Column '{col}' is missing from actives_methods")

    if "n_retrieved_inactives" not in actives_bsi.columns:
        raise ValueError("Column 'n_retrieved_inactives' is missing from actives_bsi")

    # --------------------------------------------------
    # Prepare inactives from standard methods
    # --------------------------------------------------
    df_methods_inactives = actives_methods[
        [
            "target_id",
            "method_label",
            "n_retrieved_inactives",
        ]
    ].copy()

    df_methods_inactives = (
        df_methods_inactives
        .groupby(["target_id", "method_label"], as_index=False)
        .agg(n_retrieved_inactives=("n_retrieved_inactives", "first"))
    )

    # --------------------------------------------------
    # Prepare BSI inactives
    # --------------------------------------------------
    df_bsi_inactives = actives_bsi.copy()

    if "method_label" not in df_bsi_inactives.columns:
        if bsi_method_label is None:
            raise ValueError(
                "actives_bsi has no method_label column. "
                "You must provide bsi_method_label."
            )

        df_bsi_inactives["method_label"] = bsi_method_label

    else:
        # If method_label already exists but the name used in
        # df_comparacion_total must be forced, replace it to ensure a match.
        if bsi_method_label is not None:
            df_bsi_inactives["method_label"] = bsi_method_label

    df_bsi_inactives = df_bsi_inactives[
        [
            "target_id",
            "method_label",
            "n_retrieved_inactives",
        ]
    ].copy()

    df_bsi_inactives = (
        df_bsi_inactives
        .groupby(["target_id", "method_label"], as_index=False)
        .agg(n_retrieved_inactives=("n_retrieved_inactives", "first"))
    )

    # --------------------------------------------------
    # Combine inactives by target/method
    # --------------------------------------------------
    df_inactives = pd.concat(
        [
            df_methods_inactives,
            df_bsi_inactives,
        ],
        ignore_index=True,
    )

    df_inactives = (
        df_inactives
        .groupby(["target_id", "method_label"], as_index=False)
        .agg(n_retrieved_inactives=("n_retrieved_inactives", "first"))
    )

    # --------------------------------------------------
    # Merge for method_a
    # --------------------------------------------------
    df = df.merge(
        df_inactives.rename(
            columns={
                "method_label": "method_a",
                "n_retrieved_inactives": "n_retrieved_inactives_method_a",
            }
        ),
        on=["target_id", "method_a"],
        how="left",
    )

    # --------------------------------------------------
    # Merge for method_b
    # --------------------------------------------------
    df = df.merge(
        df_inactives.rename(
            columns={
                "method_label": "method_b",
                "n_retrieved_inactives": "n_retrieved_inactives_method_b",
            }
        ),
        on=["target_id", "method_b"],
        how="left",
    )

    # --------------------------------------------------
    # Check methods without a match
    # --------------------------------------------------
    missing_a = df[df["n_retrieved_inactives_method_a"].isna()][
        ["target_id", "method_a"]
    ].drop_duplicates()

    missing_b = df[df["n_retrieved_inactives_method_b"].isna()][
        ["target_id", "method_b"]
    ].drop_duplicates()

    if len(missing_a) > 0:
        print("Warning: inactive counts are missing for some method_a values:")
        display(missing_a)

    if len(missing_b) > 0:
        print("Warning: inactive counts are missing for some method_b values:")
        display(missing_b)

    df["n_retrieved_inactives_method_a"] = (
        df["n_retrieved_inactives_method_a"].fillna(0)
    )

    df["n_retrieved_inactives_method_b"] = (
        df["n_retrieved_inactives_method_b"].fillna(0)
    )

    # --------------------------------------------------
    # Combined inactives
    # --------------------------------------------------
    # Because inactive IDs are unavailable, add counts for distinct methods.
    # This may overestimate the result when both methods retrieve the same inactives.
    df["n_retrieved_inactives_combined"] = np.where(
        df["method_a"] == df["method_b"],
        df["n_retrieved_inactives_method_a"],
        (
            df["n_retrieved_inactives_method_a"]
            + df["n_retrieved_inactives_method_b"]
        ),
    )

    # --------------------------------------------------
    # Actives / (actives + inactives)
    # --------------------------------------------------
    df["n_retrieved_total_combined"] = (
        df["n_recovered_combined"]
        + df["n_retrieved_inactives_combined"]
    )

    df["active_fraction_retrieved_combined"] = np.where(
        df["n_retrieved_total_combined"] > 0,
        df["n_recovered_combined"] / df["n_retrieved_total_combined"],
        np.nan,
    )

    df["active_pct_retrieved_combined"] = (
        100 * df["active_fraction_retrieved_combined"]
    )

    return df

def run_comparison_for_seed(
    seed,
    targets_eval,
    conteos_activos,
    percentile=99.5,
    base_bsi="./resultados_EF_repeticiones_con_inactivos_10_90_bsi",
    base_methods="./resultados_EF_repeticiones_con_inactivos_10_90",
    bsi_method_label="bsi_pf00069_target_exclusion_1024__target_seeds",
):
    """
    Run the method-to-method comparison for one seed.

    Return a DataFrame with the same structure as df_comparacion_total plus a
    seed column.
    """

    # -------------------------
    # Paths
    # -------------------------
    bsi_path = (
        f"{base_bsi}/seed_{seed}/"
        "bsi_target_exclusion_retrieved_active_sets.csv"
    )

    methods_path = (
        f"{base_methods}/seed_{seed}/"
        "retrieved_active_sets_all_methods.csv"
    )

    # -------------------------
    # Read results
    # -------------------------
    actives_bsi = pd.read_csv(bsi_path)
    actives_methods = pd.read_csv(methods_path)

    # -------------------------
    # Filter percentile
    # -------------------------
    actives_bsi = actives_bsi[
        actives_bsi["percentile"] == percentile
    ].copy()

    actives_methods = actives_methods[
        actives_methods["percentile"] == percentile
    ].copy()

    # -------------------------
    # Filter targets when applicable
    # -------------------------
    if targets_eval is not None:
        actives_bsi = actives_bsi[
            actives_bsi["target_id"].isin(targets_eval)
        ].copy()

        actives_methods = actives_methods[
            actives_methods["target_id"].isin(targets_eval)
        ].copy()

    # -------------------------
    # Method comparison
    # -------------------------
    df_comparacion_seed = compare_method_combinations(
        actives_methods=actives_methods,
        actives_bsi=actives_bsi,
        targets=targets_eval,
        bsi_method_label=bsi_method_label,
        include_single_methods=True,
    )

    df_comparacion_seed["seed"] = seed

    # -------------------------
    # Add percentage relative to n_pool_unknown_actives
    # -------------------------
    df_comparacion_seed = add_pool_active_recovery_percentage(
        df_comparacion=df_comparacion_seed,
        conteos_activos=conteos_activos,
        denominator_col="n_pool_unknown_actives",
    )

    # -------------------------
    # Add retrieved inactives and actives / (actives + inactives)
    # -------------------------
    df_comparacion_seed = add_retrieved_inactives_to_df_comparacion(
        df_comparacion_total=df_comparacion_seed,
        actives_methods=actives_methods,
        actives_bsi=actives_bsi,
        bsi_method_label=bsi_method_label,
    )

    return df_comparacion_seed
