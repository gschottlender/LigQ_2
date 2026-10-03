#!/usr/bin/env python3
"""Relabel the historical percentile-union plot without changing its protocol.

Original category assignments are read as literals from the archived notebook,
never executed. Counts are reconstructed from retained retrieved-ID sets, not
from new molecular searches. Once exported, the four full-precision plot rows
can be reused directly through --plot-data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ligq2_evaluation.constants import PRETTY_METHODS, SEEDS
from ligq2_evaluation.runtime import bootstrap_dependencies

bootstrap_dependencies()
from notebook_analysis import build_union_plot_df, plot_union_recovery_vs_purity_publication, run_union_analysis_for_seed


SELECTED_COMBINATIONS = (
    "morgan_1024_r2__target_seeds",
    "morgan_1024_r2__target_seeds + morgan_feature_1024_r2__target_seeds",
    "morgan_1024_r2__target_seeds + topological_torsion_rdkit_1024__target_seeds",
    "morgan_1024_r2__target_seeds + morgan_feature_1024_r2__target_seeds + topological_torsion_rdkit_1024__target_seeds",
)


def notebook_family_mapping(path):
    if path.suffix == ".json":
        return json.loads(path.read_text())["historical_families"]
    assignments = []
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "agrupacion_subniveles =" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "agrupacion_subniveles"
                for target in node.targets
            ):
                assignments.append(ast.literal_eval(node.value))
    if len(assignments) != 1:
        raise ValueError("Expected one literal historical category assignment in the notebook.")
    return assignments[0]


def historical_plot_rows(base_dir, notebook):
    families = notebook_family_mapping(notebook)
    frames = []
    for seed in sorted(SEEDS):
        _, _, summary = run_union_analysis_for_seed(
            base_dir, seed, families, percentile=99.5,
            denominator_col="n_pool_unknown_actives",
        )
        frames.append(summary)
        print(f"Reused retrieved sets for partition {seed}.", flush=True)
    all_seeds = pd.concat(frames, ignore_index=True)
    summary = all_seeds.groupby("method_combination_norm", as_index=False).agg(
        median_active_recovery_fraction_across_seeds=("median_of_group_median_active_recovery_fraction", "median"),
        median_active_fraction_across_seeds=("median_of_group_median_active_fraction", "median"),
    )
    return build_union_plot_df(summary, summary, SELECTED_COMBINATIONS, PRETTY_METHODS)


def main():
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=workspace / "resultados_EF_repeticiones_con_inactivos_10_90")
    parser.add_argument("--notebook", type=Path, default=workspace / "analizar_EF.ipynb")
    parser.add_argument("--plot-data", type=Path, help="Previously exported historical four-point plot CSV.")
    parser.add_argument("--output-dir", type=Path, default=workspace)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.plot_data:
        sources = [args.plot_data.expanduser().resolve()]
        rows = pd.read_csv(sources[0])
    else:
        base, notebook = args.base_dir.expanduser().resolve(), args.notebook.expanduser().resolve()
        sources = [notebook] + [base / f"seed_{seed}" / name for seed in SEEDS
                               for name in ("retrieved_active_sets_all_methods.csv", "target_total_counts.csv")]
        rows = historical_plot_rows(base, notebook)
    if list(rows.method_combination_norm) != list(SELECTED_COMBINATIONS):
        raise ValueError("Plot rows do not contain the four historical combinations in their original order.")
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plot_union_recovery_vs_purity_publication(rows, savepath=str(output / "union_recovery_vs_purity.png"))
    for suffix in ("pdf", "svg"):
        fig.savefig(output / f"union_recovery_vs_purity.{suffix}", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    rows.to_csv(output / "union_recovery_vs_purity_plot_data.csv", index=False)
    metadata = {
        "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "axes": {"x": ax.get_xlabel(), "y": ax.get_ylabel()},
        "protocol": "Union/deduplication of each method's percentile-99.5 retrieved sets; variable total budget.",
        "aggregation": "Median targets within original notebook categories per partition; median categories; median five partitions.",
        "category_mapping": "Historical notebook assignment retained; no reassignment during relabeling.",
        "difference_from_fixed_budget": "No @N labels: unlike mean-rank fusion, the union does not impose a fixed compound budget.",
    }
    (output / "union_recovery_vs_purity_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(rows[["label", "total_evaluation_actives_recovered_percent", "active_fraction_among_retrieved_percent"]].to_string(index=False))
    print(f"Wrote relabeled PNG/PDF/SVG and full-precision plot data to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
