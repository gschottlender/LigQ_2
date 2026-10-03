"""Render the selected figures using the retained, unmodified plotting functions."""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

from .common import ENGINE, ROOT, command, execute, read_json, sha256, write_json


def bootstrap():
    for path in (ENGINE, ENGINE / "dependencies/evaluation_core", ENGINE / "dependencies/ligq_core"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def module(name):
    bootstrap()
    return importlib.import_module(name)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from .resources import verify_files

    manifest = read_json(ROOT / "figure_manifest.json")
    if args.source == "historical":
        source = ROOT / "reference_data"
        verify_files(source, read_json(ROOT / "reference_lock.json")["files"])
    else:
        source = args.results_dir.resolve()
    destination = args.output_dir.resolve() / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    selected = manifest["figures"] if args.figure == "all" else [r for r in manifest["figures"] if r["id"] == args.figure]
    rendered_appendices = False
    receipts = []
    for row in selected:
        figure = row["id"]
        folder = destination / "work" / row["evaluation"]
        folder.mkdir(parents=True, exist_ok=True)
        summary = source / row["summary"]
        before = sha256(summary)
        frame = pd.read_csv(summary)
        if figure in ("Figure_2", "S2"):
            functions = module("preview_category_balanced_spread")
            if figure == "Figure_2":
                frame = module("plot_main_representation_comparison").select_main_methods(frame)
                with plt.rc_context({"font.size":16, "axes.labelsize":18, "xtick.labelsize":16,
                                     "ytick.labelsize":16, "legend.fontsize":16, "legend.title_fontsize":16}):
                    functions.plot_overlay(frame, folder, methods=module("plot_main_representation_comparison").MAIN_METHODS)
            else:
                functions.plot_overlay(frame, folder)
            stem = folder / "category_balanced_ef_iqr_overlay"
        elif figure == "Figure_3":
            functions = module("plot_identity_ligand_rescue_comparison")
            functions.plot_comparison(frame, identity_floor=55.0, ligand_budget=50,
                                      output_dir=folder, dpi=600, font_size=16)
            stem = folder / "fixed_k_domain_and_identity_ligand_rescue_cumulative_EF_publication"
        elif figure == "S1":
            functions = module("plot_retrieved_hit_distributions")
            stem = folder / "retrieved_hits_distributions"
            functions.panel(frame, stem, "ECFP4 hits and non-retrieved background | percentile 99",
                "Equal category weights; equal target weights within category; equal partition weights within target.",
                ("png", "pdf", "svg"), False)
        elif figure == "S3":
            module("18_run_active_preprocessing_sensitivity").plot_summary(frame, folder)
            stem = folder / "active_preprocessing_sensitivity_EF0.5"
        elif figure == "S4":
            execute(command("plot_historical_union_precision_recall.py", "--plot-data", summary, "--output-dir", folder))
            stem = folder / "union_recovery_vs_purity"
        elif figure == "S5":
            execute(command("plot_fixed_budget_recovery_vs_purity.py", "--input", summary,
                            "--output-dir", folder, "--policies", "mean", "--top-n", "4"))
            stem = folder / "fixed_budget_recovery_vs_purity_mean"
        elif figure == "S6":
            stem = folder / "ecfp4_transfer_strategy_comparison"
            module("plot_ecfp4_transfer_strategy_comparison").plot_comparison(frame, stem, font_size=16)
        elif figure == "S7":
            module("run_full_domain_benchmark")._plot_comparison(frame, folder, font_size=16)
            stem = folder / "nearest_neighbor_vs_full_domain_cumulative_EF_publication"
        elif figure == "S8":
            functions = module("plot_three_adaptive_vs_k5")
            labels = {"fixed_k_5":"K=5", "best_identity_floor":"Adaptive identity (≥55%)",
                      "best_ligand_budget":"Adaptive ligand threshold (≥50)",
                      "identity_ligand_rescue":"Identity ≥55% + ligand rescue ≥50"}
            with plt.rc_context({"font.size":16, "axes.labelsize":18, "xtick.labelsize":16,
                                 "ytick.labelsize":16, "legend.fontsize":16, "legend.title_fontsize":16}):
                functions.plot_comparison(frame, labels, folder, 600)
            stem = folder / "three_adaptive_vs_k5_cumulative_EF"
        else:
            folder = destination / "work/family_appendices"
            if not rendered_appendices:
                execute(command("plot_family_appendices.py",
                    "--representation-summary", source / "representation_summary/category_medians_across_partitions.csv",
                    "--representation-results-dir", source / "representations",
                    "--neighbor-summary", source / "full_domain_benchmark/neighbor_vs_full_domain_category_medians.csv",
                    "--similarity-summary", source / "distributions/figures/histograms_by_family.csv",
                    "--output-dir", folder))
                rendered_appendices = True
                shutil.copyfile(folder / "supplementary_family_appendices.pdf", destination / "supplementary_family_appendices.pdf")
            stem = folder / row["original_stem"]
        for extension in ("png", "pdf", "svg"):
            generated = Path(str(stem) + "." + extension)
            if not generated.is_file():
                raise FileNotFoundError(f"Renderer did not produce {generated}")
            shutil.copyfile(generated, destination / f"{row['filename']}.{extension}")
        if sha256(summary) != before:
            raise RuntimeError(f"Renderer changed source data: {summary}")
        frame.to_csv(destination / f"{row['filename']}_plot_data.csv", index=False)
        receipts.append({"figure":figure, "source_summary_sha256":before,
                         "output_png_sha256":sha256(destination / f"{row['filename']}.png")})
        print(f"Rendered {figure}: {row['filename']}", flush=True)
    write_json(destination / ("all_figures_receipt.json" if args.figure == "all" else f"{args.figure}_receipt.json"),
               {"source":args.source, "figures":receipts, "matplotlib_version":matplotlib.__version__})
