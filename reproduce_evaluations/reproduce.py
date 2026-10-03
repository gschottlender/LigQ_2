#!/usr/bin/env python3
"""Reproduce the result figures included in the LigQ2 manuscript and SI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from package.common import ROOT, read_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="List included figures and excluded experiments")
    for name in ("download", "import-local", "run", "validate"):
        item = actions.add_parser(name)
        item.add_argument("--data-dir", type=Path, default=ROOT / "data")
        if name in ("run", "validate"):
            item.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
        if name == "download":
            item.add_argument("--artifact-repository")
            item.add_argument("--artifact-revision")
            item.add_argument("--offline", action="store_true")
            item.add_argument("--only-benchmark", action="store_true")
            item.add_argument("--targets", help="Bounded download: comma-separated benchmark target names")
            item.add_argument("--seeds", help="Bounded download: comma-separated original partition seeds")
        if name == "import-local":
            item.add_argument("--inventory", type=Path, required=True)
            item.add_argument("--include-benchmark", action="store_true")
        if name == "run":
            item.add_argument("--evaluation", choices=("all", "representations", "neighbors", "adaptive",
                                                     "combinations", "preprocessing", "distributions"), default="all")
            item.add_argument("--resume", action="store_true")
            item.add_argument("--dry-run", action="store_true", help="Print commands; do not calculate or write")
            item.add_argument("--python", type=Path, default=Path(sys.executable), help="Main historical calculation interpreter")
            item.add_argument("--chemistry-python", type=Path, help="Historical RDKit 2025.03.3 interpreter for preprocessing and properties")
        if name == "validate":
            item.add_argument("--figures-only", action="store_true")
    item = actions.add_parser("export-resources", help="Prepare frozen resources for publication; never upload")
    item.add_argument("--inventory", type=Path, required=True)
    item.add_argument("--output-dir", type=Path, default=ROOT / "publication_export")
    item = actions.add_parser("plot")
    item.add_argument("--figure", default="all", choices=("all", "Figure_2", "Figure_3", *(f"S{i}" for i in range(1, 9)), "A1", "A2", "A3"))
    item.add_argument("--source", choices=("historical", "recalculated"), default="historical")
    item.add_argument("--results-dir", type=Path, default=ROOT / "outputs/calculations")
    item.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    item = actions.add_parser("smoke-test", help="Small synthetic tests, no full evaluation")
    item.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke_test")
    item.add_argument("--inventory", type=Path, help="Also recalculate one real target/partition from verified local inputs")
    item.add_argument("--calculation-python", type=Path, help="Historical main interpreter for the bounded real calculation")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "list":
        for row in read_json(ROOT / "figure_manifest.json")["figures"]:
            print(f"{row['id']:10} {row['title']}")
        print("Excluded: workflow, computational performance, table/text-only experiments.")
    elif args.action in ("download", "import-local", "export-resources"):
        from package import resources
        if args.action == "download":
            resources.download(args.data_dir, artifact_repository=args.artifact_repository,
                artifact_revision=args.artifact_revision, offline=args.offline, only_benchmark=args.only_benchmark,
                targets=args.targets.split(",") if args.targets else None,
                seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else None)
        elif args.action == "import-local":
            resources.import_local(args.inventory, args.data_dir, include_benchmark=args.include_benchmark)
        else:
            size = resources.export_resources(args.inventory, args.output_dir)
            print(f"Prepared {size:,} bytes for publication. Nothing was uploaded.")
    elif args.action == "run":
        from package.workflow import run
        run(args)
    elif args.action == "plot":
        from package.figures import plot
        plot(args)
    elif args.action == "validate":
        from package.validation import validate
        validate(args)
    else:
        from package.validation import smoke_test
        smoke_test(args.output_dir)
        if args.inventory:
            from package.validation import real_smoke_test
            real_smoke_test(args.inventory, args.output_dir, args.calculation_python or Path(sys.executable))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
