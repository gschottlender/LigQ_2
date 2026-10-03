#!/usr/bin/env python3
"""Rebuild fixed-budget method combinations from cached benchmark rankings.

This script never recalculates molecular similarities. To create a missing
ranking cache, run 02_run_representation_benchmark.py with
--fixed-budget-combinations first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ligq2_evaluation.config import load_config, output_path, parse_csv_arg
from ligq2_evaluation.constants import METHODS, SEEDS
from ligq2_evaluation.fixed_budget import (
    POLICIES, RankingCache, analyze_cache, cache_signature, targets_from_legacy,
)
from ligq2_evaluation.runtime import prepare_output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yml")))
    parser.add_argument("--seeds", help="Comma-separated random states; default is all five benchmark partitions")
    parser.add_argument("--targets", help="Comma-separated target subset")
    parser.add_argument("--methods", help="Comma-separated representation subset; default is all seven")
    parser.add_argument("--fusion-policy", choices=(*POLICIES, "all"), default="best")
    parser.add_argument("--output-dir", type=Path,
                        help="Derived analysis directory; default is the representation output's fixed_budget_complementarity")
    parser.add_argument("--resume", action="store_true", help="Allow regenerating derived files from a valid cache")
    parser.add_argument("--force", action="store_true", help="Allow replacing derived fixed-budget files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = parse_csv_arg(args.seeds, int) or list(SEEDS)
    methods = parse_csv_arg(args.methods) or list(METHODS)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("Choose at least two distinct methods")
    base = output_path(cfg, "representations")
    targets = targets_from_legacy(base, seeds, methods, parse_csv_arg(args.targets))
    cache = RankingCache(base, cache_signature(cfg, methods))
    missing = [(seed, target, method) for seed in seeds for target in targets for method in methods
               if not cache.has(seed, target, method)]
    if missing:
        raise FileNotFoundError(
            f"Missing/stale ranking cache ({len(missing)} entries; first: {missing[:3]}). "
            "Run 02_run_representation_benchmark.py --fixed-budget-combinations --resume first."
        )
    out = args.output_dir.expanduser().resolve() if args.output_dir else base / "fixed_budget_complementarity"
    prepare_output(out, force=args.force, resume=args.resume)
    policies = list(POLICIES) if args.fusion_policy == "all" else [args.fusion_policy]
    metrics_path = analyze_cache(cache, out, seeds=seeds, targets=targets,
                                 methods=methods, policies=policies)
    print(f"Fixed-budget metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
