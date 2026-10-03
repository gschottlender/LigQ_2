"""Dependency-ordered historical calculations; no application database writes."""
from __future__ import annotations

import json
from pathlib import Path

from .common import ROOT, SEEDS, SENSITIVITY_TARGETS, command, execute, read_json, sha256, tree_signature, write_json

DEPENDENCIES = {
    "representations": (), "neighbors": (), "adaptive": ("neighbors",),
    "combinations": ("representations",), "preprocessing": ("representations",),
    "distributions": ("representations",),
}


def ordered_stages(selected):
    result = []
    def add(stage):
        for dependency in DEPENDENCIES[stage]:
            add(dependency)
        if stage not in result:
            result.append(stage)
    for stage in DEPENDENCIES if selected == "all" else (selected,):
        add(stage)
    return result


def calculation_config(data_dir, results):
    data_dir, results = Path(data_dir).resolve(), Path(results).resolve()
    frozen = data_dir / "frozen"
    return {"paths": {
        "binding_data":str(frozen / "binding_data.parquet"), "smiles_data":str(frozen / "smiles_data.parquet"),
        "targets_csv":str(frozen / "targets.csv"), "ligand_store":str(frozen / "compound_data/pdb_chembl"),
        "fasta":str(frozen / "target_sequences.fasta"), "blast_hits":str(frozen / "neighbor_blast_hits.csv"),
        "blast_db_prefix":str(data_dir / "derived/blast/target_sequences"),
    }, "outputs":{"root":str(results), "representations":"representations", "neighbors":"neighbors"},
        "provenance":{"ligq_repo":str(ROOT.parent)},
        "blast":{"n_neighbors":10, "max_evalue":1e-10, "min_bitscore":80., "min_pident":30.,
                 "min_qcovs":60., "num_threads":12}}


def commands(args):
    import csv
    results = args.output_dir.resolve() / "calculations"
    config = results / "calculation_config.json"  # JSON is valid YAML for the retained loaders.
    data = args.data_dir.resolve()
    main_python = args.python
    chemistry_python = args.chemistry_python or main_python
    seeds = ",".join(map(str, SEEDS))
    with (ROOT / "reference_data/cohorts/neighbor_targets.csv").open() as stream:
        targets = ",".join(sorted({row["target"] for row in csv.DictReader(stream)}))
    neighbors = results / "neighbors"
    blast = data / "frozen/neighbor_blast_hits.csv"
    rep = results / "representations"
    sweep = results / "neighbor_k_complete"
    domain = results / "full_domain_benchmark"
    adaptive = results / "adaptive_k"
    rescue = results / "identity_ligand_rescue"
    identity_floors = ",".join(map(str, range(10, 71, 5)))
    budgets = ",".join(map(str, range(50, 501, 50)))
    def main(script, *argv):
        return command(script, *argv, python=main_python)
    def chemistry(script, *argv):
        return command(script, *argv, python=chemistry_python)
    return {
        "representations":[
            main("02_run_representation_benchmark.py", "--config", config, "--seeds", seeds,
                 "--fixed-budget-combinations", "--fusion-policy", "mean", "--resume"),
            main("preview_category_balanced_spread.py", "--input-dir", rep,
                 "--output-dir", results / "representation_summary"),
        ],
        "neighbors":[
            main("03_run_neighbor_benchmark.py", "--config", config, "--seeds", seeds,
                 "--targets", targets, "--neighbor-counts", "2,3,4,5,10,15", "--resume"),
            main("14_complete_neighbor_k_sweep.py", "--config", config, "--retained-root", neighbors,
                 "--blast-hits", blast, "--matched-targets", ROOT / "reference_data/cohorts/neighbor_targets.csv",
                 "--output-dir", sweep, "--neighbor-counts", "2-15", "--seeds", seeds, "--device", "cpu", "--resume"),
            main("run_full_domain_benchmark.py", "--config", config, "--neighbors-dir", neighbors,
                 "--output-dir", domain, "--targets", targets, "--seeds", seeds, "--resume"),
        ],
        "adaptive":[
            main("15_analyze_adaptive_k.py", "--sweep-dir", sweep, "--blast-hits", blast,
                 "--output-dir", adaptive, "--min-k", "2", "--max-k", "15",
                 "--identity-floors", identity_floors, "--ligand-budgets", budgets, "--resume"),
            main("analyze_identity_ligand_rescue_policy.py", "--sweep-dir", sweep, "--adaptive-dir", adaptive,
                 "--blast-hits", blast, "--output-dir", rescue, "--identity-floors", identity_floors,
                 "--ligand-budgets", budgets, "--min-k", "2", "--max-k", "15", "--resume"),
            main("plot_adaptive_k_comparison.py", "--adaptive-dir", adaptive,
                 "--full-domain-input", domain / "neighbor_vs_full_domain_all_seed_rows.csv",
                 "--output-dir", results / "adaptive_k_comparison", "--resume"),
            main("plot_identity_ligand_rescue_comparison.py", "--sweep-dir", sweep, "--rescue-dir", rescue,
                 "--full-domain-input", domain / "neighbor_vs_full_domain_all_seed_rows.csv",
                 "--output-dir", results / "identity_ligand_rescue_comparison",
                 "--identity-floor", "55", "--ligand-budget", "50", "--resume"),
            main("plot_three_adaptive_vs_k5.py", "--adaptive-comparison-dir", results / "adaptive_k_comparison",
                 "--rescue-comparison-dir", results / "identity_ligand_rescue_comparison",
                 "--output-dir", results / "three_adaptive_vs_k5", "--font-size", "16"),
        ],
        "combinations":[
            main("plot_historical_union_precision_recall.py", "--base-dir", rep,
                 "--notebook", ROOT / "configs/historical_union_categories.json", "--output-dir", results / "expanded_union"),
            main("10_run_fixed_budget_complementarity.py", "--config", config, "--seeds", seeds,
                 "--fusion-policy", "mean", "--resume"),
        ],
        "preprocessing":[
            chemistry("18_run_active_preprocessing_sensitivity.py", "--config", config, "--seeds", seeds,
                      "--targets", ",".join(SENSITIVITY_TARGETS), "--butina-cutoff", ".8",
                      "--baseline-results-dir", rep, "--benchmark-dir", data / "benchmark",
                      "--output-dir", results / "preprocessing", "--resume", "--device", "auto"),
        ],
        "distributions":[
            chemistry("19_characterize_retrieved_hits.py", "--config", config, "--reference-dir", rep,
                      "--output-dir", results / "distributions", "--percentile", "99", "--seeds", seeds,
                      "--include-nonretrieved-background", "--descriptor-cache", results / "descriptor_cache",
                      "--descriptor-workers", "8", "--skip-individual", "--resume"),
            chemistry("plot_retrieved_hit_distributions.py", "--input-dir", results / "distributions", "--skip-individual"),
        ],
    }


def verify_runtime(python, *, chemistry=False):
    import subprocess
    code = "import json,numpy,pandas,rdkit,torch; print(json.dumps(dict(numpy=numpy.__version__,pandas=pandas.__version__,rdkit=rdkit.__version__,torch=torch.__version__)))"
    result = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True)
    versions = json.loads(result.stdout)
    expected = ({"numpy":"1.26.4", "pandas":"2.3.1", "rdkit":"2025.03.3"} if chemistry else
                {"numpy":"2.3.4", "pandas":"2.3.3", "rdkit":"2025.09.2", "torch":"2.9.1"})
    mismatch = {key:(versions.get(key), value) for key, value in expected.items()
                if versions.get(key, "").split("+")[0] != value}
    if mismatch:
        raise ValueError(f"Historical environment mismatch: {mismatch}. See environments/ and README.md.")
    return versions


def run(args):
    from .resources import validate_benchmark, verify_files
    stages = ordered_stages(args.evaluation)
    steps = commands(args)
    if args.dry_run:
        for stage in stages:
            print(f"\n[{stage}]")
            for argv in steps[stage]:
                print(" ".join(argv))
        if {"neighbors", "representations"} <= set(stages):
            print("Also derive the sequence/nearest/domain comparison from those results.")
        return
    validate_benchmark(args.data_dir)
    lock = read_json(ROOT / "resource_lock.json")
    verify_files(args.data_dir, lock["frozen_inputs"])
    versions = {"main":verify_runtime(args.python)}
    if {"preprocessing", "distributions"} & set(stages):
        if args.chemistry_python is None:
            raise ValueError("Pass --chemistry-python using the historical RDKit 2025.03.3 environment.")
        versions["chemistry"] = verify_runtime(args.chemistry_python, chemistry=True)
    results = args.output_dir.resolve() / "calculations"
    results.mkdir(parents=True, exist_ok=True)
    signature = {"code":tree_signature(), "resource_lock":sha256(ROOT / "resource_lock.json"),
                 "data_dir":str(args.data_dir.resolve()), "versions":versions}
    checkpoint = results / "execution_checkpoint.json"
    completed = []
    stage_artifacts = {}
    if checkpoint.exists():
        previous = read_json(checkpoint)
        # Checking both interpreters before skipping prevents reuse after dependency changes.
        prior_versions = previous["signature"]["versions"]
        signature["versions"] = {**prior_versions, **versions}
        if previous["signature"] != signature:
            raise ValueError("Inputs, code, paths or environment changed; choose a fresh output directory.")
        if not args.resume:
            raise FileExistsError("Existing execution checkpoint; use --resume or a different output directory.")
        completed = previous["completed_stages"]
        stage_artifacts = previous.get("stage_artifacts", {})
        for stage in completed:
            if not stage_artifacts.get(stage):
                raise ValueError(f"Completed stage lacks verified artifacts: {stage}")
            for name, digest in stage_artifacts[stage].items():
                path = results / name
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError(f"Completed scientific output is missing or changed: {name}")
    elif any(results.iterdir()):
        raise FileExistsError("Untracked calculation outputs found; choose a fresh output directory.")
    config = calculation_config(args.data_dir, results)
    write_json(results / "calculation_config.json", config)
    # Record input/code identity BEFORE the first resumable engine writes outputs.
    write_json(checkpoint, {"signature":signature, "completed_stages":completed, "stage_artifacts":stage_artifacts})
    for stage in stages:
        if stage in completed:
            print(f"Reusing completed {stage}")
            continue
        for argv in steps[stage]:
            execute(argv)
        if stage == "representations":
            from .validation import verify_representation_splits
            verify_representation_splits(results, args.data_dir / "benchmark")
        roots = {
            "representations":["representations", "representation_summary"],
            "neighbors":["neighbors", "neighbor_k_complete", "full_domain_benchmark"],
            "adaptive":["adaptive_k", "identity_ligand_rescue", "adaptive_k_comparison",
                        "identity_ligand_rescue_comparison", "three_adaptive_vs_k5"],
            "combinations":["expanded_union", "representations/fixed_budget_complementarity"],
            "preprocessing":["preprocessing"], "distributions":["distributions"],
        }[stage]
        artifacts = {}
        for name in roots:
            for path in sorted((results / name).rglob("*")):
                if not path.is_file() or path.suffix not in {".csv", ".npz", ".parquet"}:
                    continue
                if stage == "representations" and "fixed_budget_complementarity" in path.parts:
                    continue
                artifacts[str(path.relative_to(results))] = sha256(path)
        if not artifacts:
            raise ValueError(f"Stage produced no scientific artifacts: {stage}")
        stage_artifacts[stage] = artifacts
        completed.append(stage)
        write_json(checkpoint, {"signature":signature, "completed_stages":completed, "stage_artifacts":stage_artifacts})
    if {"neighbors", "representations"} <= set(completed):
        execute(command("plot_ecfp4_transfer_strategy_comparison.py",
            "--representation-dir", results / "representations", "--neighbors-dir", results / "neighbors",
            "--full-domain-dir", results / "full_domain_benchmark",
            "--output-dir", results / "ecfp4_transfer_strategy_comparison", "--resume", python=args.python))
    print("Calculations complete. Run validate, then plot --source recalculated.")
