"""Scientific and visual regression checks, never a claim about unrun experiments."""
from __future__ import annotations

import ast
import importlib.metadata
import os
import sys
import unittest
from pathlib import Path

from .common import ENGINE, ROOT, SEEDS, read_json, sha256, write_json


def compare_csv(actual, expected, *, mean_policy_only=False):
    import pandas as pd
    a, b = pd.read_csv(actual), pd.read_csv(expected)
    if mean_policy_only:
        a = a.loc[a.policy.eq("mean")].copy()
        b = b.loc[b.policy.eq("mean")].copy()
    if set(a.columns) != set(b.columns):
        raise ValueError(f"Column mismatch: {actual}")
    columns = list(b.columns)
    # Results are compared independently of serialization row order.
    a = a[columns].sort_values(columns).reset_index(drop=True)
    b = b.sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-8)


def verify_representation_splits(results, benchmark, *, seeds=SEEDS):
    """Check exact known/positive/background IDs against downloaded published splits."""
    import pandas as pd
    cache_roots = [p for p in (Path(results) / "representations/ranking_cache").glob("*") if p.is_dir()]
    if len(cache_roots) != 1:
        raise ValueError("Exactly one complete ranking-cache signature is required")
    for seed in seeds:
        table = pd.read_csv(Path(results) / f"representations/seed_{seed}/known_active_sets_all_methods.csv")
        for target, records in table.groupby("target_id"):
            base = Path(benchmark) / str(target) / f"random_state_{seed}"
            known = set(pd.read_csv(base / "known_actives.csv").chem_comp_id.astype(str))
            for values in records.known_active_ids:
                if set(ast.literal_eval(values)) != known:
                    raise ValueError(f"Known-active IDs changed for {target}, seed={seed}")
            positives = set(pd.read_csv(base / "evaluation_actives.csv").chem_comp_id.astype(str))
            background = set(pd.read_csv(base / "putative_inactives.csv").chem_comp_id.astype(str))
            cached = pd.read_parquet(cache_roots[0] / f"seed_{seed}" / str(target) / "pool.parquet")
            if set(cached.loc[cached.is_active, "compound_id"].astype(str)) != positives:
                raise ValueError(f"Evaluation-active IDs changed for {target}, seed={seed}")
            if set(cached.loc[~cached.is_active, "compound_id"].astype(str)) != background:
                raise ValueError(f"Background IDs changed for {target}, seed={seed}")


def image_comparison(actual, expected):
    from PIL import Image, ImageChops, ImageStat
    with Image.open(actual) as a, Image.open(expected) as b:
        if a.size != b.size:
            return {"exact_pixels":False, "actual_size":list(a.size), "reference_size":list(b.size)}
        difference = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        return {"exact_pixels":difference.getbbox() is None, "actual_size":list(a.size),
                "mean_absolute_channel_difference":ImageStat.Stat(difference).mean}


def validate(args):
    from .resources import validate_benchmark, verify_files
    manifest = read_json(ROOT / "figure_manifest.json")
    report = {"scientific_results":[], "figures":[], "full_recalculation_certified":False}
    errors = []
    if not args.figures_only:
        try:
            report["benchmark"] = validate_benchmark(args.data_dir)
            verify_files(args.data_dir, read_json(ROOT / "resource_lock.json")["frozen_inputs"])
            results = args.output_dir / "calculations"
            verify_representation_splits(results, args.data_dir / "benchmark")
            import pandas as pd
            expected_sets = pd.read_csv(ROOT / "reference_data/cohorts/retrieved_set_digests.csv")
            for seed in SEEDS:
                actual_sets = pd.read_csv(results / f"representations/seed_{seed}/retrieved_active_sets_all_methods.csv")
                verify_retrieved_sets(actual_sets, seed, expected_sets)
            for record in read_json(ROOT / "reference_lock.json")["files"]:
                name = record["path"]
                if name.endswith(".csv") and not name.startswith("cohorts/"):
                    actual = results / name
                    try:
                        compare_csv(actual, ROOT / "reference_data" / name,
                                    mean_policy_only=name.endswith("fixed_budget_complementarity/method_combination_summary.csv"))
                        report["scientific_results"].append({"file":name, "matches":True})
                    except (OSError, ValueError, AssertionError) as exc:
                        errors.append(f"{name}: {exc}")
            report["full_recalculation_certified"] = not errors
        except (OSError, ValueError, AssertionError) as exc:
            errors.append(str(exc))
    for row in manifest["figures"]:
        actual = args.output_dir / "figures" / f"{row['filename']}.png"
        reference = ROOT / "reference_data/images" / f"{row['id']}.png"
        if not actual.is_file():
            errors.append(f"Figure not rendered: {row['id']}")
            continue
        comparison = image_comparison(actual, reference)
        report["figures"].append({"figure":row["id"], **comparison})
        if not comparison["exact_pixels"]:
            errors.append(f"Raster differs from historical figure: {row['id']}. Check the rendering environment.")
    report["errors"] = errors
    report["all_checks_passed"] = not errors
    write_json(args.output_dir / "validation_report.json", report)
    if errors:
        raise ValueError("Validation did not pass:\n" + "\n".join(errors[:20]))
    print("Validation passed. Figure-only checks do not certify a full recalculation.")


def smoke_test(output_dir):
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    for path in (ENGINE, ENGINE / "dependencies/evaluation_core", ENGINE / "dependencies/ligq_core", ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    versions = {}
    for name in ("numpy", "pandas", "rdkit", "matplotlib", "torch"):
        try:
            import importlib
            versions[name] = importlib.import_module(name).__version__
        except (ImportError, AttributeError):
            versions[name] = None
    write_json(Path(output_dir) / "smoke_test_report.json", {
        "tests_run":result.testsRun, "errors":len(result.errors), "failures":len(result.failures),
        "passed":result.wasSuccessful(), "versions":versions, "full_benchmark_started":False,
    })
    if not result.wasSuccessful():
        raise ValueError("Small reproducibility tests failed")


def real_smoke_test(inventory, output_dir, python):
    """One real PGH1 cell: original clustering, scoring, EF, and exact pool IDs."""
    import pandas as pd
    from .common import command, execute
    from .workflow import verify_runtime

    versions = verify_runtime(python)
    source = read_json(inventory)["sources"]
    data_paths = {
        "binding_data":"frozen/binding_data.parquet", "smiles_data":"frozen/smiles_data.parquet",
        "targets_csv":"frozen/targets.csv", "fasta":"frozen/target_sequences.fasta",
        "blast_hits":"frozen/neighbor_blast_hits.csv",
    }
    out = Path(output_dir).resolve() / "real_pgh1_seed42"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"paths":{key:source[value] for key, value in data_paths.items()},
           "outputs":{"root":str(out), "representations":"representations"},
           "provenance":{"ligq_repo":str(ROOT.parent)}}
    cfg["paths"]["ligand_store"] = str(Path(source["frozen/compound_data/pdb_chembl/ligands.parquet"]).parent)
    cfg["paths"]["blast_db_prefix"] = str(out / "unused_blast")
    config_path = out / "config.json"
    write_json(config_path, cfg)
    execute(command("02_run_representation_benchmark.py", "--config", config_path,
        "--targets", "pgh1", "--seeds", "42", "--methods", "morgan_1024_r2", "--resume", python=python))
    actual = pd.read_csv(out / "representations/seed_42/df_long_target_all_methods.csv")
    historical = pd.read_csv(ROOT / "reference_data/representations/seed_42/df_long_target_all_methods.csv")
    historical = historical.loc[historical.target.eq("pgh1") & historical.method_label.eq("morgan_1024_r2")]
    pd.testing.assert_frame_equal(actual.sort_values("percentile").reset_index(drop=True),
        historical[actual.columns].sort_values("percentile").reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-8)
    known = pd.read_csv(out / "representations/seed_42/known_active_sets_all_methods.csv")
    expected_known = pd.read_csv(source["benchmark/pgh1/random_state_42/known_actives.csv"])
    if set(ast.literal_eval(known.known_active_ids.iloc[0])) != set(expected_known.chem_comp_id.astype(str)):
        raise ValueError("Bounded calculation changed published known-active IDs")
    retrieved = pd.read_csv(out / "representations/seed_42/retrieved_active_sets_all_methods.csv")
    expected_hashes = pd.read_csv(ROOT / "reference_data/cohorts/retrieved_set_digests.csv")
    verify_retrieved_sets(retrieved, 42, expected_hashes)
    write_json(Path(output_dir) / "real_smoke_test_report.json", {
        "target":"pgh1", "seed":42, "method":"morgan_1024_r2", "versions":versions,
        "EF_rows_match_historical":True, "known_active_IDs_match_published":True,
        "retrieved_ID_sets_match_historical":True, "full_benchmark_started":False,
    })


def set_digest(values):
    import hashlib
    return hashlib.sha256("\n".join(sorted(set(map(str, values)))).encode()).hexdigest()


def verify_retrieved_sets(frame, seed, expected):
    keys = ["seed", "target_id", "method_label", "percentile"]
    indexed = expected.set_index(keys)
    for row in frame.itertuples():
        key = (seed, row.target_id, row.method_label, row.percentile)
        record = indexed.loc[key]
        for field, column in (("active_digest", "retrieved_active_ids"), ("inactive_digest", "retrieved_inactive_ids")):
            values = getattr(row, column)
            if set_digest(ast.literal_eval(values) if isinstance(values, str) else values) != record[field]:
                raise ValueError(f"Retrieved ligand IDs changed for {key}")
