from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ligq2_evaluation.fixed_budget import RankingCache, analyze_cache, select_compounds


METHOD_A = "morgan_1024_r2"
METHOD_B = "morgan_feature_1024_r2"
METHOD_C = "topological_torsion_rdkit_1024"


class FixedBudgetTests(unittest.TestCase):
    def test_optional_score_sink_does_not_change_legacy_evaluation(self):
        import armado_datasets_modified as core

        ids = ["known", "active", "inactive1", "inactive2", "inactive3", "inactive4"]
        targets = pd.DataFrame({"target": ["cdk2"], "uniprot_id": ["Q"]})
        binding = pd.DataFrame({"uniprot_id": ["Q", "Q"], "pfam_id": ["PF1", "PF1"],
                                "chem_comp_id": ["known", "active"]})
        smiles = pd.DataFrame({"chem_comp_id": ids, "smiles": ["C"] * len(ids)})
        store = SimpleNamespace(ligands=smiles)
        representation = SimpleNamespace(id_to_idx={value: index for index, value in enumerate(ids)})
        lookup = {"active": .9, "inactive1": .8, "inactive2": .5,
                  "inactive3": .2, "inactive4": .1}

        def split_function(**_kwargs):
            return SimpleNamespace(known_ids=["known"], test_active_ids=["active"])

        def score_function(_seed_ids, pool_ids, **_kwargs):
            return np.asarray([lookup[value] for value in pool_ids], dtype=np.float32)

        kwargs = dict(targets_dude=targets, binding_data=binding, smiles=smiles, store=store,
                      neighbor_ranked_ids_by_target={}, rep_eval=representation,
                      metric_eval="tanimoto", method_label=METHOD_A,
                      use_neighbor_seeds=False, min_known_for_eval=1,
                      split_kwargs={"random_state": 3}, split_function=split_function,
                      verbose=False)
        observed = []
        with patch.object(core, "max_score_for_pool_ids", side_effect=score_function):
            original = core.run_ef_eval_target_vs_neighbors(**kwargs)
            with_sink = core.run_ef_eval_target_vs_neighbors(
                **kwargs, target_score_sink=lambda **row: observed.append(row))
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["target"], "cdk2")
        self.assertEqual(len(observed[0]["scores"]), 5)
        for left, right in zip(original, with_sink):
            pd.testing.assert_frame_equal(left, right)

    def test_best_mean_and_balanced_are_label_blind_and_budget_exact(self):
        ids = np.asarray(list("abcdef"))
        first = np.asarray([1, 2, 3, 4, 5, 6])
        second = np.asarray([6, 1, 2, 5, 3, 4])
        third = np.asarray([3, 6, 5, 4, 2, 1])
        rankings = {
            METHOD_A: {"scores": np.asarray([.9, .8, .7, .6, .5, .4]),
                       "ranks": first, "cut": .7},
            METHOD_B: {"scores": np.asarray([.1, .95, .85, .2, .8, .3]),
                       "ranks": second, "cut": .8},
            METHOD_C: {"scores": np.asarray([.7, .4, .5, .6, .8, .9]),
                       "ranks": third, "cut": .7},
        }
        best, _, _ = select_compounds(ids, rankings, (METHOD_A, METHOD_B), "best", 2)
        mean, _, _ = select_compounds(ids, rankings, (METHOD_A, METHOD_B), "mean", 2)
        self.assertEqual(ids[best].tolist(), ["b", "a"])
        self.assertEqual(ids[mean].tolist(), ["b", "c"])
        balanced, _, sources = select_compounds(ids, rankings, (METHOD_A, METHOD_B), "balanced", 4)
        self.assertEqual(ids[balanced].tolist(), ["a", "b", "c", "e"])
        self.assertEqual(np.bincount(sources, minlength=2).tolist(), [2, 2])
        triple, _, sources = select_compounds(
            ids, rankings, (METHOD_A, METHOD_B, METHOD_C), "balanced", 5)
        self.assertEqual(len(set(ids[triple])), 5)
        self.assertEqual(np.bincount(sources, minlength=3).tolist(), [2, 2, 1])
        with self.assertRaises(ValueError):
            select_compounds(ids, rankings, (METHOD_A, METHOD_B), "best", 5)

    def test_cache_reuses_scores_and_detects_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RankingCache(Path(directory), "test_signature")
            for method, scores in ((METHOD_A, [.9, .8, .2, .1]),
                                   (METHOD_B, [.1, .8, .9, .2])):
                cache.save(seed=3, target="cdk2", method=method,
                           pool_ids=["d", "b", "a", "c"], positive_ids=["a"],
                           seed_ids=["known1"], scores=scores, score_cut_995=.8)
            self.assertTrue(cache.has(3, "cdk2", METHOD_A))
            pool, rankings = cache.load(3, "cdk2", [METHOD_A, METHOD_B])
            self.assertEqual(pool.compound_id.tolist(), ["a", "b", "c", "d"])
            self.assertEqual(rankings[METHOD_A]["ranks"].tolist(), [3, 2, 4, 1])
            metadata = cache.root / "seed_3/cdk2/morgan_1024_r2.json"
            record = json.loads(metadata.read_text(encoding="utf-8"))
            record["signature"] = "stale"
            metadata.write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(cache.has(3, "cdk2", METHOD_A))

    def test_analysis_writes_all_strategies_without_full_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = RankingCache(root, "test_signature")
            ids = [f"compound_{i}" for i in range(20)]
            for method, values in ((METHOD_A, np.linspace(1, 0, 20)),
                                   (METHOD_B, np.linspace(0, 1, 20)),
                                   (METHOD_C, np.roll(np.linspace(1, 0, 20), 5))):
                cache.save(seed=3, target="cdk2", method=method, pool_ids=ids,
                           positive_ids=[ids[0], ids[-1]], seed_ids=["known"],
                           scores=values, score_cut_995=float(np.percentile(values, 99.5)))
            out = root / "fixed_budget"
            analyze_cache(cache, out, seeds=[3], targets=["cdk2"],
                          methods=[METHOD_A, METHOD_B, METHOD_C],
                          policies=["best", "mean", "balanced"])
            metrics = pd.read_csv(out / "target_partition_budget_metrics.csv")
            self.assertEqual(len(metrics), 21)  # 3 policies × (3 singles + 3 pairs + 1 triple)
            self.assertTrue((metrics.budget == 1).all())
            self.assertTrue((out / "fixed_budget_highlighted_recall.pdf").is_file())
            analyze_cache(cache, out, seeds=[3], targets=["cdk2"],
                          methods=[METHOD_A, METHOD_B, METHOD_C], policies=["mean"])
            self.assertEqual(len(pd.read_csv(out / "target_partition_budget_metrics.csv")), 21)
            for policy in ("best", "mean", "balanced"):
                selected = pd.read_parquet(out / f"selected_rankings/{policy}/seed_3/cdk2.parquet")
                self.assertEqual(len(selected), 7)

    def test_historical_validation_rejects_changed_sets(self):
        script = Path(__file__).resolve().parents[1] / "engine" / "02_run_representation_benchmark.py"
        spec = importlib.util.spec_from_file_location("representation_benchmark_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            long = pd.DataFrame([{"target": "cdk2", "method": f"{METHOD_A}__target_seeds",
                                  "percentile": 99.5, "EF_cumulative": 5.0}])
            long.assign(method_label=METHOD_A).to_csv(seed_dir / "df_long_target_all_methods.csv", index=False)
            retrieved = pd.DataFrame([{
                "target_id": "cdk2", "method_label": f"{METHOD_A}__target_seeds",
                "percentile": 99.5, "retrieved_active_ids": ["a"],
                "retrieved_inactive_ids": ["b"],
            }])
            retrieved.to_csv(seed_dir / "retrieved_active_sets_all_methods.csv", index=False)
            module._assert_historical_unchanged(seed_dir, METHOD_A, ["cdk2"], long, retrieved)
            changed = retrieved.copy()
            changed.at[0, "retrieved_active_ids"] = ["x"]
            with self.assertRaises(AssertionError):
                module._assert_historical_unchanged(seed_dir, METHOD_A, ["cdk2"], long, changed)


if __name__ == "__main__":
    unittest.main()
