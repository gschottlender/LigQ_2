from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ligq2_evaluation.full_domain import build_full_domain_sources, slice_binding_for_target


class FullDomainTests(unittest.TestCase):
    def test_sources_share_any_query_pfam_and_exclude_all_benchmark_targets(self):
        targets = pd.DataFrame(
            {"target": ["query", "other"], "uniprot_id": ["Q", "T"]}
        )
        binding = pd.DataFrame(
            {
                "uniprot_id": ["Q", "Q", "T", "A", "A", "B", "C", "D"],
                "pfam_id": ["PF1", "PF2", "PF1", "PF1", "PF3", "PF2", "PF3", "PF1"],
                "chem_comp_id": ["q1", "q2", "t1", "a1", "a2", "b1", "c1", None],
            }
        )

        source = build_full_domain_sources(targets, binding, {"query"})["query"]
        self.assertEqual(source.pfam_ids, ("PF1", "PF2"))
        self.assertEqual(source.source_uniprot_ids, ("A", "B"))
        self.assertEqual(source.n_domain_proteins, 5)
        self.assertEqual(source.n_excluded_benchmark_proteins, 2)

    def test_target_slice_preserves_active_and_background_blacklists(self):
        targets = pd.DataFrame({"target": ["query"], "uniprot_id": ["Q"]})
        binding = pd.DataFrame(
            {
                "uniprot_id": ["Q", "Q", "A", "A", "B", "C"],
                "pfam_id": ["PF1", None, "PF1", "PF2", None, "PF3"],
                "chem_comp_id": ["q1", "q2", "a1", "a2", "b1", "c1"],
            }
        )
        source = build_full_domain_sources(targets, binding, {"query"})["query"]
        reduced = slice_binding_for_target(binding, source)

        def target_actives(table):
            return set(table.loc[table.uniprot_id.eq("Q"), "chem_comp_id"])

        def background_blacklist(table):
            pfams = set(table.loc[table.uniprot_id.eq("Q"), "pfam_id"].astype(str))
            return set(table.loc[table.pfam_id.astype(str).isin(pfams), "chem_comp_id"])

        self.assertEqual(target_actives(reduced), target_actives(binding))
        self.assertEqual(background_blacklist(reduced), background_blacklist(binding))
        self.assertEqual(
            set(reduced.loc[reduced.uniprot_id.eq("A"), "chem_comp_id"]),
            {"a1", "a2"},
        )

    def test_completed_domain_cells_join_retained_k_rows_in_the_figure(self):
        script = Path(__file__).resolve().parents[1] / "engine" / "run_full_domain_benchmark.py"
        spec = importlib.util.spec_from_file_location("full_domain_benchmark_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        targets = {"cdk2", "thrb"}
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for seed in module.SEEDS:
                for target in sorted(targets):
                    paths = module._cell_paths(output, seed, target)
                    paths["base"].mkdir(parents=True)
                    paths["status"].write_text(
                        json.dumps({"seed": seed, "target": target, "eligible": True, "full_seed_budget": True}),
                        encoding="utf-8",
                    )
                    domain_value = 30.0 if target == "cdk2" else 10.0
                    pd.DataFrame(
                        [
                            {
                                "target": target,
                                "seed": seed,
                                "strategy": module.DOMAIN_LABEL,
                                "percentile": percentile,
                                "EF_cumulative": domain_value,
                                "n_seed_requested": 10,
                                "n_seed_effective": 10,
                            }
                            for percentile in module.RAW_PERCENTILES
                        ]
                    ).to_csv(paths["metrics"], index=False)
                    pd.DataFrame([{"target": target, "seed": seed}]).to_csv(
                        paths["metadata"], index=False
                    )
                    for count in module.PLOTTED_NEIGHBOR_COUNTS:
                        for percentile in module.RAW_PERCENTILES:
                            rows.append(
                                {
                                    "target": target,
                                    "seed": seed,
                                    "neighbor_count_sweep": count,
                                    "percentile": percentile,
                                    "EF_cumulative": float(count),
                                    "n_seed_requested": 10,
                                    "n_seed_effective": 10,
                                }
                            )

            matched = module._combine_and_plot(pd.DataFrame(rows), targets, output)
            self.assertEqual(matched, 2)
            spread = pd.read_csv(output / "neighbor_vs_full_domain_category_balanced_spread.csv")
            value = spread[
                spread.strategy.eq(module.DOMAIN_LABEL) & spread.percentile.eq(99.5)
            ].iloc[0]
            self.assertEqual(value["category_balanced_median"], 20.0)
            self.assertEqual(value["category_q25"], 15.0)
            self.assertEqual(value["category_q75"], 25.0)
            self.assertEqual(value["n_categories"], 2)
            self.assertTrue(
                (output / "nearest_neighbor_vs_full_domain_cumulative_EF_publication.pdf").is_file()
            )
            family_values = pd.read_csv(output / "neighbor_vs_full_domain_category_medians.csv")
            kinase_domain = family_values[
                family_values.strategy.eq(module.DOMAIN_LABEL)
                & family_values.percentile.eq(99.5)
                & family_values.familia.eq("Quinasas")
            ].iloc[0]
            self.assertEqual(kinase_domain["category_median_EF_cumulative"], 30.0)
            self.assertEqual(kinase_domain["n_targets"], 1)
            self.assertTrue(
                (output / "nearest_neighbor_vs_full_domain_EF_by_protein_group.pdf").is_file()
            )
            self.assertTrue(
                (output / "family_figures" / "nearest_neighbor_vs_full_domain_kinases.png").is_file()
            )
            self.assertTrue(
                (output / "family_figures" / "nearest_neighbor_vs_full_domain_proteases.svg").is_file()
            )


if __name__ == "__main__":
    unittest.main()
