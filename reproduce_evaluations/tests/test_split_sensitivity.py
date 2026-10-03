from __future__ import annotations

import unittest

import pandas as pd

from ligq2_evaluation.split_sensitivity import (
    PROTOCOL_BM,
    PROTOCOL_FCFP4,
    split_bemis_murcko_representatives,
    split_fcfp4_butina_representatives,
)


class SplitSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.smiles = pd.DataFrame(
            {
                "chem_comp_id": ["benzene", "toluene", "ethylbenzene", "cyclohexane", "ethanol", "propanol"],
                "smiles": ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "C1CCCCC1", "CCO", "CCCO"],
            }
        )
        self.ids = self.smiles["chem_comp_id"].tolist()

    def test_bemis_murcko_uses_one_ring_scaffold_representative_and_keeps_acyclics_separate(self):
        split = split_bemis_murcko_representatives(
            self.ids,
            self.smiles,
            n_known=1,
            n_test=3,
            max_total_actives=None,
            random_state=42,
        )
        self.assertEqual(split.diagnostics["split_mode"], PROTOCOL_BM)
        self.assertEqual(split.diagnostics["n_scaffolds_nonempty"], 2)
        self.assertEqual(split.diagnostics["n_acyclic_structure_groups"], 2)
        self.assertEqual(split.diagnostics["n_representatives"], 4)
        self.assertEqual(len(set(split.known_ids) & set(split.test_active_ids)), 0)
        self.assertEqual(len(split.known_ids), 1)
        self.assertEqual(len(split.test_active_ids), 3)

    def test_fcfp4_butina_split_is_deterministic(self):
        kwargs = dict(
            n_known=2,
            n_test=2,
            max_total_actives=None,
            butina_cutoff=0.8,
            butina_radius=2,
            butina_nbits=1024,
            random_state=8,
        )
        first = split_fcfp4_butina_representatives(self.ids, self.smiles, **kwargs)
        second = split_fcfp4_butina_representatives(self.ids, self.smiles, **kwargs)
        self.assertEqual(first.diagnostics["split_mode"], PROTOCOL_FCFP4)
        self.assertEqual(first.known_ids, second.known_ids)
        self.assertEqual(first.test_active_ids, second.test_active_ids)
        self.assertEqual(len(set(first.known_ids) & set(first.test_active_ids)), 0)
        self.assertEqual(first.diagnostics["butina_cutoff"], 0.8)


if __name__ == "__main__":
    unittest.main()
