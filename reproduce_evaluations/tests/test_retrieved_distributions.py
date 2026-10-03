import importlib
import unittest

import numpy as np
import pandas as pd
from rdkit import rdBase

from ligq2_evaluation.retrieved_background import classify_candidates


class RetrievedDistributionTests(unittest.TestCase):
    def test_inclusive_cutoff_and_no_false_negatives_in_background(self):
        self.assertEqual(classify_candidates([True, True, False, False],
                         [.7, .6, .7, .6], .7).tolist(),
                         ["TP", "FN", "putative_FP", "putative_TN"])

    def test_properties_keep_original_difference_and_invalid_smiles(self):
        analysis = importlib.import_module("19_characterize_retrieved_hits")
        values = analysis.molecular_properties("CC(=O)O")
        self.assertEqual(values["hba_minus_hbd"], values["hba"]-values["hbd"])
        self.assertTrue(values["valid_structure"])
        with rdBase.BlockLogs():
            self.assertFalse(analysis.molecular_properties("not-a-smiles")["valid_structure"])

    def test_equal_weights_are_means_of_normalized_histograms(self):
        analysis = importlib.import_module("19_characterize_retrieved_hits")
        records = []
        for target, family, seed, value, size in [
            ("a1", "A", 3, .1, 100), ("a1", "A", 8, .9, 1),
            ("a2", "A", 3, .1, 10), ("b1", "B", 3, .9, 20)]:
            for group in analysis.GROUPS:
                records.extend({"target":target, "protein_family":family,
                    "partition":seed, "group":group, "similarity":value} for _ in range(size))
        histograms, issues = analysis.build_histograms(pd.DataFrame(records), {"similarity":[0, .5, 1]})
        self.assertFalse(issues)
        _, _, overall = analysis.aggregate_histograms(histograms)
        np.testing.assert_allclose(overall.loc[overall.bin.eq(0), "fraction"], .375)


if __name__ == "__main__":
    unittest.main()
