import inspect
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from routers.jobs import (
    BSI_CLI_METRIC,
    BSI_DEFAULT_THRESHOLD,
    BSI_REPRESENTATION,
    _build_search_args,
    start_search,
)


def _option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


class SearchArgumentTests(unittest.TestCase):
    def test_search_endpoint_exposes_bsi_defaults(self) -> None:
        parameters = inspect.signature(start_search).parameters

        self.assertFalse(parameters["use_bsi"].default.default)
        self.assertEqual(parameters["bsi_threshold"].default.default, 0.98)

    def test_bsi_search_forces_morgan_and_uses_bsi_cutoff(self) -> None:
        args = _build_search_args(
            fasta_path=Path("queries.fasta"),
            output_dir=Path("results/bsi"),
            ligand_provider="zinc",
            search_representation="chemberta_zinc_base_768",
            search_metric="bsi",
            search_threshold=0.4,
            search_threshold_max=0.8,
            use_sequence=True,
            use_nearest_k=True,
            nearest_k=5,
            use_domains=True,
            known_only=False,
            use_bsi=True,
            bsi_threshold=BSI_DEFAULT_THRESHOLD,
        )

        self.assertEqual(_option_value(args, "--search-representation"), BSI_REPRESENTATION)
        self.assertEqual(_option_value(args, "--search-metric"), BSI_CLI_METRIC)
        self.assertEqual(_option_value(args, "--bsi-threshold"), "0.98")
        self.assertIn("--bsi", args)
        self.assertNotIn("--search-threshold", args)
        self.assertNotIn("--search-threshold-max", args)

    def test_structural_search_preserves_selected_representation_and_cutoffs(self) -> None:
        args = _build_search_args(
            fasta_path=Path("queries.fasta"),
            output_dir=Path("results/structural"),
            ligand_provider="zinc",
            search_representation="chemberta_zinc_base_768",
            search_metric="cosine",
            search_threshold=0.94,
            search_threshold_max=1.0,
            use_sequence=True,
            use_nearest_k=False,
            nearest_k=5,
            use_domains=False,
            known_only=False,
            use_bsi=False,
            bsi_threshold=BSI_DEFAULT_THRESHOLD,
        )

        self.assertEqual(
            _option_value(args, "--search-representation"),
            "chemberta_zinc_base_768",
        )
        self.assertEqual(_option_value(args, "--search-metric"), "cosine")
        self.assertEqual(_option_value(args, "--search-threshold"), "0.94")
        self.assertEqual(_option_value(args, "--search-threshold-max"), "1.0")
        self.assertNotIn("--bsi", args)
        self.assertNotIn("--bsi-threshold", args)


if __name__ == "__main__":
    unittest.main()
