from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_processing import predicted_cache
from query_processing.results_tables import build_predicted_binding_data_incremental


class EmptyProvider:
    provider_name = "test"

    def compute_for_protein(self, prot: str, known_binding: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def cache_method_signature(self) -> dict:
        return {
            "provider": self.provider_name,
            "search_representation": "test_representation",
            "search_metric": "tanimoto",
        }

    def cache_coverage(self) -> tuple[float, None]:
        return 0.5, None

    def with_cache_coverage(self, threshold_min: float | None, threshold_max: float | None):
        return self

    def database_fingerprint(self, data_dir: Path) -> str:
        return "test-fingerprint"


class PredictedCacheProgressTests(unittest.TestCase):
    def test_incremental_builder_reports_every_processed_protein(self) -> None:
        events: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            build_predicted_binding_data_incremental(
                proteins_to_process=["P1", "P2", "P3"],
                cache_dir=Path(temp_dir),
                provider=EmptyProvider(),
                known_binding=pd.DataFrame(columns=["uniprot_id"]),
                resume=False,
                progress_callback=lambda current, total: events.append((current, total)),
            )

        self.assertEqual(events, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_cache_progress_starts_with_requested_proteins_already_cached(self) -> None:
        events: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            cache_dir = data_dir / "cache"
            cache_dir.mkdir(parents=True)
            (cache_dir / "cached_proteins.json").write_text('["cached"]')

            def fake_build(*, proteins_to_process, progress_callback, **kwargs) -> None:
                self.assertEqual(proteins_to_process, ["new"])
                progress_callback(0, 1)
                progress_callback(1, 1)

            with patch.object(
                predicted_cache,
                "_discover_compatible_cache",
                return_value=(cache_dir, None),
            ), patch.object(
                predicted_cache,
                "build_predicted_binding_data_incremental",
                side_effect=fake_build,
            ):
                predicted_cache.ensure_provider_cache(
                    data_dir=data_dir,
                    provider=EmptyProvider(),
                    known_binding=pd.DataFrame({"uniprot_id": ["cached", "new"]}),
                    proteins_needed={"cached", "new"},
                    load_dataframe=False,
                    progress_callback=lambda current, total: events.append((current, total)),
                )

        self.assertEqual(events[0], (1, 2))
        self.assertEqual(events[-1], (2, 2))


if __name__ == "__main__":
    unittest.main()
