import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core import state
from models.job import Job, JobStatus
from routers import results


class ClearHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_history_deletes_old_results_and_preserves_active_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            old_result = results_dir / "old_search"
            active_result = results_dir / "active_search"
            old_result.mkdir(parents=True)
            active_result.mkdir(parents=True)
            (old_result / "search_results_summary.tsv").write_text("qseqid\nold\n")
            (active_result / "partial.tsv").write_text("running\n")

            job_id = "active-history-test"
            await state.set_job(
                Job(
                    job_id=job_id,
                    job_type="search",
                    status=JobStatus.running,
                    created_at=datetime.now(timezone.utc),
                    output_dir=str(active_result),
                )
            )

            try:
                with patch.object(results, "RESULTS_DIR", results_dir):
                    response = await results.clear_results_history()
            finally:
                await state.delete_job(job_id)

            self.assertEqual(response["deleted_count"], 1)
            self.assertEqual(response["deleted_results"], ["old_search"])
            self.assertEqual(response["skipped_active"], ["active_search"])
            self.assertEqual(response["failed_results"], [])
            self.assertFalse(old_result.exists())
            self.assertTrue(active_result.exists())

    async def test_clear_history_handles_missing_results_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_results = Path(tmp_dir) / "missing"
            with patch.object(results, "RESULTS_DIR", missing_results):
                response = await results.clear_results_history()

        self.assertEqual(response["deleted_count"], 0)
        self.assertEqual(response["deleted_results"], [])
        self.assertEqual(response["skipped_active"], [])
        self.assertEqual(response["failed_results"], [])


if __name__ == "__main__":
    unittest.main()
