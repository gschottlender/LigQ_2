from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPOSITORY_ROOT / "gui" / "backend"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
os.environ["LIGQ_DEPLOYMENT_MODE"] = "web"

from core import state  # noqa: E402
from core.policy import policy_payload  # noqa: E402
from models.job import Job, JobStatus  # noqa: E402
from query_processing.predicted_cache import (  # noqa: E402
    ReadOnlyCacheError,
    load_provider_cache_read_only,
)
from services.uploads import inspect_fasta_details  # noqa: E402
from services.web_access import require_job_access  # noqa: E402
from services import search_artifacts  # noqa: E402
import validate_web_data  # noqa: E402
import run_ligq_2  # noqa: E402


class _FakeProvider:
    provider_name = "zinc"

    def cache_coverage(self):
        return 0.4, 1.0

    def cache_method_signature(self):
        return {
            "search_representation": "morgan_1024_r2",
            "search_metric": "tanimoto",
        }

    def database_fingerprint(self, _data_dir):
        return "test-fingerprint"

    def database_fingerprint_version(self):
        return 2


class WebPolicyTests(unittest.TestCase):
    def test_policy_is_restricted_and_requires_both_cache_floors(self):
        payload = policy_payload()
        self.assertEqual(payload["mode"], "web")
        self.assertFalse(payload["allow_resource_management"])
        self.assertFalse(payload["allow_bsi"])
        self.assertEqual(payload["search"]["provider"], "zinc")
        self.assertEqual(payload["search"]["max_fasta_sequences"], 100)
        self.assertEqual(payload["search"]["rate_limit_count"], 20)
        self.assertEqual(
            {
                item["name"]: item["cache_threshold_min"]
                for item in payload["search"]["representations"]
            },
            {
                "morgan_1024_r2": 0.4,
                "morgan_feature_1024_r2": 0.5,
            },
        )

    def test_fasta_inspection_counts_limits_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "queries.faa"
            fasta.write_text(
                ">same first\nACDE\n>unique\nFGHI\n>same duplicate\nKLMN\n",
                encoding="utf-8",
            )
            inspection = inspect_fasta_details(fasta)

        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.sequence_count, 3)
        self.assertEqual(inspection.total_residues, 12)
        self.assertEqual(inspection.duplicate_ids, ["same"])

    def test_read_only_cache_loader_does_not_create_missing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            before = set(data_dir.rglob("*"))
            with patch(
                "query_processing.predicted_cache._discover_compatible_cache",
                return_value=(None, None),
            ):
                with self.assertRaises(ReadOnlyCacheError):
                    load_provider_cache_read_only(
                        data_dir,
                        _FakeProvider(),
                        {"P12345"},
                    )
            after = set(data_dir.rglob("*"))

        self.assertEqual(before, after)

    def test_web_readiness_fails_if_either_cache_package_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            for relative in ("core.dat", "ecfp.dat"):
                path = data_dir / relative
                path.write_text("present", encoding="utf-8")
            with patch.object(
                validate_web_data, "CORE_DATA_PATHS", ("core.dat",)
            ), patch.object(
                validate_web_data, "ECFP_CACHE_PATHS", ("ecfp.dat",)
            ), patch.object(
                validate_web_data, "FCFP_CACHE_PATHS", ("fcfp.dat",)
            ):
                status = validate_web_data.inspect_web_data(data_dir)

        self.assertFalse(status["ready"])
        self.assertTrue(any("fcfp.dat" in error for error in status["errors"]))

    def test_command_line_read_only_guards_remain_opt_in(self):
        with patch.object(
            sys,
            "argv",
            ["run_ligq_2.py", "-i", "queries.faa", "-o", "results"],
        ):
            args = run_ligq_2.parse_args()
        self.assertFalse(args.data_read_only)
        self.assertFalse(args.predicted_cache_read_only)

    def test_job_access_is_scoped_to_anonymous_session(self):
        from datetime import datetime, timezone

        request = Request({"type": "http", "headers": []})
        request.state.session_hash = "owner-a"
        job = Job(
            job_id="private-job",
            job_type="search",
            status=JobStatus.completed,
            created_at=datetime.now(timezone.utc),
            owner_session_hash="owner-a",
        )
        self.assertIs(require_job_access(request, job), job)
        request.state.session_hash = "owner-b"
        with self.assertRaises(HTTPException) as raised:
            require_job_access(request, job)
        self.assertEqual(raised.exception.status_code, 404)

    def test_cancel_cleanup_removes_local_search_artifacts(self):
        from datetime import datetime, timezone

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uploads_dir = root / "uploads"
            temp_results_dir = root / "temp_results"
            results_dir = root / "results"
            input_path = uploads_dir / "local-job.fasta"
            temp_path = temp_results_dir / input_path.stem
            output_dir = results_dir / "local-run"

            input_path.parent.mkdir(parents=True)
            input_path.write_text(">query\nACDE\n", encoding="utf-8")
            temp_path.mkdir(parents=True)
            (temp_path / "partial.tmp").write_text("partial", encoding="utf-8")
            output_dir.mkdir(parents=True)
            (output_dir / "partial.csv").write_text("partial", encoding="utf-8")

            job = Job(
                job_id="local-job",
                job_type="search",
                status=JobStatus.cancelled,
                created_at=datetime.now(timezone.utc),
                owner_session_hash=None,
                input_path=str(input_path),
                output_dir=str(output_dir),
            )
            with patch.object(search_artifacts, "UPLOADS_DIR", uploads_dir), patch.object(
                search_artifacts, "TEMP_RESULTS_DIR", temp_results_dir
            ), patch.object(search_artifacts, "RESULTS_DIR", results_dir):
                search_artifacts.cleanup_search_artifacts(job, remove_results=True)

            self.assertFalse(input_path.exists())
            self.assertFalse(temp_path.exists())
            self.assertFalse(output_dir.exists())


class ExclusiveAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        state.jobs.clear()

    async def asyncTearDown(self):
        state.jobs.clear()

    async def test_only_one_web_search_can_be_admitted(self):
        from datetime import datetime, timezone

        first = Job(
            job_id="first",
            job_type="search",
            status=JobStatus.queued,
            created_at=datetime.now(timezone.utc),
            owner_session_hash="owner-a",
        )
        second = first.model_copy(
            update={"job_id": "second", "owner_session_hash": "owner-b"}
        )

        self.assertTrue(await state.try_set_exclusive_web_search(first))
        self.assertFalse(await state.try_set_exclusive_web_search(second))


if __name__ == "__main__":
    unittest.main()
