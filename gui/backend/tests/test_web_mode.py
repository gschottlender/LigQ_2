from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from starlette.requests import Request

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPOSITORY_ROOT / "gui" / "backend"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
os.environ["LIGQ_DEPLOYMENT_MODE"] = "web"

from core import state  # noqa: E402
from core.config import (  # noqa: E402
    _secret_from_env_or_file,
    _validate_web_session_secret,
)
from core.policy import policy_payload  # noqa: E402
from models.job import Job, JobStatus  # noqa: E402
from query_processing.predicted_cache import (  # noqa: E402
    ReadOnlyCacheError,
    load_provider_cache_read_only,
)
from services.uploads import inspect_fasta_details  # noqa: E402
from services.web_access import require_job_access  # noqa: E402
from services import search_artifacts, web_readiness  # noqa: E402
from services.setup_service import setup_job_args  # noqa: E402
from ligq_support import validate_web_data, web_validation_receipt  # noqa: E402
from routers import jobs as jobs_router  # noqa: E402
import main as backend_main  # noqa: E402
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
    def test_secret_file_takes_precedence_over_environment_value(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "session_secret"
            secret_path.write_text("file-secret\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "TEST_SESSION_SECRET": "environment-secret",
                    "TEST_SESSION_SECRET_FILE": str(secret_path),
                },
            ):
                value = _secret_from_env_or_file(
                    "TEST_SESSION_SECRET",
                    "TEST_SESSION_SECRET_FILE",
                    "default-secret",
                )

        self.assertEqual(value, "file-secret")

    def test_secure_web_mode_rejects_a_weak_session_secret(self):
        with self.assertRaises(RuntimeError):
            _validate_web_session_secret(
                deployment_mode="web",
                cookie_secure=True,
                secret="too-short",
            )

    def test_local_mode_keeps_the_development_secret_compatible(self):
        _validate_web_session_secret(
            deployment_mode="local",
            cookie_secure=False,
            secret="development",
        )

    def test_policy_is_restricted_and_requires_both_cache_floors(self):
        payload = policy_payload()
        self.assertEqual(payload["mode"], "web")
        self.assertFalse(payload["allow_resource_management"])
        self.assertFalse(payload["allow_bsi"])
        self.assertEqual(payload["search"]["provider"], "zinc")
        self.assertEqual(payload["search"]["allowed_methods"], ["sequence", "nearest_k"])
        self.assertEqual(payload["search"]["nearest_k_max"], 10)
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

    def test_successful_deep_validation_receipt_survives_runtime_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            required_paths = ("core.dat", "cache/manifest.json")
            for relative in required_paths:
                path = data_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"contents for {relative}", encoding="utf-8")

            with patch.object(
                web_validation_receipt,
                "required_web_data_paths",
                return_value=required_paths,
            ):
                receipt_path = web_validation_receipt.write_web_validation_receipt(
                    data_dir,
                    {
                        "ready": True,
                        "checks": {
                            "ecfp": {"ready": True, "message": "validated"},
                            "fcfp": {"ready": True, "message": "validated"},
                        },
                    },
                )
                first_status = web_validation_receipt.inspect_web_validation_receipt(
                    data_dir
                )
                # A new inspection has no in-memory state and models a backend restart.
                restarted_status = (
                    web_validation_receipt.inspect_web_validation_receipt(data_dir)
                )

        self.assertTrue(receipt_path.name.endswith("validation.json"))
        self.assertTrue(first_status["ready"])
        self.assertTrue(restarted_status["ready"])
        self.assertTrue(restarted_status["checks"]["receipt"]["ready"])

    def test_runtime_receipt_rejects_data_modified_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            required_path = data_dir / "core.dat"
            required_path.write_text("validated", encoding="utf-8")

            with patch.object(
                web_validation_receipt,
                "required_web_data_paths",
                return_value=("core.dat",),
            ):
                web_validation_receipt.write_web_validation_receipt(
                    data_dir,
                    {"ready": True, "checks": {}},
                )
                required_path.write_text("changed after validation", encoding="utf-8")
                status = web_validation_receipt.inspect_web_validation_receipt(data_dir)

        self.assertFalse(status["ready"])
        self.assertIn("changed", status["errors"][0])

    def test_runtime_receipt_reports_missing_and_corrupt_receipts_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            required_path = data_dir / "core.dat"
            required_path.write_text("present", encoding="utf-8")

            with patch.object(
                web_validation_receipt,
                "required_web_data_paths",
                return_value=("core.dat",),
            ):
                missing_status = (
                    web_validation_receipt.inspect_web_validation_receipt(data_dir)
                )
                web_validation_receipt.validation_receipt_path(data_dir).write_text(
                    "not json",
                    encoding="utf-8",
                )
                corrupt_status = (
                    web_validation_receipt.inspect_web_validation_receipt(data_dir)
                )

        self.assertFalse(missing_status["ready"])
        self.assertIn(
            "Administrative web-data validation is required",
            missing_status["errors"][0],
        )
        self.assertFalse(corrupt_status["ready"])
        self.assertIn("unreadable", corrupt_status["errors"][0])

    def test_validator_cli_writes_receipt_only_after_deep_success(self):
        status = {
            "ready": True,
            "mode": "web",
            "checks": {"core": {"ready": True, "message": "validated"}},
            "errors": [],
        }
        with patch.object(
            sys,
            "argv",
            [
                "validate_web_data.py",
                "--data-dir",
                "/validated/data",
                "--write-receipt",
            ],
        ), patch.object(
            validate_web_data, "inspect_web_data", return_value=status
        ), patch.object(
            validate_web_data,
            "write_web_validation_receipt",
            return_value=Path("/validated/data/.ligq-web-validation.json"),
        ) as write_receipt:
            with self.assertRaises(SystemExit) as raised:
                validate_web_data.main()

        self.assertEqual(raised.exception.code, 0)
        write_receipt.assert_called_once()

    def test_failed_deep_validation_does_not_replace_an_existing_receipt(self):
        status = {
            "ready": False,
            "mode": "web",
            "checks": {"core": {"ready": False, "message": "failed"}},
            "errors": ["deep validation failed"],
        }
        with patch.object(
            sys,
            "argv",
            [
                "validate_web_data.py",
                "--data-dir",
                "/validated/data",
                "--write-receipt",
            ],
        ), patch.object(
            validate_web_data, "inspect_web_data", return_value=status
        ), patch.object(
            validate_web_data, "write_web_validation_receipt"
        ) as write_receipt:
            with self.assertRaises(SystemExit) as raised:
                validate_web_data.main()

        self.assertEqual(raised.exception.code, 1)
        write_receipt.assert_not_called()

    def test_setup_job_runs_the_packaged_support_module(self):
        self.assertEqual(
            setup_job_args()[:2],
            ["-m", "ligq_support.prepare_ligq_2_data"],
        )

    def test_command_line_read_only_guards_remain_opt_in(self):
        with patch.object(
            sys,
            "argv",
            ["run_ligq_2.py", "-i", "queries.faa", "-o", "results"],
        ):
            args = run_ligq_2.parse_args()
        self.assertFalse(args.data_read_only)
        self.assertFalse(args.predicted_cache_read_only)

    def test_web_search_command_uses_one_worker_without_changing_local_default(self):
        common = {
            "fasta_path": Path("/tmp/queries.faa"),
            "output_dir": Path("/tmp/results"),
            "ligand_provider": "zinc",
            "search_representation": "morgan_1024_r2",
            "search_metric": "tanimoto",
            "search_threshold": 0.4,
            "search_threshold_max": 1.0,
            "use_sequence": True,
            "use_nearest_k": True,
            "nearest_k": 5,
            "use_domains": False,
            "known_only": False,
            "use_bsi": False,
            "bsi_threshold": 0.98,
        }

        web_args = jobs_router._build_search_args(
            **common,
            immutable_web_data=True,
        )
        local_args = jobs_router._build_search_args(
            **common,
            immutable_web_data=False,
        )

        worker_flag = web_args.index("--n-workers")
        self.assertEqual(web_args[worker_flag + 1], "1")
        self.assertNotIn("--n-workers", local_args)

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


class WebSearchMethodPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def _start_search(self, **overrides):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/jobs/search",
                "headers": [],
                "client": ("127.0.0.1", 12345),
            }
        )
        params = {
            "request": request,
            "fasta_file": UploadFile(
                filename="queries.fasta",
                file=io.BytesIO(b">query\\nACDE\\n"),
            ),
            "ligand_provider": "zinc",
            "search_representation": "morgan_1024_r2",
            "search_metric": "tanimoto",
            "search_threshold": 0.4,
            "search_threshold_max": 1.0,
            "use_sequence": True,
            "use_nearest_k": True,
            "nearest_k": 5,
            "use_domains": False,
            "known_only": False,
            "use_bsi": False,
            "bsi_threshold": 0.98,
        }
        params.update(overrides)
        with patch.object(
            jobs_router,
            "inspect_web_readiness",
            AsyncMock(return_value={"ready": True}),
        ), patch.object(
            jobs_router,
            "rate_limit_status",
            AsyncMock(return_value=(True, 0)),
        ):
            return await jobs_router.start_search(**params)

    async def test_domain_search_is_rejected_in_web_mode(self):
        response = await self._start_search(use_domains=True)
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(payload["error"], "search_policy_violation")
        self.assertEqual(payload["details"]["field"], "use_domains")

    async def test_nearest_k_above_ten_is_rejected_in_web_mode(self):
        response = await self._start_search(nearest_k=11)
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(payload["error"], "search_policy_violation")
        self.assertEqual(payload["details"]["field"], "nearest_k")


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


class BackendStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_readiness_is_warmed_before_the_app_accepts_requests(self):
        readiness = AsyncMock(
            return_value={"ready": True, "mode": "web", "errors": []}
        )
        initialize = AsyncMock()
        cleanup_resources = AsyncMock()
        cleanup_searches = AsyncMock()
        start_worker = AsyncMock()
        stop_worker = AsyncMock()
        start_cleanup = AsyncMock()
        stop_cleanup = AsyncMock()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(backend_main, "DATABASES_DIR", root / "databases"), patch.object(
                backend_main, "RESULTS_DIR", root / "results"
            ), patch.object(
                backend_main, "UPLOADS_DIR", root / "uploads"
            ), patch.object(
                backend_main, "TEMP_RESULTS_DIR", root / "temp_results"
            ), patch.object(
                backend_main, "STATE_DIR", root / "state"
            ), patch.object(
                backend_main, "is_web_mode", return_value=True
            ), patch.object(
                backend_main, "inspect_web_readiness", readiness
            ), patch.object(
                backend_main.state, "initialize", initialize
            ), patch.object(
                backend_main, "cleanup_stale_resource_jobs", cleanup_resources
            ), patch.object(
                backend_main, "cleanup_stale_web_search_jobs", cleanup_searches
            ), patch.object(
                backend_main, "start_worker", start_worker
            ), patch.object(
                backend_main, "stop_worker", stop_worker
            ), patch.object(
                backend_main, "start_web_cleanup", start_cleanup
            ), patch.object(
                backend_main, "stop_web_cleanup", stop_cleanup
            ):
                async with backend_main.lifespan(backend_main.app):
                    readiness.assert_awaited_once_with(force=True)
                    initialize.assert_awaited_once()
                    start_worker.assert_awaited_once()
                    start_cleanup.assert_awaited_once()

        stop_cleanup.assert_awaited_once()
        stop_worker.assert_awaited_once()

    async def test_local_startup_skips_public_data_validation(self):
        readiness = AsyncMock()
        with patch.object(
            backend_main, "is_web_mode", return_value=False
        ), patch.object(
            backend_main, "inspect_web_readiness", readiness
        ):
            await backend_main._warm_web_readiness_cache()

        readiness.assert_not_awaited()


class RuntimeWebReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        web_readiness.clear_web_readiness_cache()

    async def asyncTearDown(self):
        web_readiness.clear_web_readiness_cache()

    async def test_runtime_uses_the_fast_persistent_receipt_inspection(self):
        expected = {"ready": True, "mode": "web", "checks": {}, "errors": []}
        with patch.object(
            web_readiness, "is_web_mode", return_value=True
        ), patch.object(
            web_readiness,
            "inspect_web_validation_receipt",
            return_value=expected,
        ) as inspect_receipt:
            status = await web_readiness.inspect_web_readiness(force=True)

        self.assertEqual(status, expected)
        inspect_receipt.assert_called_once_with(web_readiness.DATABASES_DIR)

    async def test_runtime_never_emits_a_blank_error_for_receipt_failures(self):
        with patch.object(
            web_readiness, "is_web_mode", return_value=True
        ), patch.object(
            web_readiness,
            "inspect_web_validation_receipt",
            side_effect=TimeoutError(),
        ):
            status = await web_readiness.inspect_web_readiness(force=True)

        self.assertFalse(status["ready"])
        self.assertEqual(
            status["errors"],
            ["Web data receipt inspection failed: TimeoutError"],
        )


if __name__ == "__main__":
    unittest.main()
