from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core import state
from models.job import Job, JobStatus
from services import job_runner
from services.uploads import UploadTooLargeError, inspect_fasta, save_upload_stream


def _job(job_id: str, status: JobStatus = JobStatus.queued) -> Job:
    return Job(
        job_id=job_id,
        job_type="search",
        status=status,
        created_at=datetime.now(timezone.utc),
    )


def test_job_state_survives_restart_and_interrupts_unfinished_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(state, "JOB_DB_PATH", tmp_path / "state" / "jobs.sqlite3")
        state.jobs.clear()
        state._initialized = False
        await state.initialize()
        await state.set_job(_job("running-job", JobStatus.running))
        await state.set_job(_job("completed-job", JobStatus.completed))

        state.jobs.clear()
        state._initialized = False
        await state.initialize()

        interrupted = await state.get_job("running-job")
        completed = await state.get_job("completed-job")
        assert interrupted is not None
        assert interrupted.status == JobStatus.interrupted
        assert interrupted.failure is not None
        assert completed is not None
        assert completed.status == JobStatus.completed

    try:
        asyncio.run(scenario())
    finally:
        state.jobs.clear()
        state.processes.clear()
        state._initialized = False


def test_queue_runs_only_one_heavy_job_at_a_time(monkeypatch) -> None:
    async def scenario() -> None:
        active = 0
        maximum_active = 0
        order: list[str] = []

        async def fake_run_job(job_id, _args, _output_dir=None, _n_queries=0):
            nonlocal active, maximum_active
            await state.update_job(job_id, status=JobStatus.running)
            active += 1
            maximum_active = max(maximum_active, active)
            order.append(job_id)
            await asyncio.sleep(0.02)
            active -= 1
            await state.update_job(job_id, status=JobStatus.completed)

        monkeypatch.setattr(job_runner, "run_job", fake_run_job)
        state.jobs.clear()
        state._initialized = False
        await state.set_job(_job("first"))
        await state.set_job(_job("second"))
        await job_runner.enqueue_job("first", ["first.py"])
        await job_runner.enqueue_job("second", ["second.py"])
        assert job_runner._job_queue is not None
        await asyncio.wait_for(job_runner._job_queue.join(), timeout=2)
        await job_runner.stop_worker()

        assert maximum_active == 1
        assert order == ["first", "second"]

    asyncio.run(scenario())


def test_uploads_are_streamed_and_size_limited(tmp_path: Path) -> None:
    async def scenario() -> None:
        class AsyncUpload:
            def __init__(self, content: bytes) -> None:
                self.content = content
                self.offset = 0

            async def read(self, size: int) -> bytes:
                chunk = self.content[self.offset:self.offset + size]
                self.offset += len(chunk)
                return chunk

            async def close(self) -> None:
                return None

        destination = tmp_path / "upload.bin"
        upload = AsyncUpload(b"0123456789")
        with pytest.raises(UploadTooLargeError):
            await save_upload_stream(upload, destination, max_bytes=5, chunk_size=2)  # type: ignore[arg-type]
        assert not destination.exists()

    asyncio.run(scenario())


def test_fasta_is_inspected_from_disk(tmp_path: Path) -> None:
    fasta = tmp_path / "queries.fasta"
    fasta.write_text(">query_1 description\nMPEPTIDE\n>query_2\nAAAA\n")
    valid, query_ids = inspect_fasta(fasta)
    assert valid
    assert query_ids == ["query_1", "query_2"]
