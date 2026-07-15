from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.config import JOB_SHUTDOWN_GRACE_SECONDS, PIPELINE_ROOT
from core import state
from models.job import Job, JobFailure, JobProgress, JobStatus

import logging
logger = logging.getLogger(__name__)

# Parses "Block 3: processed queries 1-100 / 250" — the only per-chunk progress signal
_BLOCK3_RE = re.compile(r"Block 3: processed queries \d+-(\d+) / (\d+)")
_TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)<(\d+):(\d+)")

_WARNING_TOKENS = ("warning", "no domains found", "no known ligands", "skipped")

_BUILDING_RE = re.compile(r"\[INFO\] Building representation '(.+?)' in: .+/compound_data/(.+)")
_PROGRESS_PREFIX = "LIGQ_PROGRESS "
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_STOPPED_STATUSES = {JobStatus.cancelled, JobStatus.interrupted}


@dataclass(frozen=True)
class _QueuedJob:
    job_id: str
    args: list[str]
    output_dir: Optional[Path] = None
    n_queries: int = 0


_job_queue: asyncio.Queue[_QueuedJob] | None = None
_worker_task: asyncio.Task | None = None
_stopping = False


def _subprocess_env() -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "0"}
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = Path(conda_prefix) / "lib"
        if conda_lib.is_dir():
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{conda_lib}{os.pathsep}{existing}" if existing else str(conda_lib)
            )
    return env


def _parse_progress_event(line: str) -> JobProgress | None:
    if not line.startswith(_PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(_PROGRESS_PREFIX):])
        return JobProgress.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _build_job_failure(job: Job | None, message: str) -> tuple[JobFailure, str]:
    progress = job.progress if job else None
    label = progress.label if progress else (job.progress_message if job else "")
    label = label or "Starting job"
    failure = JobFailure(
        step=progress.step if progress else None,
        label=label,
        step_index=progress.step_index if progress else None,
        step_count=progress.step_count if progress else None,
        message=message,
    )
    if failure.step_index is not None and failure.step_count is not None:
        error = (
            f"Failed at step {failure.step_index}/{failure.step_count}: "
            f"{failure.label}. {message}"
        )
    else:
        error = f"Failed while {failure.label.lower()}. {message}"
    return failure, error


async def _watch_fs(job_id: str, output_dir: Path, n_queries: int) -> None:
    """Poll search_results/ every 2 s and track completed query directories.

    For legacy jobs without structured events, completed queries also map to the
    50-99 percent band.
    """
    sr_dir = output_dir / "search_results"
    while True:
        await asyncio.sleep(2)
        job = await state.get_job(job_id)
        if job is None or job.status not in (JobStatus.running, JobStatus.partial_results):
            break
        if not sr_dir.exists():
            continue
        completed = [d.name for d in sr_dir.iterdir() if d.is_dir()]
        if not completed:
            continue
        updates: dict = {
            "status": JobStatus.partial_results,
            "completed_queries": completed,
        }
        if n_queries > 0 and job.progress is None:
            pct = 50 + int(len(completed) / n_queries * 49)
            if pct > (job.progress_percent or 0):
                updates["progress_percent"] = pct
        await state.update_job(job_id, **updates)


async def _tail_stdout(
    job_id: str,
    process: asyncio.subprocess.Process,
    output_dir: Optional[Path],
    stderr_tail: deque[str],
    stderr_done: asyncio.Event,
) -> None:
    warnings: list[str] = []
    has_structured_progress = False

    assert process.stdout is not None
    while True:
        try:
            async for raw in process.stdout:
                chunk = raw.decode("utf-8", errors="replace")
                for line in chunk.replace('\r', '\n').splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    logger.info("[job %s] %s", job_id, line)

                    progress = _parse_progress_event(line)
                    if progress is not None:
                        has_structured_progress = True
                        current_job = await state.get_job(job_id)
                        pct = max(current_job.progress_percent or 0, progress.percent) if current_job else progress.percent
                        if pct != progress.percent:
                            progress = progress.model_copy(update={"percent": pct})
                        await state.update_job(
                            job_id,
                            progress=progress,
                            progress_percent=pct,
                            progress_message=progress.label,
                        )
                        continue

                    updates: dict = {}
                    lower = line.lower()

                    if not has_structured_progress:
                        updates["progress_message"] = line

                    if not has_structured_progress and "block 1" in lower:
                        updates["progress_percent"] = 10
                    elif not has_structured_progress and "block 2" in lower:
                        updates["progress_percent"] = 40
                    elif not has_structured_progress and "block 3" in lower:
                        m3 = _BLOCK3_RE.search(line)
                        if m3:
                            end_num, total_num = int(m3.group(1)), int(m3.group(2))
                            updates["progress_percent"] = 50 + int(end_num / total_num * 49)
                        else:
                            updates.setdefault("progress_percent", 50)

                    m_tqdm = _TQDM_RE.search(line)
                    if m_tqdm:
                        pct = int(m_tqdm.group(1))
                        current = int(m_tqdm.group(2))
                        total = int(m_tqdm.group(3))
                        eta_min = int(m_tqdm.group(6))
                        eta_sec = int(m_tqdm.group(7))
                        if has_structured_progress:
                            current_job = await state.get_job(job_id)
                            if current_job and current_job.progress:
                                updates["progress"] = current_job.progress.model_copy(
                                    update={
                                        "current": current,
                                        "total": total,
                                        "eta_seconds": eta_min * 60 + eta_sec,
                                    }
                                )
                        elif pct < 100:
                            eta_str = f"{eta_min}m{eta_sec:02d}s" if eta_min > 0 else f"{eta_sec}s"
                            updates["progress_percent"] = pct
                            updates["progress_message"] = f"{pct}% · ETA {eta_str}"

                    m_building = _BUILDING_RE.search(line) if not has_structured_progress else None
                    if m_building:
                        rep_name = m_building.group(1)
                        db_name = m_building.group(2)
                        if db_name == "pdb_chembl":
                            updates["progress_percent"] = 52
                            updates["progress_message"] = f"Building '{rep_name}' for local compatibility (pdb_chembl)…"
                        else:
                            updates["progress_percent"] = 0
                            updates["progress_message"] = f"Building '{rep_name}' for {db_name}…"

                    if any(tok in lower for tok in _WARNING_TOKENS):
                        warnings = [*warnings, line]
                        updates["warnings"] = warnings

                    if updates:
                        await state.update_job(job_id, **updates)

        except ValueError as e:
            logger.warning("[job %s] ValueError (line too long): %s", job_id, str(e))
            continue
        else:
            break  # stdout esgotado normalmente

    await process.wait()
    await stderr_done.wait()
    rc = process.returncode
    finished_at = datetime.now(timezone.utc)
    logger.info("[job %s] Process finished with return code: %s", job_id, rc)  

    job = await state.get_job(job_id)
    if job is None:
        return

    elapsed = (
        (finished_at - job.started_at).total_seconds() if job.started_at else None
    )

    if job.status in _STOPPED_STATUSES:
        await state.update_job(
            job_id,
            finished_at=job.finished_at or finished_at,
            elapsed_seconds=job.elapsed_seconds if job.elapsed_seconds is not None else elapsed,
        )
        async with state._lock:
            state.processes.pop(job_id, None)
        return

    if rc != 0:
        detail = stderr_tail[-1] if stderr_tail else f"Process exited with code {rc}."
        detail = detail[:1000]
        failure, error = _build_job_failure(job, detail)
        await state.update_job(
            job_id,
            status=JobStatus.failed,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
            error=error,
            failure=failure,
        )
    else:
        new_status = (
            JobStatus.completed_with_warnings if warnings else JobStatus.completed
        )
        completed_queries = list(job.completed_queries)
        if output_dir:
            sr_dir = output_dir / "search_results"
            if sr_dir.exists():
                completed_queries = [d.name for d in sr_dir.iterdir() if d.is_dir()]

        completed_progress = None
        if job.progress is not None:
            completed_progress = job.progress.model_copy(
                update={
                    "step": "completed",
                    "label": "Completed",
                    "step_index": job.progress.step_count,
                    "percent": 100,
                    "eta_seconds": 0,
                }
            )
        await state.update_job(
            job_id,
            status=new_status,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
            warnings=warnings,
            completed_queries=completed_queries,
            progress_percent=100,
            progress=completed_progress,
        )

    async with state._lock:
        state.processes.pop(job_id, None)


async def run_job(job_id: str, args: list[str], output_dir: Optional[Path] = None, n_queries: int = 0,) -> None:
    logger.info("[job %s] Starting with args: %s", job_id, args)
    logger.info("[job %s] Python: %s", job_id, sys.executable)
    logger.info("[job %s] CWD: %s", job_id, str(PIPELINE_ROOT))

    job = await state.get_job(job_id)
    if job is None or job.status != JobStatus.queued:
        return

    await state.update_job(
        job_id,
        status=JobStatus.running,
        started_at=datetime.now(timezone.utc),
    )

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PIPELINE_ROOT),
            limit=1024 * 1024 * 10,
            env=_subprocess_env(),
            start_new_session=(os.name == "posix"),
        )
        async with state._lock:
            state.processes[job_id] = process

        current_job = await state.get_job(job_id)
        if current_job is not None and current_job.status in _STOPPED_STATUSES:
            await _terminate_process(process, grace_seconds=1)

        stderr_tail: deque[str] = deque(maxlen=50)
        stderr_done = asyncio.Event()

        async def _tail_stderr(proc: asyncio.subprocess.Process) -> None:
            assert proc.stderr is not None
            try:
                while True:
                    try:
                        async for raw in proc.stderr:
                            chunk = raw.decode("utf-8", errors="replace")
                            for line in chunk.replace('\r', '\n').splitlines():
                                line = line.strip()
                                if line:
                                    stderr_tail.append(_ANSI_ESCAPE_RE.sub("", line))
                                    logger.info("[job %s] STDERR: %s", job_id, line)
                    except ValueError:
                        continue
                    else:
                        break
            finally:
                stderr_done.set()

        stderr_task = asyncio.create_task(_tail_stderr(process))        

        watcher: Optional[asyncio.Task] = None
        if output_dir:
            watcher = asyncio.create_task(_watch_fs(job_id, output_dir, n_queries))

        try:
            await _tail_stdout(job_id, process, output_dir, stderr_tail, stderr_done)
        finally:
            if process.returncode is None:
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
            else:
                await stderr_task

            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        job = await state.get_job(job_id)
        if job is not None and job.status in _STOPPED_STATUSES:
            async with state._lock:
                state.processes.pop(job_id, None)
            return
        failure, error = _build_job_failure(job, str(exc))
        await state.update_job(
            job_id,
            status=JobStatus.failed,
            finished_at=datetime.now(timezone.utc),
            error=error,
            failure=failure,
        )
        async with state._lock:
            state.processes.pop(job_id, None)


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = JOB_SHUTDOWN_GRACE_SECONDS,
) -> None:
    if process.returncode is not None:
        return

    try:
        if os.name == "posix" and process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass

    try:
        if os.name == "posix" and process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await process.wait()


async def terminate_job_process(job_id: str, *, grace_seconds: float = 5) -> None:
    async with state._lock:
        process = state.processes.get(job_id)
    if process is not None:
        await _terminate_process(process, grace_seconds=grace_seconds)


async def _queue_worker() -> None:
    assert _job_queue is not None
    while True:
        queued_job = await _job_queue.get()
        try:
            job = await state.get_job(queued_job.job_id)
            if job is None or job.status != JobStatus.queued:
                continue
            await run_job(
                queued_job.job_id,
                queued_job.args,
                queued_job.output_dir,
                queued_job.n_queries,
            )
        finally:
            _job_queue.task_done()


async def start_worker() -> None:
    global _job_queue, _worker_task, _stopping
    _stopping = False
    if _worker_task is None or _worker_task.done():
        _job_queue = asyncio.Queue()
        _worker_task = asyncio.create_task(_queue_worker(), name="ligq-job-worker")


async def enqueue_job(
    job_id: str,
    args: list[str],
    output_dir: Optional[Path] = None,
    n_queries: int = 0,
) -> None:
    if _stopping:
        raise RuntimeError("The job queue is shutting down.")
    await start_worker()
    assert _job_queue is not None
    await _job_queue.put(_QueuedJob(job_id, args, output_dir, n_queries))


async def stop_worker() -> None:
    global _job_queue, _worker_task, _stopping
    _stopping = True
    await state.mark_unfinished_interrupted(
        "The backend stopped before this job finished."
    )

    async with state._lock:
        running_processes = list(state.processes.values())
    if running_processes:
        await asyncio.gather(
            *(_terminate_process(process) for process in running_processes),
            return_exceptions=True,
        )

    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
    _job_queue = None
