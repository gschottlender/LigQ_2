import asyncio
import re, os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.config import PIPELINE_ROOT
from core import state
from models.job import JobStatus

import logging
logger = logging.getLogger(__name__)

# Parses "Block 3: processed queries 1-100 / 250" — the only per-chunk progress signal
_BLOCK3_RE = re.compile(r"Block 3: processed queries \d+-(\d+) / (\d+)")
_TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)<(\d+):(\d+)")

_WARNING_TOKENS = ("warning", "no domains found", "no known ligands", "skipped")

_BUILDING_RE = re.compile(r"\[INFO\] Building representation '(.+?)' in: .+/compound_data/(.+)")

async def _watch_fs(job_id: str, output_dir: Path, n_queries: int) -> None:
    """Poll search_results/ every 2 s and push incremental progress_percent updates.

    Maps completed-query-dir count to the 50–99 % band so the earlier BLAST/HMMER
    stage progress (10 % and 40 %) is never overwritten going backwards.
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
        if n_queries > 0:
            pct = 50 + int(len(completed) / n_queries * 49)
            if pct > (job.progress_percent or 0):
                updates["progress_percent"] = pct
        await state.update_job(job_id, **updates)


async def _tail_stdout( job_id: str, process: asyncio.subprocess.Process, output_dir: Optional[Path],) -> None:
    warnings: list[str] = []

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

                    updates: dict = {"progress_message": line}
                    lower = line.lower()

                    if "block 1" in lower:
                        updates["progress_percent"] = 10
                    elif "block 2" in lower:
                        updates["progress_percent"] = 40
                    elif "block 3" in lower:
                        m3 = _BLOCK3_RE.search(line)
                        if m3:
                            end_num, total_num = int(m3.group(1)), int(m3.group(2))
                            updates["progress_percent"] = 50 + int(end_num / total_num * 49)
                        else:
                            updates.setdefault("progress_percent", 50)

                    m_tqdm = _TQDM_RE.search(line)
                    if m_tqdm:
                        pct = int(m_tqdm.group(1))
                        if pct < 100:
                            eta_min = int(m_tqdm.group(6))
                            eta_sec = int(m_tqdm.group(7))
                            eta_str = f"{eta_min}m{eta_sec:02d}s" if eta_min > 0 else f"{eta_sec}s"
                            updates["progress_percent"] = pct
                            updates["progress_message"] = f"{pct}% · ETA {eta_str}"

                    m_building = _BUILDING_RE.search(line)
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

                    await state.update_job(job_id, **updates)

        except ValueError as e:
            logger.warning("[job %s] ValueError (line too long): %s", job_id, str(e))
            continue
        else:
            break  # stdout esgotado normalmente

    await process.wait()
    rc = process.returncode
    finished_at = datetime.now(timezone.utc)
    logger.info("[job %s] Process finished with return code: %s", job_id, rc)  

    job = await state.get_job(job_id)
    if job is None:
        return

    elapsed = (
        (finished_at - job.started_at).total_seconds() if job.started_at else None
    )

    if rc != 0:
        await state.update_job(
            job_id,
            status=JobStatus.failed,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
            error=(
                f"Process exited with code {rc}. "
                f"Last message: {job.progress_message or '(no output)'}"
            ),
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

        await state.update_job(
            job_id,
            status=new_status,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
            warnings=warnings,
            completed_queries=completed_queries,
            progress_percent=100,
        )

    async with state._lock:
        state.processes.pop(job_id, None)


async def run_job(job_id: str, args: list[str], output_dir: Optional[Path] = None, n_queries: int = 0,) -> None:
    logger.info("[job %s] Starting with args: %s", job_id, args)
    logger.info("[job %s] Python: %s", job_id, sys.executable)
    logger.info("[job %s] CWD: %s", job_id, str(PIPELINE_ROOT))

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
            env={**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "0"},
        )
        async with state._lock:
            state.processes[job_id] = process

        async def _tail_stderr(proc: asyncio.subprocess.Process) -> None:
            assert proc.stderr is not None
            while True:
                try:
                    async for raw in proc.stderr:
                        chunk = raw.decode("utf-8", errors="replace")
                        for line in chunk.replace('\r', '\n').splitlines():
                            line = line.strip()
                            if line:
                                logger.info("[job %s] STDERR: %s", job_id, line)
                except ValueError:
                    continue
                else:
                    break

        stderr_task = asyncio.create_task(_tail_stderr(process))        

        watcher: Optional[asyncio.Task] = None
        if output_dir:
            watcher = asyncio.create_task(_watch_fs(job_id, output_dir, n_queries))

        try:
            await _tail_stdout(job_id, process, output_dir)
        finally:
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        await state.update_job(
            job_id,
            status=JobStatus.failed,
            finished_at=datetime.now(timezone.utc),
            error=str(exc),
        )