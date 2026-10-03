#!/usr/bin/env python3
"""Stop only the isolated full-publication worker; preserve all downloaded files."""
import argparse
import datetime as dt
import os
import signal
from pathlib import Path

from package.common import ROOT, read_json, write_json


def stop(job):
    job = Path(job).resolve()
    path = job / "end_to_end_status.json"
    state = read_json(path)
    if state["status"] not in ("running", "launching"):
        print(f"No running full test: {state['status']}")
        return
    pid = state["pid"]
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        arguments = (proc / "cmdline").read_bytes().split(b"\0")
        expected = str(ROOT / "test_publication.py").encode()
        if expected not in arguments or b"--worker" not in arguments:
            raise ValueError("Recorded PID is not this package's full-test worker; no process was stopped")
        if os.getpgid(pid) != pid:
            raise ValueError("Worker is not an isolated process-group leader; refusing group termination")
        os.killpg(pid, signal.SIGTERM)
    state.update(status="stopped_by_user", stopped_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 stop_reason="Replaced by bounded validation at the user's request",
                 full_recalculation_certified=False, downloaded_files_preserved=True)
    write_json(path, state)
    print(f"Stopped full-test process group {pid}. All downloaded files were preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=ROOT / "outputs/huggingface_end_to_end")
    args = parser.parse_args()
    stop(args.job_dir)
