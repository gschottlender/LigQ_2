#!/usr/bin/env python3
"""Fresh HF download, full recalculation, rendering and regression verification."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

from package.common import ROOT, read_json, write_json


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pipeline(args):
    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=True)
    state_path = job / "end_to_end_status.json"
    if state_path.exists():
        previous = read_json(state_path)
        if previous.get("status") != "launching" and not args.resume:
            raise FileExistsError("This job already has a status record; use --resume or a fresh job directory")
    data, output = job / "data", job / "reproduction"
    state = {"status":"running", "started_utc":now(), "pid":os.getpid(),
             "job_directory":str(job), "artifact":read_json(ROOT / "artifact_lock.json"),
             "data_source":"immutable Hugging Face download; no local source import",
             "completed_steps":[], "full_recalculation_certified":False}
    write_json(state_path, state)
    environment = dict(os.environ)
    # A separate initially empty HF cache proves availability independently of local source files.
    environment["HF_HOME"] = str(job / "hf_cache")
    environment["HF_HUB_CACHE"] = str(job / "hf_cache/hub")
    environment["HF_XET_CACHE"] = str(job / "hf_cache/xet")
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(job / "mpl_cache")
    environment["PYTHONUNBUFFERED"] = "1"
    cli = ROOT / "reproduce.py"
    steps = [
        ("unit_tests", [str(args.chemistry_python), str(cli), "smoke-test", "--output-dir", str(job / "unit_tests")]),
        ("download", [str(args.python), str(cli), "download", "--data-dir", str(data)]),
        ("historical_figure_rendering", [str(args.chemistry_python), str(cli), "plot", "--source", "historical",
             "--output-dir", str(job / "historical_figures")]),
        ("historical_pixel_validation", [str(args.chemistry_python), str(cli), "validate", "--figures-only",
             "--output-dir", str(job / "historical_figures")]),
        ("full_calculation", [str(args.python), str(cli), "run", "--evaluation", "all", "--resume",
             "--data-dir", str(data), "--output-dir", str(output), "--python", str(args.python),
             "--chemistry-python", str(args.chemistry_python)]),
        ("recalculated_figure_rendering", [str(args.chemistry_python), str(cli), "plot", "--source", "recalculated",
             "--results-dir", str(output / "calculations"), "--output-dir", str(output)]),
        ("full_scientific_and_pixel_validation", [str(args.chemistry_python), str(cli), "validate",
             "--data-dir", str(data), "--output-dir", str(output)]),
    ]
    write_json(job / "command_plan.json", [{"step":name, "command":argv} for name, argv in steps])
    try:
        for name, argv in steps:
            state.update(current_step=name, step_started_utc=now())
            write_json(state_path, state)
            print(f"\n[{now()}] {name}: {' '.join(argv)}", flush=True)
            started = time.monotonic()
            subprocess.run(argv, check=True, cwd=ROOT, env=environment)
            state["completed_steps"].append({"step":name, "elapsed_seconds":time.monotonic()-started})
            write_json(state_path, state)
        report = read_json(output / "validation_report.json")
        if not report["all_checks_passed"] or not report["full_recalculation_certified"]:
            raise ValueError("Full scientific verification was not certified")
        state.update(status="passed", finished_utc=now(), full_recalculation_certified=True,
                     validation_report=str(output / "validation_report.json"))
        write_json(state_path, state)
    except BaseException as exc:
        state.update(status="failed", finished_utc=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=ROOT / "outputs/huggingface_end_to_end")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--chemistry-python", type=Path, required=True)
    parser.add_argument("--launch", action="store_true", help="Keep running independently of the interactive session")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.python, args.chemistry_python = args.python.resolve(), args.chemistry_python.resolve()
    job = args.job_dir.resolve()
    if args.launch:
        job.mkdir(parents=True, exist_ok=True)
        state = job / "end_to_end_status.json"
        if state.exists():
            old = read_json(state)
            if old.get("status") in ("launching", "running"):
                try:
                    os.kill(old["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("The existing end-to-end worker is still running")
            if not args.resume:
                raise FileExistsError("An earlier job exists; use --resume or another --job-dir")
        command = [str(sys.executable), str(Path(__file__).resolve()), "--worker", "--job-dir", str(job),
                   "--python", str(args.python), "--chemistry-python", str(args.chemistry_python)]
        if args.resume:
            command.append("--resume")
        with (job / "end_to_end.log").open("a", encoding="utf-8") as log:
            # Write before spawn so the parent never replaces a worker's newer state.
            write_json(state, {"status":"launching", "pid":os.getpid(), "requested_utc":now()})
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started end-to-end worker PID {process.pid}. Status: {state}. Log: {job / 'end_to_end.log'}")
    else:
        pipeline(args)


if __name__ == "__main__":
    main()
