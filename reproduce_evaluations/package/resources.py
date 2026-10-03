"""Verified local import, publication export and immutable HF downloads."""
from __future__ import annotations

from pathlib import Path

from .common import ROOT, SEEDS, contained, copy_verified, read_json, require_revision, sha256, write_json


def verify_files(directory, records):
    errors = []
    for record in records:
        path = contained(directory, record["path"])
        if not path.is_file():
            errors.append(f"missing: {record['path']}")
        elif path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            errors.append(f"changed: {record['path']}")
    if errors:
        raise ValueError("Resource verification failed:\n" + "\n".join(errors[:25]))


def import_local(inventory, data_dir, *, include_benchmark=False):
    inventory = read_json(inventory)
    lock = read_json(ROOT / "resource_lock.json")
    records = list(lock["frozen_inputs"])
    if include_benchmark:
        records += lock["benchmark_files"]
    for record in records:
        source = inventory["sources"].get(record["path"])
        if source is None:
            raise ValueError(f"No local source for {record['path']}")
        copy_verified(source, contained(data_dir, record["path"]), record["sha256"])
    write_json(Path(data_dir) / "resource_receipt.json", {
        "mode": "verified_local_import", "resource_lock_sha256": sha256(ROOT / "resource_lock.json"),
        "includes_benchmark": include_benchmark,
    })


def export_resources(inventory, output_dir):
    # Publication preparation only: no authentication or upload APIs.
    inventory = read_json(inventory)
    lock = read_json(ROOT / "resource_lock.json")
    output_dir = Path(output_dir).resolve()
    for record in lock["frozen_inputs"]:
        copy_verified(inventory["sources"][record["path"]],
                      contained(output_dir, record["path"]), record["sha256"])
    write_json(output_dir / "resource_lock.json", lock)
    return sum(r["size_bytes"] for r in lock["frozen_inputs"])


def benchmark_records(lock, targets=None, seeds=None):
    records = lock["benchmark_files"]
    if targets is None and seeds is None:
        return records
    available = {r["path"].split("/")[1] for r in records if len(r["path"].split("/")) == 4}
    wanted = set(targets or available)
    selected_seeds = set(seeds or SEEDS)
    if not wanted or wanted - available:
        raise ValueError(f"Unknown benchmark targets: {sorted(wanted - available)}")
    if not selected_seeds or selected_seeds - set(SEEDS):
        raise ValueError("A bounded download must use original partition seeds")
    return [r for r in records if len(r["path"].split("/")) != 4 or
            (r["path"].split("/")[1] in wanted and
             r["path"].split("/")[2] in {f"random_state_{s}" for s in selected_seeds})]


def download(data_dir, *, artifact_repository=None, artifact_revision=None, offline=False,
             only_benchmark=False, targets=None, seeds=None):
    lock = read_json(ROOT / "resource_lock.json")
    selected_benchmark = benchmark_records(lock, targets, seeds)
    if artifact_repository is None and artifact_revision is None and (ROOT / "artifact_lock.json").is_file():
        artifact = read_json(ROOT / "artifact_lock.json")
        if artifact["resource_lock_sha256"] != sha256(ROOT / "resource_lock.json"):
            raise ValueError("Published artifact lock does not match the scientific input inventory")
        artifact_repository, artifact_revision = artifact["repository"], artifact["revision"]
    if bool(artifact_repository) != bool(artifact_revision):
        raise ValueError("Supply both --artifact-repository and --artifact-revision, or neither")
    frozen = lock["frozen_inputs"]
    missing_frozen = any(not contained(data_dir, r["path"]).is_file() for r in frozen)
    if not only_benchmark and missing_frozen and not (artifact_repository and artifact_revision):
        raise ValueError("The exact historical input bundle has no configured immutable publication. Use import-local, or "
                         "supply --artifact-repository and --artifact-revision after publishing the prepared "
                         "bundle. No current database or regenerated embedding is substituted.")
    if not only_benchmark and not missing_frozen:
        verify_files(data_dir, frozen)
    from huggingface_hub import hf_hub_download
    benchmark = lock["benchmark"]
    require_revision(benchmark["revision"])
    downloaded = []
    groups = [(selected_benchmark, benchmark["repository"], benchmark["revision"], benchmark["remote_prefix"])]
    if not only_benchmark and missing_frozen:
        require_revision(artifact_revision)
        groups.append((frozen, artifact_repository, artifact_revision, ""))
    for records, repository, revision, prefix in groups:
        print(f"Verifying/downloading {len(records)} files from {repository}@{revision}", flush=True)
        processed = 0
        for record in records:
            processed += 1
            if processed == 1 or processed % 50 == 0 or processed == len(records):
                print(f"  {processed}/{len(records)}: {record['path']}", flush=True)
            target = contained(data_dir, record["path"])
            if target.is_file() and sha256(target) == record["sha256"]:
                continue
            if target.exists():
                raise ValueError(f"Existing download differs from the publication input: {target}")
            name = record.get("remote_path", record["path"])
            name = "/".join(s for s in (prefix.strip("/"), name) if s)
            cached = hf_hub_download(repository, name, repo_type="dataset", revision=revision,
                                     local_files_only=offline)
            copy_verified(cached, target, record["sha256"])
            downloaded.append(record["path"])
    verify_files(data_dir, selected_benchmark)
    if not only_benchmark:
        verify_files(data_dir, frozen)
    write_json(Path(data_dir) / "resource_receipt.json", {
        "mode": "immutable_huggingface", "benchmark": benchmark,
        "artifact_repository": artifact_repository, "artifact_revision": artifact_revision,
        "resource_lock_sha256": sha256(ROOT / "resource_lock.json"), "downloaded_files": downloaded,
        "benchmark_subset": {"targets":targets, "seeds":seeds},
        "complete_benchmark_requested": targets is None and seeds is None,
    })


def validate_benchmark(data_dir):
    import pandas as pd
    benchmark = Path(data_dir) / "benchmark"
    table = pd.read_csv(benchmark / "manifest.csv")
    if table.duplicated(["target_id", "random_state"]).any():
        raise ValueError("Duplicate benchmark target/partition")
    if table.target_id.nunique() != 61 or len(table) != 305:
        raise ValueError("The publication requires exactly 61 targets and 305 partitions")
    for target, rows in table.groupby("target_id"):
        if set(rows.random_state) != set(SEEDS):
            raise ValueError(f"Incomplete partition set for {target}")
    verify_files(data_dir, read_json(ROOT / "resource_lock.json")["benchmark_files"])
    return {"targets": 61, "partitions": 305, "file_hashes_verified": True}
