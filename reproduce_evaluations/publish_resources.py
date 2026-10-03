#!/usr/bin/env python3
"""Maintainer-only publication of the verified, explicitly allowlisted input bundle."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from package.common import ROOT, contained, read_json, require_revision, sha256, write_json
from package.resources import verify_files


def remote_matches(record, sibling, local_path):
    if sibling.size != record["size_bytes"]:
        return False
    if sibling.lfs is not None:
        return sibling.lfs.sha256 == record["sha256"]
    payload = local_path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() == sibling.blob_id


def publish(directory, repository):
    from huggingface_hub import CommitOperationAdd, HfApi
    directory = Path(directory).resolve()
    lock = read_json(directory / "resource_lock.json")
    verify_files(directory, lock["frozen_inputs"])
    if lock != read_json(ROOT / "resource_lock.json"):
        raise ValueError("Exported and package resource locks differ")
    api = HfApi()
    account = api.whoami()["name"]
    if repository != f"{account}/LigQ_2_evaluations":
        raise ValueError("This maintainer command only publishes the account's LigQ_2_evaluations dataset")
    exists = api.repo_exists(repository, repo_type="dataset")
    if not exists:
        api.create_repo(repository, repo_type="dataset", private=False, exist_ok=False)
    info = api.dataset_info(repository, files_metadata=True)
    if info.private:
        raise ValueError("An existing private dataset will not be made public automatically")
    siblings = {s.rfilename: s for s in info.siblings}
    operations = []
    for record in lock["frozen_inputs"]:
        path = contained(directory, record["path"])
        if record["path"] in siblings:
            if not remote_matches(record, siblings[record["path"]], path):
                raise FileExistsError(f"Refusing to overwrite a different remote input: {record['path']}")
        else:
            operations.append(CommitOperationAdd(path_in_repo=record["path"], path_or_fileobj=path))
    # These two public documents are the only non-resource uploads. No local paths or DOCX files.
    for name, path in (("resource_lock.json", directory / "resource_lock.json"),
                       ("README.md", ROOT / "configs/huggingface_dataset_card.md")):
        if name in siblings:
            data = path.read_bytes()
            record = {"size_bytes":len(data), "sha256":hashlib.sha256(data).hexdigest()}
            if not remote_matches(record, siblings[name], path):
                raise FileExistsError(f"Refusing to overwrite a different remote document: {name}")
        else:
            operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=path))
    if operations:
        commit = api.create_commit(repository, repo_type="dataset", operations=operations,
            parent_commit=info.sha, commit_message="Publish frozen LigQ2 evaluation inputs with SHA-256 inventory")
        revision = commit.oid
    else:
        revision = info.sha
    require_revision(revision)
    remote = api.dataset_info(repository, revision=revision, files_metadata=True)
    siblings = {s.rfilename:s for s in remote.siblings}
    for record in lock["frozen_inputs"]:
        if record["path"] not in siblings or not remote_matches(
                record, siblings[record["path"]], contained(directory, record["path"])):
            raise ValueError(f"Uploaded input verification failed: {record['path']}")
    receipt = {"repository":repository, "revision":revision, "repo_type":"dataset",
               "files":len(lock["frozen_inputs"]),
               "size_bytes":sum(r["size_bytes"] for r in lock["frozen_inputs"]),
               "resource_lock_sha256":sha256(ROOT / "resource_lock.json"),
               "remote_input_hashes_verified":True}
    write_json(ROOT / "artifact_lock.json", receipt)
    print(f"Published and verified: https://huggingface.co/datasets/{repository}/tree/{revision}", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="gschottlender/LigQ_2_evaluations")
    parser.add_argument("--directory", type=Path, default=ROOT / "publication_export")
    args = parser.parse_args()
    publish(args.directory, args.repository)
