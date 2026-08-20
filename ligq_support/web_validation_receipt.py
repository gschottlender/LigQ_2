from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RECEIPT_FILENAME = ".ligq-web-validation.json"
RECEIPT_SCHEMA_VERSION = 1
# Increment this contract version whenever deep validation semantics change in a
# way that cannot be detected from the required-file inventory alone.
WEB_DATA_VALIDATOR_VERSION = 1
SMALL_FILE_HASH_LIMIT = 1024 * 1024


def required_web_data_paths() -> tuple[str, ...]:
    # Keep the runtime check independent from the expensive provider imports used
    # by the administrative validator.
    from ligq_support.prepare_ligq_2_data import (
        CORE_DATA_PATHS,
        ECFP_CACHE_PATHS,
        FCFP_CACHE_PATHS,
    )

    return tuple(
        dict.fromkeys((*CORE_DATA_PATHS, *ECFP_CACHE_PATHS, *FCFP_CACHE_PATHS))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_web_data_inventory(
    data_dir: Path,
    *,
    required_paths: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    data_dir = Path(data_dir)
    paths = (
        tuple(required_paths)
        if required_paths is not None
        else required_web_data_paths()
    )
    inventory: list[dict[str, Any]] = []
    missing: list[str] = []

    for relative in paths:
        path = data_dir / relative
        try:
            stat = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            missing.append(relative)
            continue

        if not path.is_file():
            missing.append(relative)
            continue

        item: dict[str, Any] = {
            "path": relative,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= SMALL_FILE_HASH_LIMIT:
            item["sha256"] = _sha256(path)
        inventory.append(item)

    return inventory, missing


def validation_receipt_path(
    data_dir: Path,
    *,
    receipt_dir: Path | None = None,
) -> Path:
    directory = Path(receipt_dir) if receipt_dir is not None else Path(data_dir)
    return directory / RECEIPT_FILENAME


def write_web_validation_receipt(
    data_dir: Path,
    status: dict[str, Any],
    *,
    receipt_dir: Path | None = None,
) -> Path:
    if not status.get("ready"):
        raise ValueError("A validation receipt can only be written for ready web data.")

    data_dir = Path(data_dir)
    inventory, missing = build_web_data_inventory(data_dir)
    if missing:
        raise RuntimeError(
            "Required web data disappeared before the validation receipt was written: "
            + ", ".join(missing[:5])
        )

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "validator_version": WEB_DATA_VALIDATOR_VERSION,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "required_files": inventory,
        "checks": status.get("checks", {}),
    }
    path = validation_receipt_path(data_dir, receipt_dir=receipt_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _not_ready(message: str) -> dict[str, Any]:
    return {
        "ready": False,
        "mode": "web",
        "checks": {
            "receipt": {
                "ready": False,
                "message": message,
            }
        },
        "errors": [message],
    }


def inspect_web_validation_receipt(
    data_dir: Path,
    *,
    receipt_dir: Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    path = validation_receipt_path(data_dir, receipt_dir=receipt_dir)
    if not path.is_file():
        return _not_ready(
            "Administrative web-data validation is required. Run the "
            "validate-data service before starting the public service."
        )

    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        detail = str(exc).strip() or type(exc).__name__
        return _not_ready(
            f"The administrative web-data validation receipt is unreadable: {detail}. "
            "Run the administrative validator again."
        )

    if not isinstance(receipt, dict):
        return _not_ready(
            "The administrative web-data validation receipt has an invalid format. "
            "Run the administrative validator again."
        )
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        return _not_ready(
            "The administrative web-data validation receipt is incompatible with "
            "this LigQ 2 version. Run the administrative validator again."
        )
    if receipt.get("validator_version") != WEB_DATA_VALIDATOR_VERSION:
        return _not_ready(
            "The installed web data was validated with a different validator version. "
            "Run the administrative validator again."
        )

    current_inventory, missing = build_web_data_inventory(data_dir)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        return _not_ready(
            f"Required web data is missing: {preview}{suffix}. Run data preparation "
            "and the administrative validator again."
        )
    if receipt.get("required_files") != current_inventory:
        return _not_ready(
            "The installed web data changed after its last successful administrative "
            "validation. Run the administrative validator again."
        )

    validated_at = receipt.get("validated_at")
    if not isinstance(validated_at, str) or not validated_at:
        return _not_ready(
            "The administrative web-data validation receipt is incomplete. Run the "
            "administrative validator again."
        )

    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    checks = dict(checks)
    checks["receipt"] = {
        "ready": True,
        "message": f"Administrative web-data validation succeeded at {validated_at}.",
    }
    return {
        "ready": True,
        "mode": "web",
        "checks": checks,
        "errors": [],
        "validated_at": validated_at,
    }
