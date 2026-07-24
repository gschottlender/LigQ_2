#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ligq_support.prepare_ligq_2_data import (
    CORE_DATA_PATHS,
    ECFP_CACHE_PATHS,
    FCFP_CACHE_PATHS,
)
from query_processing.ligand_providers import build_provider
from query_processing.predicted_cache import load_provider_cache_read_only


WEB_CACHE_CONFIGS = (
    ("ecfp", "morgan_1024_r2", 0.4),
    ("fcfp", "morgan_feature_1024_r2", 0.5),
)


def _fasta_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            identifier = line[1:].strip().split(maxsplit=1)[0]
            if identifier:
                identifiers.add(identifier)
    return identifiers


def inspect_web_data(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    required_paths = tuple(
        dict.fromkeys((*CORE_DATA_PATHS, *ECFP_CACHE_PATHS, *FCFP_CACHE_PATHS))
    )
    missing = [relative for relative in required_paths if not (data_dir / relative).is_file()]
    checks: dict[str, dict] = {
        "core": {
            "ready": not missing,
            "message": (
                "Required databases and representations are installed."
                if not missing
                else f"{len(missing)} required file(s) are missing."
            ),
        }
    }
    errors: list[str] = []
    if missing:
        errors.append(
            "Missing required web data: "
            + ", ".join(missing[:5])
            + ("" if len(missing) <= 5 else f" and {len(missing) - 5} more")
        )
        for name, _representation, _threshold in WEB_CACHE_CONFIGS:
            checks[name] = {
                "ready": False,
                "message": "Required web files are incomplete.",
            }
        return {
            "ready": False,
            "mode": "web",
            "checks": checks,
            "errors": errors,
        }

    target_fasta = data_dir / "sequences" / "target_sequences.fasta"
    proteins = _fasta_ids(target_fasta)
    if not proteins:
        errors.append("The installed target protein FASTA contains no identifiers.")

    for name, representation, threshold in WEB_CACHE_CONFIGS:
        try:
            provider = build_provider(
                provider_name="zinc",
                data_dir=data_dir,
                search_representation=representation,
                search_metric="tanimoto",
                search_threshold=threshold,
                search_threshold_max=1.0,
                cluster_threshold=0.8,
                search_per_iteration_topk=1000,
                search_global_topk=10000,
                search_device="cpu",
            )
            cache_path = load_provider_cache_read_only(
                data_dir=data_dir,
                provider=provider,
                proteins_needed=proteins,
                load_dataframe=False,
            )
            checks[name] = {
                "ready": True,
                "message": (
                    f"{representation} cache covers {len(proteins)} installed proteins."
                ),
                "cache_path": str(Path(cache_path).relative_to(data_dir)),
            }
        except Exception as exc:
            message = str(exc)
            checks[name] = {"ready": False, "message": message}
            errors.append(f"{name.upper()}: {message}")

    return {
        "ready": not errors and all(check["ready"] for check in checks.values()),
        "mode": "web",
        "checks": checks,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable data package required by LigQ 2 web mode."
    )
    parser.add_argument("--data-dir", default="databases")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = inspect_web_data(Path(args.data_dir))
    if args.json:
        print(json.dumps(status))
    else:
        print("ready" if status["ready"] else "not ready")
        for name, check in status["checks"].items():
            marker = "OK" if check["ready"] else "FAIL"
            print(f"[{marker}] {name}: {check['message']}")
    raise SystemExit(0 if status["ready"] else 1)


if __name__ == "__main__":
    main()
