from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import output_path, resolve_path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = SCRIPT_ROOT / "dependencies"


def bootstrap_dependencies() -> None:
    for path in (
        DEPENDENCIES / "ligq_core",
        DEPENDENCIES / "evaluation_core",
        DEPENDENCIES / "bsi",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


bootstrap_dependencies()


@dataclass
class EvaluationContext:
    targets: pd.DataFrame
    binding: pd.DataFrame
    smiles: pd.DataFrame
    store: object
    representations: dict[str, object]
    neighbor_rankings: dict[str, list[str]] | None = None


def filter_targets(targets: pd.DataFrame, selected: Iterable[str] | None) -> pd.DataFrame:
    if not selected:
        return targets.copy()
    wanted = {str(value).lower() for value in selected}
    result = targets[targets["target"].astype(str).str.lower().isin(wanted)].copy()
    missing = sorted(wanted - set(result["target"].astype(str).str.lower()))
    if missing:
        raise ValueError(f"Unknown target(s): {', '.join(missing)}")
    return result


def load_context(cfg: dict, methods: Iterable[str], selected_targets=None, with_neighbors=False) -> EvaluationContext:
    from compound_helpers import LigandStore

    targets = pd.read_csv(resolve_path(cfg, "paths", "targets_csv"))
    targets = filter_targets(targets, selected_targets)
    binding = pd.read_parquet(resolve_path(cfg, "paths", "binding_data"))
    smiles = pd.read_parquet(resolve_path(cfg, "paths", "smiles_data"))
    store = LigandStore(resolve_path(cfg, "paths", "ligand_store"))

    needed = list(dict.fromkeys([*methods, "morgan_1024_r2"]))
    representations = {name: store.load_representation(name) for name in needed}
    neighbors = build_neighbor_rankings(cfg, targets) if with_neighbors else None
    return EvaluationContext(targets, binding, smiles, store, representations, neighbors)


def build_neighbor_rankings(cfg: dict, targets: pd.DataFrame):
    # The published benchmark differs from the live platform. Reuse the exact
    # archived ranking after quality filters and ALL benchmark-ID exclusions.
    if cfg.get("paths", {}).get("blast_hits"):
        hits = pd.read_csv(resolve_path(cfg, "paths", "blast_hits"))
        required = {"target_name", "neighbor_rank", "sseqid"}
        require_columns(hits, required, "Frozen BLAST rankings")
        if hits.duplicated(["target_name", "neighbor_rank"]).any():
            raise ValueError("Duplicate target/rank in frozen BLAST table")
        return {str(target): frame.sort_values("neighbor_rank")["sseqid"].astype(str).tolist()
                for target, frame in hits.groupby("target_name", sort=False)}
    from clustering_proteinas import BlastSeedConfig, build_neighbor_rankings_for_all_targets

    # A bounded calculation must not shrink the benchmark exclusion universe.
    targets = pd.read_csv(resolve_path(cfg, "paths", "targets_csv"))

    target_to_uniprot = {
        target: group["uniprot_id"].dropna().astype(str).tolist()
        for target, group in targets.groupby("target", sort=False)
    }
    blast = cfg.get("blast", {})
    config = BlastSeedConfig(
        n_neighbors=int(blast.get("n_neighbors", 10)),
        max_evalue=float(blast.get("max_evalue", 1e-10)),
        min_bitscore=float(blast.get("min_bitscore", 80.0)),
        min_pident=float(blast.get("min_pident", 30.0)),
        min_qcovs=float(blast.get("min_qcovs", 60.0)),
        num_threads=int(blast.get("num_threads", 12)),
    )
    rankings, _, _, _ = build_neighbor_rankings_for_all_targets(
        target_to_uniprot=target_to_uniprot,
        fasta_db_path=str(resolve_path(cfg, "paths", "fasta")),
        db_prefix=str(resolve_path(cfg, "paths", "blast_db_prefix")),
        config=config,
        keep_only_top_n=None,
        verbose=True,
    )
    return rankings


def prepare_output(path: Path, *, force=False, resume=False) -> None:
    if path.exists() and any(path.iterdir()) and not (force or resume):
        raise FileExistsError(f"Output directory is not empty: {path}. Use --resume or --force.")
    path.mkdir(parents=True, exist_ok=True)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(path: Path, cfg: dict, stage: str, extra: dict | None = None) -> None:
    from .provenance import sha256_file

    versions = {"python": platform.python_version()}
    for module_name in ("numpy", "pandas", "pyarrow", "rdkit", "torch", "sklearn", "matplotlib"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {type(exc).__name__}"
    resolved_config = {key: value for key, value in cfg.items() if not key.startswith("_")}
    resolved_config = dict(resolved_config)
    resolved_config["paths"] = {key: str(resolve_path(cfg, "paths", key)) for key in cfg.get("paths", {})}
    resolved_config["provenance"] = {key: str(resolve_path(cfg, "provenance", key)) for key in cfg.get("provenance", {})}
    resolved_config["outputs"] = {key: str(output_path(cfg, key)) for key in cfg.get("outputs", {})}
    sources = DEPENDENCIES / "SOURCES.json"
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg["_config_path"]),
        "resolved_config": resolved_config,
        "versions": versions,
        "ligq_commit": git_revision(resolve_path(cfg, "provenance", "ligq_repo")),
        "bsi_commit": (git_revision(resolve_path(cfg, "provenance", "bsi_repo"))
                       if cfg.get("provenance", {}).get("bsi_repo") else None),
        "vendored_sources_manifest": str(sources),
        "vendored_sources_manifest_sha256": sha256_file(sources),
    }
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
