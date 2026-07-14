"""
add_new_representation.py

Build an additional HuggingFace representation for a target compound database
(default: ZINC). Also ensures the same representation exists in the local
PDB+ChEMBL compound database so both spaces stay compatible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import pyarrow.parquet as pq

from compound_processing.compound_helpers import (
    build_huggingface_representation,
    build_rdkit_representation,
)
from progress_reporting import ProgressEmitter


DEFAULT_LOCAL_ROOT = Path("compound_data/pdb_chembl")
DEFAULT_ZINC_ROOT = Path("compound_data/zinc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an additional HuggingFace representation for one compound "
            "database and ensure it also exists in the local PDB+ChEMBL base."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="databases",
        help="Root output directory containing compound_data/.",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="zinc",
        choices=["zinc", "local"],
        help=(
            "Primary base where the representation will be created. "
            "Default: zinc"
        ),
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help=(
            "Optional custom base name under compound_data/<base-name>/. "
            "When provided, it overrides --base."
        ),
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default=None,
        help=(
            "Optional explicit compound database root containing ligands.parquet "
            "and reps/. When provided, it overrides --base-name and --base."
        ),
    )
    parser.add_argument(
        "--rep-name",
        type=str,
        default=None,
        help=(
            "Optional representation name (saved under reps/<rep-name>.dat + "
            ".meta.json). When omitted, a default name is derived from the "
            "representation settings."
        ),
    )
    parser.add_argument(
        "--representation-type",
        type=str,
        default="huggingface",
        choices=["huggingface", "rdkit"],
        help="Type of representation to build.",
    )
    parser.add_argument(
        "--rdkit-fp-kind",
        type=str,
        default="ap",
        choices=["ap", "topological_torsion", "rdkit", "morgan_feature", "maccs"],
        help="RDKit fingerprint kind when --representation-type=rdkit.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="HuggingFace model ID to build the representation.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        help=(
            "Expected representation dimension. For HuggingFace, defaults to the "
            "model hidden_size when omitted; if provided, it must match hidden_size. "
            "For RDKit fingerprints, this is required."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=14,
        help="Batch size for representation computation.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of CPU workers for RDKit fingerprints (default: all CPUs).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500,
        help="Chunk size for multiprocessing imap in RDKit fingerprints.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer max_length.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_attention_mask",
        choices=["mean_attention_mask", "cls"],
        help="Pooling strategy for HuggingFace representation.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow HuggingFace to execute custom code from the model repository "
            "when loading tokenizer/model. Required for some models such as "
            "ibm-research/MoLFormer-XL-both-10pct."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Optional HuggingFace model revision/commit to pin when loading the "
            "representation model."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the representation already exists.",
    )
    parser.add_argument(
        "--ensure-local-compatible",
        action="store_true",
        help=(
            "Also build the same representation in compound_data/pdb_chembl. "
            "This is enabled automatically for legacy --base zinc behavior."
        ),
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def rep_meta_path(root: Path, rep_name: str) -> Path:
    return root / "reps" / f"{rep_name}.meta.json"


def representation_exists(root: Path, rep_name: str) -> bool:
    return rep_meta_path(root, rep_name).exists()


def ensure_ligands_exist(root: Path) -> None:
    ligands_path = root / "ligands.parquet"
    if not ligands_path.exists():
        raise FileNotFoundError(
            f"ligands.parquet not found at {ligands_path}. "
            "Generate this base first before adding a representation."
        )


def resolve_representation_name(
    representation_type: str,
    rep_name: Optional[str],
    rdkit_fp_kind: str,
    n_bits: Optional[int],
) -> str:
    if rep_name:
        return rep_name
    if representation_type == "huggingface":
        return "chemberta_zinc_base_768"
    if representation_type != "rdkit":
        raise ValueError(f"Unsupported representation_type: {representation_type}")
    if rdkit_fp_kind == "maccs":
        return "maccs_167"
    if n_bits is None:
        raise ValueError("--n-bits must be provided for RDKit fingerprints.")
    if rdkit_fp_kind == "ap":
        return f"atom_pair_{int(n_bits)}"
    if rdkit_fp_kind == "topological_torsion":
        return f"topological_torsion_{int(n_bits)}"
    if rdkit_fp_kind == "rdkit":
        return f"rdkit_daylight_{int(n_bits)}"
    if rdkit_fp_kind == "morgan_feature":
        return f"morgan_feature_{int(n_bits)}_r2"
    raise ValueError(f"Unsupported rdkit_fp_kind: {rdkit_fp_kind}")


def ligand_count(root: Path) -> int:
    ensure_ligands_exist(root)
    return int(pq.ParquetFile(root / "ligands.parquet").metadata.num_rows)


def build_representation_if_needed(
    root: Path,
    representation_type: str,
    rep_name: Optional[str],
    model_id: str,
    n_bits: Optional[int],
    batch_size: int,
    max_length: Optional[int],
    pooling: str,
    rdkit_fp_kind: str,
    n_jobs: Optional[int],
    chunksize: int,
    force: bool,
    trust_remote_code: bool,
    revision: Optional[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    ensure_ligands_exist(root)

    resolved_name = resolve_representation_name(
        representation_type,
        rep_name,
        rdkit_fp_kind,
        n_bits,
    )

    if representation_exists(root, resolved_name) and not force:
        print(f"[INFO] Representation '{resolved_name}' already exists in {root}. Skipping.")
        return

    print(f"[INFO] Building representation '{resolved_name}' in: {root}")
    if representation_type == "huggingface":
        build_huggingface_representation(
            root=root,
            n_bits=n_bits,
            batch_size=batch_size,
            name=resolved_name,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
            trust_remote_code=trust_remote_code,
            revision=revision,
            progress_callback=progress_callback,
        )
    elif representation_type == "rdkit":
        if n_bits is None:
            raise ValueError("--n-bits must be provided for RDKit fingerprints.")
        build_rdkit_representation(
            root=root,
            fp_kind=rdkit_fp_kind,
            n_bits=int(n_bits),
            batch_size=batch_size,
            name=resolved_name,
            n_jobs=n_jobs,
            chunksize=chunksize,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unsupported representation_type: {representation_type}")


def main() -> None:
    args = parse_args()
    progress = ProgressEmitter(enabled=args.progress_json)

    output_dir = Path(args.output_dir).resolve()
    local_root = output_dir / DEFAULT_LOCAL_ROOT
    zinc_root = output_dir / DEFAULT_ZINC_ROOT

    using_legacy_zinc = (
        args.target_root is None
        and args.base_name is None
        and args.base == "zinc"
    )
    if args.target_root is not None:
        primary_root = Path(args.target_root).resolve()
    elif args.base_name is not None:
        primary_root = output_dir / "compound_data" / args.base_name
    else:
        primary_root = zinc_root if args.base == "zinc" else local_root

    ensure_local_compatible = bool(args.ensure_local_compatible or using_legacy_zinc)
    resolved_name = resolve_representation_name(
        args.representation_type,
        args.rep_name,
        args.rdkit_fp_kind,
        args.n_bits,
    )

    requested_builds: list[tuple[Path, bool]] = [(primary_root, args.force)]
    if ensure_local_compatible and primary_root != local_root:
        requested_builds.append((local_root, False))

    pending_builds: list[tuple[Path, bool, int]] = []
    for root, force in requested_builds:
        ensure_ligands_exist(root)
        if representation_exists(root, resolved_name) and not force:
            print(f"[INFO] Representation '{resolved_name}' already exists in {root}. Skipping.")
            continue
        pending_builds.append((root, force, ligand_count(root)))

    step_count = len(pending_builds) + 2
    progress.emit(
        step="preparing",
        label="Preparing representation build",
        step_index=1,
        step_count=step_count,
        percent=1,
        context=resolved_name,
    )

    total_work = sum(n for _, _, n in pending_builds)
    completed_work = 0
    for phase_index, (root, force, n_ligands) in enumerate(pending_builds, start=2):
        initial_label = (
            f"Loading model for {root.name}"
            if args.representation_type == "huggingface"
            else f"Computing fingerprints for {root.name}"
        )
        initial_percent = 5 + round(completed_work / max(total_work, 1) * 90)
        progress.emit(
            step=f"building_{root.name}",
            label=initial_label,
            step_index=phase_index,
            step_count=step_count,
            percent=initial_percent,
            current=0,
            total=n_ligands,
            unit="compounds",
            context=root.name,
        )

        def phase_progress(current: int, total: int, *, _root=root, _completed=completed_work) -> None:
            overall_current = _completed + current
            progress.emit(
                step=f"building_{_root.name}",
                label=f"Computing '{resolved_name}' for {_root.name}",
                step_index=phase_index,
                step_count=step_count,
                percent=5 + round(overall_current / max(total_work, 1) * 90),
                current=current,
                total=total,
                unit="compounds",
                context=_root.name,
            )

        build_representation_if_needed(
            root=root,
            representation_type=args.representation_type,
            rep_name=resolved_name,
            model_id=args.model_id,
            n_bits=args.n_bits,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            rdkit_fp_kind=args.rdkit_fp_kind,
            n_jobs=args.n_jobs,
            chunksize=args.chunksize,
            force=force,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            progress_callback=phase_progress,
        )
        completed_work += n_ligands

    progress.emit(
        step="finalizing",
        label="Finalizing representation",
        step_index=step_count,
        step_count=step_count,
        percent=99,
        context=resolved_name,
    )


if __name__ == "__main__":
    main()
