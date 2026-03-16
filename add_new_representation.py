"""
add_new_representation.py

Build an additional HuggingFace representation for a target compound database
(default: ZINC). Also ensures the same representation exists in the local
PDB+ChEMBL compound database so both spaces stay compatible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from compound_processing.compound_helpers import (
    build_huggingmolecules_representation,
    build_huggingface_representation,
    build_rdkit_representation,
)


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
        "--rep-name",
        type=str,
        default="chemberta_zinc_base_768",
        help="Representation name (saved under reps/<rep-name>.dat + .meta.json).",
    )
    parser.add_argument(
        "--representation-type",
        type=str,
        default="huggingface",
        choices=["huggingface", "rdkit", "huggingmolecules"],
        help="Type of representation to build.",
    )
    parser.add_argument(
        "--rdkit-fp-kind",
        type=str,
        default="ap",
        choices=["ap", "topological_torsion", "rdkit", "maccs"],
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
        default=768,
        help="Expected embedding dimension (must match model hidden_size).",
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
        "--hm-model-type",
        type=str,
        default="grover",
        choices=["grover", "rmat"],
        help="HuggingMolecules model family when --representation-type=huggingmolecules.",
    )
    parser.add_argument(
        "--hm-model-id",
        type=str,
        default="grover_base",
        help="HuggingMolecules model identifier/checkpoint.",
    )
    parser.add_argument(
        "--hm-env-dir",
        type=str,
        default=".microenvs/huggingmolecules",
        help="Reusable local micro-environment directory for HuggingMolecules.",
    )
    parser.add_argument(
        "--hm-repo-url",
        type=str,
        default="https://github.com/gmum/huggingmolecules.git",
        help="Git URL used to install HuggingMolecules into the micro-environment.",
    )
    parser.add_argument(
        "--hm-repo-ref",
        type=str,
        default="",
        help="Optional git ref (branch/tag/commit) appended as @<ref> during installation.",
    )
    parser.add_argument(
        "--hm-force-install",
        action="store_true",
        help="Force reinstall dependencies inside the HuggingMolecules micro-environment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the representation already exists.",
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


def build_representation_if_needed(
    root: Path,
    representation_type: str,
    rep_name: str,
    model_id: str,
    n_bits: Optional[int],
    batch_size: int,
    max_length: Optional[int],
    pooling: str,
    hm_model_type: str,
    hm_model_id: str,
    hm_env_dir: str,
    hm_repo_url: str,
    hm_repo_ref: str,
    hm_force_install: bool,
    rdkit_fp_kind: str,
    n_jobs: Optional[int],
    chunksize: int,
    force: bool,
) -> None:
    ensure_ligands_exist(root)

    if representation_exists(root, rep_name) and not force:
        print(f"[INFO] Representation '{rep_name}' already exists in {root}. Skipping.")
        return

    print(f"[INFO] Building representation '{rep_name}' in: {root}")
    if representation_type == "huggingface":
        build_huggingface_representation(
            root=root,
            n_bits=n_bits,
            batch_size=batch_size,
            name=rep_name,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
        )
    elif representation_type == "rdkit":
        if n_bits is None:
            raise ValueError("--n-bits must be provided for RDKit fingerprints.")
        build_rdkit_representation(
            root=root,
            fp_kind=rdkit_fp_kind,
            n_bits=int(n_bits),
            batch_size=batch_size,
            name=rep_name,
            n_jobs=n_jobs,
            chunksize=chunksize,
        )
    elif representation_type == "huggingmolecules":
        build_huggingmolecules_representation(
            root=root,
            n_bits=n_bits,
            batch_size=batch_size,
            name=rep_name,
            model_type=hm_model_type,
            model_id=hm_model_id,
            max_length=max_length,
            microenv_dir=Path(hm_env_dir),
            hm_repo_url=hm_repo_url,
            hm_repo_ref=hm_repo_ref or None,
            force_install=hm_force_install,
        )
    else:
        raise ValueError(f"Unsupported representation_type: {representation_type}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    local_root = output_dir / DEFAULT_LOCAL_ROOT
    zinc_root = output_dir / DEFAULT_ZINC_ROOT

    primary_root = zinc_root if args.base == "zinc" else local_root

    # 1) Build on selected base
    build_representation_if_needed(
        root=primary_root,
        representation_type=args.representation_type,
        rep_name=args.rep_name,
        model_id=args.model_id,
        n_bits=args.n_bits,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        hm_model_type=args.hm_model_type,
        hm_model_id=args.hm_model_id,
        hm_env_dir=args.hm_env_dir,
        hm_repo_url=args.hm_repo_url,
        hm_repo_ref=args.hm_repo_ref,
        hm_force_install=args.hm_force_install,
        rdkit_fp_kind=args.rdkit_fp_kind,
        n_jobs=args.n_jobs,
        chunksize=args.chunksize,
        force=args.force,
    )

    # 2) Ensure same representation exists in local base
    if primary_root != local_root:
        build_representation_if_needed(
            root=local_root,
            representation_type=args.representation_type,
            rep_name=args.rep_name,
            model_id=args.model_id,
            n_bits=args.n_bits,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            hm_model_type=args.hm_model_type,
            hm_model_id=args.hm_model_id,
            hm_env_dir=args.hm_env_dir,
            hm_repo_url=args.hm_repo_url,
            hm_repo_ref=args.hm_repo_ref,
            hm_force_install=False,
            rdkit_fp_kind=args.rdkit_fp_kind,
            n_jobs=args.n_jobs,
            chunksize=args.chunksize,
            force=False,
        )


if __name__ == "__main__":
    main()
