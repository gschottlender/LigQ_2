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

from compound_processing.compound_helpers import build_huggingface_representation


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
        help="Batch size for embedding computation.",
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
    rep_name: str,
    model_id: str,
    n_bits: Optional[int],
    batch_size: int,
    max_length: Optional[int],
    pooling: str,
    force: bool,
) -> None:
    ensure_ligands_exist(root)

    if representation_exists(root, rep_name) and not force:
        print(f"[INFO] Representation '{rep_name}' already exists in {root}. Skipping.")
        return

    print(f"[INFO] Building representation '{rep_name}' in: {root}")
    build_huggingface_representation(
        root=root,
        n_bits=n_bits,
        batch_size=batch_size,
        name=rep_name,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    local_root = output_dir / DEFAULT_LOCAL_ROOT
    zinc_root = output_dir / DEFAULT_ZINC_ROOT

    primary_root = zinc_root if args.base == "zinc" else local_root

    # 1) Build on selected base
    build_representation_if_needed(
        root=primary_root,
        rep_name=args.rep_name,
        model_id=args.model_id,
        n_bits=args.n_bits,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        force=args.force,
    )

    # 2) Ensure same representation exists in local base
    if primary_root != local_root:
        build_representation_if_needed(
            root=local_root,
            rep_name=args.rep_name,
            model_id=args.model_id,
            n_bits=args.n_bits,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            force=False,
        )


if __name__ == "__main__":
    main()
