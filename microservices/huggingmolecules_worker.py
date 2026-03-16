from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def _load_huggingmolecules_encoder(model_type: str, model_id: str, device: torch.device):
    """
    Best-effort adapter for HuggingMolecules.

    If API changes in upstream huggingmolecules, users can replace this function
    with a local adapter while keeping the same worker contract.
    """
    model_type = model_type.lower().strip()
    try:
        if model_type == "grover":
            from huggingmolecules import GroverConfig, GroverFeaturizer, GroverModel

            config = GroverConfig.from_pretrained(model_id)
            featurizer = GroverFeaturizer.from_pretrained(model_id)
            model = GroverModel.from_pretrained(model_id, config=config).to(device)
        elif model_type == "rmat":
            from huggingmolecules import RMatConfig, RMatFeaturizer, RMatModel

            config = RMatConfig.from_pretrained(model_id)
            featurizer = RMatFeaturizer.from_pretrained(model_id)
            model = RMatModel.from_pretrained(model_id, config=config).to(device)
        else:
            raise ValueError(f"Unsupported HuggingMolecules model_type: {model_type}")
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize HuggingMolecules backend. Verify model_type/model_id and package compatibility."
        ) from exc

    model.eval()

    def _encode(smiles_batch: list[str]) -> np.ndarray:
        with torch.inference_mode():
            feats = featurizer(smiles_batch)
            if isinstance(feats, dict):
                feats = {k: v.to(device) if hasattr(v, "to") else v for k, v in feats.items()}
                out = model(**feats)
            else:
                out = model(feats)

            if isinstance(out, dict):
                for key in ("pooler_output", "embeddings", "last_hidden_state", "logits"):
                    if key in out:
                        tensor = out[key]
                        break
                else:
                    raise RuntimeError("Could not infer embedding tensor from HuggingMolecules dict output.")
            else:
                tensor = out

            if tensor.ndim == 3:
                tensor = tensor.mean(dim=1)
            if tensor.ndim != 2:
                raise RuntimeError(f"Unexpected HuggingMolecules output ndim={tensor.ndim}; expected 2.")

            return tensor.detach().to(torch.float16).cpu().numpy()

    return _encode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuggingMolecules representation worker")
    p.add_argument("--root", required=True, type=str)
    p.add_argument("--rep-name", required=True, type=str)
    p.add_argument("--model-type", required=True, choices=["grover", "rmat"])
    p.add_argument("--model-id", required=True, type=str)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-bits", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    reps_dir = root / "reps"
    reps_dir.mkdir(parents=True, exist_ok=True)

    lig_path = root / "ligands.parquet"
    ligs = pd.read_parquet(lig_path, columns=["smiles"])
    smiles_all = ["" if pd.isna(x) else str(x).strip() for x in ligs["smiles"].tolist()]
    n = len(smiles_all)
    if n == 0:
        raise ValueError("ligands.parquet is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _load_huggingmolecules_encoder(args.model_type, args.model_id, device)

    # infer dim
    probe = None
    for s in smiles_all:
        if s:
            probe = s
            break
    if probe is None:
        raise ValueError("All smiles are empty/invalid")

    probe_emb = encoder([probe])
    if probe_emb.ndim != 2 or probe_emb.shape[0] != 1:
        raise RuntimeError(f"Unexpected probe embedding shape: {probe_emb.shape}")
    dim = int(probe_emb.shape[1])

    if args.n_bits is not None and int(args.n_bits) != dim:
        raise ValueError(
            f"n_bits={args.n_bits} does not match HuggingMolecules embedding dim={dim}."
        )

    data_path = reps_dir / f"{args.rep_name}.dat"
    mm = np.memmap(data_path, mode="w+", dtype=np.float16, shape=(n, dim))

    invalid_smiles = 0
    embed_failures = 0
    t0 = time.perf_counter()

    total_batches = (n + args.batch_size - 1) // args.batch_size
    for start in tqdm(range(0, n, args.batch_size), total=total_batches, desc=f"[{root.name}] {args.rep_name}"):
        end = min(start + args.batch_size, n)
        batch = smiles_all[start:end]
        out = np.zeros((end - start, dim), dtype=np.float16)

        valid_idx = [i for i, smi in enumerate(batch) if smi]
        valid_smiles = [batch[i] for i in valid_idx]
        invalid_smiles += (len(batch) - len(valid_smiles))

        if valid_smiles:
            try:
                emb = encoder(valid_smiles)
                if emb.shape != (len(valid_smiles), dim):
                    raise RuntimeError(f"Unexpected embedding shape {emb.shape}")
                out[np.asarray(valid_idx)] = emb.astype(np.float16, copy=False)
            except Exception:
                for idx, smi in zip(valid_idx, valid_smiles):
                    try:
                        out[idx] = encoder([smi])[0].astype(np.float16, copy=False)
                    except Exception:
                        embed_failures += 1

        mm[start:end, :] = out

    mm.flush()
    elapsed = float(time.perf_counter() - t0)

    meta = {
        "name": args.rep_name,
        "file": f"{args.rep_name}.dat",
        "dtype": "float16",
        "dim": int(dim),
        "packed_bits": False,
        "packed_dim": None,
        "n_ligands": int(n),
        "model_id": args.model_id,
        "model_type": args.model_type,
        "max_length": args.max_length,
        "elapsed_seconds": elapsed,
        "ligands_per_second": float(n / elapsed) if elapsed > 0 else 0.0,
        "invalid_smiles": int(invalid_smiles),
        "invalid_smiles_fraction": float(invalid_smiles / n),
        "embed_failures": int(embed_failures),
    }

    meta_path = reps_dir / f"{args.rep_name}.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
