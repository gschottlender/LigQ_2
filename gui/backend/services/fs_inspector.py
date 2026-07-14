import json
from pathlib import Path
from core.config import COMPOUND_DATA_DIR, PIPELINE_ROOT

_FINGERPRINT_TYPES = {"morgan", "rdkit", "maccs", "topological", "atompair", "avalon", "ecfp", "fcfp"}
_COSINE_KEYWORDS = {"chemberta", "huggingface", "bert", "transformer", "embedding", "molformer"}
_THRESHOLD_DEFAULTS_PATH = PIPELINE_ROOT / "search_threshold_defaults.json"


def load_search_threshold_defaults() -> dict[str, float]:
    try:
        data = json.loads(_THRESHOLD_DEFAULTS_PATH.read_text())
        return {str(name): float(value) for name, value in data.items()}
    except (OSError, ValueError, TypeError):
        return {}


def get_metric_from_manifest(rep_path: Path) -> str:
    """Return the similarity metric for a representation.

    Checks, in order:
      1. ``search_metric`` key in the JSON sidecar (future-proof explicit field).
      2. ``fingerprint_type`` in the sidecar — known FP types → tanimoto, anything
         else (embeddings) → cosine.
      3. Name-based keyword heuristic as last resort.

    The sidecar is ``{stem}.meta.json`` alongside the ``.dat`` file (actual layout)
    or ``manifest.json`` inside a representation sub-directory (alternative layout).
    """
    candidates = [
        rep_path.with_suffix(".meta.json"),       # actual: {name}.meta.json next to .dat
        rep_path.parent / "manifest.json",        # alternative: manifest.json in same dir
    ]
    for meta in candidates:
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                if "search_metric" in data:
                    return str(data["search_metric"])
                fp_type = data.get("fingerprint_type", "").lower()
                if fp_type in _FINGERPRINT_TYPES:
                    return "tanimoto"
                if fp_type:
                    return "cosine"
            except Exception:
                pass

    name = rep_path.stem.lower()
    return "cosine" if any(kw in name for kw in _COSINE_KEYWORDS) else "tanimoto"


def list_databases() -> list[str]:
    if not COMPOUND_DATA_DIR.exists():
        return []
    return sorted(
        d.name
        for d in COMPOUND_DATA_DIR.iterdir()
        if d.is_dir() and d.name != "pdb_chembl" and (d / "ligands.parquet").exists()
    )


def list_representations(db_name: str) -> list[dict]:
    reps_dir = COMPOUND_DATA_DIR / db_name / "reps"
    if not reps_dir.exists():
        return []
    defaults = load_search_threshold_defaults()
    return [
        {
            "name": f.stem,
            "metric": get_metric_from_manifest(f),
            "default_threshold": defaults.get(f.stem),
        }
        for f in sorted(reps_dir.iterdir())
        if f.is_file() and f.suffix == ".dat"
    ]


def database_exists(db_name: str) -> bool:
    return (COMPOUND_DATA_DIR / db_name / "ligands.parquet").exists()


def representation_exists(db_name: str, rep_name: str) -> bool:
    return (COMPOUND_DATA_DIR / db_name / "reps" / f"{rep_name}.dat").exists()


def read_file_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_schema(str(path)).names
    import pandas as pd
    sep = "\t" if suffix == ".tsv" else ","
    return list(pd.read_csv(path, nrows=0, sep=sep).columns)
