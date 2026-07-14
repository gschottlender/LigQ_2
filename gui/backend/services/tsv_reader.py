import ast
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

import logging
logger = logging.getLogger(__name__)

_LIST_COLUMNS = {"binding_sites", "pdb_ids"}

def _nan_to_none(val: Any) -> Any:
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _parse_list_value(v: Any) -> list:
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    if not isinstance(v, str) or not v.strip():
        return []
    v = v.strip()
    if v.startswith("["):
        quoted_values = re.findall(r"['\"]([^'\"]+)['\"]", v)
        if quoted_values:
            return [x.strip() for x in quoted_values if x.strip()]
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        inner = v[1:-1].strip()
        return [
            x.strip("'\"")
            for x in re.split(r"[\s,;]+", inner)
            if x.strip("'\"")
        ]
    return [x.strip() for x in re.split(r"[;,]", v) if x.strip()]


def _clean_row(row: dict) -> dict:
    return {k: _nan_to_none(v) for k, v in row.items()}


def read_tsv_paginated(path, page=1, per_page=20, filters=None, sort_by=None, sort_dir="asc"):
    empty_pagination = {"page": page, "per_page": per_page, "total": 0, "total_pages": 0}
    if not path.exists():
        return {"data": [], "pagination": empty_pagination}

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return {"data": [], "pagination": empty_pagination}

    if filters:
        for col, val in filters.items():
            if col in df.columns and val and val != "all":
                df = df[df[col] == val]

    for col in _LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_value)

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir.lower() != "desc"), na_position="last")

    total = len(df)
    total_pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    data = [_clean_row(r) for r in df.iloc[start : start + per_page].to_dict(orient="records")]

    return {
        "data": data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
    }


def _count_by_search_type(path: Path) -> dict[str, int]:
    """Count rows per search_type value in a TSV. Reads only the search_type column."""
    counts: dict[str, int] = {"sequence": 0, "nearest_k": 0, "domain": 0}
    if not path.exists():
        return counts
    try:
        df = pd.read_csv(path, sep="\t", usecols=["search_type"])
        for st, grp in df.groupby("search_type", sort=False):
            if str(st) in counts:
                counts[str(st)] = len(grp)
    except Exception:
        pass
    return counts


def _summary_row_from_dir(qdir: Path) -> dict:
    """Build a summary row by reading per-search-type counts from a query result dir."""
    prot = _count_by_search_type(qdir / "protein_ranking.tsv")
    known = _count_by_search_type(qdir / "known_ligands.tsv")
    pred = _count_by_search_type(qdir / "predicted_ligands.tsv")
    return _clean_row({
        "qseqid": qdir.name,
        "n_proteins_sequence": prot["sequence"],
        "n_proteins_nearest_k": prot["nearest_k"],
        "n_proteins_domain": prot["domain"],
        "n_known_ligands_sequence": known["sequence"],
        "n_known_ligands_nearest_k": known["nearest_k"],
        "n_known_ligands_domain": known["domain"],
        "n_predicted_ligands_sequence": pred["sequence"],
        "n_predicted_ligands_nearest_k": pred["nearest_k"],
        "n_predicted_ligands_domain": pred["domain"],
        "has_protein_ranking": (qdir / "protein_ranking.tsv").exists(),
        "has_known_ligands": (qdir / "known_ligands.tsv").exists(),
        "has_predicted_ligands": (qdir / "predicted_ligands.tsv").exists(),
    })


def read_summary(summary_path: Path, search_results_dir: Path) -> list[dict]:
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path, sep="\t")
        except Exception as exc:
            logger.error("Failed to read summary %s: %s", summary_path, exc)
            return []
        
        rows = []
        for row in df.to_dict(orient="records"):
            qseqid = row.get("qseqid", "")
            qdir = search_results_dir / qseqid
            row["has_protein_ranking"] = (qdir / "protein_ranking.tsv").exists()
            row["has_known_ligands"] = (qdir / "known_ligands.tsv").exists()
            row["has_predicted_ligands"] = (qdir / "predicted_ligands.tsv").exists()
            rows.append(_clean_row(row))
        return rows

    # Summary TSV not written yet — build incrementally from completed query dirs
    if not search_results_dir.exists():
        return []
    return [
        _summary_row_from_dir(qdir)
        for qdir in sorted(search_results_dir.iterdir())
        if qdir.is_dir()
    ]
