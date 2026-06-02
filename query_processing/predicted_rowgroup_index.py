from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROW_GROUP_INDEX_FILENAME = "predicted_binding_rowgroup_index.json"
ROW_GROUP_INDEX_VERSION = 1


def default_row_group_index_path(parquet_path: str | Path) -> Path:
    return Path(parquet_path).with_name(ROW_GROUP_INDEX_FILENAME)


def _parquet_fingerprint(parquet_path: Path, parquet_file: pq.ParquetFile) -> dict[str, Any]:
    stat = parquet_path.stat()
    return {
        "parquet_size": stat.st_size,
        "parquet_mtime_ns": stat.st_mtime_ns,
        "num_row_groups": parquet_file.num_row_groups,
        "num_rows": parquet_file.metadata.num_rows,
    }


def _payload_matches_parquet(payload: dict[str, Any], parquet_path: Path, parquet_file: pq.ParquetFile) -> bool:
    if payload.get("version") != ROW_GROUP_INDEX_VERSION:
        return False
    for key, value in _parquet_fingerprint(parquet_path, parquet_file).items():
        if key == "parquet_mtime_ns":
            continue
        if payload.get(key) != value:
            return False
    return isinstance(payload.get("row_groups_by_uniprot_id"), dict)


def load_row_group_index(
    parquet_path: str | Path,
    index_path: str | Path | None = None,
) -> dict[str, list[int]] | None:
    parquet_path = Path(parquet_path)
    index_path = Path(index_path) if index_path is not None else default_row_group_index_path(parquet_path)
    if not parquet_path.exists() or not index_path.exists():
        return None

    parquet_file = pq.ParquetFile(parquet_path)
    with open(index_path, "r") as f:
        payload = json.load(f)
    if not _payload_matches_parquet(payload, parquet_path, parquet_file):
        return None

    row_groups_by_uniprot_id = payload["row_groups_by_uniprot_id"]
    return {
        str(uniprot_id): [int(row_group) for row_group in row_groups]
        for uniprot_id, row_groups in row_groups_by_uniprot_id.items()
    }


def build_row_group_index(
    parquet_path: str | Path,
    index_path: str | Path | None = None,
) -> dict[str, list[int]]:
    parquet_path = Path(parquet_path)
    index_path = Path(index_path) if index_path is not None else default_row_group_index_path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)
    if "uniprot_id" not in parquet_file.schema_arrow.names:
        raise ValueError(f"Parquet file must contain 'uniprot_id': {parquet_path}")

    uniprot_column_index = parquet_file.schema_arrow.names.index("uniprot_id")
    row_groups_by_uniprot_id: dict[str, list[int]] = {}

    for row_group_index in range(parquet_file.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(uniprot_column_index)
        stats = column.statistics
        if stats is not None and stats.min is not None and stats.max is not None and stats.min == stats.max:
            protein_ids = [str(stats.min)]
        else:
            table = parquet_file.read_row_group(row_group_index, columns=["uniprot_id"])
            unique_values = pc.unique(table["uniprot_id"]).to_pylist()
            protein_ids = [str(value) for value in unique_values if value is not None]

        for protein_id in protein_ids:
            row_groups_by_uniprot_id.setdefault(protein_id, []).append(row_group_index)

    payload = {
        "version": ROW_GROUP_INDEX_VERSION,
        **_parquet_fingerprint(parquet_path, parquet_file),
        "row_groups_by_uniprot_id": row_groups_by_uniprot_id,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(payload, f)
    os.replace(temp_path, index_path)
    return row_groups_by_uniprot_id


def load_or_build_row_group_index(
    parquet_path: str | Path,
    index_path: str | Path | None = None,
) -> dict[str, list[int]]:
    index = load_row_group_index(parquet_path, index_path=index_path)
    if index is not None:
        return index
    return build_row_group_index(parquet_path, index_path=index_path)
