from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from core.config import MAX_UPLOAD_BYTES


class UploadTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class FastaInspection:
    valid: bool
    query_ids: list[str]
    sequence_count: int
    total_residues: int
    duplicate_ids: list[str]


async def save_upload_stream(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int | None = MAX_UPLOAD_BYTES,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Copy an UploadFile to disk without retaining the full payload in RAM."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(chunk_size):
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise UploadTooLargeError(
                        f"The uploaded file exceeds the {max_bytes} byte limit."
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return total


def inspect_fasta(path: Path) -> tuple[bool, list[str]]:
    inspection = inspect_fasta_details(path)
    return inspection.valid, inspection.query_ids


def inspect_fasta_details(path: Path) -> FastaInspection:
    has_header = False
    has_sequence = False
    query_ids: list[str] = []
    total_residues = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                has_header = True
                header = line[1:].strip()
                if header:
                    query_ids.append(header.split()[0])
            else:
                has_sequence = True
                total_residues += len("".join(line.split()))

    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for query_id in query_ids:
        if query_id in seen and query_id not in duplicate_ids:
            duplicate_ids.append(query_id)
        seen.add(query_id)
    return FastaInspection(
        valid=has_header and has_sequence and bool(query_ids),
        query_ids=query_ids,
        sequence_count=len(query_ids),
        total_residues=total_residues,
        duplicate_ids=duplicate_ids,
    )
