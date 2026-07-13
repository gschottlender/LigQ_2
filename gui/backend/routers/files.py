from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from core.config import UPLOADS_DIR
from services.fs_inspector import read_file_columns

router = APIRouter(prefix="/api/files", tags=["files"])

_ALLOWED_EXTENSIONS = {".smi", ".csv", ".tsv", ".parquet"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            detail={
                "error": "invalid_file_type",
                "message": f"Unsupported file type '{suffix}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
                "details": None,
            },
        )

    file_id = str(uuid.uuid4())
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{file_id}{suffix}"
    dest.write_bytes(await file.read())

    columns: list[str] = []
    if suffix != ".smi":
        try:
            columns = read_file_columns(dest)
        except Exception:
            columns = []

    return {"file_id": file_id, "filename": file.filename, "columns": columns}