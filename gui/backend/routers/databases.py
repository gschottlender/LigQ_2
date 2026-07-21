from fastapi import APIRouter, HTTPException
from core.config import UPLOADS_DIR
from services.fs_inspector import (
    database_exists,
    list_databases,
    list_representations,
    read_file_columns,
)

router = APIRouter(prefix="/api/databases", tags=["databases"])


@router.get("")
async def get_databases():
    return {"databases": list_databases()}


@router.get("/{name}/representations")
async def get_representations(name: str):
    if not database_exists(name):
        raise HTTPException(
            404,
            detail={
                "error": "database_not_found",
                "message": f"Database '{name}' not found in databases/compound_data/",
                "details": None,
            },
        )
    return {"representations": list_representations(name)}


@router.get("/{name}/columns")
async def get_columns(name: str):
    for ext in (".parquet", ".csv", ".tsv"):
        path = UPLOADS_DIR / f"{name}{ext}"
        if path.exists():
            try:
                return {"columns": read_file_columns(path)}
            except Exception as exc:
                raise HTTPException(500, detail={"error": "read_error", "message": str(exc), "details": None})

    raise HTTPException(
        404,
        detail={
            "error": "file_not_found",
            "message": f"No uploaded file found with name '{name}'.",
            "details": None,
        },
    )