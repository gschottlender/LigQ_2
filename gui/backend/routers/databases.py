from fastapi import APIRouter, HTTPException
from core.config import UPLOADS_DIR
from services.fs_inspector import (
    database_exists,
    list_databases,
    list_representations,
    read_file_columns,
)
from core.policy import WEB_REPRESENTATIONS, is_web_mode
from services.web_access import require_resource_management

router = APIRouter(prefix="/api/databases", tags=["databases"])


@router.get("")
async def get_databases():
    if is_web_mode():
        return {"databases": ["zinc"] if database_exists("zinc") else []}
    return {"databases": list_databases()}


@router.get("/{name}/representations")
async def get_representations(name: str):
    if is_web_mode() and name != "zinc":
        raise HTTPException(
            404,
            detail={
                "error": "database_not_found",
                "message": "The requested database is not available.",
                "details": None,
            },
        )
    if not database_exists(name):
        raise HTTPException(
            404,
            detail={
                "error": "database_not_found",
                "message": f"Database '{name}' not found in databases/compound_data/",
                "details": None,
            },
        )
    representations = list_representations(name)
    if is_web_mode():
        allowed = {item.name for item in WEB_REPRESENTATIONS}
        representations = [
            item
            for item in representations
            if item["name"] in allowed and item["metric"] == "tanimoto"
        ]
    return {"representations": representations}


@router.get("/{name}/columns")
async def get_columns(name: str):
    require_resource_management()
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
