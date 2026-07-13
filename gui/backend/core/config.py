import os
from pathlib import Path

# gui/backend/core/config.py → parents[3] is the pipeline root
PIPELINE_ROOT: Path = Path(__file__).resolve().parents[3]

DATABASES_DIR: Path = PIPELINE_ROOT / "databases"
COMPOUND_DATA_DIR: Path = DATABASES_DIR / "compound_data"
RESULTS_DIR: Path = PIPELINE_ROOT / "results"
UPLOADS_DIR: Path = PIPELINE_ROOT / "gui" / "backend" / "uploads"

ALLOWED_ORIGINS: list[str] = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")