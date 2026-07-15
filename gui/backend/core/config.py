import os
from pathlib import Path

_SOURCE_PIPELINE_ROOT = Path(__file__).resolve().parents[3]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


# These defaults preserve the native Conda layout. Docker overrides them with
# mounted volume paths, so the scientific pipeline never needs Docker-specific
# branches.
PIPELINE_ROOT: Path = _path_from_env("LIGQ_PIPELINE_ROOT", _SOURCE_PIPELINE_ROOT)
DATABASES_DIR: Path = _path_from_env("LIGQ_DATABASES_DIR", PIPELINE_ROOT / "databases")
COMPOUND_DATA_DIR: Path = DATABASES_DIR / "compound_data"
RESULTS_DIR: Path = _path_from_env("LIGQ_RESULTS_DIR", PIPELINE_ROOT / "results")
UPLOADS_DIR: Path = _path_from_env(
    "LIGQ_UPLOADS_DIR", PIPELINE_ROOT / "gui" / "backend" / "uploads"
)
TEMP_RESULTS_DIR: Path = _path_from_env(
    "LIGQ_TEMP_RESULTS_DIR", PIPELINE_ROOT / "temp_results"
)
STATE_DIR: Path = _path_from_env(
    "LIGQ_STATE_DIR", PIPELINE_ROOT / "gui" / "backend" / "state"
)
JOB_DB_PATH: Path = STATE_DIR / "jobs.sqlite3"

MAX_UPLOAD_BYTES: int = int(os.environ.get("LIGQ_MAX_UPLOAD_BYTES", str(20 * 1024**3)))
JOB_SHUTDOWN_GRACE_SECONDS: float = float(
    os.environ.get("LIGQ_JOB_SHUTDOWN_GRACE_SECONDS", "15")
)

ALLOWED_ORIGINS: list[str] = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")
