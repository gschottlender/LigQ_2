import os
from pathlib import Path

_SOURCE_PIPELINE_ROOT = Path(__file__).resolve().parents[3]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

DEPLOYMENT_MODE: str = os.environ.get("LIGQ_DEPLOYMENT_MODE", "local").strip().lower()
if DEPLOYMENT_MODE not in {"local", "web"}:
    raise RuntimeError("LIGQ_DEPLOYMENT_MODE must be either 'local' or 'web'.")

WEB_MAX_FASTA_SEQUENCES: int = int(os.environ.get("LIGQ_WEB_MAX_FASTA_SEQUENCES", "100"))
WEB_MAX_FASTA_RESIDUES: int = int(os.environ.get("LIGQ_WEB_MAX_FASTA_RESIDUES", "500000"))
WEB_MAX_FASTA_BYTES: int = int(os.environ.get("LIGQ_WEB_MAX_FASTA_BYTES", str(5 * 1024**2)))
WEB_SEARCH_TIMEOUT_SECONDS: int = int(os.environ.get("LIGQ_WEB_SEARCH_TIMEOUT_SECONDS", "3600"))
WEB_RESULT_RETENTION_SECONDS: int = int(os.environ.get("LIGQ_WEB_RESULT_RETENTION_SECONDS", "7200"))
WEB_RATE_LIMIT_COUNT: int = int(os.environ.get("LIGQ_WEB_RATE_LIMIT_COUNT", "20"))
WEB_RATE_LIMIT_WINDOW_SECONDS: int = int(
    os.environ.get("LIGQ_WEB_RATE_LIMIT_WINDOW_SECONDS", "3600")
)
WEB_SESSION_COOKIE_SECURE: bool = _bool_from_env("LIGQ_SESSION_COOKIE_SECURE", False)
WEB_TRUST_PROXY_HEADERS: bool = _bool_from_env("LIGQ_TRUST_PROXY_HEADERS", False)
WEB_SESSION_SECRET: str = os.environ.get(
    "LIGQ_SESSION_SECRET",
    "ligq2-local-web-test-secret-change-before-public-deployment",
)

ALLOWED_ORIGINS: list[str] = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")
