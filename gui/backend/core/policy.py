from __future__ import annotations

from dataclasses import asdict, dataclass

from core.config import (
    DEPLOYMENT_MODE,
    MAX_UPLOAD_BYTES,
    WEB_MAX_FASTA_BYTES,
    WEB_MAX_FASTA_RESIDUES,
    WEB_MAX_FASTA_SEQUENCES,
    WEB_RATE_LIMIT_COUNT,
    WEB_RATE_LIMIT_WINDOW_SECONDS,
    WEB_RESULT_RETENTION_SECONDS,
    WEB_SEARCH_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class WebRepresentationPolicy:
    name: str
    label: str
    metric: str
    cache_threshold_min: float
    default_threshold: float


WEB_REPRESENTATIONS = (
    WebRepresentationPolicy(
        name="morgan_1024_r2",
        label="Morgan ECFP",
        metric="tanimoto",
        cache_threshold_min=0.4,
        default_threshold=0.42,
    ),
    WebRepresentationPolicy(
        name="morgan_feature_1024_r2",
        label="Morgan Feature FCFP",
        metric="tanimoto",
        cache_threshold_min=0.5,
        default_threshold=0.51,
    ),
)


def is_web_mode() -> bool:
    return DEPLOYMENT_MODE == "web"


def web_representation_policy(name: str) -> WebRepresentationPolicy | None:
    return next((item for item in WEB_REPRESENTATIONS if item.name == name), None)


def policy_payload() -> dict:
    if not is_web_mode():
        return {
            "mode": "local",
            "allow_resource_management": True,
            "allow_setup_download": True,
            "allow_bsi": True,
            "history_scope": "global",
            "search": {
                "default_mode": "zinc",
                "allowed_modes": ["zinc", "known_only"],
                "provider": None,
                "representations": [],
                "allowed_methods": ["sequence", "nearest_k", "domain"],
                "nearest_k_min": 1,
                "nearest_k_max": 15,
                "nearest_k_default": 5,
                "max_fasta_sequences": None,
                "max_fasta_bytes": MAX_UPLOAD_BYTES,
                "max_fasta_residues": None,
                "max_active_jobs": None,
                "queue_limit": None,
                "timeout_seconds": None,
                "rate_limit_count": None,
                "rate_limit_window_seconds": None,
                "result_retention_seconds": None,
                "predicted_cache_mode": "read_write",
            },
        }

    return {
        "mode": "web",
        "allow_resource_management": False,
        "allow_setup_download": False,
        "allow_bsi": False,
        "history_scope": "session",
        "search": {
            "default_mode": "zinc",
            "allowed_modes": ["zinc", "known_only"],
            "provider": "zinc",
            "representations": [asdict(item) for item in WEB_REPRESENTATIONS],
            "allowed_methods": ["sequence", "nearest_k", "domain"],
            "nearest_k_min": 1,
            "nearest_k_max": 15,
            "nearest_k_default": 5,
            "max_fasta_sequences": WEB_MAX_FASTA_SEQUENCES,
            "max_fasta_bytes": WEB_MAX_FASTA_BYTES,
            "max_fasta_residues": WEB_MAX_FASTA_RESIDUES,
            "max_active_jobs": 1,
            "queue_limit": 0,
            "timeout_seconds": WEB_SEARCH_TIMEOUT_SECONDS,
            "rate_limit_count": WEB_RATE_LIMIT_COUNT,
            "rate_limit_window_seconds": WEB_RATE_LIMIT_WINDOW_SECONDS,
            "result_retention_seconds": WEB_RESULT_RETENTION_SECONDS,
            "predicted_cache_mode": "read_only",
        },
    }
