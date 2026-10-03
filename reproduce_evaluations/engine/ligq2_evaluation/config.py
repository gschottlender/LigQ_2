from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["_config_path"] = config_path
    cfg["_config_dir"] = config_path.parent
    return cfg


def resolve_path(cfg: dict[str, Any], section: str, key: str) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(cfg[section][key])))
    path = Path(raw)
    if not path.is_absolute():
        path = Path(cfg["_config_dir"]) / path
    return path.resolve()


def output_path(cfg: dict[str, Any], key: str) -> Path:
    root = resolve_path(cfg, "outputs", "root")
    if key == "root":
        return root
    value = Path(str(cfg["outputs"].get(key, key)))
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def split_kwargs(seed: int) -> dict[str, Any]:
    return {
        "max_total_actives": 1000,
        "known_frac": 0.1,
        "known_min": 10,
        "known_max": 200,
        "test_to_known_ratio": 10,
        "test_min": 50,
        "test_max": 1000,
        "nontrivial_tanimoto_cutoff": 0.8,
        "fallback_if_short": "allow_short",
        "random_state": int(seed),
    }


def parse_csv_arg(value: str | None, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]
