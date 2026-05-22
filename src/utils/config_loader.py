from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


def _resolve_paths(settings: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(settings)

    for key, relative_path in resolved.get("profiles", {}).items():
        resolved["profiles"][key] = str((PROJECT_ROOT / relative_path).resolve())

    logging_settings = resolved.get("logging", {})
    for key in ("jsonl_dir", "sqlite_path"):
        if key in logging_settings:
            logging_settings[key] = str((PROJECT_ROOT / logging_settings[key]).resolve())

    browser_settings = resolved.get("browser", {})
    if "app_profile" in browser_settings:
        browser_settings["app_profile"] = str((PROJECT_ROOT / browser_settings["app_profile"]).resolve())

    return resolved


def load_settings() -> dict[str, Any]:
    settings = _load_yaml(PROJECT_ROOT / "config" / "settings.yaml")
    model_sites = _load_yaml(PROJECT_ROOT / "config" / "model_sites.yaml")
    settings["model_sites"] = model_sites
    settings["project_root"] = str(PROJECT_ROOT)
    return _resolve_paths(settings)
