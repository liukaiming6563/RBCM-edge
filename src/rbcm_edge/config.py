"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    """Resolve a path that may be absolute or relative to the project root."""
    value = Path(value)
    if value.is_absolute():
        return value
    return Path(project_root) / value
