"""I/O helpers shared by MEA and model scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_auto(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV files saved with either UTF-8 or UTF-8-BOM."""
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with stable indentation for reproducibility."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
