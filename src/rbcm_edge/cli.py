"""Small argparse helpers used by runnable scripts.

Scripts keep `DEFAULT_ARGS` dictionaries so exploratory runs can be controlled
either from command line flags or by editing default values inside the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping


def add_path_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: str | Path,
    help_text: str,
) -> None:
    """Add a Path argument with a readable default."""
    parser.add_argument(
        f"--{name}",
        type=Path,
        default=Path(default),
        help=f"{help_text} Default: {default}",
    )


def namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse output to a regular dictionary for logging or YAML."""
    return vars(args).copy()


def format_defaults(defaults: Mapping[str, Any]) -> str:
    """Format default arguments for diagnostic console output."""
    return "\n".join(f"  {key}: {value}" for key, value in defaults.items())
