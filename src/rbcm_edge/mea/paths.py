"""Path conventions for the local MEA data tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXPERIMENTS: dict[str, str] = {
    "000031": "double_edge",
    "000032": "single_edge",
    "000033": "bar",
    "000034": "double_edge",
    "000035": "single_edge",
    "000036": "bar",
    "000037": "double_edge",
    "000038": "single_edge",
    "000039": "bar",
}


@dataclass(frozen=True)
class MeaProjectPaths:
    """Resolve paths for Kilosort outputs and stimulus metadata."""

    data_root: Path
    output_root: Path

    def run_root(self, run_id: str) -> Path:
        """Return the top-level folder for one experiment run."""
        return self.data_root / str(run_id)

    def kilosort_dir(self, run_id: str) -> Path:
        """Return the folder containing Kilosort output files for a run."""
        run_root = self.run_root(run_id)
        nested = run_root / "kilosort4"
        if nested.exists():
            return nested
        return run_root

    def stimulus_dir(self, paradigm: str) -> Path:
        """Return the stimulus metadata folder for a paradigm."""
        return self.data_root / "sti_info" / paradigm

    def per_run_output_dir(self, run_id: str, paradigm: str) -> Path:
        """Return the output folder for one run and paradigm."""
        return self.output_root / "per_experiment" / f"{run_id}_{paradigm}"
