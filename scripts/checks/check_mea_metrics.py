"""Run a small sanity check for MEA metric helpers.

This is a normal Python script, not a test-framework file. It is designed for PyCharm:
right-click the file and choose Run, or edit `DEFAULT_ARGS` below before running.
"""

from __future__ import annotations

import argparse

import numpy as np

from rbcm_edge.mea.metrics import modulation_index

DEFAULT_ARGS = {
    "epsilon": 1e-6,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments while keeping editable defaults in the file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_ARGS["epsilon"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Check the expected signs and finite behavior of modulation index."""
    enhanced = modulation_index(3.0, 1.0, epsilon=args.epsilon)
    suppressed = modulation_index(1.0, 3.0, epsilon=args.epsilon)
    neutral = modulation_index(0.0, 0.0, epsilon=args.epsilon)

    assert enhanced > 0, f"Expected positive MI for enhancement, got {enhanced}"
    assert suppressed < 0, f"Expected negative MI for suppression, got {suppressed}"
    assert np.isfinite(neutral), f"Expected finite MI at zero response, got {neutral}"

    print("MEA metric check passed")
    print(f"  enhanced MI:  {enhanced:.4f}")
    print(f"  suppressed MI: {suppressed:.4f}")
    print(f"  neutral MI:    {neutral:.4f}")


if __name__ == "__main__":
    main(parse_args())
