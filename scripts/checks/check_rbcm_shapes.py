"""Run a small shape check for the RBCM module.

This is a normal Python script, not a test-framework file. If PyTorch is not
installed in the selected interpreter, the script prints a clear message instead
of relying on test-framework-specific skip behavior.
"""

from __future__ import annotations

import argparse

DEFAULT_ARGS = {
    "batch_size": 2,
    "channels": 16,
    "height": 32,
    "width": 32,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments while keeping editable defaults in the file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--channels", type=int, default=DEFAULT_ARGS["channels"])
    parser.add_argument("--height", type=int, default=DEFAULT_ARGS["height"])
    parser.add_argument("--width", type=int, default=DEFAULT_ARGS["width"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Instantiate RBCM and verify output tensor shapes and gate range."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in the current interpreter. Install torch or "
            "choose your PyTorch environment in PyCharm before running this check."
        ) from exc

    from rbcm_edge.models.modules import RBCM

    module = RBCM(channels=args.channels)
    x = torch.randn(args.batch_size, args.channels, args.height, args.width)
    with torch.no_grad():
        out = module(x)

    assert out["feature"].shape == x.shape, out["feature"].shape
    assert out["gate"].shape == x.shape, out["gate"].shape
    assert out["gate"].min().item() >= -1.0
    assert out["gate"].max().item() <= 1.0

    print("RBCM shape check passed")
    for key, value in out.items():
        print(f"  {key:10s}: {tuple(value.shape)}")
    print(f"  gate range: [{out['gate'].min().item():.3f}, {out['gate'].max().item():.3f}]")


if __name__ == "__main__":
    main(parse_args())
