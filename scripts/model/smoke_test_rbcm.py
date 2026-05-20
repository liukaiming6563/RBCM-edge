"""Smoke-test the RBCM-Fuse model skeleton."""

from __future__ import annotations

import argparse

import torch

from rbcm_edge.models.networks import RBCMFuseEdgeNet

DEFAULT_ARGS = {
    "batch_size": 2,
    "height": 128,
    "width": 128,
    "feature_channels": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--height", type=int, default=DEFAULT_ARGS["height"])
    parser.add_argument("--width", type=int, default=DEFAULT_ARGS["width"])
    parser.add_argument("--feature-channels", type=int, default=DEFAULT_ARGS["feature_channels"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model = RBCMFuseEdgeNet(feature_channels=args.feature_channels)
    x = torch.randn(args.batch_size, 3, args.height, args.width)
    with torch.no_grad():
        out = model(x)
    print("Smoke test passed")
    for key, value in out.items():
        print(f"  {key:15s}: {tuple(value.shape)}")
    print(f"  gate range: [{out['gate'].min().item():.3f}, {out['gate'].max().item():.3f}]")


if __name__ == "__main__":
    main(parse_args())
