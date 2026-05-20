from __future__ import annotations

import pytest


def test_rbcm_output_shapes() -> None:
    torch = pytest.importorskip("torch")
    from rbcm_edge.models.modules import RBCM

    module = RBCM(channels=16)
    x = torch.randn(2, 16, 32, 32)
    out = module(x)
    assert out["feature"].shape == x.shape
    assert out["gate"].shape == x.shape
    assert out["gate"].min().item() >= -1.0
    assert out["gate"].max().item() <= 1.0
