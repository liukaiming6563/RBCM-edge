"""Retinal Boundary-Context Modulation module."""

from __future__ import annotations

import torch
from torch import nn


class RBCM(nn.Module):
    """Retinal Boundary-Context Modulation.

    The module separates local edge features and boundary-context features, then
    predicts a signed gate in `[-1, 1]`. Positive gate values enhance local edge
    response, negative values suppress it, and values near zero keep it stable.
    """

    def __init__(self, channels: int, reduction: int = 2, alpha_init: float = 0.1) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.ctx_d1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False
        )
        self.ctx_d2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
        )
        self.ctx_d4 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels, bias=False
        )
        self.context_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return modulated feature, signed gate, local feature, and context feature."""
        f_local = self.local_branch(x)
        c1 = self.ctx_d1(x)
        c2 = self.ctx_d2(x)
        c4 = self.ctx_d4(x)
        f_context = self.context_fusion(torch.cat([c1, c2, c4], dim=1))
        gate_input = torch.cat([f_local, f_context, f_context - f_local], dim=1)
        g_context = self.gate(gate_input)
        feature = f_local + self.alpha * f_local * g_context
        return {
            "feature": feature,
            "gate": g_context,
            "local": f_local,
            "context": f_context,
        }
