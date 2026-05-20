"""Initial RBCM-Fuse edge detection network."""

from __future__ import annotations

import torch
from torch import nn

from rbcm_edge.models.modules import RBCM


class TinyEdgeBackbone(nn.Module):
    """Small convolutional backbone for smoke tests and pipeline debugging."""

    def __init__(self, in_channels: int = 3, feature_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RBCMFuseEdgeNet(nn.Module):
    """RBCM-Fuse architecture skeleton."""

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 32,
        rbcm_reduction: int = 2,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TinyEdgeBackbone(in_channels, feature_channels)
        self.rbcm = RBCM(feature_channels, reduction=rbcm_reduction, alpha_init=alpha_init)
        self.final_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.local_head = nn.Conv2d(feature_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self.backbone(x)
        rbcm_out = self.rbcm(fused)
        final_logits = self.final_head(rbcm_out["feature"])
        local_logits = self.local_head(rbcm_out["local"])
        return {
            "logits": final_logits,
            "local_logits": local_logits,
            "gate": rbcm_out["gate"],
            "feature": rbcm_out["feature"],
            "local_feature": rbcm_out["local"],
            "context_feature": rbcm_out["context"],
        }
