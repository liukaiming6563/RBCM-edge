"""Loss functions for edge detection with RBCM."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted BCE for sparse edge maps."""
    target = target.float()
    positive = target.sum().clamp_min(1.0)
    negative = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = negative / positive
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Dice loss computed on sigmoid probabilities."""
    prob = torch.sigmoid(logits)
    target = target.float()
    numerator = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + epsilon
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + epsilon
    return (1.0 - numerator / denominator).mean()


class EdgeDetectionLoss(nn.Module):
    """Combined loss used by the first RBCM experiments."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        local_weight: float = 0.3,
        gate_sparsity_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.local_weight = local_weight
        self.gate_sparsity_weight = gate_sparsity_weight

    def forward(
        self,
        final_logits: torch.Tensor,
        target: torch.Tensor,
        local_logits: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        final_bce = weighted_bce_with_logits(final_logits, target)
        final_dice = dice_loss_with_logits(final_logits, target)
        total = final_bce + self.dice_weight * final_dice
        local_loss = torch.zeros((), device=final_logits.device)
        if local_logits is not None:
            local_loss = weighted_bce_with_logits(local_logits, target) + dice_loss_with_logits(
                local_logits, target
            )
            total = total + self.local_weight * local_loss
        gate_loss = torch.zeros((), device=final_logits.device)
        if gate is not None:
            gate_loss = gate.abs().mean()
            total = total + self.gate_sparsity_weight * gate_loss
        return {
            "total": total,
            "final_bce": final_bce,
            "final_dice": final_dice,
            "local": local_loss,
            "gate_sparsity": gate_loss,
        }
