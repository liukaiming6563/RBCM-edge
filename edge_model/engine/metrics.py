"""Evaluation metrics for binary edge probability maps."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def sigmoid_to_numpy(logits: torch.Tensor) -> np.ndarray:
    """Convert `[B, 1, H, W]` logits to NumPy probabilities."""
    return torch.sigmoid(logits).detach().cpu().numpy()


def edge_metrics_from_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute approximate ODS, OIS, and AP for edge maps.

    This is a pure-Python benchmark approximation suitable for development and
    ablations. If final publication needs strict BSDS evaluation compatibility,
    a dataset-specific official evaluator can be plugged in later.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    flat_prob = np.concatenate([p.reshape(-1) for p in probabilities])
    flat_target = np.concatenate([t.reshape(-1) for t in targets]).astype(np.uint8)

    ods = max(_f1_at_threshold(flat_prob, flat_target, threshold) for threshold in thresholds)

    per_image_best = []
    for prob, target in zip(probabilities, targets):
        prob_flat = prob.reshape(-1)
        target_flat = target.reshape(-1).astype(np.uint8)
        per_image_best.append(
            max(_f1_at_threshold(prob_flat, target_flat, threshold) for threshold in thresholds)
        )
    ois = float(np.mean(per_image_best)) if per_image_best else 0.0

    if flat_target.max() == flat_target.min():
        ap = 0.0
    else:
        ap = float(average_precision_score(flat_target, flat_prob))

    return {"ODS": float(ods), "OIS": float(ois), "AP": ap}


def _f1_at_threshold(prob: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """Compute binary F1 at one probability threshold."""
    pred = prob >= threshold
    truth = target > 0
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))
