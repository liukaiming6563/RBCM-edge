"""Training and evaluation loops for edge detection."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from edge_model.engine.metrics import edge_metrics_from_arrays, sigmoid_to_numpy
from edge_model.engine.visualize import (
    save_gate_heatmap,
    save_probability_map,
    save_triplet_visualization,
)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    epoch: int,
    log_interval: int,
) -> dict[str, float]:
    """Train one epoch and return averaged loss values."""
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["edge"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            outputs = model(images)
            losses = criterion(
                final_logits=outputs["logits"],
                target=targets,
                local_logits=outputs.get("local_logits"),
                gate=outputs.get("gate"),
            )

        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            optimizer.step()

        steps += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        if steps % max(1, log_interval) == 0:
            progress.set_postfix(total=totals["total"] / steps)

    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module | None,
    device: torch.device,
    visual_dir: Path | None = None,
    pred_dir: Path | None = None,
    gate_dir: Path | None = None,
    max_visual_samples: int = 0,
) -> dict[str, float]:
    """Evaluate a model and optionally save predictions and gate maps."""
    model.eval()
    probabilities: list = []
    targets: list = []
    loss_total = 0.0
    loss_steps = 0
    saved = 0

    progress = tqdm(loader, desc="eval", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["edge"].to(device, non_blocking=True)
        outputs = model(images)

        if criterion is not None:
            losses = criterion(
                final_logits=outputs["logits"],
                target=target,
                local_logits=outputs.get("local_logits"),
                gate=outputs.get("gate"),
            )
            loss_total += float(losses["total"].detach().cpu())
            loss_steps += 1

        batch_prob = sigmoid_to_numpy(outputs["logits"])
        batch_target = target.detach().cpu().numpy()
        for idx in range(batch_prob.shape[0]):
            prob = batch_prob[idx, 0]
            truth = batch_target[idx, 0]
            probabilities.append(prob)
            targets.append(truth)

            sample_id = str(batch["sample_id"][idx])
            if pred_dir is not None:
                save_probability_map(prob, pred_dir / f"{sample_id}.png")
            if gate_dir is not None and "gate" in outputs:
                save_gate_heatmap(outputs["gate"][idx], gate_dir / f"{sample_id}.png")
            if visual_dir is not None and saved < max_visual_samples:
                save_triplet_visualization(
                    image_tensor=batch["image"][idx].cpu(),
                    target_tensor=batch["edge"][idx].cpu(),
                    probability=prob,
                    output_path=visual_dir / f"{sample_id}.png",
                )
                saved += 1

    metrics = edge_metrics_from_arrays(probabilities, targets)
    if loss_steps:
        metrics["loss"] = loss_total / loss_steps
    return metrics


def append_metrics_csv(path: str | Path, row: dict) -> None:
    """Append one metrics row to CSV, writing the header when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
