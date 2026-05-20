"""Model factory for edge detection experiments."""

from __future__ import annotations

from rbcm_edge.models.networks import RBCMFuseEdgeNet


def build_model(config: dict):
    """Build the configured edge detection model.

    The first complete engineering version supports `rbcm_fuse`. TEED and
    PiDiNet adapters can later be added behind this same factory so training
    scripts do not change.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "rbcm_fuse")
    if name != "rbcm_fuse":
        raise ValueError(f"Unsupported model name for initial scaffold: {name}")

    return RBCMFuseEdgeNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        feature_channels=int(model_cfg.get("feature_channels", 32)),
        rbcm_reduction=int(model_cfg.get("rbcm_reduction", 2)),
        alpha_init=float(model_cfg.get("alpha_init", 0.1)),
    )
