"""Weights & Biases integration, isolated behind four functions.

Previously four private helpers inside ``train.py``. Behaviour is unchanged:
every call is a no-op when logging is disabled or the package is missing, and
every W&B call is wrapped so a logging failure can never abort a training run.

Nothing here touches the model, the optimiser or any RNG.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn

from mmfusion.config import WandbConfig

logger = logging.getLogger(__name__)

__all__ = ["init_run", "watch", "log", "finish"]

_WATCH_LOG_FREQ = 100


def init_run(wandb_cfg: WandbConfig, run_config: dict):
    """Start a run, or return None when disabled / wandb not installed."""
    if not wandb_cfg.enabled:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed - skipping WandB logging.")
        return None

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.run_name,
        tags=wandb_cfg.tags,
        config=run_config,
        reinit=True,
    )


def watch(run, model: nn.Module, wandb_cfg: WandbConfig) -> None:
    """Attach parameter (or gradient) histogram logging."""
    if run is None:
        return
    try:
        import wandb

        if wandb_cfg.log_gradients:
            wandb.watch(model, log="all", log_freq=_WATCH_LOG_FREQ)
        else:
            wandb.watch(model, log="parameters", log_freq=_WATCH_LOG_FREQ)
    except Exception:
        pass


def log(run, metrics: dict, step: int) -> None:
    """Log a metrics dict at *step*, swallowing any transport error."""
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:
        pass


def finish(run) -> None:
    """Close the run, swallowing any transport error."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
