"""Exponential moving average of model parameters.

Lifted unchanged from ``train.py``. The shadow tracks only parameters with
``requires_grad``, so freezing part of the encoder changes what is averaged —
that interaction is inherited, not introduced.

No RNG is consumed here, so EMA can be toggled without affecting the training
stream; it only changes which weights evaluation sees.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["create_ema_state", "update_ema_state", "ema_scope"]


def create_ema_state(model: nn.Module, decay: float) -> dict:
    """Snapshot trainable parameters as the initial shadow."""
    shadow = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    return {"decay": decay, "shadow": shadow}


@torch.no_grad()
def update_ema_state(model: nn.Module, ema_state: Optional[dict]) -> None:
    """``shadow = decay * shadow + (1 - decay) * param``."""
    if ema_state is None:
        return
    d = ema_state["decay"]
    for n, p in model.named_parameters():
        if p.requires_grad:
            ema_state["shadow"][n].mul_(d).add_(p.detach(), alpha=1.0 - d)


@contextmanager
def ema_scope(model: nn.Module, ema_state: Optional[dict]):
    """Temporarily swap the shadow weights in, then restore the live ones."""
    if ema_state is None:
        yield
        return
    backup = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and n in ema_state["shadow"]:
                backup[n] = p.detach().clone()
                p.copy_(ema_state["shadow"][n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup:
                    p.copy_(backup[n])
