"""Loss functions for survival, regression and classification.

Consolidates the losses that were split between ``losses_metrics.py`` (Cox,
MSE, MAE, Huber, BCE, CE) and the notebook cells (quantile, weighted Huber).
Every implementation is byte-equivalent to its origin; only the location moved.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "cox_ph_loss",
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "quantile_loss",
    "weighted_huber",
    "bce_loss",
    "ce_loss",
    "resolve_loss_fn",
    "resolve_mil_loss_fn",
]


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

def cox_ph_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Breslow approximation of the Cox partial-likelihood loss.

    Parameters
    ----------
    risk  : [B] (or [B, 1]) predicted log-hazard scores
    time  : [B] observed times
    event : [B] 1 = event occurred, 0 = censored
    """
    risk = risk.view(-1)
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    event_sum = event.sum().clamp_min(1.0)
    return -((risk - log_cumsum) * event).sum() / event_sum


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.view(target.shape), target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred.view(target.shape), target)


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    return F.huber_loss(pred.view(target.shape), target, delta=delta)


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Pinball loss. ``tau=0.5`` is median regression.

    Origin: ``Pathology_only_TCGR_GHI.ipynb``. Used for the GHI assay only.
    """
    error = target - pred
    loss = torch.max((tau - 1) * error, tau * error)
    return loss.mean()


def weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Per-sample-weighted smooth L1.

    Origin: ``Pathology_only_TCGR_GHI.ipynb``. Note that this uses
    ``F.smooth_l1_loss`` (beta=1.0) whereas :func:`huber_loss` uses
    ``F.huber_loss`` (delta=1.0); for beta == delta == 1.0 the two differ by a
    factor of ``delta``. That asymmetry is inherited, not introduced.
    """
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).mean()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy (single or multi-label).

    logits : [B] or [B, C]  (raw scores, not sigmoid)
    labels : [B] or [B, C]  (float)
    """
    return F.binary_cross_entropy_with_logits(
        logits.view(labels.shape).float(),
        labels.float(),
        pos_weight=pos_weight,
    )


def ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Categorical cross-entropy. logits [B, C], labels [B] (long)."""
    return F.cross_entropy(logits, labels.long(), weight=weight)


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

_AUTO_BY_TASK: Dict[str, str] = {
    "survival": "cox_ph",
    "regression": "mse",
}


def resolve_loss_fn(
    task_type: str,
    loss_fn_name: str,
    num_classes: int = 2,
) -> Callable[[torch.Tensor, dict], torch.Tensor]:
    """Return ``loss_fn(pred, batch) -> scalar`` for the fusion training loop.

    Behaviourally identical to the former ``losses_metrics._resolve_loss_fn``,
    including the ``auto`` mapping (survival→cox_ph, regression→mse,
    classification→bce when ``num_classes <= 2`` else ce).
    """
    if loss_fn_name != "auto":
        name = loss_fn_name
    elif task_type == "classification":
        name = "bce" if num_classes <= 2 else "ce"
    else:
        # Unknown task types raise KeyError here, as the original dict-index did.
        name = _AUTO_BY_TASK[task_type]

    if name == "cox_ph":
        def fn(pred, batch):
            return cox_ph_loss(pred, batch["time"], batch["event"])
    elif name == "mse":
        def fn(pred, batch):
            return mse_loss(pred, batch["target"])
    elif name == "mae":
        def fn(pred, batch):
            return mae_loss(pred, batch["target"])
    elif name == "huber":
        def fn(pred, batch):
            return huber_loss(pred, batch["target"])
    elif name == "bce":
        def fn(pred, batch):
            return bce_loss(pred, batch["label"].float())
    elif name == "ce":
        def fn(pred, batch):
            return ce_loss(pred, batch["label"])
    else:
        raise ValueError(f"Unknown loss_fn: {name}")
    return fn


def resolve_mil_loss_fn(
    loss_type: str,
    tau: float = 0.5,
) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """Return ``loss_fn(pred, target, weight) -> scalar`` for the MIL loop.

    Mirrors the if/elif ladder that was inlined in the notebooks' ``run_epoch``,
    including the ``ValueError`` raised when ``weighted_huber`` is selected but
    no sample weights were supplied.
    """
    if loss_type == "huber":
        def fn(pred, target, weight=None):
            return huber_loss(pred, target)
    elif loss_type == "quantile":
        def fn(pred, target, weight=None):
            return quantile_loss(pred, target, tau=tau)
    elif loss_type == "weighted_huber":
        def fn(pred, target, weight=None):
            if weight is None:
                raise ValueError("Weighted Huber requires sample weights in dataset")
            return weighted_huber(pred, target, weight)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_type}")
    return fn
