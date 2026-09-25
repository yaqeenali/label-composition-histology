"""Training loop for the multimodal fusion arm.

``fit`` is the entry point, carried over from ``train.py``. The per-epoch
sequence is unchanged and load-bearing::

    train_one_epoch -> EMA update -> evaluate(val) -> evaluate(test)
                    -> scheduler.step -> early-stopping check

RNG CONTRACT
------------
Three things draw from the global torch RNG here, and the third is easy to miss:

1. dropout and modality dropout, in ``model.train()`` only;
2. the shuffled batch order of the training loader;
3. **every** ``DataLoader.__iter__`` call, shuffled or not — PyTorch seeds each
   iterator with one draw from the global generator.

(3) means the number of evaluation passes is part of the RNG contract. Adding or
removing one shifts the stream for everything that follows, so
``WandbConfig.log_test_every_epoch=False`` changes trained weights. It is *not*
a logging-only switch; the divergence is measured in
``tests/test_parity_training.py::test_test_logging_toggle_changes_results``.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmfusion.config import TaskConfig, TrainConfig, WandbConfig
from mmfusion.losses import resolve_loss_fn
from mmfusion.metrics import compute_metrics, default_es_metric
from mmfusion.training import wandb_logging as wb
from mmfusion.training.ema import create_ema_state, ema_scope, update_ema_state

logger = logging.getLogger(__name__)

__all__ = ["fit", "train_one_epoch", "evaluate", "split_batch"]

RESERVED_KEYS = {"case_id", "time", "event", "target", "label"}


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def split_batch(batch: dict) -> Tuple[Dict, Dict]:
    """Separate model inputs (modalities and their masks) from targets."""
    modalities: Dict[str, Optional[torch.Tensor]] = {}
    masks: Dict[str, Optional[torch.Tensor]] = {}
    for k, v in batch.items():
        if k in RESERVED_KEYS:
            continue
        if k.endswith("_mask"):
            masks[k[:-5]] = v
        else:
            modalities[k] = v
    return modalities, masks


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    grad_clip: float = 0.0,
) -> dict:
    """One pass over the training loader. Mean loss over batches."""
    model.train()
    losses = []
    for batch in loader:
        batch = _to_device(batch, device)
        modalities, masks = split_batch(batch)
        pred = model(modalities, masks)
        loss = loss_fn(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return {"loss": float(np.mean(losses)) if losses else float("nan")}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    task_cfg: TaskConfig,
    device: torch.device,
) -> dict:
    """Loss plus task metrics over one loader. Consumes no RNG."""
    model.eval()
    losses, all_preds = [], []
    accum: Dict[str, list] = {"time": [], "event": [], "target": [], "label": []}

    for batch in loader:
        batch = _to_device(batch, device)
        modalities, masks = split_batch(batch)
        pred = model(modalities, masks)
        loss = loss_fn(pred, batch)
        losses.append(loss.item())
        all_preds.append(pred.detach().cpu().numpy())

        for key in accum:
            if key in batch:
                accum[key].append(batch[key].detach().cpu().numpy())

    if not losses:
        return {"loss": float("nan")}

    mean_loss = float(np.mean(losses))
    preds_np = np.concatenate(all_preds, axis=0)
    accum_np = {k: np.concatenate(v) for k, v in accum.items() if v}
    metrics = compute_metrics(preds_np, accum_np, task_cfg.task_type, task_cfg.num_classes)
    metrics["loss"] = mean_loss
    return metrics


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class _EarlyStopping:
    """Best-score tracker with patience.

    Extracted from the body of ``fit`` unchanged, including the details that
    matter: NaN scores are skipped without counting toward patience, the
    comparison is strict (``>`` / ``<``) so the first of equal scores wins, and
    the best state is a CPU clone of the full ``state_dict``.
    """

    def __init__(self, patience: int, higher_is_better: bool):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.no_improve = 0

    def update(self, score, model: nn.Module) -> bool:
        """Record *score*; return True when training should stop."""
        if score is None or np.isnan(float(score)):
            return False
        better = (
            self.best_score is None
            or (score > self.best_score if self.higher_is_better else score < self.best_score)
        )
        if better:
            self.best_score = score
            self.no_improve = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        self.no_improve += 1
        return self.no_improve >= self.patience


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def fit(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    train_cfg: TrainConfig,
    task_cfg: TaskConfig,
    wandb_cfg: Optional[WandbConfig] = None,
    run_config: Optional[dict] = None,
    device: Optional[str] = None,
) -> tuple:
    """Train and return ``(final_metrics, history)``.

    ``loaders`` needs at least ``train``; ``val`` and ``test`` are optional.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    logger.info(f"Training on device: {device}")

    loss_fn = resolve_loss_fn(task_cfg.task_type, task_cfg.loss_fn, task_cfg.num_classes)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.lr,
        weight_decay=train_cfg.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(train_cfg.max_epochs, 1), eta_min=0.0
    )

    ema_state = create_ema_state(model, train_cfg.ema_decay) if train_cfg.ema_eval else None

    es_metric = train_cfg.es_metric
    if es_metric == "auto":
        es_metric = default_es_metric(task_cfg.task_type)
    stopper = _EarlyStopping(train_cfg.es_patience, train_cfg.es_higher_is_better)

    wandb_cfg = wandb_cfg or WandbConfig(enabled=False)
    wb_run = wb.init_run(wandb_cfg, run_config or {})
    wb.watch(wb_run, model, wandb_cfg)

    # Per-epoch evaluation. The original iterated exactly ("val", "test"); adding
    # a pass for a custom es_split would insert an extra DataLoader.__iter__ and
    # therefore an extra global-RNG draw, changing trained weights. So we never
    # add one — we warn instead, matching the original's silent no-op.
    eval_splits = ("val", "test") if wandb_cfg.log_test_every_epoch else ("val",)
    if train_cfg.early_stopping and train_cfg.es_split not in eval_splits:
        logger.warning(
            "early_stopping is on with es_split=%r, which is not evaluated each "
            "epoch (%s). No epoch will be selected and the final weights are the "
            "last epoch's - matching the original behaviour. Evaluating it here "
            "would add an RNG draw and change results.",
            train_cfg.es_split, list(eval_splits),
        )

    history = []
    for epoch in range(train_cfg.max_epochs):
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, loss_fn,
            device, grad_clip=train_cfg.grad_clip,
        )
        update_ema_state(model, ema_state)

        epoch_row: dict = {"epoch": epoch, "train_loss": train_metrics["loss"]}
        log_dict: dict = {"epoch": epoch, "train/loss": train_metrics["loss"]}

        with ema_scope(model, ema_state):
            for split in eval_splits:
                if split not in loaders:
                    continue
                m = evaluate(model, loaders[split], loss_fn, task_cfg, device)
                for k, v in m.items():
                    epoch_row[f"{split}_{k}"] = v
                    log_dict[f"{split}/{k}"] = v

        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else train_cfg.lr
        log_dict["train/lr"] = lr_now

        history.append(epoch_row)
        scheduler.step()

        wb.log(wb_run, log_dict, step=epoch)

        if train_cfg.early_stopping and train_cfg.es_split in loaders:
            key = f"{train_cfg.es_split}_{es_metric}"
            if key in epoch_row and stopper.update(epoch_row[key], model):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement in '{key}' for {stopper.no_improve} epochs)"
                )
                break

        _log_epoch(epoch, train_cfg.max_epochs, epoch_row)

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    final: dict = {}
    eval_ema = ema_state if stopper.best_state is None else None
    with ema_scope(model, eval_ema):
        for split, loader in loaders.items():
            m = evaluate(model, loader, loss_fn, task_cfg, device)
            for k, v in m.items():
                final[f"{split}_{k}"] = v

    wb.log(wb_run, {f"final/{k}": v for k, v in final.items()}, step=train_cfg.max_epochs)
    wb.finish(wb_run)

    return final, history


def _log_epoch(epoch: int, max_epochs: int, row: dict) -> None:
    parts = [f"Epoch {epoch + 1:>4}/{max_epochs}"]
    for k, v in row.items():
        if k == "epoch":
            continue
        parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    logger.info("  ".join(parts))
