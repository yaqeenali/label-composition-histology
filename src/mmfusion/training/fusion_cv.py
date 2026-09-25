"""Cross-validation driver for the fusion arm.

The orchestration that used to live in ``run_training.run_fold`` / ``main``:
build datasets per fold, infer modality dims, build the model, optionally
transfer weights, train, and write summary/history/config/checkpoint.

Split out from the argparse layer so it can be called from a notebook or a test
without a command line.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mmfusion.config import (
    ModalityConfig,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    TransferConfig,
    WandbConfig,
)
from mmfusion.data.multimodal import (
    MultiModalDataset,
    infer_dims_from_dataset,
    make_collate_fn,
)
from mmfusion.models import build_model
from mmfusion.training.fusion import fit
from mmfusion.transfer import (
    freeze_by_prefix,
    freeze_encoder,
    load_pretrained_encoder,
    print_trainable_params,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

__all__ = ["parse_modalities", "run_fold", "run_all_folds", "summarise_folds"]


# ---------------------------------------------------------------------------
# Modality spec parsing
# ---------------------------------------------------------------------------

def parse_modalities(
    spec_str: str,
    tabular_csv: Optional[str] = None,
) -> List[ModalityConfig]:
    """Parse ``name:type:source_type[:path]`` specs, comma separated.

    ``type``        bag | vector
    ``source_type`` pt_dir | tabular

    The path is split with ``maxsplit=3`` so Windows drive letters survive.
    A ``tabular`` spec is always forced to ``modality_type='vector'``, as before.
    """
    configs: List[ModalityConfig] = []
    if not spec_str.strip():
        return configs
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":", 3)
        if len(pieces) < 3:
            raise ValueError(
                f"Invalid modality spec '{part}'. "
                "Expected format: name:type:source_type[:path]"
            )
        name, mod_type, src_type = (p.strip() for p in pieces[:3])
        path = pieces[3].strip() if len(pieces) >= 4 else None

        if src_type == "pt_dir":
            if path is None:
                raise ValueError(f"Modality '{name}': pt_dir source requires a path.")
            configs.append(ModalityConfig(
                name=name, modality_type=mod_type,
                source_type="pt_dir", source_dir=path,
            ))
        elif src_type == "tabular":
            configs.append(ModalityConfig(
                name=name, modality_type="vector",
                source_type="tabular", source_csv=path or tabular_csv,
            ))
        else:
            raise ValueError(f"Unknown source_type '{src_type}' in modality '{name}'.")
    return configs


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def run_fold(
    fold_dir: Path,
    split_names: List[str],
    mod_cfgs: List[ModalityConfig],
    task_cfg: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    transfer_cfg: TransferConfig,
    wandb_cfg: WandbConfig,
    results_dir: Path,
    tabular_csv: Optional[str] = None,
    freeze_encoder_flag: bool = False,
) -> dict:
    """Train one fold and write its artifacts; return the final metrics."""
    # Dims are re-inferred per fold because per-fold embedding directories can
    # differ in width. Resetting them here is what makes that work.
    for mc in mod_cfgs:
        mc.dim = None

    datasets: Dict[str, MultiModalDataset] = {}
    for sp in split_names:
        csv_path = fold_dir / f"{sp}.csv"
        if not csv_path.exists():
            continue
        datasets[sp] = MultiModalDataset(
            split_csv=str(csv_path),
            modality_configs=mod_cfgs,
            task_config=task_cfg,
            tabular_csv=tabular_csv,
            fold_name=fold_dir.name,
        )

    if "train" not in datasets:
        raise RuntimeError(f"No train.csv found under {fold_dir}")

    modality_dims = infer_dims_from_dataset(datasets["train"])
    modality_types = datasets["train"].modality_types
    logger.info(f"Modality dims: {modality_dims}")

    model = build_model(model_cfg, task_cfg, modality_dims, modality_types)

    if transfer_cfg.enabled and transfer_cfg.checkpoint_path:
        load_pretrained_encoder(model, transfer_cfg.checkpoint_path)
        if freeze_encoder_flag:
            freeze_encoder(model)
        elif transfer_cfg.freeze_prefixes:
            freeze_by_prefix(model, transfer_cfg.freeze_prefixes)
    elif freeze_encoder_flag:
        freeze_encoder(model)

    print_trainable_params(model)

    collate = make_collate_fn(mod_cfgs, task_cfg)
    loaders = {
        sp: DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(sp == "train"),
            num_workers=train_cfg.num_workers,
            collate_fn=collate,
        )
        for sp, ds in datasets.items()
    }

    wb = WandbConfig(
        enabled=wandb_cfg.enabled,
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        run_name=f"{wandb_cfg.run_name or 'run'}_{fold_dir.name}" if wandb_cfg.enabled else None,
        tags=wandb_cfg.tags + [fold_dir.name],
        log_system=wandb_cfg.log_system,
        log_gradients=wandb_cfg.log_gradients,
        log_test_every_epoch=wandb_cfg.log_test_every_epoch,
    )

    run_config = {
        "fold": fold_dir.name,
        "modalities": [mc.name for mc in mod_cfgs],
        "task_type": task_cfg.task_type,
        "model_type": model_cfg.model_type,
    }

    final, history = fit(
        model=model, loaders=loaders,
        train_cfg=train_cfg, task_cfg=task_cfg,
        wandb_cfg=wb, run_config=run_config,
    )

    fold_out = results_dir / fold_dir.name
    fold_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([final]).to_csv(fold_out / "summary.csv", index=False)
    pd.DataFrame(history).to_csv(fold_out / "history.csv", index=False)

    cfg_record = {
        "modalities": [
            {k: getattr(mc, k) for k in
             ("name", "modality_type", "source_type", "source_dir", "source_csv")}
            for mc in mod_cfgs
        ],
        "task": task_cfg.__dict__,
        "model": model_cfg.__dict__,
        "train": train_cfg.__dict__,
    }
    with open(fold_out / "config.json", "w") as f:
        json.dump(cfg_record, f, indent=2, default=str)

    save_checkpoint(
        model, str(fold_out / "best_model.pt"),
        extra={"fold": fold_dir.name, "final_metrics": final},
    )
    return final


# ---------------------------------------------------------------------------
# All folds
# ---------------------------------------------------------------------------

def run_all_folds(
    splits_root: Path,
    results_dir: Path,
    split_names: List[str],
    mod_cfgs: List[ModalityConfig],
    task_cfg: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    transfer_cfg: TransferConfig,
    wandb_cfg: WandbConfig,
    tabular_csv: Optional[str] = None,
    freeze_encoder_flag: bool = False,
) -> pd.DataFrame:
    """Run every ``fold_*`` directory under *splits_root*."""
    fold_dirs = sorted(p for p in splits_root.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories found in {splits_root}")

    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fd in fold_dirs:
        logger.info(f"\n{'='*60}\n  {fd.name}\n{'='*60}")
        row = run_fold(
            fold_dir=fd, split_names=split_names, mod_cfgs=mod_cfgs,
            task_cfg=task_cfg, model_cfg=model_cfg, train_cfg=train_cfg,
            transfer_cfg=transfer_cfg, wandb_cfg=wandb_cfg,
            results_dir=results_dir, tabular_csv=tabular_csv,
            freeze_encoder_flag=freeze_encoder_flag,
        )
        row["fold"] = fd.name
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "all_folds_summary.csv", index=False)
    summarise_folds(df).to_csv(results_dir / "all_folds_mean_std.csv", index=False)
    return df


def summarise_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD of every numeric column except ``fold``."""
    stats = {"metric": [], "mean": [], "std": []}
    for c in (col for col in df.columns if col != "fold"):
        vals = pd.to_numeric(df[c], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        stats["metric"].append(c)
        stats["mean"].append(float(np.mean(vals)))
        stats["std"].append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
    return pd.DataFrame(stats)
