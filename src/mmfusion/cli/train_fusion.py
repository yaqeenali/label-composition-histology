"""CLI: train multimodal fusion models across k-folds.

Replaces ``run_training.py``. Argument names, defaults and semantics are
unchanged so existing shell scripts keep working, with two exceptions that are
strict supersets of the old behaviour:

``--seed``
    The old entry point hard-coded ``set_seed(242)`` while every doc string and
    ``make_splits`` used 42. The default here stays 242 so existing commands
    reproduce, but the value is now visible and recorded.

``--es_higher_is_better / --no_es_higher_is_better``
    The old flag was ``action="store_true", default=True``, so it could never be
    turned off — early stopping on a loss-type metric selected the *worst*
    epoch. The default is still True; the negative form is new.

Both are logged in ``REFACTOR_NUMERICS.md`` (N-02, D-03).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mmfusion.config import (
    ModelConfig,
    TaskConfig,
    TrainConfig,
    TransferConfig,
    WandbConfig,
)
from mmfusion.seeding import set_seed
from mmfusion.training.fusion_cv import parse_modalities, run_all_folds

logger = logging.getLogger(__name__)

LEGACY_SEED = 242
"""Seed the original ``run_training.py`` used at its entry point."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train multimodal fusion models across k-folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--splits_root", required=True,
                   help="Directory containing fold_0/, fold_1/, ... sub-directories.")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--split_names", default="train,val,test",
                   help="Comma-separated split CSV names (without .csv).")
    p.add_argument("--tabular_csv", default=None,
                   help="Shared CSV for modalities with source_type='tabular'.")

    # Modalities
    p.add_argument("--modalities", required=True,
                   help="Comma-separated specs: name:type:source_type[:path]. "
                        "type = bag|vector, source_type = pt_dir|tabular.")

    # Task
    p.add_argument("--task_type", default="regression",
                   choices=["survival", "regression", "classification"])
    p.add_argument("--case_col", default="CLID")
    p.add_argument("--time_col", default="survival_months")
    p.add_argument("--censor_col", default="censorship")
    p.add_argument("--target_cols", default="target",
                   help="Comma-separated target columns (regression/classification).")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--loss_fn", default="auto",
                   choices=["auto", "cox_ph", "mse", "mae", "huber", "bce", "ce"])

    # Model
    p.add_argument("--model_type", default="crossattentionfusion",
                   choices=["abmil", "modalityfusion", "crossattentionfusion"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_queries", type=int, default=4)
    p.add_argument("--post_sab_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modality_dropout", type=float, default=0.1)

    # Training
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=LEGACY_SEED,
                   help="Global seed set once before the first fold.")
    p.add_argument("--ema_eval", action="store_true", default=False)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--early_stopping", action="store_true", default=False)
    p.add_argument("--es_patience", type=int, default=15)
    p.add_argument("--es_metric", default="auto")
    p.add_argument("--es_split", default="val")
    p.add_argument("--es_higher_is_better", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Direction of the early-stopping metric. Use "
                        "--no-es-higher-is-better for loss-type metrics.")

    # Transfer learning
    p.add_argument("--transfer_checkpoint", default=None,
                   help="Checkpoint used to initialise the encoder.")
    p.add_argument("--freeze_encoder", action="store_true", default=False,
                   help="Freeze everything except the prediction head.")
    p.add_argument("--freeze_prefixes", default="",
                   help="Comma-separated parameter-name prefixes to freeze.")

    # WandB
    p.add_argument("--wandb_enabled", action="store_true", default=False)
    p.add_argument("--wandb_project", default="mmfusion")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", default="")
    p.add_argument("--wandb_log_gradients", action="store_true", default=False)
    p.add_argument("--log_test_every_epoch", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Evaluate and log the test split each epoch. Leaving this "
                        "on reproduces the existing W&B history; turning it off "
                        "keeps the test split unseen during development and does "
                        "not change training (evaluation draws no randomness).")

    return p


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    mod_cfgs = parse_modalities(args.modalities, tabular_csv=args.tabular_csv)

    task_cfg = TaskConfig(
        task_type=args.task_type,
        case_col=args.case_col,
        time_col=args.time_col,
        censor_col=args.censor_col,
        target_cols=[c.strip() for c in args.target_cols.split(",") if c.strip()],
        num_classes=args.num_classes,
        loss_fn=args.loss_fn,
    )
    model_cfg = ModelConfig(
        model_type=args.model_type, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, num_queries=args.num_queries,
        post_sab_layers=args.post_sab_layers, dropout=args.dropout,
        modality_dropout=args.modality_dropout,
    )
    train_cfg = TrainConfig(
        max_epochs=args.max_epochs, lr=args.lr, wd=args.wd,
        grad_clip=args.grad_clip, batch_size=args.batch_size,
        num_workers=args.num_workers, ema_eval=args.ema_eval,
        ema_decay=args.ema_decay, early_stopping=args.early_stopping,
        es_patience=args.es_patience, es_metric=args.es_metric,
        es_split=args.es_split, es_higher_is_better=args.es_higher_is_better,
    )
    transfer_cfg = TransferConfig(
        enabled=bool(args.transfer_checkpoint),
        checkpoint_path=args.transfer_checkpoint,
        freeze_prefixes=[p.strip() for p in args.freeze_prefixes.split(",") if p.strip()],
    )
    wandb_cfg = WandbConfig(
        enabled=args.wandb_enabled, project=args.wandb_project,
        entity=args.wandb_entity, run_name=args.wandb_run_name,
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
        log_gradients=args.wandb_log_gradients,
        log_test_every_epoch=args.log_test_every_epoch,
    )

    df = run_all_folds(
        splits_root=Path(args.splits_root),
        results_dir=Path(args.results_dir),
        split_names=[s.strip() for s in args.split_names.split(",") if s.strip()],
        mod_cfgs=mod_cfgs,
        task_cfg=task_cfg, model_cfg=model_cfg, train_cfg=train_cfg,
        transfer_cfg=transfer_cfg, wandb_cfg=wandb_cfg,
        tabular_csv=args.tabular_csv,
        freeze_encoder_flag=args.freeze_encoder,
    )

    from mmfusion.training.fusion_cv import summarise_folds

    print("\n=== Cross-fold summary ===")
    print(summarise_folds(df).to_string(index=False))


if __name__ == "__main__":
    main()
