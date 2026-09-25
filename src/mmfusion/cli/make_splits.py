"""CLI: create stratified k-fold train/val/test CSV splits."""
from __future__ import annotations

import argparse

from mmfusion.data.splits import generate_splits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create stratified k-fold train/val/test CSV splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata_csv", required=True, help="Input CSV with all cases.")
    p.add_argument("--out_dir", required=True, help="Root directory for fold outputs.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Fraction of training data held out for validation. 0 to skip.")
    p.add_argument("--case_col", default="CLID")
    p.add_argument("--min_samples_per_stratum", type=int, default=5,
                   help="Strata smaller than this are merged into '__RARE__'.")
    p.add_argument("--task_type", default="regression",
                   choices=["survival", "regression", "classification"])
    p.add_argument("--censor_col", default="censorship",
                   help="1=censored / 0=event (survival only).")
    p.add_argument("--target_col", default="target",
                   help="Continuous column binned for stratification (regression only).")
    p.add_argument("--regression_bins", type=int, default=5)
    p.add_argument("--label_col", default="label", help="Class label (classification only).")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    generate_splits(
        metadata_csv=args.metadata_csv,
        out_dir=args.out_dir,
        task_type=args.task_type,
        case_col=args.case_col,
        n_folds=args.n_folds,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_samples_per_stratum=args.min_samples_per_stratum,
        censor_col=args.censor_col,
        target_col=args.target_col,
        regression_bins=args.regression_bins,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()
