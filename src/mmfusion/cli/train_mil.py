"""CLI: cross-validated gated-attention MIL training for one assay.

Everything that used to be edited inside a notebook cell now comes from an
assay config:

    mmfusion-train-mil --config configs/assays/ghi.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mmfusion.config import load_config
from mmfusion.paths import PathConfig
from mmfusion.seeding import DEFAULT_SEED, set_seed
from mmfusion.training.mil import run_cv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run cross-validated MIL training for one assay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Assay config (YAML or JSON).")
    p.add_argument("--splits_dir", default=None, help="Override the config's splits_dir.")
    p.add_argument("--results_dir", default=None, help="Override the config's results_dir.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Seed set once before the first fold, as in the notebooks.")
    p.add_argument("--rewrite_root", action="append", default=[], metavar="RECORDED=LOCAL",
                   help="Rebase paths stored in the split CSVs, e.g. "
                        r"'C:\Users\me\data=/data'. Repeatable.")
    p.add_argument("--no_artifacts", action="store_true",
                   help="Skip embeddings, prediction CSVs and plots; metrics are unaffected.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    if args.splits_dir:
        cfg.splits_dir = args.splits_dir
    if args.results_dir:
        cfg.results_dir = args.results_dir

    rewrites = {}
    for item in args.rewrite_root:
        recorded, _, local = item.partition("=")
        if not local:
            raise SystemExit(f"--rewrite_root expects RECORDED=LOCAL, got {item!r}")
        rewrites[recorded] = local
    path_cfg = PathConfig(rewrites=rewrites)

    set_seed(args.seed)
    result = run_cv(
        cfg,
        write_artifacts=not args.no_artifacts,
        path_resolver=path_cfg.resolve,
        verbose=not args.quiet,
    )

    print("\n===== PER-FOLD SUMMARY (mean +/- SD across folds) =====")
    print(result.per_fold_summary().to_string())
    if cfg.results_dir and not args.no_artifacts:
        print(f"\nWritten to {Path(cfg.results_dir).resolve()}")


if __name__ == "__main__":
    main()
