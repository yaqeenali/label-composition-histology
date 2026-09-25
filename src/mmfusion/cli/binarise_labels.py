"""CLI: derive the binary risk label used for the classification metrics.

Writes to a separate directory by default so ``data/`` stays read-only; pass
``--in_place`` to reproduce the notebooks' behaviour of writing ``*_bin.csv``
back into the split folders.
"""
from __future__ import annotations

import argparse

from mmfusion.config import load_config
from mmfusion.data.labels import LABEL_MAPS, write_binary_splits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Derive binary risk labels from an assay's risk-group column.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="Assay config (YAML/JSON). Supplies every option below.")
    p.add_argument("--splits_dir")
    p.add_argument("--out_dir", default=None,
                   help="Where to write. Defaults to <splits_dir>_binary.")
    p.add_argument("--in_place", action="store_true",
                   help="Write *_bin.csv into the split directories (legacy behaviour).")
    p.add_argument("--group_col", help="Source risk-group column.")
    p.add_argument("--binary_col", help="Name of the derived column.")
    p.add_argument("--preset", choices=sorted(LABEL_MAPS), help="Use a built-in label map.")
    p.add_argument("--suffix", default="_bin")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
        splits_dir = args.splits_dir or cfg.splits_dir
        group_col, binary_col, label_map = cfg.group_col, cfg.binary_target_col, cfg.label_map
    else:
        if not (args.splits_dir and args.group_col and args.binary_col):
            raise SystemExit("Provide --config, or all of --splits_dir --group_col --binary_col.")
        splits_dir = args.splits_dir
        group_col, binary_col = args.group_col, args.binary_col
        if not args.preset:
            raise SystemExit("--preset is required when not using --config.")
        label_map = LABEL_MAPS[args.preset]

    if args.in_place:
        out_dir = None
    elif args.out_dir:
        out_dir = args.out_dir
    elif args.config and cfg.binary_splits_dir:
        out_dir = cfg.binary_splits_dir
    else:
        out_dir = f"{splits_dir.rstrip('/')}_binary"

    written = write_binary_splits(
        splits_dir, group_col, binary_col, label_map,
        out_dir=out_dir, suffix=args.suffix,
    )

    # Training reads `binary_splits_dir or splits_dir`. If those disagree, the run
    # would silently consume whatever *_bin.csv already sits beside the splits.
    if out_dir is not None and args.config and str(written) != (cfg.binary_splits_dir or cfg.splits_dir):
        raise SystemExit(
            f"\nWrote derived labels to {written}, but {args.config} would train from "
            f"{cfg.binary_splits_dir or cfg.splits_dir}.\n"
            f"Set `binary_splits_dir: {written}` in that config, or pass --in_place "
            f"to write beside the splits as the notebooks did."
        )


if __name__ == "__main__":
    main()
