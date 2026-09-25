"""Derivation of binary labels from assay risk-group columns.

Replaces the copy-pasted "cell 1" of each pathology notebook, which read the
split CSVs, mapped a group column to 0/1, and wrote ``*_bin.csv`` back **into
the split directory**. The mapping is unchanged; the write target is now a
caller-chosen directory so ``data/`` can stay read-only.

.. warning::
   Every mapping below folds the *intermediate* risk group in with *high*. That
   is the original choice and it is preserved, but it is a modelling decision
   rather than a convention — see the audit, finding S4.

   The derived label is a deterministic function of the same quantity the model
   regresses, so the resulting "classification" metrics are a second view of the
   regression, not independent evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

__all__ = ["LABEL_MAPS", "add_binary_label", "write_binary_splits"]

LABEL_MAPS: Dict[str, Dict[str, int]] = {
    "ghi": {"High": 1, "Intermediate": 1, "Low": 0},
    "mammaprint": {"NKI70_Good": 1, "NKI70_Bad": 0},
    "ror_p": {"high": 1, "med": 1, "low": 0},
    "ror_s": {"high": 1, "med": 1, "low": 0},
}


def add_binary_label(
    df: pd.DataFrame,
    group_col: str,
    binary_col: str,
    label_map: Dict[str, int],
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Return a copy of *df* with ``binary_col`` derived from ``group_col``.

    The original printed a warning for unmapped values and then called
    ``.astype(int)``, which raises on the resulting NaN. That surface is kept by
    default; ``strict=True`` raises a clear error instead of an opaque cast
    failure.
    """
    out = df.copy()
    out[binary_col] = out[group_col].map(label_map)

    if out[binary_col].isna().any():
        unmapped = sorted(out.loc[out[binary_col].isna(), group_col].dropna().unique(), key=str)
        message = (
            f"Unmapped values in column '{group_col}': {unmapped}. "
            f"Known values: {sorted(label_map)}"
        )
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")

    out[binary_col] = out[binary_col].astype(int)
    return out


def write_binary_splits(
    splits_dir: str,
    group_col: str,
    binary_col: str,
    label_map: Dict[str, int],
    *,
    out_dir: Optional[str] = None,
    splits: Iterable[str] = ("train", "val", "test"),
    suffix: str = "_bin",
    verbose: bool = True,
) -> Path:
    """Write ``{split}{suffix}.csv`` for every fold.

    ``out_dir`` defaults to ``splits_dir`` (the original in-place behaviour).
    Pass a separate directory to keep the source splits immutable; the CLI
    does so by default.
    """
    src = Path(splits_dir)
    dst = Path(out_dir) if out_dir is not None else src
    for fold_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        target_dir = dst / fold_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            src_file = fold_dir / f"{split}.csv"
            if not src_file.exists():
                # The originals had no guard and raised. Skipping silently would
                # produce an incomplete set, so say so.
                print(f"Warning: {src_file} not found; no {split}{suffix}.csv written.")
                continue
            df = add_binary_label(pd.read_csv(src_file), group_col, binary_col, label_map)
            out_file = target_dir / f"{split}{suffix}.csv"
            df.to_csv(out_file, index=False)
            if verbose:
                print(f"Saved: {out_file}")
    return dst
