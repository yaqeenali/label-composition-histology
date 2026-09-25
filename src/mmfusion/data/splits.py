"""Stratified k-fold split generation.

The logic of ``make_splits.py`` moved here unchanged; only the argparse shell
was separated out (it now lives in :mod:`mmfusion.cli.make_splits`). Splitter
construction, random states and the rare-stratum merge are identical, so
regenerating splits with the same arguments reproduces the same fold membership
— asserted by ``tests/test_parity_splits.py``.

Stratification by task:

survival        event indicator (0/1)
regression      equal-frequency quantile bins of the first target column
classification  the label column itself
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

__all__ = [
    "make_strata",
    "merge_rare_strata",
    "strata_distribution",
    "generate_splits",
]


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def _strata_survival(df: pd.DataFrame, censor_col: str) -> np.ndarray:
    return (1 - df[censor_col].astype(int)).values


def _strata_regression(df: pd.DataFrame, target_col: str, n_bins: int = 5) -> np.ndarray:
    vals = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    bins = pd.qcut(vals, q=n_bins, labels=False, duplicates="drop")
    return bins.values.astype(int)


def _strata_classification(df: pd.DataFrame, label_col: str) -> np.ndarray:
    return df[label_col].fillna("__MISSING__").astype(str).values


def make_strata(
    df: pd.DataFrame,
    task_type: str,
    *,
    censor_col: str = "censorship",
    target_col: str = "target",
    label_col: str = "label",
    regression_bins: int = 5,
) -> np.ndarray:
    """Stratification labels for the given task type."""
    if task_type == "survival":
        return _strata_survival(df, censor_col)
    if task_type == "regression":
        return _strata_regression(df, target_col, n_bins=regression_bins)
    if task_type == "classification":
        return _strata_classification(df, label_col)
    raise ValueError(f"Unknown task_type: {task_type}")


def merge_rare_strata(strata: np.ndarray, min_count: int) -> np.ndarray:
    """Collapse strata with fewer than *min_count* members into ``__RARE__``."""
    s = pd.Series(strata).fillna("__MISSING__").astype(str)
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    if len(rare) > 0:
        s = s.where(~s.isin(rare), "__RARE__")
    return s.values


def strata_distribution(strata: np.ndarray) -> Dict[str, float]:
    """Relative frequency of each stratum, for the per-fold log lines."""
    vc = pd.Series(strata).value_counts(normalize=True).sort_values(ascending=False)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_splits(
    metadata_csv: str,
    out_dir: str,
    *,
    task_type: str = "regression",
    case_col: str = "CLID",
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
    min_samples_per_stratum: int = 5,
    censor_col: str = "censorship",
    target_col: str = "target",
    regression_bins: int = 5,
    label_col: str = "label",
    verbose: bool = True,
) -> Path:
    """Write ``fold_k/{train,val,test}.csv`` under *out_dir*; return that path.

    RNG CONTRACT — unchanged from the original:

    * outer split uses ``random_state=seed``
    * the validation carve-out uses ``random_state=seed + k``
    * the non-stratifiable fallback uses ``np.random.default_rng(seed + k)``
    """
    df = pd.read_csv(metadata_csv)

    if case_col not in df.columns and "CLID" in df.columns:
        if verbose:
            print("[make_splits] --case_col not found, using detected column 'CLID'.")
        case_col = "CLID"

    df = df.drop_duplicates(subset=[case_col]).reset_index(drop=True)

    strata = make_strata(
        df, task_type,
        censor_col=censor_col, target_col=target_col,
        label_col=label_col, regression_bins=regression_bins,
    )
    strata = merge_rare_strata(strata, min_count=min_samples_per_stratum)

    strata_series = pd.Series(strata)
    min_class_count = int(strata_series.value_counts().min())
    n_unique = int(strata_series.nunique())

    use_stratified = (n_unique > 1) and (min_class_count >= 2)
    effective_folds = n_folds
    if use_stratified and min_class_count < n_folds:
        effective_folds = min_class_count
        if verbose:
            print(
                f"[make_splits] Reducing n_folds from {n_folds} to {effective_folds} "
                "because at least one stratum is too small."
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if use_stratified:
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        split_iter: Iterable[Tuple[np.ndarray, np.ndarray]] = splitter.split(df, strata)
    else:
        if verbose:
            print("[make_splits] Stratification unavailable (single/too-small class); using plain KFold.")
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(df)

    for k, (tr_idx, te_idx) in enumerate(split_iter):
        fold_dir = out / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        train_strata = strata[tr_idx]
        test_strata = strata[te_idx]
        val_df = None
        val_strata = None

        if val_fraction > 0.0:
            n_val = max(1, int(len(train_df) * val_fraction))
            all_idx = np.arange(len(train_df))

            can_stratify_val = (
                use_stratified
                and pd.Series(train_strata).nunique() > 1
                and n_val >= pd.Series(train_strata).nunique()
                and pd.Series(train_strata).value_counts().min() >= 2
            )

            if can_stratify_val:
                train_idx, val_idx = train_test_split(
                    all_idx,
                    test_size=n_val,
                    random_state=seed + k,
                    shuffle=True,
                    stratify=train_strata,
                )
            else:
                rng = np.random.default_rng(seed + k)
                val_idx = rng.choice(len(train_df), n_val, replace=False)
                train_idx = np.setdiff1d(all_idx, val_idx)

            val_df = train_df.iloc[val_idx].reset_index(drop=True)
            val_strata = train_strata[val_idx]
            train_df = train_df.iloc[train_idx].reset_index(drop=True)
            train_strata = train_strata[train_idx]
            val_df.to_csv(fold_dir / "val.csv", index=False)

        train_df.to_csv(fold_dir / "train.csv", index=False)
        test_df.to_csv(fold_dir / "test.csv", index=False)

        if verbose:
            n_val_written = len(val_df) if val_df is not None else 0
            print(
                f"fold_{k}: train={len(train_df)}, val={n_val_written}, test={len(test_df)}, "
                f"train_dist={strata_distribution(train_strata)}, "
                f"test_dist={strata_distribution(test_strata)}"
            )
            if val_strata is not None:
                print(f"fold_{k}: val_dist={strata_distribution(val_strata)}")

    return out
