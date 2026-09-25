"""Per-fold embedding generation for tabular feature sources.

The clinical and radiomics notebooks each contained the same loop: for every
fold, fit a :class:`~mmfusion.features.clinical.ClinicalPreprocessor` on the
training cases only, then transform every split and write ``{case_id}.pt`` plus
a ``batch.pt`` bundle and a manifest. That loop is here once, parameterised.

The fold-local fit is the important property and it is preserved exactly: the
preprocessor never sees validation or test rows.

.. note::
   Where a case has several source rows (a patient with more than one MRI
   lesion), ``.iloc[0]`` selects the first, matching the notebooks. That is an
   arbitrary choice and it disagrees with the radiomics *feature-selection*
   notebook, which averaged rows per lesion name. The disagreement is inherited;
   ``row_selector`` makes it explicit and overridable rather than implicit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
import torch

from mmfusion.features.clinical import ClinicalPreprocessor

__all__ = ["normalize_clid", "build_fold_embeddings", "first_row"]


def normalize_clid(s: pd.Series) -> pd.Series:
    """Reduce a TCGA barcode to its first three tokens (``TCGA-XX-XXXX``)."""
    return s.astype(str).apply(lambda x: "-".join(x.split("-")[:3]))


def first_row(rows: pd.DataFrame) -> pd.Series:
    """Default row selector: take the first matching source row."""
    return rows.iloc[0]


def build_fold_embeddings(
    splits_dir: str,
    source_df: pd.DataFrame,
    out_dir: str,
    numerical_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    *,
    source_case_col: str = "CLID",
    split_case_col: str = "CLID",
    splits: Iterable[str] = ("train", "val", "test"),
    row_selector: Callable[[pd.DataFrame], pd.Series] = first_row,
    write_batch: bool = True,
    verbose: bool = True,
) -> Path:
    """Fit per fold on train, transform every split, write ``.pt`` files.

    Layout produced under *out_dir*::

        fold_k/
          preprocessor.joblib
          feature_names.json          # expanded (one-hot) names
          selected_features.txt
          train/  {case}.pt  batch.pt  manifest.csv
          val/    ...
          test/   ...

    Returns the output root.
    """
    categorical_cols = list(categorical_cols or [])
    src = source_df.copy()
    src[source_case_col] = normalize_clid(src[source_case_col])

    splits_root = Path(splits_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for fold_path in sorted(p for p in splits_root.iterdir() if p.is_dir()):
        fold = fold_path.name
        if verbose:
            print(f"\nProcessing {fold}")
        fold_out = out_root / fold
        fold_out.mkdir(parents=True, exist_ok=True)

        train_csv = fold_path / "train.csv"
        if not train_csv.exists():
            print(f"  [WARN] Missing train.csv in {fold_path}, skipping fold.")
            continue

        train_df = pd.read_csv(train_csv)
        if split_case_col not in train_df.columns:
            raise ValueError(f"[{fold}] train.csv missing '{split_case_col}' column")
        train_ids = set(normalize_clid(train_df[split_case_col]))

        # ---- fit on the training cases only ---------------------------------
        train_src = src[src[source_case_col].isin(train_ids)]
        if train_src.empty:
            print(f"  [WARN] No overlapping cases between train.csv and source for {fold}.")
            continue

        proc = ClinicalPreprocessor(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            case_col=source_case_col,
        )
        proc.fit(train_src)
        proc.save(str(fold_out / "preprocessor.joblib"))
        if verbose:
            print(f"  Saved preprocessor -> {fold_out / 'preprocessor.joblib'}")
            print(f"  output_dim = {proc.output_dim}")

        feature_names = proc.get_feature_names()
        (fold_out / "feature_names.json").write_text(
            json.dumps(feature_names, indent=2), encoding="utf-8"
        )
        (fold_out / "selected_features.txt").write_text(
            "Numerical:\n" + "\n".join(numerical_cols)
            + "\n\nCategorical:\n" + "\n".join(categorical_cols),
            encoding="utf-8",
        )

        # ---- transform every split -----------------------------------------
        for split in splits:
            split_csv = fold_path / f"{split}.csv"
            if not split_csv.exists():
                if verbose:
                    print(f"  [{split}] split file not found, skipping.")
                continue

            split_df = pd.read_csv(split_csv)
            if split_case_col not in split_df.columns:
                print(f"  [WARN] {split}.csv missing '{split_case_col}'; skipping.")
                continue

            split_out = fold_out / split
            split_out.mkdir(parents=True, exist_ok=True)

            tensors, ids = [], []
            for case_id in normalize_clid(split_df[split_case_col]).unique():
                rows = src[src[source_case_col] == case_id]
                if rows.empty:
                    continue
                tensor = proc.transform_to_tensor(row_selector(rows))
                torch.save(tensor, split_out / f"{case_id}.pt")
                t = tensor.detach().cpu()
                tensors.append(t.view(-1) if t.ndim > 1 else t)
                ids.append(str(case_id))

            if verbose:
                print(f"  {split}: saved {len(ids)} per-case embeddings")

            if write_batch and ids:
                X_batch = torch.stack(tensors, dim=0)
                torch.save(
                    {"X": X_batch, "ids": ids, "feature_names": feature_names},
                    split_out / "batch.pt",
                )
                pd.DataFrame({split_case_col: ids}).to_csv(
                    split_out / "manifest.csv", index=False
                )
                if verbose:
                    print(f"  {split}: saved batch.pt shape={tuple(X_batch.shape)}")

    return out_root
