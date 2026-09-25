"""Clinical / tabular feature preprocessing and embedding.

Carried over from ``clinical_encoder.py``. The sklearn pipeline is unchanged —
median imputation and standardisation for numeric columns, most-frequent
imputation and one-hot encoding (``handle_unknown='ignore'``) for categorical —
and it is still fitted on the training split only.

One addition: :meth:`ClinicalPreprocessor.get_feature_names`. The clinical
notebook already called this method inside a bare ``try/except``; because it did
not exist, every fold silently fell back to writing the *raw* column names into
``feature_names.json`` while the saved tensors were the wider one-hot expansion.
The method now exists, so the manifest describes the tensors. This changes the
contents of a metadata file and nothing else — no tensor, no metric. See
``REFACTOR_NUMERICS.md`` entry D-02.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["ClinicalPreprocessor", "ClinicalEmbedder", "prepare_clinical_embeddings"]


# ===========================================================================
# Stage 1 — sklearn preprocessor
# ===========================================================================

class ClinicalPreprocessor:
    """Fit/transform pipeline for mixed numerical and categorical columns.

    Columns absent from the fitted frame are dropped with a warning rather than
    raising, which is how the original behaved and why a typo in a column name
    silently shrinks the feature vector. :meth:`fitted_columns` is provided so
    callers can check what was actually used.
    """

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        case_col: str = "case_id",
        default_num: float = 0.0,
    ):
        self.numerical_cols = list(numerical_cols)
        self.categorical_cols = list(categorical_cols)
        self.case_col = case_col
        self.default_num = default_num

        self._preprocessor = None
        self._output_dim: Optional[int] = None
        self._is_fitted: bool = False
        self._fitted_numerical: List[str] = []
        self._fitted_categorical: List[str] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        """Fit on *df*; only the declared columns are used."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        transformers = []

        if self.numerical_cols:
            present = [c for c in self.numerical_cols if c in df.columns]
            if present:
                num_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
                transformers.append(("num", num_pipe, present))
                self._fitted_numerical = present
                if len(present) < len(self.numerical_cols):
                    missing = set(self.numerical_cols) - set(present)
                    logger.warning(f"Numerical columns not found in DataFrame: {missing}")

        if self.categorical_cols:
            present = [c for c in self.categorical_cols if c in df.columns]
            if present:
                cat_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float32,
                    )),
                ])
                transformers.append(("cat", cat_pipe, present))
                self._fitted_categorical = present
                if len(present) < len(self.categorical_cols):
                    missing = set(self.categorical_cols) - set(present)
                    logger.warning(f"Categorical columns not found in DataFrame: {missing}")

        if not transformers:
            raise ValueError(
                "No declared columns found in the provided DataFrame. "
                "Check numerical_cols and categorical_cols."
            )

        self._preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        self._preprocessor.fit(df)
        dummy = self._preprocessor.transform(df.iloc[:1])
        self._output_dim = int(dummy.shape[1])
        self._is_fitted = True
        logger.info(f"ClinicalPreprocessor fitted. output_dim={self._output_dim}")
        return self

    def fit_from_csv(
        self,
        csv_path: str,
        case_ids: Optional[List[str]] = None,
    ) -> "ClinicalPreprocessor":
        """Load a CSV, optionally subset to *case_ids*, then fit."""
        df = pd.read_csv(csv_path)
        if case_ids is not None:
            df = df[df[self.case_col].astype(str).isin([str(c) for c in case_ids])]
            if len(df) == 0:
                raise ValueError(f"None of the provided case_ids were found in {csv_path}.")
        return self.fit(df)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform_row(self, row: Union[pd.Series, Dict]) -> np.ndarray:
        """One row -> float32 array [D]."""
        self._require_fitted()
        if isinstance(row, dict):
            row = pd.Series(row)
        return self._preprocessor.transform(pd.DataFrame([row])).astype(np.float32).squeeze(0)

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """A frame -> float32 array [N, D]."""
        self._require_fitted()
        return self._preprocessor.transform(df).astype(np.float32)

    def transform_to_tensor(self, row: Union[pd.Series, Dict]) -> torch.Tensor:
        """One row -> FloatTensor [D]."""
        return torch.tensor(self.transform_row(row), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Expanded output feature names, one per column of the transform.

        NEW. Falls back to positional names if the fitted transformers cannot
        report their own, so it never raises on a fitted preprocessor.
        """
        self._require_fitted()
        try:
            return [str(n) for n in self._preprocessor.get_feature_names_out()]
        except Exception:
            return [f"f{i}" for i in range(self.output_dim)]

    def fitted_columns(self) -> Dict[str, List[str]]:
        """The declared columns that were actually present at fit time."""
        return {
            "numerical": list(self._fitted_numerical),
            "categorical": list(self._fitted_categorical),
        }

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted pipeline with joblib."""
        import joblib

        os.makedirs(Path(path).parent, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self._preprocessor,
                "output_dim": self._output_dim,
                "numerical_cols": self.numerical_cols,
                "categorical_cols": self.categorical_cols,
                "case_col": self.case_col,
                "is_fitted": self._is_fitted,
                "fitted_numerical": self._fitted_numerical,
                "fitted_categorical": self._fitted_categorical,
            },
            path,
        )
        logger.info(f"ClinicalPreprocessor saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "ClinicalPreprocessor":
        """Restore a preprocessor saved by :meth:`save`."""
        import joblib

        state = joblib.load(path)
        obj = cls(
            numerical_cols=state["numerical_cols"],
            categorical_cols=state["categorical_cols"],
            case_col=state["case_col"],
        )
        obj._preprocessor = state["preprocessor"]
        obj._output_dim = state["output_dim"]
        obj._is_fitted = state["is_fitted"]
        obj._fitted_numerical = state.get("fitted_numerical", [])
        obj._fitted_categorical = state.get("fitted_categorical", [])
        logger.info(f"ClinicalPreprocessor loaded from {path} (output_dim={obj._output_dim})")
        return obj

    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("output_dim is unknown until the preprocessor is fitted.")
        return self._output_dim

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        status = f"output_dim={self._output_dim}" if self._is_fitted else "not fitted"
        return (
            f"ClinicalPreprocessor(num={len(self.numerical_cols)}, "
            f"cat={len(self.categorical_cols)}, {status})"
        )


# ===========================================================================
# Stage 2 — torch embedding module
# ===========================================================================

class ClinicalEmbedder(nn.Module):
    """MLP mapping a preprocessed clinical vector to ``d_model``.

    LayerNorm -> Linear -> GELU -> Dropout -> Linear -> LayerNorm.
    Trained end to end; slots into ``build_model(custom_projectors={'clin': ...})``.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else max(input_dim, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, input_dim] -> [B, d_model]."""
        return self.net(x)


# ===========================================================================
# Convenience: CSV -> per-case .pt files
# ===========================================================================

def prepare_clinical_embeddings(
    clinical_csv: str,
    out_dir: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    case_col: str = "case_id",
    train_case_ids: Optional[List[str]] = None,
    preprocessor_save_path: Optional[str] = None,
) -> Tuple["ClinicalPreprocessor", int]:
    """Fit on the training cases, then write ``{case_id}.pt`` for every row.

    ``train_case_ids`` restricts fitting to the training split, which is what
    keeps this leakage-free; passing None fits on everything.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(clinical_csv)
    full_df[case_col] = full_df[case_col].astype(str)

    if train_case_ids is not None:
        train_ids_str = [str(i) for i in train_case_ids]
        train_df = full_df[full_df[case_col].isin(train_ids_str)]
        if len(train_df) == 0:
            raise ValueError("None of the train_case_ids match rows in the CSV.")
    else:
        train_df = full_df

    proc = ClinicalPreprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        case_col=case_col,
    )
    proc.fit(train_df)

    if preprocessor_save_path:
        proc.save(preprocessor_save_path)

    n_written = n_failed = 0
    for _, row in full_df.iterrows():
        case_id = str(row[case_col])
        try:
            torch.save(proc.transform_to_tensor(row), out_path / f"{case_id}.pt")
            n_written += 1
        except Exception as exc:
            logger.warning(f"  Failed to process case '{case_id}': {exc}")
            n_failed += 1

    logger.info(
        f"prepare_clinical_embeddings: wrote {n_written} .pt files to {out_dir} "
        f"({n_failed} failed). output_dim={proc.output_dim}"
    )
    return proc, proc.output_dim
