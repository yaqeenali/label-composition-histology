"""Whole-slide MIL dataset and its preprocessing helpers.

Everything here previously existed only inside the ``Pathology_only_TCGR_*``
notebooks, redefined in each. This module is the single home for it.

RNG CONTRACT
------------
:meth:`WSIMILDataset.__getitem__` draws from the global torch RNG via
``torch.randperm`` whenever ``num_patches`` is set and a slide has more patches
than that. It fires on *every* access, including evaluation, so test predictions
depend on the draw. That is the original behaviour and is preserved; see
``REFACTOR_NUMERICS.md`` entry N-10 for the deterministic alternative, which is
opt-in and off by default.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = [
    "WSIMILDataset",
    "compute_global_feature_stats",
    "compute_sample_weights",
    "assign_weights",
    "load_slide",
]

PathResolver = Callable[[str], str]


def _identity(path: str) -> str:
    return path


def load_slide(path: str) -> dict:
    """Load one Trident feature bundle: ``{'data': {'features', 'coords'}, 'meta'}``."""
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_global_feature_stats(
    df: pd.DataFrame,
    path_col: str = "pt_path",
    path_resolver: PathResolver = _identity,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std of patch features across the given slides.

    Fitted on the training split only by every caller. Concatenates every patch
    of every slide in memory, exactly as before — for 57 slides of ~10k patches
    at 1024 dims this is several GB, which is why it is worth knowing about.

    The ``+ 1e-6`` on the std and the use of ``torch.std`` (unbiased, n-1) are
    both preserved.
    """
    all_feats = []
    for _, row in df.iterrows():
        loaded = load_slide(path_resolver(row[path_col]))
        feats = loaded["data"]["features"].float()
        all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0)
    mean = all_feats.mean(0)
    std = all_feats.std(0) + 1e-6
    return mean, std


def compute_sample_weights(y: np.ndarray, bins: int = 10) -> np.ndarray:
    """Inverse-frequency weights from a histogram of the target.

    .. warning::
       BUG-PRESERVED: the original allocated the output with
       ``np.zeros_like(y)``, so an integer-typed target truncated every weight
       to zero. That is reproduced. Pass ``y`` as float (as the notebooks do,
       since the assay scores are floats) to get the intended behaviour.
       See ``REFACTOR_NUMERICS.md`` entry N-11.
    """
    counts, bin_edges = np.histogram(y, bins=bins)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean()

    sample_weights = np.zeros_like(y)  # BUG-PRESERVED: dtype follows y
    for i in range(len(counts)):
        if i < len(counts) - 1:
            mask = (y >= bin_edges[i]) & (y < bin_edges[i + 1])
        else:
            mask = (y >= bin_edges[i]) & (y <= bin_edges[-1])
        sample_weights[mask] = weights[i]
    return sample_weights


def assign_weights(y: np.ndarray, edges: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """Apply training-set bin edges and weights to another split.

    Same ``np.zeros_like`` behaviour as :func:`compute_sample_weights`.
    """
    w = np.zeros_like(y)  # BUG-PRESERVED
    for i in range(len(inv_freq)):
        if i < len(inv_freq) - 1:
            mask = (y >= edges[i]) & (y < edges[i + 1])
        else:
            mask = (y >= edges[i]) & (y <= edges[-1])
        w[mask] = inv_freq[i]
    return w


class WSIMILDataset(Dataset):
    """One whole-slide feature bag per item.

    Parameters
    ----------
    df : split dataframe; must carry ``path_col`` and ``target_col``
    target_col : continuous target column
    scaler : fitted ``StandardScaler`` for the target (fit on train only)
    feat_mean, feat_std : patch-feature statistics (fit on train only)
    num_patches : if set, randomly subsample slides larger than this
    sample_weights : per-item weights; when given, ``__getitem__`` returns a
        4-tuple instead of a 3-tuple, which is how the loop detects them
    path_col : column holding the ``.pt`` path (default ``pt_path``)
    path_resolver : maps a recorded path onto this machine; identity by default
    deterministic_subsample : NEW, default ``False``. When ``True``, patch
        subsampling uses a per-item generator seeded from ``subsample_seed`` and
        the item index, making evaluation reproducible. This CHANGES RESULTS and
        is therefore off unless explicitly requested.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        scaler=None,
        feat_mean: Optional[torch.Tensor] = None,
        feat_std: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        sample_weights: Optional[np.ndarray] = None,
        path_col: str = "pt_path",
        path_resolver: PathResolver = _identity,
        deterministic_subsample: bool = False,
        subsample_seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.scaler = scaler
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.num_patches = num_patches
        self.sample_weights = sample_weights
        self.path_col = path_col
        self.path_resolver = path_resolver
        self.deterministic_subsample = deterministic_subsample
        self.subsample_seed = subsample_seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        loaded = load_slide(self.path_resolver(row[self.path_col]))
        feats = loaded["data"]["features"].float()
        coords = loaded["data"]["coords"].int()

        if self.num_patches and feats.shape[0] > self.num_patches:
            if self.deterministic_subsample:
                gen = torch.Generator().manual_seed(self.subsample_seed + idx)
                ids = torch.randperm(feats.shape[0], generator=gen)[: self.num_patches]
            else:
                # RNG-ORDER: global torch RNG, one draw per oversized slide per access
                ids = torch.randperm(feats.shape[0])[: self.num_patches]
            feats = feats[ids]
            coords = coords[ids]

        if self.feat_mean is not None:
            feats = (feats - self.feat_mean) / self.feat_std

        y = np.array([[row[self.target_col]]], dtype="float32")
        if self.scaler is not None:
            y = self.scaler.transform(y)

        y_tensor = torch.tensor(y[0])

        if self.sample_weights is not None:
            weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
            return feats, coords, y_tensor, weight
        return feats, coords, y_tensor
