"""Cross-validated gated-attention MIL training.

This is the pipeline that produced everything in ``Results/``. It previously
existed only as a copy-pasted cell inside four notebooks, which had drifted
apart in four ways (model width, dropout, weight decay, patch subsampling and
the model-selection rule). All four variants are reproduced here exactly; which
one runs is decided by :class:`mmfusion.config.AssayConfig`, not by editing code.

RNG CONTRACT — the order below is load-bearing and must not be reordered
-----------------------------------------------------------------------
Per fold, drawing from the global torch RNG then the global NumPy RNG:

1. ``GatedABMIL(...)``                     5 Linear inits
2. for each epoch: train pass, then val pass
3. one further val pass (threshold tuning)
4. train, val, test passes with ``return_embeddings=True``
5. ``permutation_test``  (1000 NumPy draws)
6. ``bootstrap_ci``      (2000 NumPy draws)

Every pass over a loader draws ``torch.randperm`` once per slide that exceeds
``num_patches`` — including evaluation passes, because subsampling lives in the
dataset rather than the loop. Dropout draws only in ``model.train()``.

Artifact writing (embeddings, prediction CSVs, plots) consumes no RNG, so
``write_artifacts=False`` yields bit-identical metrics. That is what makes the
parity tests fast, and it is asserted in
``tests/test_parity_mil.py::test_artifacts_flag_is_numerically_neutral``.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mmfusion.config import AssayConfig
from mmfusion.data.mil import (
    WSIMILDataset,
    assign_weights,
    compute_global_feature_stats,
    compute_sample_weights,
)
from mmfusion.losses import resolve_mil_loss_fn
from mmfusion.metrics import (
    binary_report,
    find_best_threshold,
    pearson_r,
    regression_report,
)
from mmfusion.models.gated_abmil import GatedABMIL
from mmfusion.seeding import resolve_device

__all__ = ["run_epoch", "run_fold", "run_cv", "FoldResult", "CVResult"]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Everything one fold produces."""
    fold: str
    metrics: Dict[str, float]
    test_preds: np.ndarray
    test_targets: np.ndarray
    val_preds: np.ndarray
    val_binary: Optional[np.ndarray]
    test_binary: Optional[np.ndarray]
    best_threshold: float
    attentions: List[np.ndarray] = field(default_factory=list)
    coords: List[np.ndarray] = field(default_factory=list)


@dataclass
class CVResult:
    """Per-fold results plus the pooled ("global") summary.

    .. note::
       ``global_metrics`` pools predictions from all folds — five different
       models — into one correlation. That is what the notebooks reported and it
       is preserved verbatim. ``per_fold_summary`` is new and computes nothing
       new: it is the mean and SD of numbers already in ``folds``.
    """
    folds: List[FoldResult]
    global_metrics: Dict[str, float]

    def per_fold_summary(self) -> pd.DataFrame:
        """Mean / SD / min / max of each metric across folds."""
        df = pd.DataFrame([f.metrics for f in self.folds], index=[f.fold for f in self.folds])
        numeric = df.select_dtypes(include="number")
        return pd.DataFrame({
            "mean": numeric.mean(),
            "std": numeric.std(ddof=1),
            "min": numeric.min(),
            "max": numeric.max(),
            "n_folds": numeric.notna().sum(),
        })


# ---------------------------------------------------------------------------
# One pass over a loader
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: Optional[torch.optim.Optimizer] = None,
    *,
    loss_fn: Callable,
    return_embeddings: bool = False,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
):
    """Train (when *opt* is given) or evaluate one pass. Batch size is 1.

    Returns ``(mean_loss, preds, targets)``, or with ``return_embeddings=True``
    ``(mean_loss, preds, targets, embeddings, attentions, coords)``.

    Predictions are in the *scaled* target space; callers inverse-transform.
    """
    device = device if device is not None else resolve_device()
    train = opt is not None
    model.train() if train else model.eval()

    preds, targets, losses = [], [], []
    embeddings, attns, coords_list = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if len(batch) == 4:                       # sample weights present
                feats, coords, target, weight = batch
                weight = weight.to(device)
            else:
                feats, coords, target = batch
                weight = None

            feats = feats[0].to(device)
            target = target.to(device).view(1)
            if train:
                opt.zero_grad()

            if return_embeddings:
                pred, attn, emb = model(feats, return_emb=True)
                embeddings.append(emb)
                attns.append(attn.cpu().numpy())
                coords_list.append(coords[0].numpy())
            else:
                pred, _ = model(feats)

            pred = pred.view(1)
            loss = loss_fn(pred, target, weight)

            if train:
                loss.backward()
                if grad_clip > 0:   # notebooks always passed 1.0; guard added
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            preds.append(pred.item())
            targets.append(target.item())
            losses.append(loss.item())

    if return_embeddings:
        return (np.mean(losses), np.array(preds), np.array(targets),
                embeddings, attns, coords_list)
    return np.mean(losses), np.array(preds), np.array(targets)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

class _ModelSelector:
    """Keeps the best epoch's weights under one of two divergent rules.

    ``val_loss``     lower is better, compared with ``<=`` so the *last* of
                     equal scores wins (GHI notebook).
    ``val_pearson``  higher is better, compared with ``>`` so the *first* of
                     equal scores wins (all other notebooks).

    The asymmetric tie-breaking is deliberate: it is what the two code paths
    did, and with 20 epochs on 9-10 validation patients ties are not exotic.
    """

    def __init__(self, metric: str):
        if metric not in ("val_loss", "val_pearson"):
            raise ValueError(
                f"Unknown selection_metric '{metric}'. "
                "Expected 'val_loss' or 'val_pearson'."
            )
        self.metric = metric
        self.best_score = float("inf") if metric == "val_loss" else -1
        self.best_state: Optional[dict] = None

    def update(self, model: torch.nn.Module, val_loss: float, val_r: float) -> None:
        if self.metric == "val_loss":
            better = val_loss <= self.best_score
            score = val_loss
        else:
            better = val_r > self.best_score
            score = val_r
        if better:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def _build_sample_weights(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    weight_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-frequency weights for the ``weighted_huber`` path."""
    train_targets_orig = train_df[target_col].values
    train_weights = compute_sample_weights(train_targets_orig, bins=weight_bins)

    counts, bin_edges = np.histogram(train_targets_orig, bins=weight_bins)
    inv_freq = 1.0 / (counts + 1e-6)
    inv_freq = inv_freq / inv_freq.mean()

    val_weights = assign_weights(val_df[target_col].values, bin_edges, inv_freq)
    test_weights = assign_weights(test_df[target_col].values, bin_edges, inv_freq)
    return train_weights, val_weights, test_weights


def run_fold(
    cfg: AssayConfig,
    fold: str,
    frames: Dict[str, pd.DataFrame],
    *,
    fold_dir: Optional[Path] = None,
    write_artifacts: bool = True,
    device: Optional[torch.device] = None,
    path_resolver: Callable[[str], str] = lambda p: p,
    case_col: str = "CLID",
    verbose: bool = True,
) -> FoldResult:
    """Train and evaluate one fold. See the module RNG contract for ordering."""
    from sklearn.preprocessing import StandardScaler

    device = device if device is not None else resolve_device()
    train_df, val_df, test_df = frames["train"], frames["val"], frames["test"]
    target_col = cfg.target_col
    tcfg = cfg.train

    # ---- fold-local fitting (train split only) ------------------------------
    scaler = StandardScaler().fit(train_df[[target_col]].values)
    feat_mean, feat_std = compute_global_feature_stats(train_df, path_resolver=path_resolver)

    train_w = val_w = test_w = None
    if tcfg.loss_type == "weighted_huber":
        train_w, val_w, test_w = _build_sample_weights(
            train_df, val_df, test_df, target_col, tcfg.weight_bins
        )

    def _dataset(df, weights):
        return WSIMILDataset(
            df, target_col, scaler, feat_mean, feat_std,
            num_patches=tcfg.num_patches,
            sample_weights=weights,
            path_resolver=path_resolver,
        )

    loaders = {
        "train": DataLoader(_dataset(train_df, train_w), batch_size=tcfg.batch_size,
                            shuffle=tcfg.shuffle_train),
        "val": DataLoader(_dataset(val_df, val_w), batch_size=tcfg.batch_size, shuffle=False),
        "test": DataLoader(_dataset(test_df, test_w), batch_size=tcfg.batch_size, shuffle=False),
    }

    # ---- model (RNG step 1) -------------------------------------------------
    model = GatedABMIL(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), tcfg.lr, weight_decay=tcfg.weight_decay)
    loss_fn = resolve_mil_loss_fn(tcfg.loss_type, tau=tcfg.quantile_tau)
    selector = _ModelSelector(tcfg.selection_metric)

    def _epoch(split, optimizer=None, return_embeddings=False):
        return run_epoch(
            model, loaders[split], optimizer,
            loss_fn=loss_fn, return_embeddings=return_embeddings,
            grad_clip=tcfg.grad_clip, device=device,
        )

    def _unscale(arr: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()

    # ---- training (RNG step 2) ---------------------------------------------
    for e in range(tcfg.epochs):
        tl, _, _ = _epoch("train", opt)
        vl, preds_val, targets_val = _epoch("val")
        r = pearson_r(_unscale(targets_val), _unscale(preds_val))
        if verbose:
            print(f"Epoch {e+1} train {tl:.3f} val {vl:.3f} r {r:.3f}")
        selector.update(model, vl, r)

    model.load_state_dict(selector.best_state)

    # ---- threshold tuning pass (RNG step 3) --------------------------------
    _, preds_val_raw, _ = _epoch("val")
    preds_val = _unscale(preds_val_raw.ravel())

    best_thresh = 0.5
    val_binary = None
    if cfg.binary_target_col is not None:
        val_binary = val_df[cfg.binary_target_col].values.astype(float)
        best_thresh, best_f1_val = find_best_threshold(val_binary, preds_val)
        if verbose:
            print(f"  Best threshold on validation: {best_thresh:.4f} (F1={best_f1_val:.4f})")

    # ---- embedding / prediction passes (RNG step 4) ------------------------
    split_outputs = {}
    for split in ("train", "val", "test"):
        _, preds, targets, embeddings, attns, coords_list = _epoch(split, return_embeddings=True)
        preds, targets = _unscale(preds), _unscale(targets)
        split_outputs[split] = (preds, targets, embeddings, attns, coords_list)
        if write_artifacts and fold_dir is not None:
            _write_split_artifacts(
                fold_dir, split, frames[split][case_col].values,
                preds, targets, embeddings,
            )

    preds_arr, targets_arr, _, attns_test, coords_test = split_outputs["test"]

    if write_artifacts and fold_dir is not None:
        _write_fold_plots(
            fold_dir, fold, target_col, preds_arr, targets_arr,
            test_df[case_col].values, attns_test, coords_test,
        )

    # ---- metrics (RNG steps 5 and 6) ---------------------------------------
    metrics = regression_report(targets_arr, preds_arr)

    test_binary = None
    if cfg.binary_target_col is not None:
        test_binary = test_df[cfg.binary_target_col].values.astype(float)
        if len(np.unique(test_binary)) != 2 and verbose:
            print(
                f"  Warning: Test set in fold {fold} has only one class "
                f"({np.unique(test_binary)[0]}). AUC set to NaN."
            )
        metrics.update(binary_report(test_binary, preds_arr, best_thresh))
        if verbose:
            print(
                f"  Binary metrics (threshold={best_thresh:.4f}): "
                f"AUC={metrics['Binary_AUC']:.4f}, "
                f"Acc={metrics['Binary_Accuracy']:.4f}, F1={metrics['Binary_F1']:.4f}"
            )
    elif cfg.binary_class:
        # Legacy path: threshold the *target* at 0.5 to synthesise labels.
        derived = (targets_arr > 0.5).astype(int)
        metrics.update(binary_report(derived, preds_arr, 0.5, prefix="", include_threshold=False))

    if write_artifacts and fold_dir is not None:
        pd.DataFrame([metrics]).to_csv(fold_dir / f"metrics_fold_{fold}.csv", index=False)

    return FoldResult(
        fold=fold,
        metrics=metrics,
        test_preds=preds_arr,
        test_targets=targets_arr,
        val_preds=preds_val,
        val_binary=val_binary,
        test_binary=test_binary,
        best_threshold=best_thresh,
        attentions=attns_test,
        coords=coords_test,
    )


# ---------------------------------------------------------------------------
# Artifact writing (no RNG)
# ---------------------------------------------------------------------------

def _write_split_artifacts(fold_dir, split, case_ids, preds, targets, embeddings) -> None:
    split_dir = Path(fold_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, clid in enumerate(case_ids):
        torch.save(embeddings[i].squeeze(), split_dir / f"{clid}.pt")
    pd.DataFrame({"CLID": case_ids, "true": targets, "pred": preds}).to_csv(
        split_dir / f"{split}_preds.csv", index=False
    )


def _write_fold_plots(
    fold_dir, fold, target_col, preds, targets, case_ids, attns, coords,
) -> None:
    """Per-patient and aggregate attention maps plus the four regression plots.

    The original wrote the aggregate attention heatmap twice to each of two
    paths (a copy-paste duplication). The duplicate writes are dropped here: the
    same figure was rendered from the same arrays to the same filenames, so the
    files on disk are unchanged.
    """
    from mmfusion.viz import (
        attention_heatmap, bland_altman, calibration_plot, residual_plot, scatter_plot,
    )

    fold_dir = Path(fold_dir)
    per_patient = fold_dir / "test_attention_per_patient"
    per_patient.mkdir(parents=True, exist_ok=True)
    for i, clid in enumerate(case_ids):
        attention_heatmap(
            attns[i], coords[i],
            save_path=str(per_patient / f"{clid}.png"),
            title=f"Patient {clid} Attention",
        )

    all_attn = np.concatenate(attns)
    all_coords = np.concatenate(coords, axis=0)

    attention_heatmap(
        all_attn, all_coords,
        save_path=str(fold_dir / f"attention_Fold{fold}.png"),
        title=f"Fold {fold} Attention Heatmap",
    )
    test_attn_dir = fold_dir / "test_attention"
    test_attn_dir.mkdir(parents=True, exist_ok=True)
    attention_heatmap(
        all_attn, all_coords,
        save_path=str(test_attn_dir / f"attention_Fold{fold}.png"),
        title=f"Fold {fold} Test Attention Heatmap",
    )

    # Argument order matches the original calls exactly (preds first).
    scatter_plot(preds, targets, target_col, str(fold_dir), fold)
    calibration_plot(preds, targets, str(fold_dir), fold)
    bland_altman(preds, targets, str(fold_dir), fold)
    residual_plot(preds, targets, target_col, str(fold_dir), fold)


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------

def _read_fold_frames(splits_dir: Path, fold: str, suffix: str) -> Dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(splits_dir / fold / f"{split}{suffix}.csv")
        for split in ("train", "val", "test")
    }


def run_cv(
    cfg: AssayConfig,
    *,
    write_artifacts: bool = True,
    device: Optional[torch.device] = None,
    path_resolver: Callable[[str], str] = lambda p: p,
    case_col: str = "CLID",
    verbose: bool = True,
) -> CVResult:
    """Run every fold under ``cfg.splits_dir`` and pool the results.

    Folds are visited in sorted directory order, as before. Non-directory
    entries are skipped — the original would crash on a stray ``.DS_Store``, so
    any run that previously succeeded saw only fold directories and the order is
    unchanged.
    """
    # Read the derived *_bin.csv from binary_splits_dir when the config sets it,
    # so `mmfusion-binarise` writing out of place cannot leave training silently
    # consuming stale in-place labels.
    splits_dir = Path(cfg.binary_splits_dir or cfg.splits_dir)
    results_dir = Path(cfg.results_dir) if cfg.results_dir else None
    if write_artifacts and results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

    fold_names = sorted(p.name for p in splits_dir.iterdir() if p.is_dir())

    folds: List[FoldResult] = []
    for fold in fold_names:
        if verbose:
            print(f"\n===== FOLD {fold} =====")
        fold_dir = (results_dir / fold) if results_dir is not None else None
        if write_artifacts and fold_dir is not None:
            fold_dir.mkdir(parents=True, exist_ok=True)

        frames = _read_fold_frames(splits_dir, fold, cfg.split_suffix)
        result = run_fold(
            cfg, fold, frames,
            fold_dir=fold_dir, write_artifacts=write_artifacts,
            device=device, path_resolver=path_resolver,
            case_col=case_col, verbose=verbose,
        )
        if verbose:
            print(f"Fold {fold} metrics:")
            for k, v in result.metrics.items():
                print(f"  {k}: {v:.4f}")
        folds.append(result)

    global_metrics = _pool_global_metrics(cfg, folds, verbose=verbose)

    if write_artifacts and results_dir is not None:
        pd.DataFrame([global_metrics]).to_csv(results_dir / "global_metrics.csv", index=False)
        CVResult(folds, global_metrics).per_fold_summary().to_csv(
            results_dir / "per_fold_summary.csv"
        )

    if verbose:
        print("\n===== GLOBAL METRICS =====")
        for k, v in global_metrics.items():
            print(f"{k}: {v:.4f}")

    return CVResult(folds=folds, global_metrics=global_metrics)


def _pool_global_metrics(
    cfg: AssayConfig,
    folds: List[FoldResult],
    *,
    verbose: bool = True,
) -> Dict[str, float]:
    """Concatenate every fold's test predictions and score them as one set.

    Preserved exactly, including the fact that this mixes five separately
    trained models and reports no interval. ``with_resampling=False`` matches
    the original, which computed no permutation test or CI at this level.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    all_preds = np.concatenate([f.test_preds for f in folds]) if folds else np.array([])
    all_targets = np.concatenate([f.test_targets for f in folds]) if folds else np.array([])

    global_metrics = regression_report(all_targets, all_preds, with_resampling=False)

    if cfg.binary_target_col is not None and any(f.test_binary is not None for f in folds):
        all_binary = np.concatenate([f.test_binary for f in folds])
        all_val_preds = np.concatenate([f.val_preds for f in folds])
        all_val_binary = np.concatenate([f.val_binary for f in folds])

        if len(np.unique(all_binary)) == 2:
            global_auc = roc_auc_score(all_binary, all_preds)
        else:
            global_auc = np.nan
            if verbose:
                print("Warning: Global test set has only one class. AUC set to NaN.")

        if len(all_val_preds) > 0:
            if len(np.unique(all_val_binary)) == 2:
                global_thresh, _ = find_best_threshold(all_val_binary, all_val_preds)
            else:
                global_thresh = 0.5
                if verbose:
                    print("Warning: Validation set has only one class. Using default threshold 0.5.")
        else:
            thresholds = [f.best_threshold for f in folds]
            global_thresh = np.mean(thresholds) if thresholds else 0.5

        global_preds_bin = (all_preds >= global_thresh).astype(int)
        global_metrics.update({
            "Binary_AUC": global_auc,
            "Binary_Accuracy": accuracy_score(all_binary, global_preds_bin),
            "Binary_F1": f1_score(all_binary, global_preds_bin),
            "Global_Best_Threshold": global_thresh,
        })

    # Two independent ifs, as in the original - not elif. The branches are
    # mutually exclusive in practice but the control flow is preserved.
    if cfg.binary_class and cfg.binary_target_col is None:
        derived = (all_targets > 0.5).astype(int)
        preds_bin = (all_preds > 0.5).astype(int)
        global_metrics.update({
            "AUC": roc_auc_score(derived, all_preds),
            "Accuracy": accuracy_score(derived, preds_bin),
            "F1": f1_score(derived, preds_bin),
        })

    return global_metrics
