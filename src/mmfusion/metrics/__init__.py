"""Evaluation metrics.

Public surface is unchanged from ``losses_metrics.py``; the implementations now
live in :mod:`mmfusion.metrics.core` (point metrics) and
:mod:`mmfusion.metrics.inference` (resampling, thresholds).

``regression_report`` and ``binary_report`` are new *containers* only. They hold
the metric sequence that was previously inlined in the notebooks' ``run_cv``.
Both the set of metrics and — critically — the order in which the two
resampling routines are called are preserved, because those routines draw from
the global RNG and reordering them would change every p-value and CI.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from mmfusion.metrics.core import (
    accuracy,
    auroc,
    concordance_ccc,
    concordance_index,
    f1_score_macro,
    pearson_r,
    r2_score,
    spearman_r,
)
from mmfusion.metrics.inference import (
    bootstrap_ci,
    find_best_threshold,
    permutation_test,
    permutation_test_paired,
)

__all__ = [
    "accuracy",
    "auroc",
    "bootstrap_ci",
    "binary_report",
    "compute_metrics",
    "concordance_ccc",
    "concordance_index",
    "default_es_metric",
    "f1_score_macro",
    "find_best_threshold",
    "pearson_r",
    "permutation_test",
    "permutation_test_paired",
    "r2_score",
    "regression_report",
    "spearman_r",
]


def compute_metrics(
    preds: np.ndarray,
    batch_accum: dict,
    task_type: str,
    num_classes: int = 2,
) -> dict:
    """Metrics for the fusion training loop, from accumulated NumPy arrays.

    Key insertion order is load-bearing: it determines the column order of
    ``summary.csv`` and ``history.csv``.
    """
    metrics: Dict[str, float] = {}
    if task_type == "survival":
        t = batch_accum["time"]
        e = batch_accum["event"]
        metrics["c_index"] = concordance_index(t, e, preds.squeeze())
    elif task_type == "regression":
        y = batch_accum["target"]
        p = preds.squeeze() if y.ndim == 1 else preds
        metrics["r2"] = r2_score(y.squeeze(), p.squeeze())
        metrics["pearson"] = pearson_r(y.squeeze(), p.squeeze())
        metrics["mae"] = float(np.mean(np.abs(y.squeeze() - p.squeeze())))
    elif task_type == "classification":
        y = batch_accum["label"]
        if num_classes <= 2:
            scores = torch.sigmoid(torch.tensor(preds)).numpy()
            y_pred = (scores > 0.5).astype(int)
            metrics["auroc"] = auroc(y, scores)
        else:
            scores = torch.softmax(torch.tensor(preds), dim=-1).numpy()
            y_pred = np.argmax(scores, axis=-1)
            metrics["auroc"] = auroc(y, scores)
        metrics["accuracy"] = accuracy(y, y_pred)
        metrics["f1_macro"] = f1_score_macro(y, y_pred)
    return metrics


def default_es_metric(task_type: str) -> str:
    """Default early-stopping metric per task."""
    return {
        "survival": "c_index",
        "regression": "r2",
        "classification": "auroc",
    }.get(task_type, "loss")


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    with_resampling: bool = True,
) -> Dict[str, float]:
    """The nine regression metrics the MIL pipeline records, in original order.

    RNG CONTRACT: ``permutation_test`` (1000 global draws) runs *before*
    ``bootstrap_ci`` (2000 global draws), matching the notebooks. Swapping them
    would change both results. See ``REFACTOR_NUMERICS.md`` entry N-05.
    """
    from sklearn.metrics import mean_squared_error

    report: Dict[str, float] = {
        "Pearson": pearson_r(y_true, y_pred),
        "Spearman": spearman_r(y_true, y_pred),
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "CCC": concordance_ccc(y_true, y_pred),
    }
    if with_resampling:
        _, p = permutation_test(y_true, y_pred)      # 1000 draws — must come first
        ci_low, ci_high = bootstrap_ci(y_true, y_pred)  # 2000 draws
        report["Permutation_p"] = p
        report["Bootstrap_CI_low"] = ci_low
        report["Bootstrap_CI_high"] = ci_high
    return report


def binary_report(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    prefix: str = "Binary_",
    include_threshold: bool = True,
) -> Dict[str, float]:
    """AUC / accuracy / F1 from thresholded continuous predictions.

    AUC is NaN unless both classes are present, matching the notebooks' guard.
    ``accuracy_score`` and ``f1_score`` come from sklearn here exactly as they
    did there (note: *not* the NaN-swallowing wrappers in
    :mod:`mmfusion.metrics.core`).
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    unique = np.unique(y_true_bin)
    if len(unique) == 2:
        auc = roc_auc_score(y_true_bin, y_score)
    else:
        auc = np.nan

    y_pred_bin = (y_score > threshold).astype(int)
    out = {
        f"{prefix}AUC": auc,
        f"{prefix}Accuracy": accuracy_score(y_true_bin, y_pred_bin),
        f"{prefix}F1": f1_score(y_true_bin, y_pred_bin),
    }
    if include_threshold:
        out["Best_Threshold"] = threshold
    return out


def majority_class_baseline(y_true_bin: np.ndarray) -> Dict[str, float]:
    """Prevalence and the all-positive baseline for accuracy and F1.

    NEW — computes no model quantity and changes nothing that already exists.
    Added because the recorded binary metrics are not interpretable without it
    (on the GHI assay the model scores below this baseline).
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true_bin = np.asarray(y_true_bin)
    all_pos = np.ones_like(y_true_bin)
    return {
        "prevalence": float(np.mean(y_true_bin)),
        "baseline_accuracy": float(accuracy_score(y_true_bin, all_pos)),
        "baseline_f1": float(f1_score(y_true_bin, all_pos)),
    }
