"""Point metrics for survival, regression and classification.

All implementations are carried over unchanged from ``losses_metrics.py``.
Two properties are deliberately preserved because results in ``Results/`` depend
on them:

* ``r2_score`` is hand-rolled rather than ``sklearn.metrics.r2_score``. The two
  agree to floating-point noise on finite input but differ in the degenerate
  ``ss_tot == 0`` case (this returns NaN, sklearn returns 0.0 or 1.0).
* ``auroc`` and ``f1_score_macro`` swallow *every* exception and return NaN.
  That silently converts a single-class test fold into a missing value rather
  than an error. See ``REFACTOR_NUMERICS.md`` entry N-06.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "concordance_index",
    "r2_score",
    "pearson_r",
    "spearman_r",
    "concordance_ccc",
    "accuracy",
    "auroc",
    "f1_score_macro",
]


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

def concordance_index(
    event_time: np.ndarray,
    event_observed: np.ndarray,
    risk_score: np.ndarray,
) -> float:
    """Harrell's C-index. O(n^2) — evaluation only."""
    n = len(event_time)
    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = event_time[i], event_time[j]
            ei, ej = event_observed[i], event_observed[j]
            ri, rj = risk_score[i], risk_score[j]
            if ti == tj:
                continue
            if ti < tj and ei == 1:
                comparable += 1
                if ri > rj:
                    concordant += 1
                elif ri == rj:
                    concordant += 0.5
            elif tj < ti and ej == 1:
                comparable += 1
                if rj > ri:
                    concordant += 1
                elif ri == rj:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Returns NaN when the target has zero variance."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation. Returns NaN if either input is constant."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation.

    The scipy import stays function-local, exactly as in the original. Hoisting
    it would turn a missing-scipy install from a call-time error into an
    import-time error for the whole package.
    """
    from scipy.stats import spearmanr

    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    Uses the *biased* (population) variance from ``np.var`` and the 1e-8
    denominator guard of the original.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy."""
    return float(np.mean(y_true == y_pred))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve. Returns NaN on any failure (e.g. one class only)."""
    try:
        from sklearn.metrics import roc_auc_score

        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except Exception:
        return float("nan")


def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1. Returns NaN on any failure."""
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return float("nan")
