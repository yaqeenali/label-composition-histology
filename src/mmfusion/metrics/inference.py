"""Resampling-based inference and operating-point selection.

.. warning::

   ``permutation_test`` in this module reproduces a **defect** in the original
   ``losses_metrics.py``: it builds its null distribution by correlating
   ``y_true`` with a permutation of ``y_true``, so the predictions never enter
   the null. Every ``Permutation_p`` value in ``Results/`` was produced this
   way. It is preserved here verbatim because correcting it changes published
   numbers, which is out of scope for a refactor. The corrected version is
   provided alongside as :func:`permutation_test_paired` but is **not** wired
   into any pipeline. See ``REFACTOR_NUMERICS.md`` entries N-03 and N-04.

Both resampling routines draw from the **global** NumPy RNG, exactly as before.
They therefore consume from the same stream the training loop uses, and their
results depend on how many draws preceded them. An optional ``rng`` argument is
available for new code; passing it changes the numbers, so it defaults to
``None`` (legacy behaviour).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mmfusion.metrics.core import pearson_r

__all__ = [
    "bootstrap_ci",
    "permutation_test",
    "permutation_test_paired",
    "find_best_threshold",
]


def bootstrap_ci(
    y_true,
    y_pred,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for Pearson r.

    With ``rng=None`` this draws from the global NumPy RNG using
    ``np.random.choice``, identical to the original. Supplying ``rng`` makes the
    result reproducible but *different*; no legacy caller does so.
    """
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        if rng is None:
            idx = np.random.choice(n, n, replace=True)
        else:
            idx = rng.choice(n, n, replace=True)
        r = pearson_r(y_true[idx], y_pred[idx])
        stats.append(r)
    stats = np.array(stats)
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper


def permutation_test(
    y_true,
    y_pred,
    n: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Permutation p-value for Pearson r — **as originally implemented**.

    BUG-PRESERVED: the null is built from ``pearson_r(y_true, permutation(
    y_true))``, so ``y_pred`` contributes only to the observed statistic. The
    reference distribution therefore reflects the target's own marginal rather
    than the truth/prediction pairing under test.

    Kept because ``Results/*/metrics_fold_*.csv`` contains p-values from this
    function. Use :func:`permutation_test_paired` for new analyses.

    Returns
    -------
    (observed_r, p_value)
    """
    real = pearson_r(y_true, y_pred)
    sims = []
    for _ in range(n):
        if rng is None:
            permuted = np.random.permutation(y_true)
        else:
            permuted = rng.permutation(y_true)
        sims.append(pearson_r(y_true, permuted))  # BUG-PRESERVED: not y_pred
    sims = np.array(sims)
    p = (np.sum(np.abs(sims) >= abs(real)) + 1) / (n + 1)
    return real, p


def permutation_test_paired(
    y_true,
    y_pred,
    n: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Correct two-sided permutation test: shuffles the truth/prediction pairing.

    NOT used by any existing pipeline and NOT wired into the MIL loop — calling
    it in place of :func:`permutation_test` would change published p-values.
    Provided so the fix is available the moment it is asked for.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    real = pearson_r(y_true, y_pred)
    gen = rng if rng is not None else np.random
    sims = np.array([
        pearson_r(y_true, gen.permutation(y_pred)) for _ in range(n)
    ])
    p = (np.sum(np.abs(sims) >= abs(real)) + 1) / (n + 1)
    return real, p


def find_best_threshold(y_true, y_scores) -> Tuple[float, float]:
    """Threshold maximising F1, selected on the supplied (validation) set.

    Origin: the pathology notebooks. Falls back to 0.5 when only one class is
    present, returning the F1 that threshold happens to give.
    """
    from sklearn.metrics import f1_score, precision_recall_curve

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        # Only one class — cannot optimise a threshold; use 0.5
        return 0.5, f1_score(y_true, (y_scores > 0.5).astype(int))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1
