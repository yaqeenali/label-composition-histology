"""Diagnostic plots.

Moved out of ``losses_metrics.py``, which mixed matplotlib with the metric
definitions and imported pyplot at module scope — meaning any import of a metric
pulled in a GUI backend. Here the backend is forced to Agg on first use, so the
functions work headless and in tests.

Signatures are unchanged, including the argument orders the callers rely on
(``scatter_plot(preds, targets, ...)`` — predictions first). No plot computes a
reported number and none consumes RNG.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

__all__ = [
    "scatter_plot",
    "calibration_plot",
    "bland_altman",
    "residual_plot",
    "attention_heatmap",
]


def _plt():
    """Return pyplot with a headless backend selected if none is set yet."""
    import matplotlib

    if matplotlib.get_backend().lower() not in {"agg", "pdf", "svg", "ps"}:
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass
    import matplotlib.pyplot as plt

    return plt


def scatter_plot(y, t, name, save_d, f) -> None:
    """Predicted (``y``) against true (``t``), titled with Pearson r."""
    from mmfusion.metrics.core import pearson_r

    plt = _plt()
    plt.figure()
    plt.scatter(t, y, alpha=.6)
    r = pearson_r(t, y)
    plt.title(f"{name} Fold{f} r={r:.2f}")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.savefig(f"{save_d}/scatter_{f}.png")
    plt.close()


def calibration_plot(y, t, save_d, f) -> None:
    """Predictions against truth with the identity line."""
    plt = _plt()
    plt.figure()
    plt.plot([t.min(), t.max()], [t.min(), t.max()])
    plt.scatter(t, y, alpha=.6)
    plt.title("Calibration")
    plt.savefig(f"{save_d}/calibration_{f}.png")
    plt.close()


def bland_altman(y, t, save_d, f) -> None:
    """Difference against mean, with the 1.96-SD limits of agreement."""
    plt = _plt()
    m = (y + t) / 2
    d = y - t
    plt.figure()
    plt.scatter(m, d, alpha=.6)
    plt.axhline(d.mean())
    plt.axhline(d.mean() + 1.96 * d.std(), linestyle="--")
    plt.axhline(d.mean() - 1.96 * d.std(), linestyle="--")
    plt.title("Bland-Altman")
    plt.savefig(f"{save_d}/bland_{f}.png")
    plt.close()


def residual_plot(y_pred, y_true, name, save_d, f) -> None:
    """Residuals (true - predicted) against predicted."""
    plt = _plt()
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residuals - {name} Fold{f}")
    plt.savefig(f"{save_d}/residual_{f}.png")
    plt.close()


def attention_heatmap(
    attn_weights,
    coords: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap",
) -> None:
    """Patch attention, either as a trace (no coords) or a spatial map.

    Weights are min-max normalised per call, so maps are comparable in shape but
    not in absolute magnitude across slides.
    """
    plt = _plt()
    attn_weights = np.array(attn_weights)
    attn_weights = (attn_weights - attn_weights.min()) / (
        attn_weights.max() - attn_weights.min() + 1e-8
    )

    if coords is None:
        plt.figure(figsize=(10, 3))
        plt.plot(attn_weights)
        plt.title(title)
        plt.ylabel("Attention")
        plt.xlabel("Patch index")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    coords = np.array(coords)
    plt.figure(figsize=(7, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=attn_weights, cmap="jet", s=40, alpha=0.85)
    plt.colorbar(label="Attention weight")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
