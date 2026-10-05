#!/usr/bin/env python3
"""Submission figures for the Journal of Imaging Informatics in Medicine.

Fig 1   cohort flow
Fig 2   study design and model architecture
Fig 3   the histology gain over the clinicopathological model, and how it
        tracks agreement between each label and ER status (2 panels)

Springer requirements followed: sans-serif labels (Liberation Sans is
metrically Arial), 600 dpi combination art, RGB, all lines >= 0.3 pt; each
figure is written as PNG (for embedding) and as LZW-compressed TIFF (for
upload); captions live in the manuscript rather than in the image.

Colours are the CVD-safe pair #1f6fb2 / #c1662b, validated with the dataviz
palette checker; identity is carried by direct labels as well as by hue, so the
figures survive greyscale printing.

    python scripts/make_submission_figures.py --results Results_locked/modal --out figs
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MODEL, CLIN = "#1f6fb2", "#c1662b"
INK, MUTED, RULE, FILL = "#111111", "#555555", "#b8b8b8", "#f2f4f7"
DPI = 600
SANS = "Liberation Sans"

plt.rcParams.update({
    "font.family": SANS, "font.size": 9,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

ORDER = ["GHI (Oncotype)", "ROR-P", "ROR-S", "MammaPrint"]
SHORT = {"GHI (Oncotype)": "Oncotype", "ROR-P": "ROR-P",
         "ROR-S": "ROR-S", "MammaPrint": "MammaPrint"}


def pfmt(x: float) -> str:
    if x < 0.001:
        return "p < 0.001"
    return f"p = {Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)}"


# --------------------------------------------------------------------------- #
# Fig 1 - study design and architecture
# --------------------------------------------------------------------------- #

def fig1_design(out: Path) -> None:
    """Three panels: representation, model, evaluation design.

    The radiomics arm sits inside the evaluation panel rather than in one of its
    own -- it is simply a fourth predictor evaluated on the same folds, and
    giving it a separate panel both crowded the figure and implied it was a
    separate study.
    """
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc=FILL, ec=RULE, size=8.5, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=INK, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color=MUTED, lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color=color, linewidth=lw,
                                     shrinkA=0, shrinkB=0))

    def panel(x, y, letter, title):
        ax.text(x, y, letter, fontsize=11, fontweight="bold", va="center")
        ax.text(x + 4, y, title, fontsize=9.5, fontweight="bold", va="center")

    # ---- (a) representation pipeline -------------------------------------
    panel(1, 96, "a", "Slide representation")
    yb, hb = 78, 12
    box(1, yb, 17, hb, "Diagnostic H&E\nwhole-slide image")
    box(21.5, yb, 17, hb, "Tiling\n256 px at 20$\\times$\nmedian 19,376\npatches", size=8)
    box(42, yb, 17, hb, "UNI encoder\n1,024-d embedding\nper patch")
    box(62.5, yb, 17, hb, "Fixed random pool\n512 patches per slide")
    box(83, yb, 16, hb, "Slide bag\n512 $\\times$ 1024", fc="white", ec=MODEL, lw=1.2)
    for x in (18, 38.5, 59, 79.5):
        arrow(x, yb + hb / 2, x + 3.4, yb + hb / 2)

    # ---- (b) model -------------------------------------------------------
    panel(1, 71, "b", "Gated-attention multiple-instance model")
    yc, hc = 52, 13
    box(1, yc, 17, hc, "Slide bag\n512 $\\times$ 1024", fc="white", ec=MODEL, lw=1.2)
    box(21.5, yc, 19, hc, "Shared projection\n$h_i = \\mathrm{ReLU}(W x_i)$")
    box(44, yc, 24, hc, "Gated attention\n"
                        r"$a_i \propto \exp\{w^\top(\tanh(Vh_i)\odot\sigma(Uh_i))\}$", size=7.0)
    box(71.5, yc, 13.5, hc, "Pooling\n$z=\\sum_i a_i h_i$")
    box(88, yc, 11, hc, "Regression\nhead", fc="white", ec=MODEL, lw=1.2)
    for x in (18, 40.5, 68, 85):
        arrow(x, yc + hc / 2, x + 3.4, yc + hc / 2)
    ax.text(93.5, yc - 2.6, "predicted signature score", ha="center", va="top",
            fontsize=8, color=MODEL)
    arrow(93.5, yc, 93.5, yc - 2.2, color=MODEL)

    # ---- (c) evaluation design ------------------------------------------
    panel(1, 43, "c", "Evaluation design, identical for all four signatures")
    box(1, 12, 22, 14, "One shared partition\n5-fold, multi-label\nstratified on all "
                       "four\ntargets and ER status", fc="white", ec=INK, lw=1.1)

    arms = [
        (33.0, "Histology model\ngated-attention network", "white", MODEL, 1.2),
        (24.5, "ER status alone", FILL, RULE, 0.8),
        (16.0, "Clinicopathological model\nage, ER, PR, HER2, nodes, stage", "white", CLIN, 1.2),
        (4.0, "MRI radiomics\nnested feature and penalty selection", FILL, RULE, 0.8),
    ]
    for y, text, fc, ec, lw in arms:
        box(29.5, y, 32, 7.6, text, fc=fc, ec=ec, lw=lw, size=8)
        arrow(23.2, 19, 29.1, y + 3.8)
    for y, _, _, _, _ in arms[:3]:
        arrow(61.9, y + 3.8, 67.1, 25.5)
    arrow(61.9, 7.8, 67.1, 7.8)

    box(67.5, 18, 31.5, 15, "Paired bootstrap on the same\npatients, 4,000 resamples\n"
                            "histology $-$ comparator\n$\\Delta$AUC with 95% CI",
        fc="white", ec=INK, lw=1.1, size=8)
    box(67.5, 3, 31.5, 9.6, "AUC under nested selection;\noptimism = non-nested $-$ nested",
        fc="white", ec=RULE, lw=0.8, size=8)

    fig.savefig(out / "Fig2.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig S1 - cohort flow
# --------------------------------------------------------------------------- #

def fig2_flow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 5.2))
    ax.set_xlim(-1, 101); ax.set_ylim(0, 125); ax.axis("off")

    def box(x, y, w, h, title, n, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor="white", edgecolor=INK if bold else MUTED,
                                    linewidth=1.3 if bold else 0.8))
        ax.text(x + 3, y + h / 2, title, fontsize=9,
                fontweight="bold" if bold else "normal", va="center")
        ax.text(x + w - 3, y + h / 2, f"n = {n}", fontsize=9.5, ha="right", va="center",
                fontweight="bold" if bold else "normal")

    def excl(x, y, w, h, text, n):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor="white", edgecolor=RULE,
                                    linewidth=0.7, linestyle=(0, (3, 3))))
        ax.text(x + 3, y + h / 2, text, fontsize=8, color=MUTED, va="center")
        ax.text(x + w - 3, y + h / 2, f"n = {n}", fontsize=8.5, color=MUTED,
                ha="right", va="center")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color=MUTED, linewidth=0.8))

    box(0, 105, 52, 15, "Recomputed signature scores", 100)
    box(0, 80, 52, 15, "In the radiogenomics subset", 84)
    box(0, 55, 52, 15, "With a diagnostic slide", 83)
    box(0, 30, 52, 15, "Analysis cohort", 82, bold=True)
    box(0, 5, 52, 15, "With complete covariates", 81, bold=True)
    excl(58, 94, 42, 9, "Outside the subset", 16)
    excl(58, 69, 42, 9, "No diagnostic slide", 1)
    excl(58, 44, 42, 9, "Neoadjuvant treatment", 1)
    excl(58, 19, 42, 9, "HER2 equivocal on both tests", 1)

    for y in (105, 80, 55, 30):
        arrow(26, y, 26, y - 9.4)
        arrow(26, y - 6.5, 57.4, y - 6.5)

    ax.text(0, -1.5, "Slides were processed only for the radiogenomics subset (91 patients with MRI);\n"
                     "seven of its 90 slides belong to patients without signature scores.",
            fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "Fig1.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2 - the gain, and what predicts it
# --------------------------------------------------------------------------- #

ALT_LABEL = {"GHI (Oncotype)": "Oncotype\nHigh only", "ROR-P": "ROR-P\nhigh only",
             "ROR-S": "ROR-S\nhigh only"}


def fig3_mechanism(res: Path, out: Path) -> None:
    cmp_ = pd.read_csv(res / "review" / "comparators.csv").set_index("assay")
    alt = pd.read_csv(res / "review" / "binarisation_sensitivity.csv").set_index("assay")
    order = cmp_.d_vs_clin_predeclared.sort_values(ascending=True).index.tolist()

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})

    # (a) gain over the clinicopathological model, pre-declared comparator
    # filled, tuned comparator hollow
    axL.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    for i, a in enumerate(order):
        r = cmp_.loc[a]
        lo, hi = [float(v) for v in r.CI_clin_predeclared.strip("[]").split(",")]
        axL.plot([lo, hi], [i, i], color=MODEL, linewidth=1.8, zorder=2)
        for xb in (lo, hi):
            axL.plot([xb, xb], [i - .09, i + .09], color=MODEL, linewidth=1.8, zorder=2)
        axL.plot([r.d_vs_clin_predeclared], [i], "o", color=MODEL, markersize=6.5,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        axL.plot([r.d_vs_clin_tuned], [i - 0.28], "D", markerfacecolor="white",
                 markeredgecolor=MODEL, markersize=4.2, markeredgewidth=1.0, zorder=3)
        axL.text(0.905, i, f"{r.d_vs_clin_predeclared:+.3f}".replace("-", "−") + f"  {pfmt(r.p_clin_predeclared)}",
                 ha="right", va="center", fontsize=8)
    axL.set_yticks(range(len(order))); axL.set_yticklabels([SHORT[a] for a in order])
    axL.set_xlim(-0.22, 0.92)
    axL.set_xticks([-0.2, 0.0, 0.2, 0.4])
    axL.set_xticklabels(["−0.2", "0", "0.2", "0.4"])
    axL.set_ylim(-0.7, len(order) - 0.4)
    axL.set_xlabel("Gain in AUC over the clinicopathological model (95% CI)")
    axL.grid(axis="x", color=RULE, linewidth=0.4, alpha=0.6); axL.set_axisbelow(True)
    axL.set_title("a", loc="left", fontsize=11, fontweight="bold", pad=6)
    axL.tick_params(axis="y", length=0)
    axL.legend(handles=[Line2D([], [], marker="o", linestyle="", color=MODEL, markersize=6,
                               label="C = 1 comparator (95% CI)"),
                        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                               markeredgecolor=MODEL, markersize=4.2, label="tuned-penalty comparator")],
               loc="lower right", frameon=False, fontsize=7, handletextpad=0.3,
               borderaxespad=0.1, labelcolor=INK)

    # (b) the gain against how much of the label ER status already is,
    # for the four primary labels and three alternative cuts
    axR.axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    for a in ORDER:
        r = cmp_.loc[a]
        lo, hi = [float(v) for v in r.CI_clin_predeclared.strip("[]").split(",")]
        axR.plot([r.AUC_ER_fixed, r.AUC_ER_fixed], [lo, hi], color=MODEL, linewidth=0.9, zorder=2)
        axR.plot([r.AUC_ER_fixed], [r.d_vs_clin_predeclared], "o", color=MODEL, markersize=6.5,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    for a in alt.index:
        r = alt.loc[a]
        lo, hi = [float(v) for v in r.CI.strip("[]").split(",")]
        axR.plot([r.AUC_ER_fixed, r.AUC_ER_fixed], [lo, hi], color=MODEL, linewidth=0.9, zorder=2)
        axR.plot([r.AUC_ER_fixed], [r.d_vs_clin_predeclared], "D", markerfacecolor="white",
                 markeredgecolor=MODEL, markersize=4.6, markeredgewidth=1.0, zorder=3)
    # direct labels with short leaders, placed by hand so none collides
    def lab(text, xy, xytext, ha, va="center", color=INK, size=7.5, leader=True):
        axR.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=size, color=color,
                     linespacing=1.1, zorder=5,
                     arrowprops=dict(arrowstyle="-", color=RULE, linewidth=0.6,
                                     shrinkA=0, shrinkB=3) if leader else None)
    pt = lambda a: (cmp_.loc[a, "AUC_ER_fixed"], cmp_.loc[a, "d_vs_clin_predeclared"])
    at = lambda a: (alt.loc[a, "AUC_ER_fixed"], alt.loc[a, "d_vs_clin_predeclared"])
    lab("Oncotype", pt("GHI (Oncotype)"), (0.615, 0.385), "left")
    lab("ROR-P", pt("ROR-P"), (0.635, 0.235), "left")
    lab("ROR-S", pt("ROR-S"), (0.693, 0.185), "left")
    lab("MammaPrint", pt("MammaPrint"), (0.862, -0.012), "left")
    lab("Oncotype,\nHigh only", at("GHI (Oncotype)"), (0.535, 0.05), "left", color=MUTED, size=6.8)
    lab("ROR-P,\nhigh only", at("ROR-P"), (0.752, 0.105), "left", color=MUTED, size=6.8)
    lab("ROR-S,\nhigh only", at("ROR-S"), (0.80, -0.11), "right", color=MUTED, size=6.8)
    axR.set_xlim(0.52, 0.96); axR.set_ylim(-0.17, 0.47)
    axR.set_xticks([0.6, 0.7, 0.8, 0.9])
    axR.set_xlabel("AUC of ER status alone against the label")
    axR.set_ylabel("Gain in AUC over the\nclinicopathological model")
    axR.grid(color=RULE, linewidth=0.4, alpha=0.6); axR.set_axisbelow(True)
    axR.set_title("b", loc="left", fontsize=11, fontweight="bold", pad=6)
    axR.legend(handles=[Line2D([], [], marker="o", linestyle="", color=MODEL, markersize=6,
                               label="group call in score file"),
                        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                               markeredgecolor=MODEL, markersize=4.4, label="alternative cut")],
               loc="upper right", frameon=False, fontsize=7, handletextpad=0.3,
               borderaxespad=0.1, labelcolor=INK)

    fig.savefig(out / "Fig3.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1_design(out); fig2_flow(out); fig3_mechanism(res, out)
    (out / "FigS1.png").unlink(missing_ok=True)
    from PIL import Image
    for f in sorted(out.glob("Fig*.png")):
        im = Image.open(f).convert("RGB")
        tif = f.with_suffix(".tif")
        im.save(tif, compression="tiff_lzw", dpi=(DPI, DPI))
        w_mm = im.size[0] / DPI * 25.4
        print(f"{f.name} / {tif.name}  {im.size[0]}x{im.size[1]} px, {w_mm:.0f} mm wide at {DPI} dpi")


if __name__ == "__main__":
    main()
