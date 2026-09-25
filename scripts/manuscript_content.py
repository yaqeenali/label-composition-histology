#!/usr/bin/env python3
"""Single source of truth for the manuscript.

Emits ``manuscript.json``: prose, declarations, references, figure captions, and
every table generated from the locked result CSVs.  ``build_docx.js`` and
``build_latex.py`` both render from that file, so the Word and LaTeX submissions
cannot disagree with each other or with the analysis.

Target journal: JCO Clinical Cancer Informatics (ASCO).  Original Report:
structured abstract (PURPOSE / METHODS / RESULTS / CONCLUSION, <= 275 words),
body <= 3,000 words, at most six tables and figures in the main text, single-
blind review, and a Data Supplement.  Everything that does not fit the main
text lives in the supplement, so nothing is lost; the validator enforces the
limits.

Citations are written as ``[[key]]`` or ``[[key1, key2]]`` and numbered by
order of first appearance (main text, then supplement) when the JSON is
emitted, so adding or moving a citation can never leave the numbering stale.

Numbers come from three places under the locked results directory:
  locked_floor_comparison.csv / locked_subgroups.csv   the pre-declared analysis
  continuous_and_calibration.csv, table1.csv           continuous metrics, cohort
  review/*.csv                                         review-response analyses,
                                                       provisional comparison,
                                                       repeated cross-validation

    python scripts/manuscript_content.py --results Results_locked/modal \\
        --radiomics Results_locked/radiomics --out manuscript.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

ORDER = ["GHI (Oncotype)", "ROR-P", "ROR-S", "MammaPrint"]
LONG = {"GHI (Oncotype)": "Oncotype (GHI-RS)", "ROR-P": "PAM50 ROR-P",
        "ROR-S": "PAM50 ROR-S", "MammaPrint": "MammaPrint (NKI70)"}
SHORT = {"GHI (Oncotype)": "Oncotype", "ROR-P": "ROR-P",
         "ROR-S": "ROR-S", "MammaPrint": "MammaPrint"}
CUT = {"GHI (Oncotype)": "High or Intermediate vs Low", "ROR-P": "high or medium vs low",
       "ROR-S": "high or medium vs low", "MammaPrint": "Good vs Bad"}
ALT_CUT = {"GHI (Oncotype)": "High vs Intermediate or Low", "ROR-P": "high vs medium or low",
           "ROR-S": "high vs medium or low"}

SHORTEN = {
    "Post (prior bilateral ovariectomy OR >12 mo since LMP with no prior hysterectomy)": "Postmenopausal",
    "Pre (<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)": "Premenopausal",
    "Peri (6-12 months since last menstrual period)": "Perimenopausal",
    "Indeterminate (neither Pre or Postmenopausal)": "Indeterminate",
    "Infiltrating Ductal Carcinoma": "Invasive ductal",
    "Infiltrating Lobular Carcinoma": "Invasive lobular",
    "Mixed Histology (please specify)": "Mixed",
    "Other, specify": "Other",
}


def pv(x: float) -> str:
    if x < 0.001:
        return "<0.001"
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def signed(x: float, dp: int = 3) -> str:
    return f"−{abs(x):.{dp}f}" if x < 0 else f"+{x:.{dp}f}"


def bound(x: float) -> str:
    return f"−{abs(x):.3f}" if x < 0 else f"{x:.3f}"


def ci_text(s: str) -> str:
    lo, hi = [float(v) for v in s.strip("[]").split(",")]
    return f"{bound(lo)} to {bound(hi)}"


def ci_dash(s: str) -> str:
    lo, hi = [float(v) for v in s.strip("[]").split(",")]
    return f"{bound(lo)}–{bound(hi)}"


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def build_tables(res: Path, rad: Path) -> list:
    cmp_ = pd.read_csv(res / "review" / "comparators.csv").set_index("assay")
    alt = pd.read_csv(res / "review" / "binarisation_sensitivity.csv").set_index("assay")
    sub = pd.read_csv(res / "review" / "subgroups.csv")
    cal = pd.read_csv(res / "continuous_and_calibration.csv").set_index("assay")
    rdx = pd.read_csv(rad / "radiomics_nested_cv.csv")
    prov = pd.read_csv(res / "review" / "provisional_per_target.csv").set_index("assay")
    rcv = pd.read_csv(res / "review" / "repeated_cv_summary.csv").set_index("assay")

    # cohort
    rows1 = []
    for r in list(csv.reader((res / "table1.csv").open()))[1:]:
        raw = r[0].strip()
        rows1.append({"indent": r[0].startswith("    "),
                      "cells": [SHORTEN.get(raw, raw), r[1], r[2], r[3]]})

    # primary endpoint, pre-declared comparators
    rows2 = []
    for a in ORDER:
        r = cmp_.loc[a]
        rows2.append([LONG[a], f"{r.AUC_model:.3f} ({ci_dash(r.model_CI)})",
                      f"{r.AUC_ER_fixed:.3f}", f"{r.AUC_clin_predeclared:.3f}",
                      f"{signed(r.d_vs_clin_predeclared)} ({ci_text(r.CI_clin_predeclared)})",
                      pv(r.p_clin_predeclared), pv(r.p_clin_predeclared_holm)])

    # label composition, primary labels and alternative cuts
    comp = []
    for a in ORDER:
        r = cmp_.loc[a]
        comp.append((r.AUC_ER_fixed, SHORT[a], CUT[a], f"{int(r.positives)} / {int(r.negatives)}",
                     r.kappa_ER, r.d_vs_clin_predeclared, r.p_clin_predeclared, False))
    for a in alt.index:
        r = alt.loc[a]
        comp.append((r.AUC_ER_fixed, SHORT[a], ALT_CUT[a], f"{int(r.positives)} / {int(r.negatives)}",
                     r.kappa_ER, r.d_vs_clin_predeclared, r.p, True))
    rows3 = [[name, cut + (" (alternative)" if is_alt else ""), bal, f"{er:.3f}", f"{k:.3f}",
              f"{signed(d)}; {pv(p)}"]
             for er, name, cut, bal, k, d, p, is_alt in sorted(comp, key=lambda t: t[0])]

    # comparator sensitivity and incremental value, signatures as columns
    R = {a: cmp_.loc[a] for a in ORDER}
    orci = lambda r: r.OR_adjusted_CI.strip("[]").replace(", ", " to ")
    rows4 = [
        {"indent": False, "cells": ["Events per variable"] + [f"{R[a].events_per_variable:.1f}" for a in ORDER]},
        {"indent": False, "cells": ["Clinical model, penalty tuned"] + [""] * 4},
        {"indent": True, "cells": ["AUC, pooled"] + [f"{R[a].AUC_clin_tuned:.3f}" for a in ORDER]},
        {"indent": True, "cells": ["AUC per fold, mean ± SD"] + [f"{R[a].AUC_clin_tuned_foldmean:.3f} ± {R[a].AUC_clin_tuned_foldsd:.3f}" for a in ORDER]},
        {"indent": True, "cells": ["Histology gain (95% CI)"] + [f"{signed(R[a].d_vs_clin_tuned)} ({ci_text(R[a].CI_clin_tuned)})" for a in ORDER]},
        {"indent": True, "cells": ["p; Holm p"] + [f"{pv(R[a].p_clin_tuned)}; {pv(R[a].p_clin_tuned_holm)}" for a in ORDER]},
        {"indent": False, "cells": ["Clinical model + histology score"] + [""] * 4},
        {"indent": True, "cells": ["AUC, pooled"] + [f"{R[a].AUC_stacked:.3f}" for a in ORDER]},
        {"indent": True, "cells": ["AUC per fold, mean ± SD"] + [f"{R[a].AUC_stacked_foldmean:.3f} ± {R[a].AUC_stacked_foldsd:.3f}" for a in ORDER]},
        {"indent": True, "cells": ["Incremental gain (95% CI)"] + [f"{signed(R[a].d_incremental)} ({ci_text(R[a].CI_incremental)})" for a in ORDER]},
        {"indent": True, "cells": ["p; Holm p"] + [f"{pv(R[a].p_incremental)}; {pv(R[a].p_incremental_holm)}" for a in ORDER]},
        {"indent": False, "cells": ["Adjusted association, full data"] + [""] * 4},
        {"indent": True, "cells": ["Odds ratio per SD (95% CI)"] + [f"{R[a].OR_adjusted:.2f} ({orci(R[a])})" for a in ORDER]},
        {"indent": True, "cells": ["Likelihood-ratio p; Holm p"] + [f"{pv(R[a].p_adjusted_LRT)}; {pv(R[a].p_adjusted_LRT_holm)}" for a in ORDER]},
    ]

    # S1 calibration and stability
    rowsS1 = []
    for a in ORDER:
        r = cal.loc[a]
        rowsS1.append([SHORT[a], f"{r.Pearson_r:.3f} ({r.r_CI.strip('[]').replace(', ', '–')})",
                       f"{r.AUC_perfold_mean:.3f} ± {r.AUC_perfold_sd:.3f}",
                       f"{r.calib_slope:.3f}", f"{r.calib_intercept:.2f}"])

    # S2 subgroups
    lbl = {"ALL": "All", "ER+": "ER-positive", "ER+/HER2-": "ER-positive, HER2-negative"}
    sub = sub.set_index("assay").loc[ORDER].reset_index()
    rowsS2, seen = [], None
    for _, r in sub.iterrows():
        head = SHORT[r.assay] if r.assay != seen else ""
        seen = r.assay
        f = lambda v: "not evaluable" if pd.isna(v) else f"{v:.3f}"
        rowsS2.append([head, lbl[r.cohort], f"{int(r.n)} ({int(r.pos)})", f(r.AUC_model),
                       f(r.AUC_clin_predeclared), f(r.AUC_clin_tuned)])

    # S3 radiomic feature arm
    rowsS3 = [[SHORT[r.target], f"{r.AUC_nested:.3f} ({r.CI_low:.3f}–{r.CI_high:.3f})",
               f"{r.AUC_notebook_style:.3f}", signed(r.optimism)] for _, r in rdx.iterrows()]

    # S4 provisional versus locked
    rowsS4 = []
    for a in ORDER:
        p_, l_ = prov.loc[a], cmp_.loc[a]
        rowsS4.append([SHORT[a],
                       f"{p_.AUC_model:.3f}", f"{signed(p_.d_vs_clin)} ({ci_text(p_.CI_clin)}); {pv(p_.p_clin)}",
                       f"{l_.AUC_model:.3f}",
                       f"{signed(l_.d_vs_clin_predeclared)} ({ci_text(l_.CI_clin_predeclared)}); {pv(l_.p_clin_predeclared)}"])

    # S5 repeated cross-validation
    RCVKEY = {"GHI (Oncotype)": "GHI", "ROR-P": "ROR-P", "ROR-S": "ROR-S", "MammaPrint": "MammaPrint"}
    rowsS5 = []
    for a in ORDER:
        r = rcv.loc[RCVKEY[a]]
        rowsS5.append([SHORT[a],
                       f"{r.AUC_hist_mean:.3f} ± {r.AUC_hist_sd:.3f} ({r.AUC_hist_min:.3f}–{r.AUC_hist_max:.3f})",
                       f"{signed(r.margin_mean)} ± {r.margin_sd:.3f} ({signed(r.margin_min)} to {signed(r.margin_max)})",
                       f"{int(r.repeats_margin_positive)} / {int(r.repeats)}"])

    main = [
        {"label": "Table 1", "id": "tab1", "widths": [3400, 1500, 1800, 1800],
         "caption": "Cohort characteristics, overall and by the ROR-P label. Median [IQR] "
                    "for age, n (%) otherwise. HER2 follows ASCO/CAP order: FISH where "
                    "informative, otherwise immunohistochemistry. No significance tests are "
                    "reported, because the groups are defined by a computed score rather "
                    "than by an exposure.",
         "headers": ["Characteristic", "All (n = 82)", "ROR-P low (n = 32)",
                     "ROR-P high (n = 50)"],
         "rows": rows1, "structured": True},
        {"label": "Table 2", "id": "tab2", "widths": [1800, 1950, 850, 950, 2250, 700, 800],
         "caption": "Primary endpoint: histology against chance, against ER status alone "
                    "and against the pre-declared six-variable clinicopathological model "
                    "(L2-penalized logistic regression, C = 1), all on pooled held-out "
                    "predictions. ER alone is a fixed binary predictor oriented a priori "
                    "(ER-positive predicts the low-risk class). n = 81 with complete "
                    "covariates; paired bootstrap, 4,000 resamples; Holm adjustment across "
                    "the four targets. Rows are ordered by the gain column, which runs in "
                    "almost the opposite order to raw AUC.",
         "headers": ["Signature", "Histology AUC (95% CI)", "ER alone", "Clinical",
                     "Gain over clinical (95% CI)", "p", "Holm p"],
         "rows": rows2},
        {"label": "Table 3", "id": "tab3", "widths": [2900, 1600, 1600, 1600, 1600],
         "caption": "Sensitivity to the comparator, and incremental value (n = 81). Events "
                    "per variable: minority class ÷ 7, for a seven-term logistic model. "
                    "Clinical model, penalty tuned: the six-variable model with its penalty "
                    "chosen by inner three-fold cross-validation; the histology gain is "
                    "against that comparator. Clinical model + histology score: the same "
                    "model with the out-of-fold histology score as a seventh variable, "
                    "refitted inside each fold, compared with the tuned clinical model. "
                    "Adjusted association: odds ratio per SD of the histology score after "
                    "adjustment for the six variables, fitted once on all 81 patients, with "
                    "a likelihood-ratio test. Paired bootstrap, 4,000 resamples; Holm "
                    "adjustment across the four targets.",
         "headers": ["", "Oncotype", "ROR-P", "ROR-S", "MammaPrint"],
         "rows": rows4, "structured": True},
        {"label": "Table 4", "id": "tab4", "widths": [1500, 2500, 1200, 1100, 1000, 2000],
         "caption": "Label composition — how much of each binary label is ER status — and "
                    "the histology gain against it, for the four published group calls and "
                    "three alternative cuts of the same scores (n = 81). AUC of ER alone is "
                    "the balanced accuracy of ER status as a classifier of the label, with "
                    "ER-positive assigned to the low-risk class a priori; κ is Cohen's kappa "
                    "for the same agreement. Both are properties of the label, computed "
                    "without any model. Gain is against the pre-declared clinicopathological "
                    "model. “Alternative”: a cut of the same score other than the published "
                    "group call, evaluated on the same predictions. Rows are ordered by the "
                    "AUC of ER alone; the gain falls monotonically along them.",
         "headers": ["Target", "Binary label", "Positive / negative", "AUC of ER alone",
                     "κ with ER", "Gain over clinical; p"],
         "rows": rows3},
    ]
    supp = [
        {"label": "Table S1", "id": "tabS1", "widths": [1700, 2400, 2200, 1600, 1000],
         "caption": "Agreement with the continuous score, per-fold stability and "
                    "calibration (n = 82). Calibration is the regression of observed on "
                    "predicted values: a slope below one means the predictions are spread "
                    "more widely than the observed scores, a slope above one that they are "
                    "compressed toward the mean.",
         "headers": ["Signature", "Pearson r (95% CI)", "AUC per fold (mean ± SD)",
                     "Calibration slope", "Intercept"],
         "rows": rowsS1},
        {"label": "Table S2", "id": "tabS2", "widths": [1300, 2600, 850, 1400, 1650, 1500],
         "caption": "Performance restricted to ER-positive and ER-positive, HER2-negative "
                    "disease. Restriction applies to the evaluation set only; models were "
                    "developed on the full cohort. n (positives) per subgroup. Clinical "
                    "AUCs are given for the pre-declared and the tuned comparator. “Not "
                    "evaluable”: all 56 patients share one MammaPrint label, so no AUC is "
                    "defined; the ER-positive MammaPrint row rests on four negatives.",
         "headers": ["Signature", "Cohort", "n (pos.)", "Histology AUC",
                     "Clinical AUC, pre-declared", "Clinical AUC, tuned"],
         "rows": rowsS2},
        {"label": "Table S3", "id": "tabS3", "widths": [1900, 3000, 2200, 1700],
         "caption": "Pre-extracted radiomic feature arm: nested versus conventional "
                    "non-nested cross-validation, on the same patients and the same folds "
                    "as the histology arm (n = 82). Three of the four nested intervals "
                    "include chance; the ROR-S interval sits just above it. No "
                    "clinicopathological comparator is reported for this arm because the "
                    "nested AUCs are at or near chance, so a head-to-head comparison would "
                    "be uninformative.",
         "headers": ["Target", "Nested AUC (95% CI)", "Non-nested AUC", "Optimism"],
         "rows": rowsS3},
        {"label": "Table S4", "id": "tabS4", "widths": [1300, 1300, 2900, 1300, 2900],
         "caption": "Effect of the design choices: the earlier per-target pipelines against "
                    "the locked design. Provisional: each target on its own cross-validation "
                    "partition with its own tuned hyperparameters and a random patch "
                    "subsample drawn at every access (n = 82, neoadjuvant case still "
                    "included). Locked: one shared partition, one configuration, a fixed "
                    "patch pool (n = 81). The comparator, HER2 rule, bootstrap and seed are "
                    "identical in both columns; only the imaging-side design differs. Gain "
                    "is over the pre-declared clinicopathological model.",
         "headers": ["Signature", "Provisional AUC", "Provisional gain (95% CI); p",
                     "Locked AUC", "Locked gain (95% CI); p"],
         "rows": rowsS4},
        {"label": "Table S5", "id": "tabS5", "widths": [1700, 3100, 3100, 1500],
         "caption": "Repeated cross-validation. The entire locked pipeline — same fixed "
                    "configuration, same model, same fixed patch pools, same fold-honest "
                    "C = 1 clinicopathological comparator — was rerun under five "
                    "independently drawn multi-label stratified five-fold partitions "
                    "(n = 81); the first is the locked partition and reproduces the primary "
                    "numbers exactly. Values are the mean ± SD across the five partitions "
                    "with the range in brackets. This measures how much the point estimate "
                    "moves with the split, not statistical significance, and does not add "
                    "patients: the confidence intervals and multiplicity adjustment of "
                    "Table 2 are unchanged.",
         "headers": ["Signature", "Histology AUC, mean ± SD (range)",
                     "Gain over clinical, mean ± SD (range)", "Draws with gain > 0"],
         "rows": rowsS5},
    ]
    return main, supp


# --------------------------------------------------------------------------- #
# prose
# --------------------------------------------------------------------------- #

JOURNAL = "JCO Clinical Cancer Informatics"

TITLE = ("Label composition explains whether histology outperforms routine "
         "clinicopathological variables: a controlled comparison of four breast cancer "
         "risk signatures")

ABSTRACT = [
    ("PURPOSE",
     "Deep learning on hematoxylin and eosin sections predicts the Oncotype DX recurrence "
     "score beyond routine clinicopathological variables, but whether this extends to "
     "other genomic risk signatures is untested under a common design. "
     "We hypothesized that the margin by which histology outperforms routine variables is "
     "governed by label composition — how much of each binary label estrogen receptor (ER) "
     "status already explains — a property of the target, not of the imaging model."),
    ("METHODS",
     "Recomputed Oncotype DX, MammaPrint, and PAM50 ROR-S and ROR-P signatures were "
     "predicted from UNI patch embeddings by gated-attention multiple-instance learning in "
     "82 patients from The Cancer Genome Atlas, under one shared five-fold partition and "
     "configuration, and compared on identical folds by "
     "paired bootstrap against ER status alone and a six-variable clinicopathological "
     "model. Label composition was quantified as the area under the curve (AUC) of ER "
     "status alone against each label."),
    ("RESULTS",
     "Discrimination against chance ranged from AUC 0.744 to 0.892, but the margin over "
     "the clinicopathological model ran in almost the opposite order: Oncotype +0.280 "
     "(95% CI, 0.133 to 0.435), ROR-P +0.160 (0.012 to 0.300), ROR-S +0.124 (0.013 to "
     "0.246), and MammaPrint −0.042 (−0.134 to 0.034); only Oncotype survived multiplicity "
     "adjustment. The ordering was that of the ER-alone AUC (0.582, 0.610, 0.672, and "
     "0.850), as the score definitions predict; the margin fell monotonically with that "
     "AUC across seven label definitions and was stable across five partition draws. A pre-extracted radiomic feature arm on the same folds "
     "showed non-nested selection inflating AUC by up to 0.121."),
    ("CONCLUSION",
     "Whether histology outperforms routine clinicopathological variables is predictable "
     "from the labels alone. Studies predicting genomic signatures from images should "
     "report label composition and benchmark against clinicopathological variables, not "
     "chance."),
]

KEYWORDS = ["Computational pathology", "Multiple instance learning", "Breast cancer",
            "Gene expression signatures", "Model benchmarking", "Cross-validation"]

INTRO = [
    "Multigene expression assays guide adjuvant treatment in early-stage breast cancer. The 21-gene recurrence score (Oncotype DX) [[paik2004]] and the 70-gene "
    "signature (MammaPrint) [[vantveer2002]] identify patients in whom chemotherapy can be "
    "safely omitted, as shown prospectively by TAILORx [[sparano2018]] and MINDACT "
    "[[cardoso2016]], and the PAM50 risk-of-recurrence scores provide subtype-based and "
    "proliferation-weighted estimates [[parker2009]]. Cost and turnaround time limit "
    "access.",

    "Deep learning on hematoxylin and eosin (H&E) whole-slide images recovers hormone "
    "receptor status, HER2 "
    "status and intrinsic subtype [[couture2018, naik2020, labarbera2020]]. "
    "For Oncotype DX the question is largely settled: Goyal et al "
    "reported an image model at AUC 0.87 against a clinicopathological model's 0.83 in 950 "
    "patients with external validation in 405 [[goyal2024]]; Boehm et al, in 6,172 "
    "patients, reported AUC 0.89 for a multimodal model against 0.73 for a "
    "clinicopathological nomogram [[boehm2025]]; and Shamai et al, with a multimodal model "
    "fine-tuned on the TAILORx trial and validated in six external cohorts, reported AUC "
    "0.898 for high genomic risk [[shamai2026]]. Morphology carries recurrence-score "
    "information routine variables do not.",

    "Whether this generalizes across signatures is not known, and the construction of the "
    "scores suggests that it should not. The recurrence score weights a five-gene "
    "proliferation group most heavily and enters a four-gene ER group with a negative "
    "coefficient [[paik2004]]; in a predominantly ER-positive cohort, ER-group expression "
    "varies little between patients, so the variation that decides the High or "
    "Intermediate versus Low call is largely proliferative and only weakly tied to ER "
    "status. The 70-gene signature classifies by correlation with a good-prognosis profile "
    "that ER-negative tumors rarely meet — in MINDACT, hormone receptor–negative tumors "
    "were almost uniformly genomically high risk [[cardoso2016]] — so its binary label in "
    "a mixed cohort is close to a restatement of ER status. The PAM50 scores lie between: "
    "the subtype-only score (ROR-S) draws on correlation with the basal-like and "
    "HER2-enriched centroids, which track ER-negative disease, whereas the "
    "proliferation-weighted score (ROR-P) adds within-luminal variation that is "
    "independent of ER status [[parker2009]]. Because ER status is in every pathology "
    "report and visible on H&E, a label that nearly restates it offers an image model "
    "little to learn that a clinician does not already have, however high the AUC. The expected ordering of the histology margin "
    "over routine variables therefore follows from the score definitions alone — smallest "
    "for MammaPrint, intermediate for ROR-S, largest for ROR-P and Oncotype — whereas its "
    "magnitude does not.",

    "Testing this requires a constant design, and two common choices prevent it. "
    "Selecting hyperparameters separately for each target while held-out "
    "performance is visible turns the reported figure into the maximum over a search "
    "[[varma2006, cawley2010]], and evaluating targets on different partitions of the same "
    "patients confounds target with split; neither is detectable from a methods "
    "section. A second modality is available for this cohort: Li et al "
    "extracted 36 dynamic contrast-enhanced MRI radiomic features for the same patients "
    "[[li2016]]. We include them not to test whether MRI predicts the signatures but as a "
    "methodological control — to measure, on the same folds, how much of an apparent "
    "radiomics signal is an artifact of the non-nested feature selection that such "
    "pipelines commonly use [[tareke2025]].",

    "We therefore compared four recomputed signatures under one design — "
    "one shared partition, one pre-fixed configuration, one model — benchmarked each against ER status alone and a "
    "clinicopathological model, quantified label composition, tested whether histology "
    "adds to the clinical variables and measured the cost of non-nested selection in the "
    "radiomic feature arm.",
]

METHODS = [
    ("Study design and cohort", [
        "This retrospective model-development study uses public, de-identified data and "
        "required no ethical approval; reporting follows TRIPOD+AI [[collins2024]] "
        "(checklist in the Data Supplement). The cohort is the breast radiogenomics subset "
        "of The Cancer Genome Atlas [[tcga2012]] assembled by Li et al [[li2016]], for "
        "which recomputed signature scores have been published. Of 100 patients with recomputed scores, 83 also had a diagnostic "
        "whole-slide image with extracted features; one who had received neoadjuvant "
        "chemotherapy was excluded before any partition was drawn, leaving 82 (Fig S1). "
        "Follow-up comprises three recurrences and one death over a median of 42 months, "
        "so no survival or chemotherapy-benefit analysis is attempted.",
    ]),
    ("Prediction targets", [
        "The targets are genefu-style recomputations from expression data [[gendoo2016]] "
        "of the 21-gene recurrence score (GHI-RS), the 70-gene signature correlation "
        "(NKI70), and the PAM50 ROR-S and ROR-P scores. Binary labels follow the published "
        "group calls: GHI High or Intermediate versus Low; NKI70 Good versus Bad; ROR high "
        "or medium versus low. These are research recomputations, not commercial assay "
        "results or patient outcomes; every claim is at that level.",
    ]),
    ("Slide processing and model", [
        "Whole-slide images were segmented, tiled into non-overlapping 256-pixel patches "
        "at 20× and encoded into 1,024-dimensional embeddings by the UNI foundation model "
        "[[chen2024]] using the Trident toolkit [[zhang2025]]; this step was carried out "
        "beforehand as preparatory work in our group; no image processing method was "
        "developed here, and everything downstream was performed in this study. Each slide "
        "(median 19,376 patches, range 3,334 to 32,498) was reduced once, with a recorded "
        "seed, to a fixed pool of 512 patches, equalizing bag size and making evaluation "
        "deterministic (Fig 1a). A gated attention-based multiple-instance model "
        "[[ilse2018]] pools patch embeddings into a slide representation from which a "
        "regression head predicts the continuous score (Fig 1b); training details are in "
        "the Data Supplement.",
    ]),
    ("Partition and configuration", [
        "A single five-fold partition was drawn by multi-label stratification "
        "[[sechidis2011]], balanced on all four binary targets and on ER status, and used "
        "identically by every target (Fig 1c). An inner three-fold split, drawn the same way "
        "within each training set, supplied the validation fold: each outer fold trained on "
        "42 to 44 patients, selected its epoch on 21 to 23 and was "
        "scored on 15 to 18 never seen during training or selection. The configuration was fixed before the confirmatory run "
        "and is the same for every target; it is the setting shared by the majority of four "
        "earlier per-target pipelines, which had been tuned with held-out performance "
        "visible, so the rule removes dependence on this study's measured performance and "
        "on target-specific tuning, not every prior look at the data.",
    ]),
    ("Comparators", [
        "Two comparators were evaluated on the same folds. ER status alone is a single "
        "fixed binary predictor, oriented a priori so that ER-positive disease predicts the "
        "low-risk class, needing no fitting. The second is a six-variable logistic "
        "regression on age, ER, progesterone receptor and HER2 status, nodal status and "
        "ordinal AJCC stage — the routine variables available here; grade and Ki-67 are not "
        "in the public extract — L2-penalized at C = 1, "
        "pre-declared, trained on the remaining folds and used to predict each held-out "
        "fold. HER2 followed ASCO/CAP order [[wolff2018]]; one patient with uninformative "
        "HER2 testing is excluded from comparisons (n = 81). As a sensitivity analysis the "
        "penalty was instead chosen by inner cross-validation, and the apparent (in-sample) "
        "AUC was recorded. To test whether histology adds to rather than matches the "
        "clinical variables, the out-of-fold histology score was entered as a seventh "
        "variable into the tuned model, refitted inside each fold, and its association was "
        "also tested on the full data by likelihood-ratio test (odds ratio per SD); events "
        "per variable are reported for every fitted comparator [[peduzzi1996]].",
    ]),
    ("Pre-extracted radiomic feature arm", [
        "The 36 DCE-MRI radiomic features of Li et al [[li2016]] were used as published; no "
        "image analysis was performed. Median imputation, standardization, univariate F-test "
        "filtering and L2-penalized logistic regression were fitted inside each training "
        "fold, with the number of features and the penalty chosen by inner "
        "cross-validation; the non-nested procedure, choosing them on all the data before "
        "the same cross-validation, was reproduced to measure its optimism. No "
        "clinicopathological comparator is reported for this arm, because the nested AUCs "
        "are at or near chance.",
    ]),
    ("Statistical analysis", [
        "Label orientation was fixed a priori from the sign of the correlation between the "
        "continuous target and its binary label. Discrimination is reported as the AUC on "
        "pooled held-out predictions, with per-fold means and SDs alongside; pooling ranks "
        "every patient against every other, so a comparator refitted per fold can score "
        "below chance pooled but not per fold [[forman2010]]. Confidence intervals are percentile intervals from 4,000 "
        "bootstrap resamples; differences between a model and a comparator use a paired "
        "bootstrap with a two-sided empirical p value, adjusted across the four targets by "
        "Holm's procedure [[holm1979]]. Label composition is the AUC of ER status alone "
        "against each label (the balanced accuracy of ER as a classifier of it), with "
        "Cohen's κ alongside, both model-free; proportion agreement is not used because "
        "prevalence dominates it. Three alternative binarization cuts of "
        "the same scores were evaluated on the same predictions. The whole pipeline was "
        "rerun under five independently drawn partitions, as in our earlier multi-seed "
        "evaluations [[ali2026]], to measure how much the estimates move with the split. "
        "The minimum detectable difference between two AUCs on the same patients was "
        "computed from the Hanley–McNeil standard error [[hanley1982, hanley1983]] "
        "(assumptions in the Data Supplement). Analyses used Python 3.11, PyTorch 2.14, "
        "scikit-learn 1.8 and statsmodels 0.15.",
    ]),
]

RESULTS = [
    ("Cohort", [
        "The 82 analyzed patients had a median age of 53 years (IQR 45 to 63); 71 (87%) were "
        "ER-positive, 65 (79%) HER2-negative and 65 (79%) PR-positive; 41 (50%) were "
        "node-positive; 71 (87%) had invasive ductal carcinoma; and 53 (65%) were PAM50 "
        "luminal A (Table 1).",
    ]),
    ("Discrimination against chance", [
        "All four models discriminated against chance: pooled held-out AUC was 0.892 "
        "for MammaPrint, 0.834 for ROR-S, 0.815 for ROR-P and 0.744 for Oncotype, with "
        "Pearson correlations to the continuous score of 0.766, 0.695, 0.589 and 0.449 "
        "(Table 2; calibration and per-fold stability in Table S1).",
    ]),
    ("Discrimination against routine clinicopathological variables", [
        "Against the pre-declared clinical model the ordering nearly reverses "
        "(Table 2, Fig 2a). That model alone reached out-of-fold AUC 0.464 for Oncotype, "
        "0.655 for ROR-P, 0.710 for ROR-S and 0.934 for MammaPrint. Oncotype gains the "
        "most, +0.280 AUC (95% CI, 0.133 to 0.435; p < 0.001), followed by ROR-P at +0.160 "
        "(0.012 to 0.300; p = 0.036) and ROR-S at +0.124 (0.013 to 0.246; p = 0.028). After "
        "Holm adjustment only the Oncotype margin remains significant (p = 0.001; ROR-P and "
        "ROR-S p = 0.084). MammaPrint gains nothing: −0.042 (−0.134 to 0.034; p = 0.313), so "
        "the clinicopathological model numerically exceeded the imaging model and the "
        "interval excludes any gain larger than 0.034. With the penalty tuned by inner "
        "cross-validation (Table 3, Fig 2a) the margins were +0.249 (0.090 to 0.416; "
        "p = 0.002), +0.173 (0.025 to 0.314; p = 0.020), +0.088 (−0.012 to 0.194; p = 0.090) "
        "and −0.042: the same conclusions under either comparator.",
        "The Oncotype comparator is below chance under both penalties (per-fold 0.427 ± "
        "0.269 when tuned) although its apparent AUC on all 81 patients is 0.721: six "
        "variables estimated from 14 minority-class cases fit the training folds and carry "
        "nothing to the held-out ones. The routine variables contain no transferable "
        "information about the Oncotype label here; they do not predict it inversely.",
    ]),
    ("Incremental value over the clinicopathological model", [
        "Outperforming the clinical variables is not the same as adding to them (Table 3). "
        "For ROR-P the combined model improved on the clinical model by +0.138 AUC (0.014 "
        "to 0.262; p = 0.029; Holm p = 0.114) and the histology score remained strongly "
        "associated with the label after adjustment (odds ratio 3.35 per SD, 1.53 to 7.32; "
        "p < 0.001). For ROR-S the adjusted association was of the same size (3.34, 1.25 to "
        "8.88; p = 0.007) but the cross-validated increment was small (+0.043, −0.020 to "
        "0.114; p = 0.198). For Oncotype and MammaPrint the question is not answerable: with "
        "14 minority-class cases a seven-variable model has two "
        "events per variable, the Oncotype combined model was no better than the clinical "
        "model (−0.020, −0.161 to 0.133) with an uninformative adjusted odds ratio (1.43, "
        "0.66 to 3.07; p = 0.365), and the MammaPrint combined model matched a clinical model that "
        "already reaches AUC 0.934 (−0.016, −0.054 to 0.018). For those two targets only the "
        "head-to-head comparison is supported.",
    ]),
    ("Label composition", [
        "The explanation for the ordering lies in the labels (Table 4, Fig 2b). ER status "
        "alone, as a fixed predictor, reaches AUC 0.850 against the MammaPrint label "
        "(κ = 0.764), 0.672 against ROR-S (κ = 0.388), 0.610 against ROR-P (κ = 0.178) and "
        "0.582 against Oncotype (κ = 0.064) — the order anticipated from the score "
        "definitions — and the histology margin rises in exactly that order, from −0.042 "
        "to +0.280.",
        "Because the published group calls are arbitrary cuts, three alternative cuts were "
        "evaluated on the same predictions. Moving the Oncotype cut "
        "to High versus Intermediate or Low leaves ER weakly informative (AUC 0.590, "
        "κ = 0.098) and the margin large (+0.192, 0.058 to 0.340; p = 0.005). Moving the "
        "ROR cuts to high versus medium or low makes both labels far more ER-like — ER "
        "alone reaches 0.740 against ROR-P (κ = 0.511) and 0.826 against ROR-S "
        "(κ = 0.726) — and the margins collapse to +0.036 (−0.057 to 0.145; p = 0.493) and "
        "−0.030 (−0.094 to 0.025; p = 0.304) while raw discrimination against those labels "
        "rises to 0.897 and 0.928. Across all seven label definitions the margin falls "
        "monotonically as the AUC of ER alone rises. Seven points from one cohort do not "
        "establish a law; what they show is that label composition predicted whether "
        "histology outperformed routine variables here, and that changing one signature's "
        "cut moved it across that line.",
    ]),
    ("Within ER-positive disease, robustness, and the radiomic feature arm", [
        "Within ER-positive, HER2-negative disease (n = 56) the clinicopathological model "
        "for Oncotype falls to AUC 0.279 (0.297 with tuned penalty) while the histology "
        "model retains 0.673; ROR-P is stable under restriction (0.815 to 0.756), ROR-S "
        "degrades (0.834 to 0.673), and MammaPrint is not evaluable because all 56 patients "
        "share one label (Table S2). Rerunning the whole pipeline under five independent "
        "partitions (Table S5) left the margin stable — +0.213 ± 0.051 for Oncotype, "
        "+0.199 ± 0.045 for ROR-P, +0.111 ± 0.046 for ROR-S and −0.006 ± 0.048 for "
        "MammaPrint; the first three positive in all five draws, the last straddling zero "
        "(positive in two of five) — so the ordering and the MammaPrint null are properties "
        "of the design, not of one split. In the radiomic feature arm, nested "
        "cross-validation gave AUC 0.588 to 0.637 with three of four intervals including "
        "chance, and the non-nested procedure inflated it by +0.022 to +0.121 (Table S3). "
        "The earlier per-target pipelines, with their own partitions and tuning, had "
        "ordered the targets differently under the same comparator (Table S4); the shared "
        "design reordered them without new data or a better model.",
    ]),
]

DISCUSSION = [
    ("", [
        "Across four recomputed genomic risk signatures evaluated under one design, "
        "whether histology outperformed six routine clinicopathological variables varied "
        "from a margin of +0.280 AUC to none at all, and the variation tracked label "
        "composition, not the imaging model. Against the MammaPrint "
        "label, which ER status alone predicts at AUC 0.850 and which is single-class "
        "within ER-positive, HER2-negative disease, a model achieving AUC 0.892 was still "
        "numerically behind six routine variables — a distinction invisible when "
        "performance is reported against chance. The direction was predictable from the "
        "score formulas, and the observed ordering of ER-alone AUC and of the histology "
        "margin is the predicted one; the magnitude was not predictable and is what the "
        "study measures.",

        "Two questions are conflated easily. Whether an image model reaches the "
        "information in routine variables is a head-to-head comparison, which we "
        "pre-specified. Whether it adds to them needs a combined model, estimable at "
        "n = 81 only where the minority class allows: for the PAM50 scores the histology score stayed "
        "strongly associated after adjustment and for ROR-P the combined model also "
        "improved discrimination, whereas for Oncotype and MammaPrint, with 14 "
        "minority-class cases, neither the clinical nor a combined model can be estimated "
        "reliably, and our claims for those two targets are head-to-head claims only.",

        "This complements the large Oncotype studies [[goyal2024, boehm2025, shamai2026]], "
        "which report a clear increment for the one signature whose label retains most "
        "variation within ER-positive disease, and suggests where the approach will not "
        "transfer. Three costless recommendations follow: report the AUC of receptor "
        "status alone against the binarized label, or κ, because it bounds what any model "
        "can contribute; benchmark against a clinicopathological model, not chance, "
        "and say whether the image model was compared with it or added to it; and when "
        "several targets are compared, share one partition and one configuration and nest "
        "all selection inside the training folds. The magnitudes involved are not a "
        "technicality: per-target partitioning and tuning reordered the four targets "
        "(Table S4), and in the radiomic feature arm — included as a methodological "
        "control, not as a competitor to histology — non-nested selection accounted for "
        "most of the apparent signal, showing that the selection bias long described in "
        "the literature [[varma2006, cawley2010]] and recently documented for pathology "
        "foundation models [[akebli2026]] is directly observable on the same data, at the "
        "0.03 to 0.12 AUC scale at which increments are commonly reported.",
    ]),
    ("Limitations", [
        "The targets are research recomputations of signature scores, not commercial assay "
        "results and not patient outcomes; with three recurrences and one death, no "
        "prognostic or chemotherapy-benefit claim is made; clinical utility rests on the "
        "cohorts cited above. The cohort is small: the minimum detectable "
        "difference between two AUCs on the same patients is 0.135 to 0.154 depending on "
        "class balance and 0.165 to 0.221 in the ER-positive, HER2-negative subgroup, so "
        "only the Oncotype margin survives adjustment, the ROR-P and ROR-S margins are "
        "suggestive until replicated, and the confidence intervals rest on one "
        "pre-declared partition — stabilized but not narrowed by the five-partition repeat. "
        "The comparator lacks grade and Ki-67, so the Oncotype margin is against a weaker "
        "comparator than a full pathology report. Other limitations "
        "are the single cohort without external validation, a fixed 512-patch sample of "
        "each slide, a configuration inherited from pipelines that had seen these data, "
        "and the over-representation of ER-positive disease. The pathology-only arm can be "
        "extended to roughly 900 TCGA cases with no new data collection, at which point "
        "the minimum detectable difference falls to about 0.04 to 0.05 and the incremental "
        "question becomes answerable for every target.",
        "In conclusion, whether deep learning on H&E outperforms routine "
        "clinicopathological variables differs sharply between genomic risk signatures, "
        "and the difference was governed by label composition. Discrimination against "
        "chance is not an informative endpoint; studies should report label composition, "
        "benchmark against a clinicopathological model and say whether the image model was "
        "compared with it or added to it, and nest all selection inside the training folds.",
    ]),
]

SUPP_METHODS = [
    ("S1. Model and training", [
        "Patch embeddings pass through a shared linear projection and a gated attention "
        "module producing one weight per patch; the attention-weighted mean is passed to a "
        "regression head predicting the continuous signature score. Training used AdamW at "
        "learning rate 1×10⁻⁴, weight decay 1×10⁻⁴, gradient-norm clipping at 1.0, batch "
        "size one, Huber loss and 20 epochs, with the reported epoch chosen by validation "
        "Pearson correlation; the fixed configuration is hidden width 1,024 and dropout "
        "0.25. Patch standardization and target scaling were fitted on training folds "
        "only. The four earlier per-target pipelines from which this configuration was "
        "taken as the modal setting differed in width, dropout, weight decay, loss and "
        "selection rule and used a random patch subsample drawn at every access; the "
        "fixed 512-patch pool removes that divergence and the per-access randomness.",
    ]),
    ("S2. Partition", [
        "Multi-label stratification places patients one at a time, rarest label first, "
        "into whichever fold is furthest below its target count for that label, so every "
        "fold carries a similar positive rate for all four binary targets and for ER "
        "status simultaneously [[sechidis2011]]; with four targets plus ER, an ordinary "
        "stratified split produces folds in which one target has a single-class test set. "
        "The inner three-fold split for epoch selection was drawn by the same procedure "
        "inside each training set.",
    ]),
    ("S3. Comparator sensitivity and incremental analyses", [
        "The tuned comparator chose the L2 penalty from C ∈ {0.01, 0.03, 0.1, 0.3, 1} by "
        "inner three-fold cross-validation within each training fold. The stacked model "
        "entered the out-of-fold histology score, standardized, as a seventh variable into "
        "the same tuned logistic model, refitted inside each fold, and was compared with "
        "the tuned clinical model by paired bootstrap on the same folds. Because that model "
        "must be estimated from few minority-class cases, the association was also tested "
        "on all 81 patients by a likelihood-ratio test for the histology score added to the "
        "six variables, reported as an odds ratio per standard deviation. Events per "
        "variable are the minority-class count divided by seven. The tuned penalty changed the "
        "Oncotype comparator from 0.464 to 0.495 and the ROR-S comparator from 0.710 to "
        "0.746, and left ROR-P and MammaPrint essentially unchanged. For Oncotype the "
        "unadjusted association was present (odds ratio 1.93 per SD, 1.03 to 3.60; "
        "p = 0.033) but the adjusted one uninformative in either direction.",
    ]),
    ("S4. Minimum detectable difference", [
        "The minimum detectable difference between two AUCs measured on the same patients "
        "was computed from the Hanley–McNeil standard error [[hanley1982, hanley1983]] "
        "assuming both AUCs near 0.80, a correlation of 0.5 between them, two-sided "
        "α = 0.05 and 80% power, at the observed class balance of each target: 0.135 to "
        "0.154 in the full cohort, 0.165 to 0.221 in the ER-positive, HER2-negative "
        "subgroup, and about 0.04 to 0.05 at n ≈ 900 with the same class balance.",
    ]),
    ("S5. Repeated cross-validation", [
        "The entire locked pipeline — the same fixed configuration, model, fixed patch "
        "pools and fold-honest C = 1 clinicopathological comparator — was rerun under five "
        "independently drawn multi-label stratified five-fold partitions, changing only "
        "the seed that draws the outer partition and the inner epoch-selection split. The "
        "first seed is the locked seed, so repeat 0 reproduces the primary numbers exactly "
        "and serves as a check on the harness. This measures how much the point estimates "
        "move with the split; it adds no patients, so the confidence intervals and "
        "multiplicity adjustment of the primary analysis are unchanged.",
    ]),
]

SUPP_RESULTS = [
    ("S6. Calibration and stability", [
        "Calibration slopes for the continuous predictions were 0.658 (Oncotype), 0.815 "
        "(ROR-P), 0.983 (ROR-S) and 1.124 (MammaPrint) (Table S1). A slope below one means "
        "the predictions are spread more widely than the observed scores warrant — the "
        "over-dispersion expected of a model fitted to a small sample — so the Oncotype "
        "predictions are the least well calibrated and the MammaPrint predictions mildly "
        "compressed. Per-fold variability was modest for MammaPrint (AUC 0.955 ± 0.037) "
        "and larger for Oncotype (0.749 ± 0.105); per-fold SDs of 0.037 to 0.105 show that "
        "a single five-fold draw is a coarse instrument at this sample size. The MammaPrint "
        "per-fold mean sits above its pooled value because the predicted scale shifts "
        "between folds, a within-fold ranking that pooling penalizes.",
    ]),
    ("S7. Performance within ER-positive disease", [
        "Restriction to ER-positive, HER2-negative disease (n = 56) — the population in "
        "which Oncotype DX and the PAM50 scores are principally used, and most of the "
        "MammaPrint indication — applies to the evaluation set only; the models were "
        "developed on the full cohort, so these figures are not what a model developed "
        "within the indication would achieve (Table S2). This is the group in which the "
        "six variables carry least information, since two of them are constant within it, "
        "and it is where the morphological signal does the most work. Within ER-positive "
        "disease as a whole the MammaPrint AUC rests on four negatives.",
    ]),
    ("S8. Pre-extracted radiomic feature arm", [
        "Nested cross-validation gave AUC 0.593 for Oncotype (95% CI, 0.429 to 0.752), 0.628 "
        "for MammaPrint (0.454 to 0.790), 0.588 for ROR-P (0.454 to 0.721) and 0.637 for "
        "ROR-S (0.510 to 0.760); the non-nested procedure returned 0.656, 0.749, 0.610 and "
        "0.692 on the same data, an optimism of +0.022 to +0.121 AUC (Table S3). Three of "
        "the four nested intervals include chance; the ROR-S interval sits just above it. "
        "The optimism was largest for MammaPrint. On this cohort the arm has at most a marginal "
        "signal, and most of the signal the conventional procedure reports is an artifact "
        "of selecting and scoring on the same data. The arm is a methodological control: "
        "its role is to show that the selection-bias magnitude described in the literature "
        "is directly observable on the same patients and folds as the histology analysis.",
    ]),
    ("S9. Effect of the design choices", [
        "Under the earlier per-target pipelines, with per-target partitions, per-target "
        "tuned hyperparameters and a random patch subsample at every access, the same "
        "comparator gave margins of +0.155 for Oncotype (−0.005 to 0.314; p = 0.057), "
        "+0.173 for ROR-P, +0.066 for ROR-S (−0.075 to 0.205; p = 0.349) and +0.014 for "
        "MammaPrint (−0.055 to 0.082; p = 0.673): Oncotype was a marginal result, ROR-S was "
        "null and MammaPrint showed a small positive margin. Under one shared partition, "
        "one configuration and a fixed patch pool, Oncotype became the strongest result, "
        "ROR-S gained a nominally significant margin, ROR-P was unchanged and MammaPrint's "
        "margin became negative (Table S4). Separately, nesting the radiomic feature "
        "selection removed most of that arm's apparent performance.",
    ]),
    ("S10. Repeated cross-validation", [
        "Across the five partitions the histology AUC and the margin over the "
        "clinicopathological model are given as mean ± SD with the range (Table S5). The "
        "Oncotype, ROR-P and ROR-S margins were positive in all five draws; the MammaPrint "
        "margin straddled zero, positive in two of five.",
    ]),
]

DECLARATIONS = [
    ("Support", "[to complete]"),
    ("Authors' disclosures of potential conflicts of interest", "[to complete]"),
    ("Author contributions", "Conception and design: Yaqeen Ali, Johannes Gregori. Data "
                             "processing, analysis and software: Yaqeen Ali. Manuscript "
                             "writing: Yaqeen Ali. Critical revision and supervision: "
                             "Johannes Gregori. Final approval: all authors."),
    ("Acknowledgments", "We thank Vincenzo Della Mea (University of Udine) for providing the "
                        "curated clinical data file for this cohort."),
    ("Ethics", "Not required; the analysis uses public, de-identified data from The Cancer "
               "Genome Atlas."),
    ("Data sharing statement", "Whole-slide images and clinical data are available through "
                               "the NCI Genomic Data Commons and MRI data through The Cancer "
                               "Imaging Archive; recomputed signature scores derive from "
                               "published expression data [[li2016]]. All analysis code, the "
                               "locked run manifest (seeds, configuration, package versions) "
                               "and the derived result tables are available at [repository "
                               "URL]."),
]

# Structured references, rendered in JCO style by ``render_reference``.
REFS = {
    "paik2004": dict(a=["Paik S", "Shak S", "Tang G"], etal=True,
                     t="A multigene assay to predict recurrence of tamoxifen-treated, node-negative breast cancer",
                     j="N Engl J Med", v="351", p="2817-2826", y="2004"),
    "vantveer2002": dict(a=["van 't Veer LJ", "Dai H", "van de Vijver MJ"], etal=True,
                         t="Gene expression profiling predicts clinical outcome of breast cancer",
                         j="Nature", v="415", p="530-536", y="2002"),
    "sparano2018": dict(a=["Sparano JA", "Gray RJ", "Makower DF"], etal=True,
                        t="Adjuvant chemotherapy guided by a 21-gene expression assay in breast cancer",
                        j="N Engl J Med", v="379", p="111-121", y="2018"),
    "cardoso2016": dict(a=["Cardoso F", "van 't Veer LJ", "Bogaerts J"], etal=True,
                        t="70-gene signature as an aid to treatment decisions in early-stage breast cancer",
                        j="N Engl J Med", v="375", p="717-729", y="2016"),
    "parker2009": dict(a=["Parker JS", "Mullins M", "Cheang MCU"], etal=True,
                       t="Supervised risk predictor of breast cancer based on intrinsic subtypes",
                       j="J Clin Oncol", v="27", p="1160-1167", y="2009"),
    "couture2018": dict(a=["Couture HD", "Williams LA", "Geradts J"], etal=True,
                        t="Image analysis with deep learning to predict breast cancer grade, ER status, histologic subtype, and intrinsic subtype",
                        j="NPJ Breast Cancer", v="4", p="30", y="2018"),
    "naik2020": dict(a=["Naik N", "Madani A", "Esteva A"], etal=True,
                     t="Deep learning-enabled breast cancer hormonal receptor status determination from base-level H&E stains",
                     j="Nat Commun", v="11", p="5727", y="2020"),
    "labarbera2020": dict(a=["La Barbera D", "Polónia A", "Roitero K"], etal=True,
                          t="Detection of HER2 from haematoxylin-eosin slides through a cascade of deep learning classifiers via multi-instance learning",
                          j="J Imaging", v="6", p="82", y="2020"),
    "goyal2024": dict(a=["Goyal M", "Marotti JD", "Workman AA"], etal=True,
                      t="A multi-model approach integrating whole-slide imaging and clinicopathologic features to predict breast cancer recurrence risk",
                      j="NPJ Breast Cancer", v="10", p="93", y="2024"),
    "boehm2025": dict(a=["Boehm KM", "El Nahhas OSM", "Marra A"], etal=True,
                      t="Multimodal histopathologic models stratify hormone receptor-positive early breast cancer",
                      j="Nat Commun", v="16", p="2106", y="2025"),
    "shamai2026": dict(a=["Shamai G", "Cohen S", "Binenbaum Y"], etal=True,
                       t="Deep learning on histopathological images to predict breast cancer recurrence risk and chemotherapy benefit: a multicentre, model development and validation study",
                       j="Lancet Oncol", doi="10.1016/S1470-2045(25)00727-2", y="2026"),
    "varma2006": dict(a=["Varma S", "Simon R"], etal=False,
                      t="Bias in error estimation when using cross-validation for model selection",
                      j="BMC Bioinformatics", v="7", p="91", y="2006"),
    "cawley2010": dict(a=["Cawley GC", "Talbot NLC"], etal=False,
                       t="On over-fitting in model selection and subsequent selection bias in performance evaluation",
                       j="J Mach Learn Res", v="11", p="2079-2107", y="2010"),
    "li2016": dict(a=["Li H", "Zhu Y", "Burnside ES"], etal=True,
                   t="MR imaging radiomics signatures for predicting the risk of breast cancer recurrence as given by research versions of MammaPrint, Oncotype DX, and PAM50 gene assays",
                   j="Radiology", v="281", p="382-391", y="2016"),
    "tareke2025": dict(a=["Tareke TW", "Payan N", "Cochet A"], etal=True,
                       t="Automated radiomics analysis from multi-modal image segmentation for predicting triple negative breast cancer",
                       j="Annu Int Conf IEEE Eng Med Biol Soc", doi="10.1109/EMBC58623.2025.11252611", y="2025"),
    "collins2024": dict(a=["Collins GS", "Moons KGM", "Dhiman P"], etal=True,
                        t="TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods",
                        j="BMJ", v="385", p="e078378", y="2024"),
    "tcga2012": dict(a=["The Cancer Genome Atlas Network"], etal=False,
                     t="Comprehensive molecular portraits of human breast tumours",
                     j="Nature", v="490", p="61-70", y="2012"),
    "gendoo2016": dict(a=["Gendoo DMA", "Ratanasirigulchai N", "Schröder MS"], etal=True,
                       t="Genefu: An R/Bioconductor package for computation of gene expression-based signatures in breast cancer",
                       j="Bioinformatics", v="32", p="1097-1099", y="2016"),
    "chen2024": dict(a=["Chen RJ", "Ding T", "Lu MY"], etal=True,
                     t="Towards a general-purpose foundation model for computational pathology",
                     j="Nat Med", v="30", p="850-862", y="2024"),
    "zhang2025": dict(a=["Zhang A", "Jaume G", "Vaidya A"], etal=True,
                      t="Accelerating data processing and benchmarking of AI models for pathology",
                      j="arXiv", arxiv="2502.06750", y="2025"),
    "ilse2018": dict(a=["Ilse M", "Tomczak JM", "Welling M"], etal=False,
                     t="Attention-based deep multiple instance learning",
                     j="Proc Mach Learn Res", v="80", p="2127-2136", y="2018"),
    "sechidis2011": dict(a=["Sechidis K", "Tsoumakas G", "Vlahavas I"], etal=False,
                         t="On the stratification of multi-label data",
                         j="Lect Notes Comput Sci", v="6913", p="145-158", y="2011"),
    "wolff2018": dict(a=["Wolff AC", "Hammond MEH", "Allison KH"], etal=True,
                      t="Human epidermal growth factor receptor 2 testing in breast cancer: American Society of Clinical Oncology/College of American Pathologists clinical practice guideline focused update",
                      j="J Clin Oncol", v="36", p="2105-2122", y="2018"),
    "peduzzi1996": dict(a=["Peduzzi P", "Concato J", "Kemper E"], etal=True,
                        t="A simulation study of the number of events per variable in logistic regression analysis",
                        j="J Clin Epidemiol", v="49", p="1373-1379", y="1996"),
    "forman2010": dict(a=["Forman G", "Scholz M"], etal=False,
                       t="Apples-to-apples in cross-validation studies: Pitfalls in classifier performance measurement",
                       j="ACM SIGKDD Explor Newsl", v="12", p="49-57", y="2010"),
    "holm1979": dict(a=["Holm S"], etal=False,
                     t="A simple sequentially rejective multiple test procedure",
                     j="Scand J Stat", v="6", p="65-70", y="1979"),
    "ali2026": dict(a=["Ali Y", "Müller J", "Weinmann A"], etal=True,
                    t="Performance of federated versus centralized learning for mammography classification across film-digital domain shift",
                    j="Front Digit Health", doi="10.3389/fdgth.2026.1715858", y="2026"),
    "hanley1982": dict(a=["Hanley JA", "McNeil BJ"], etal=False,
                       t="The meaning and use of the area under a receiver operating characteristic (ROC) curve",
                       j="Radiology", v="143", p="29-36", y="1982"),
    "hanley1983": dict(a=["Hanley JA", "McNeil BJ"], etal=False,
                       t="A method of comparing the areas under receiver operating characteristic curves derived from the same cases",
                       j="Radiology", v="148", p="839-843", y="1983"),
    "akebli2026": dict(a=["Akebli H", "Della Mea V"], etal=False,
                       t="Domain generalization of histopathology foundation models in multicenter, multi-scanner cohorts: A comparative benchmark",
                       j="J Imaging", v="12", p="381", y="2026"),
}


def render_reference(r: dict) -> str:
    """JCO style: Authors (first three, then et al): Title. Journal vol:pages, year."""
    auth = ", ".join(r["a"]) + (", et al" if r.get("etal") else "")
    core = f"{auth}: {r['t']}. "
    if r.get("arxiv"):
        return core + f"arXiv:{r['arxiv']}, {r['y']}"
    if r.get("v"):
        return core + f"{r['j']} {r['v']}:{r['p']}, {r['y']}"
    return core + f"{r['j']}, {r['y']}. doi:{r['doi']}"


CITE = re.compile(r"\[\[([a-z0-9]+(?:\s*,\s*[a-z0-9]+)*)\]\]")


def number_citations(blocks: list[str]) -> tuple[list[str], list[str]]:
    """Replace ``[[key, key]]`` with ``[n, m]`` numbered by first appearance
    across ``blocks`` in order; consecutive runs of three or more collapse to a
    range. Returns the rewritten blocks and the ordered reference keys."""
    order: list[str] = []

    def num(key: str) -> int:
        if key not in REFS:
            raise KeyError(f"unknown reference key: {key}")
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    def fmt(nums: list[int]) -> str:
        nums = sorted(set(nums))
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(out) + "]"

    def sub(m):
        return fmt([num(k.strip()) for k in m.group(1).split(",")])

    return [CITE.sub(sub, b) for b in blocks], order


FIGURES_MAIN = [
    ("Fig 1", "Fig1.png",
     "Study design and model architecture. (a) Each diagnostic H&E whole-slide image is "
     "tiled at 20× and encoded by the UNI foundation model, then reduced once, with a seed "
     "recorded per slide, to a fixed pool of 512 patches. (b) A gated-attention "
     "multiple-instance model weights patches and pools them into a slide representation "
     "from which the continuous signature score is predicted. (c) All four signatures use "
     "one shared multi-label stratified five-fold partition. Histology is compared on "
     "identical folds against ER status alone and against a six-variable "
     "clinicopathological model by paired bootstrap; the pre-extracted radiomic feature "
     "arm is evaluated on the same folds under nested selection, with the non-nested "
     "procedure reproduced to measure its optimism."),
    ("Fig 2", "Fig2.png",
     "The histology margin over the clinicopathological model, and what predicts it. (a) "
     "Gain in AUC over the six-variable model with 95% percentile intervals from a paired "
     "bootstrap of 4,000 resamples (n = 81), for the pre-declared comparator (filled) and "
     "the comparator with its penalty tuned by inner cross-validation (open). (b) The same "
     "gain against the AUC of ER status alone as a fixed predictor of each label — label "
     "composition — for the four published group calls (filled) and three alternative "
     "cuts of the same scores (open); vertical bars are the 95% intervals of the gain. "
     "The margin falls monotonically as the label becomes more nearly a restatement of ER "
     "status."),
]
FIGURES_SUPP = [
    ("Fig S1", "FigS1.png",
     "Cohort flow. The analysis cohort is the intersection of two lists: patients with "
     "recomputed signature scores and patients with extracted whole-slide features. "
     "Neither can be enlarged from the available data."),
]


def flatten(sections):
    out = []
    for h, paras in sections:
        out.extend(paras)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--radiomics", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    # Number citations by first appearance: main text, then declarations, then
    # the supplement.  Every block is rewritten in place.
    main_blocks = (INTRO + flatten(METHODS) + flatten(RESULTS) + flatten(DISCUSSION))
    decl_blocks = [v for _, v in DECLARATIONS]
    supp_blocks = flatten(SUPP_METHODS) + flatten(SUPP_RESULTS)
    rewritten, order = number_citations(main_blocks + decl_blocks + supp_blocks)
    n_main, n_decl = len(main_blocks), len(decl_blocks)
    main_rw, decl_rw, supp_rw = (rewritten[:n_main], rewritten[n_main:n_main + n_decl],
                                 rewritten[n_main + n_decl:])
    unused = sorted(set(REFS) - set(order))
    if unused:
        raise SystemExit(f"references defined but never cited: {unused}")

    def regroup(sections, flat):
        out, i = [], 0
        for h, paras in sections:
            out.append({"h": h, "paras": flat[i:i + len(paras)]}); i += len(paras)
        return out

    i = 0
    intro = main_rw[i:i + len(INTRO)]; i += len(INTRO)
    methods = regroup(METHODS, main_rw[i:]); i += sum(len(p) for _, p in METHODS)
    results = regroup(RESULTS, main_rw[i:]); i += sum(len(p) for _, p in RESULTS)
    discussion = regroup(DISCUSSION, main_rw[i:])
    j = 0
    smeth = regroup(SUPP_METHODS, supp_rw[j:]); j += sum(len(p) for _, p in SUPP_METHODS)
    sres = regroup(SUPP_RESULTS, supp_rw[j:])

    tables_main, tables_supp = build_tables(Path(a.results), Path(a.radiomics))
    abstract_text = " ".join(t for _, t in ABSTRACT)
    doc = {
        "journal": JOURNAL,
        "title": TITLE,
        # Affiliation 1 is the corresponding address (Darmstadt UAS), per the
        # corresponding author's choice; the builders print the first-listed
        # affiliation of the corresponding author on the title page.
        "authors": [
            {"name": "Yaqeen Ali", "aff": [1, 2, 3], "corresponding": True,
             "email": "yaqeenali786@gmail.com"},
            {"name": "Johannes Gregori", "aff": [1, 3]},
        ],
        "affiliations": ["Darmstadt University of Applied Sciences, Darmstadt, Germany",
                         "Doctoral Center Applied Informatics, Darmstadt, Germany",
                         "mediri GmbH, Heidelberg, Germany"],
        "abstract": abstract_text,
        "abstract_structured": ABSTRACT,
        "abstract_words": len(abstract_text.split()),
        "keywords": KEYWORDS,
        "sections": (
            [{"h": "Introduction", "paras": intro}]
            + [{"h": "Methods", "paras": [], "subs": methods}]
            + [{"h": "Results", "paras": [], "subs": results}]
            + [{"h": "Discussion", "paras": discussion[0]["paras"], "subs": discussion[1:]}]
        ),
        "supplement": {
            "title": "Data Supplement",
            "sections": [{"h": "Supplementary Methods", "paras": [], "subs": smeth},
                         {"h": "Supplementary Results", "paras": [], "subs": sres}],
            "tables": tables_supp,
            "figures": [{"label": l, "file": f, "caption": c} for l, f, c in FIGURES_SUPP],
        },
        "declarations": [[k, v] for (k, _), v in zip(DECLARATIONS, decl_rw)],
        "references": [render_reference(REFS[k]) for k in order],
        "reference_keys": order,
        "tables": tables_main,
        "figures": [{"label": l, "file": f, "caption": c} for l, f, c in FIGURES_MAIN],
    }
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    body_words = sum(len(x.split()) for x in main_rw)
    print(f"wrote {a.out}")
    print(f"abstract: {doc['abstract_words']} words (limit 275)   body: {body_words} words (limit 3000)")
    print(f"main tables: {len(tables_main)}  main figures: {len(doc['figures'])}  "
          f"(limit 6 combined)   supp tables: {len(tables_supp)}  refs: {len(order)}")


if __name__ == "__main__":
    main()
