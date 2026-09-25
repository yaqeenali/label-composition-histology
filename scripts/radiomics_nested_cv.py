#!/usr/bin/env python3
"""Radiomics arm, rebuilt as nested cross-validation.

The problem this fixes
======================
Feature selection in ``radiology_radiomics_features_selection.ipynb`` runs
against every labelled case, and the winning feature set is then scored with the
same cross-validation that chose it.  That is selection on the evaluation data:
the reported number is the maximum over a search, not an estimate of
out-of-sample performance, and it is biased upward by an amount that grows with
the size of the search.

Here the entire pipeline -- imputation, scaling, univariate filtering and the
classifier's regularisation strength -- is fitted **inside each training fold**
and applied unchanged to the held-out fold.  Nothing about the test fold informs
any choice made about it.

The partition is the one written by ``locked_rerun.py``, so the radiology and
pathology arms are evaluated on identical patients in identical folds and can be
compared directly.  Expect the radiomics numbers to fall relative to the
notebook; that fall is the point.

    python scripts/radiomics_nested_cv.py --partition Results_locked/shared_partition.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]

TARGETS = {"GHI_bin": "GHI (Oncotype)", "Mammaprint_bin": "MammaPrint",
           "ROR_P_bin": "ROR-P", "ROR_S_bin": "ROR-S"}

#: Pre-specified search.  Small on purpose: with 82 patients a large grid buys
#: nothing but optimism, and every extra setting widens the selection bias that
#: nesting is here to remove.
GRID = [{"select__k": k, "clf__C": c}
        for k in (5, 10, 20, "all") for c in (0.01, 0.1, 1.0)]

N_INNER = 3


def make_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=10)),
        ("clf", LogisticRegression(max_iter=5000, C=1.0)),
    ])


def nested_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int = 20260906):
    """Outer folds fixed; the whole pipeline is chosen inside each training set."""
    oof = np.full(len(y), np.nan)
    chosen = []
    for k in sorted(np.unique(folds)):
        te = folds == k
        tr = ~te
        Xtr, ytr = X[tr], y[tr]
        if len(np.unique(ytr)) < 2:
            continue

        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed + int(k))
        best, best_score = None, -np.inf
        for params in GRID:
            kk = params["select__k"]
            if kk != "all" and kk > Xtr.shape[1]:
                continue
            scores = []
            for itr, iva in inner.split(Xtr, ytr):
                if len(np.unique(ytr[iva])) < 2:
                    continue
                pipe = make_pipe().set_params(**params)
                pipe.fit(Xtr[itr], ytr[itr])
                scores.append(roc_auc_score(ytr[iva], pipe.predict_proba(Xtr[iva])[:, 1]))
            if scores and np.mean(scores) > best_score:
                best_score, best = float(np.mean(scores)), params

        pipe = make_pipe().set_params(**best)
        pipe.fit(Xtr, ytr)
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        chosen.append({"fold": int(k), **best, "inner_auc": round(best_score, 3)})
    return oof, chosen


def flat_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int = 20260906):
    """The notebook's procedure: pick the best setting using all the data, then
    report that setting's own cross-validated score.  Reproduced here only so the
    optimism can be quantified rather than argued about."""
    best, best_oof, best_score = None, None, -np.inf
    for params in GRID:
        kk = params["select__k"]
        if kk != "all" and kk > X.shape[1]:
            continue
        oof = np.full(len(y), np.nan)
        for k in sorted(np.unique(folds)):
            te, tr = folds == k, folds != k
            if len(np.unique(y[tr])) < 2:
                continue
            pipe = make_pipe().set_params(**params)
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict_proba(X[te])[:, 1]
        m = np.isfinite(oof)
        if len(np.unique(y[m])) < 2:
            continue
        s = roc_auc_score(y[m], oof[m])
        if s > best_score:
            best_score, best, best_oof = s, params, oof
    return best_oof, best, best_score


def boot_ci(y, s, n=4000, seed=20260906):
    rng = np.random.default_rng(seed)
    vals = []
    for idx in rng.integers(0, len(y), size=(n, len(y))):
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partition", required=True,
                   help="shared_partition.csv written by locked_rerun.py")
    p.add_argument("--radiomics", default=str(REPO / "data/tcga_brca/radiomics.csv"))
    p.add_argument("--out", default=str(REPO / "Results_locked/radiomics"))
    args = p.parse_args(argv)

    part = pd.read_csv(args.partition)
    rad = pd.read_csv(args.radiomics)
    rad["CLID"] = rad.CLID.astype(str).str.strip().str[:12]
    d = part.merge(rad, on="CLID", how="inner")
    feat_cols = [c for c in rad.columns if c != "CLID"]
    print(f"patients in shared partition: {len(part)}   with radiomics: {len(d)}")
    print(f"radiomic features: {len(feat_cols)}\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X = d[feat_cols].values.astype(float)

    rows, all_chosen, preds = [], [], {"CLID": d.CLID.values, "fold": d.outer_fold.values}
    for col, label in TARGETS.items():
        y = d[col].values
        m = np.isfinite(y)
        yy, XX, ff = y[m].astype(int), X[m], d.outer_fold.values[m]

        oof, chosen = nested_cv(XX, yy, ff)
        ok = np.isfinite(oof)
        auc_nested = roc_auc_score(yy[ok], oof[ok])
        lo, hi = boot_ci(yy[ok], oof[ok])

        oof_flat, best_flat, auc_flat = flat_cv(XX, yy, ff)

        rows.append({"target": label, "n": int(m.sum()), "positives": int(yy.sum()),
                     "AUC_nested": round(auc_nested, 3),
                     "CI_low": round(lo, 3), "CI_high": round(hi, 3),
                     "AUC_notebook_style": round(auc_flat, 3),
                     "optimism": round(auc_flat - auc_nested, 3),
                     "notebook_pick": json.dumps(best_flat)})
        for c in chosen:
            all_chosen.append({"target": label, **c})
        col_oof = np.full(len(d), np.nan)
        col_oof[np.where(m)[0]] = oof
        preds[f"{col}_pred"] = col_oof
        print(f"{label:16s} nested {auc_nested:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"notebook-style {auc_flat:.3f}   optimism {auc_flat-auc_nested:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "radiomics_nested_cv.csv", index=False)
    pd.DataFrame(all_chosen).to_csv(out / "radiomics_selected_per_fold.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "radiomics_oof_predictions.csv", index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
