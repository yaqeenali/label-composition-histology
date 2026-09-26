#!/usr/bin/env python3
"""Analyses the cold review asked for, run on the locked predictions.

M1  Incremental value: clinical + histology versus clinical alone -- a stacked
    logistic model refitted inside each fold (prediction-side answer; NOTE: the
    training patients' histology scores here are pooled out-of-fold predictions
    from models that saw the held-out fold, so this estimate can be optimistic.
    The reported combined model is the cross-fitted one in
    crossfit_combined.py; the columns kept here are for comparison), and a
    likelihood-ratio test for the out-of-fold histology score added to the
    clinical model on the full data (association-side answer).
M3  ER alone as what it is -- one fixed binary predictor, oriented a priori
    (ER-positive predicts the low-risk class). Its AUC is the balanced accuracy
    of ER as a classifier of the label; Cohen's kappa is given alongside as
    the chance-corrected agreement. This replaces the out-of-fold "fit" of a
    single binary predictor, whose pooled AUC mixes five per-fold intercepts.
M4  A penalised clinical comparator with C chosen by inner cross-validation,
    fold-averaged AUCs alongside pooled ones, the apparent (in-sample) AUC of
    the clinical model, and events per variable -- so a below-chance pooled AUC
    can be seen for what it is.
M9  Sensitivity of the label composition and the gains to the binarisation cut.
M10 Holm-adjusted p values across the four targets.

Everything is written to <results>/review/ and nothing else under <results>/
is modified. The pre-declared comparator (L2 logistic regression, C = 1,
identical to locked_floor_comparison.py) is recomputed here with the same seed
so that the two files can be checked against each other.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

NAMES = {"ghi": "GHI (Oncotype)", "mammaprint": "MammaPrint",
         "ror_p": "ROR-P", "ror_s": "ROR-S"}
# Which class the binary label calls positive, and therefore which way ER
# status points a priori: ER-positive disease is low-risk on every signature.
ER_PREDICTS_POSITIVE = {"ghi": False,        # positive = High/Intermediate
                        "mammaprint": True,  # positive = Good
                        "ror_p": False,      # positive = high/medium
                        "ror_s": False}
COV = ["AGE", "ERpos", "PRpos", "HER2", "node_pos", "stage"]
STAGE = {"Stage I": 1, "Stage IA": 1, "Stage II": 2, "Stage IIA": 2,
         "Stage IIB": 2.5, "Stage IIIA": 3, "Stage IIIB": 3.2, "Stage IIIC": 3.5}
SEED = 20260906
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


def load_clinical(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path)
    c["CLID"] = c.PATIENT_ID.astype(str).str.strip()

    def her2(r):
        f, i = str(r.HER2_FISH_STATUS).strip(), str(r.IHC_HER2).strip()
        if f == "Positive": return 1
        if f == "Negative": return 0
        if i == "Positive": return 1
        if i == "Negative": return 0
        return np.nan

    c["HER2"] = c.apply(her2, axis=1)
    c["ERpos"] = (c.ER_STATUS_BY_IHC == "Positive").astype(int)
    c["PRpos"] = (c.PR_STATUS_BY_IHC == "Positive").astype(int)
    c["node_pos"] = (~c.AJCC_NODES_PATHOLOGIC_PN.astype(str).str.startswith("N0")).astype(int)
    c["stage"] = c.AJCC_PATHOLOGIC_TUMOR_STAGE.map(STAGE)
    return c[["CLID"] + COV]


def fold_honest(d, cols, ycol, C=1.0, tune=False):
    """Out-of-fold probabilities. With tune=True, C is chosen by an inner
    3-fold CV inside each training set (a nested comparator)."""
    out = pd.Series(index=d.index, dtype=float)
    chosen = []
    for f in sorted(d.outer_fold.unique()):
        tr, te = d[d.outer_fold != f], d[d.outer_fold == f]
        Xtr, Xte = tr[cols].values.astype(float), te[cols].values.astype(float)
        ytr = tr[ycol].values
        if len(np.unique(ytr)) < 2:
            out.loc[te.index] = 0.5; continue
        sc = StandardScaler().fit(Xtr)
        c_use = C
        if tune:
            best, best_s = None, -np.inf
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED + int(f))
            for cc in C_GRID:
                s = []
                for itr, iva in inner.split(Xtr, ytr):
                    if len(np.unique(ytr[iva])) < 2: continue
                    m = LogisticRegression(C=cc, max_iter=5000).fit(sc.transform(Xtr[itr]), ytr[itr])
                    s.append(roc_auc_score(ytr[iva], m.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
                if s and np.mean(s) > best_s:
                    best_s, best = np.mean(s), cc
            c_use = best
        chosen.append(c_use)
        m = LogisticRegression(C=c_use, max_iter=5000).fit(sc.transform(Xtr), ytr)
        out.loc[te.index] = m.predict_proba(sc.transform(Xte))[:, 1]
    return out.values, chosen


def apparent_auc(d, cols, ycol, C=1.0):
    """The in-sample AUC of the same model fitted once on all rows: what the
    comparator looks like before cross-validation removes the optimism."""
    X = d[cols].values.astype(float); y = d[ycol].values
    m = LogisticRegression(C=C, max_iter=5000).fit(StandardScaler().fit_transform(X), y)
    return roc_auc_score(y, m.predict_proba(StandardScaler().fit_transform(X))[:, 1])


def per_fold_auc(d, score, ycol):
    v = []
    for f in sorted(d.outer_fold.unique()):
        m = d.outer_fold == f
        y = d.loc[m, ycol].values
        if len(np.unique(y)) > 1:
            v.append(roc_auc_score(y, score[m.values]))
    return float(np.mean(v)), float(np.std(v, ddof=1))


def boot_ci(y, s, n=4000, seed=SEED):
    rng = np.random.default_rng(seed); v = []
    for idx in rng.integers(0, len(y), size=(n, len(y))):
        if len(np.unique(y[idx])) > 1:
            v.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_boot(y, a, b, n=4000, seed=SEED):
    rng = np.random.default_rng(seed); diffs = []
    for idx in rng.integers(0, len(y), size=(n, len(y))):
        if len(np.unique(y[idx])) < 2: continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    dd = np.array(diffs)
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    return float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)), max(p, 1 / len(dd))


def logit_or(d, cols, extra, ycol):
    """Odds ratio per SD for `extra` in a logistic model of `ycol` on
    `cols + [extra]`, fitted once on all rows, with a likelihood-ratio p value
    against the model without `extra`."""
    import statsmodels.api as sm
    X = d[cols + [extra]].values.astype(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    y = d[ycol].values.astype(int)
    try:
        m0 = sm.Logit(y, sm.add_constant(X[:, :-1]) if cols else np.ones((len(y), 1))).fit(disp=0, maxiter=200)
        m1 = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    p = float(chi2.sf(2 * (m1.llf - m0.llf), 1))
    b, se = m1.params[-1], m1.bse[-1]
    return float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se)), p


def hanley_mcneil_se(A, n1, n0):
    """Standard error of an AUC (Hanley and McNeil 1982)."""
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))


def mdd(n1, n0, A=0.80, rho=0.5, alpha=0.05, power=0.80):
    """Minimum detectable difference between two AUCs measured on the same
    patients (Hanley and McNeil 1983): both AUCs A, correlation rho between
    them, two-sided alpha, given power."""
    from scipy.stats import norm
    s = hanley_mcneil_se(A, n1, n0)
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(2 * s * s * (1 - rho))


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--clinical", required=True)
    p.add_argument("--perou", required=True)
    a = p.parse_args(argv)
    res = Path(a.results); out = res / "review"; out.mkdir(exist_ok=True)
    clin = load_clinical(Path(a.clinical))
    perou = pd.read_csv(a.perou); perou["CLID"] = perou.CLID.str.strip().str[:12]

    rows, subrows, sens = [], [], []
    for key, label in NAMES.items():
        d = pd.read_csv(res / key / "test_predictions.csv").merge(clin, on="CLID", how="left")
        d = d.dropna(subset=COV + ["binary"]).reset_index(drop=True)
        y = d["binary"].values.astype(int)
        sign = np.sign(np.corrcoef(d["true"].values, y)[0, 1])
        d["hscore"] = sign * d["pred"].values
        d["hist_z"] = (d.hscore - d.hscore.mean()) / d.hscore.std(ddof=1)
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())

        # --- comparators ---------------------------------------------------
        er_fixed = d.ERpos.values if ER_PREDICTS_POSITIVE[key] else 1 - d.ERpos.values
        er_auc = roc_auc_score(y, er_fixed)
        kappa = cohen_kappa_score(y, er_fixed)
        concord = float((y == er_fixed).mean())
        clin_pre, _ = fold_honest(d, COV, "binary", C=1.0)
        clin_tuned, chosen = fold_honest(d, COV, "binary", tune=True)
        clin_app = apparent_auc(d, COV, "binary")
        stacked, _ = fold_honest(d, COV + ["hist_z"], "binary", tune=True)

        # --- AUCs ------------------------------------------------------------
        a_h = roc_auc_score(y, d.hscore.values); h_lo, h_hi = boot_ci(y, d.hscore.values)
        a_pre, a_tun, a_st = (roc_auc_score(y, s) for s in (clin_pre, clin_tuned, stacked))
        fh, ft, fs = (per_fold_auc(d, s, "binary") for s in (d.hscore.values, clin_tuned, stacked))
        # --- paired differences ---------------------------------------------
        e_lo, e_hi, e_p = paired_boot(y, d.hscore.values, er_fixed)
        p_lo, p_hi, p_p = paired_boot(y, d.hscore.values, clin_pre)
        t_lo, t_hi, t_p = paired_boot(y, d.hscore.values, clin_tuned)
        i_lo, i_hi, i_p = paired_boot(y, stacked, clin_tuned)
        # --- association ------------------------------------------------------
        u_or, u_lo, u_hi, u_p = logit_or(d, [], "hist_z", "binary")
        a_or, a_lo, a_hi, a_p = logit_or(d, COV, "hist_z", "binary")

        rows.append({
            "assay": label, "n": len(d), "positives": n_pos, "negatives": n_neg,
            "events_per_variable": round(min(n_pos, n_neg) / (len(COV) + 1), 1),
            "AUC_model": round(a_h, 3), "model_CI": ci(h_lo, h_hi),
            "AUC_model_foldmean": round(fh[0], 3), "AUC_model_foldsd": round(fh[1], 3),
            "AUC_ER_fixed": round(er_auc, 3), "kappa_ER": round(kappa, 3),
            "pct_concordant_with_ER": round(100 * concord, 1),
            "AUC_clin_predeclared": round(a_pre, 3), "AUC_clin_apparent": round(clin_app, 3),
            "AUC_clin_tuned": round(a_tun, 3),
            "AUC_clin_tuned_foldmean": round(ft[0], 3), "AUC_clin_tuned_foldsd": round(ft[1], 3),
            "C_chosen": " ".join(str(c) for c in chosen),
            "d_vs_ER_fixed": round(a_h - er_auc, 3), "CI_ER_fixed": ci(e_lo, e_hi), "p_ER_fixed": round(e_p, 4),
            "d_vs_clin_predeclared": round(a_h - a_pre, 3), "CI_clin_predeclared": ci(p_lo, p_hi),
            "p_clin_predeclared": round(p_p, 4),
            "d_vs_clin_tuned": round(a_h - a_tun, 3), "CI_clin_tuned": ci(t_lo, t_hi), "p_clin_tuned": round(t_p, 4),
            "AUC_stacked": round(a_st, 3),
            "AUC_stacked_foldmean": round(fs[0], 3), "AUC_stacked_foldsd": round(fs[1], 3),
            "d_incremental": round(a_st - a_tun, 3), "CI_incremental": ci(i_lo, i_hi), "p_incremental": round(i_p, 4),
            "OR_unadjusted": round(u_or, 2), "OR_unadjusted_CI": f"[{u_lo:.2f}, {u_hi:.2f}]", "p_unadjusted": round(u_p, 4),
            "OR_adjusted": round(a_or, 2), "OR_adjusted_CI": f"[{a_lo:.2f}, {a_hi:.2f}]", "p_adjusted_LRT": round(a_p, 4),
        })

        # --- subgroups with both comparators ---------------------------------
        for lbl, m in [("ALL", np.ones(len(d), bool)), ("ER+", d.ERpos.values == 1),
                       ("ER+/HER2-", (d.ERpos.values == 1) & (d.HER2.values == 0))]:
            yy = y[m]
            ok = len(np.unique(yy)) > 1
            subrows.append({"assay": label, "cohort": lbl, "n": int(m.sum()), "pos": int(yy.sum()),
                            "AUC_model": round(roc_auc_score(yy, d.hscore.values[m]), 3) if ok else np.nan,
                            "AUC_clin_predeclared": round(roc_auc_score(yy, clin_pre[m]), 3) if ok else np.nan,
                            "AUC_clin_tuned": round(roc_auc_score(yy, clin_tuned[m]), 3) if ok else np.nan,
                            "AUC_stacked": round(roc_auc_score(yy, stacked[m]), 3) if ok else np.nan})

        # --- M9: alternative binarisation ------------------------------------
        g = d.merge(perou[["CLID", "GHI_RS_3Group", "ROR-P Group", "ROR-S Group"]], on="CLID")
        alt, desc, er_alt = None, "", None
        if key == "ghi":
            alt = (g.GHI_RS_3Group == "High").astype(int).values; desc = "High vs Intermediate+Low"
        elif key == "ror_p":
            alt = (g["ROR-P Group"] == "high").astype(int).values; desc = "high vs medium+low"
        elif key == "ror_s":
            alt = (g["ROR-S Group"] == "high").astype(int).values; desc = "high vs medium+low"
        if alt is not None and len(np.unique(alt)) > 1:
            g["alt"] = alt
            er_alt = 1 - g.ERpos.values
            c_alt, _ = fold_honest(g, COV, "alt", C=1.0)
            ah, ac = roc_auc_score(alt, g.hscore.values), roc_auc_score(alt, c_alt)
            lo, hi, pp = paired_boot(alt, g.hscore.values, c_alt)
            sens.append({"assay": label, "cut": desc, "positives": int(alt.sum()), "negatives": int((1 - alt).sum()),
                         "AUC_ER_fixed": round(roc_auc_score(alt, er_alt), 3),
                         "kappa_ER": round(cohen_kappa_score(alt, er_alt), 3),
                         "AUC_model": round(ah, 3), "AUC_clin_predeclared": round(ac, 3),
                         "d_vs_clin_predeclared": round(ah - ac, 3), "CI": ci(lo, hi), "p": round(pp, 4)})

    # --- minimum detectable AUC difference under stated assumptions ---------
    md = []
    S = pd.DataFrame(subrows)
    for a in NAMES.values():
        r_all = S[(S.assay == a) & (S.cohort == "ALL")].iloc[0]
        r_sub = S[(S.assay == a) & (S.cohort == "ER+/HER2-")].iloc[0]
        n1, n0 = int(r_all.pos), int(r_all.n - r_all.pos)
        prev = n1 / (n1 + n0)
        row = {"assay": a, "n": int(r_all.n), "positives": n1, "negatives": n0,
               "MDD_full": round(mdd(n1, n0), 3),
               "n_subgroup": int(r_sub.n), "MDD_subgroup": round(mdd(int(r_sub.pos), int(r_sub.n - r_sub.pos)), 3)
               if 0 < r_sub.pos < r_sub.n else np.nan,
               "MDD_n900_same_prevalence": round(mdd(int(round(900 * prev)), int(round(900 * (1 - prev)))), 3),
               "assumptions": "both AUC 0.80; rho 0.5; alpha 0.05 two-sided; power 0.80"}
        md.append(row)
    pd.DataFrame(md).to_csv(out / "mdd.csv", index=False)

    R = pd.DataFrame(rows)
    for col in ("p_ER_fixed", "p_clin_predeclared", "p_clin_tuned", "p_incremental", "p_adjusted_LRT"):
        R[col + "_holm"] = np.round(holm(R[col].values), 4)
    R.to_csv(out / "comparators.csv", index=False)
    pd.DataFrame(subrows).to_csv(out / "subgroups.csv", index=False)
    pd.DataFrame(sens).to_csv(out / "binarisation_sensitivity.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== comparators (pooled out-of-fold unless stated) ===\n")
    print(R[["assay", "n", "positives", "negatives", "events_per_variable", "AUC_model", "model_CI",
             "AUC_ER_fixed", "kappa_ER", "pct_concordant_with_ER", "AUC_clin_predeclared", "AUC_clin_apparent",
             "AUC_clin_tuned", "AUC_clin_tuned_foldmean", "AUC_clin_tuned_foldsd", "C_chosen"]].to_string(index=False))
    print("\n=== histology versus each comparator ===\n")
    print(R[["assay", "d_vs_ER_fixed", "CI_ER_fixed", "p_ER_fixed",
             "d_vs_clin_predeclared", "CI_clin_predeclared", "p_clin_predeclared", "p_clin_predeclared_holm",
             "d_vs_clin_tuned", "CI_clin_tuned", "p_clin_tuned", "p_clin_tuned_holm"]].to_string(index=False))
    print("\n=== incremental value: clinical + histology versus clinical (both tuned) ===\n")
    print(R[["assay", "AUC_stacked", "AUC_stacked_foldmean", "AUC_stacked_foldsd", "d_incremental", "CI_incremental",
             "p_incremental", "p_incremental_holm", "OR_unadjusted", "OR_unadjusted_CI", "p_unadjusted",
             "OR_adjusted", "OR_adjusted_CI", "p_adjusted_LRT", "p_adjusted_LRT_holm"]].to_string(index=False))
    print("\n=== minimum detectable AUC difference ===\n")
    print(pd.DataFrame(md).to_string(index=False))
    print("\n=== subgroups ===\n")
    print(pd.DataFrame(subrows).to_string(index=False, na_rep="single-class"))
    print("\n=== binarisation sensitivity (pre-declared comparator) ===\n")
    print(pd.DataFrame(sens).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
