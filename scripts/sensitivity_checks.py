#!/usr/bin/env python3
"""Sensitivity checks on three analysis choices, on the locked predictions.

1  Nodal coding. The comparator codes every pN category other than pN0
   (including i- and i+) as node-positive, which puts the one pNX patient in the
   node-positive group. Here the C = 1 comparator is refitted with that patient
   coded node-negative, and with that patient excluded.
2  Scaler in the tuned comparator. ``fold_honest(tune=True)`` fits the scaler on
   the whole outer training set before the inner split used to choose C. Here
   the scaler is refitted within every inner split instead.
3  Confidence intervals for the subgroup AUCs of the histology model (95%
   percentile bootstrap, 4,000 resamples), which Table S2 reported without
   intervals.

Nothing reported as a primary result is changed; outputs go to
<results>/review/sensitivity_*.csv.

    python scripts/sensitivity_checks.py --results Results_locked/modal \\
        --clinical data/tcga_brca/clinical_91_from_vincenzo.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_response_analyses import (C_GRID, COV, NAMES, SEED, boot_ci, ci,  # noqa: E402
                                      fold_honest, holm, load_clinical, paired_boot)


def tuned_scaler_inside(d, cols, ycol):
    """As fold_honest(tune=True), but the scaler is refitted within each inner split."""
    out = pd.Series(index=d.index, dtype=float); chosen = []
    for f in sorted(d.outer_fold.unique()):
        tr, te = d[d.outer_fold != f], d[d.outer_fold == f]
        Xtr, Xte = tr[cols].values.astype(float), te[cols].values.astype(float)
        ytr = tr[ycol].values
        best, best_s = None, -np.inf
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED + int(f))
        for cc in C_GRID:
            s = []
            for itr, iva in inner.split(Xtr, ytr):
                if len(np.unique(ytr[iva])) < 2:
                    continue
                sc_i = StandardScaler().fit(Xtr[itr])
                m = LogisticRegression(C=cc, max_iter=5000).fit(sc_i.transform(Xtr[itr]), ytr[itr])
                s.append(roc_auc_score(ytr[iva], m.predict_proba(sc_i.transform(Xtr[iva]))[:, 1]))
            if s and np.mean(s) > best_s:
                best_s, best = np.mean(s), cc
        chosen.append(best)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=best, max_iter=5000).fit(sc.transform(Xtr), ytr)
        out.loc[te.index] = m.predict_proba(sc.transform(Xte))[:, 1]
    return out.values, chosen


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--clinical", required=True)
    a = p.parse_args(argv)
    res = Path(a.results); out = res / "review"
    clin = load_clinical(Path(a.clinical))
    raw = pd.read_csv(a.clinical); raw["CLID"] = raw.PATIENT_ID.astype(str).str.strip()
    nx = set(raw.loc[raw.AJCC_NODES_PATHOLOGIC_PN.astype(str).str.strip() == "NX", "CLID"])

    nodal, scaler, subs = [], [], []
    for key, label in NAMES.items():
        d0 = pd.read_csv(res / key / "test_predictions.csv").merge(clin, on="CLID", how="left")
        d0 = d0.dropna(subset=COV + ["binary"]).reset_index(drop=True)
        y0 = d0["binary"].values.astype(int)
        sign = np.sign(np.corrcoef(d0["true"].values, y0)[0, 1])
        d0["hscore"] = sign * d0["pred"].values

        # 1 nodal coding
        variants = {
            "pNX node-positive (as reported)": d0,
            "pNX node-negative": d0.assign(node_pos=np.where(d0.CLID.isin(nx), 0, d0.node_pos)),
            "pNX patient excluded": d0[~d0.CLID.isin(nx)].reset_index(drop=True),
        }
        for vname, d in variants.items():
            y = d["binary"].values.astype(int)
            c1, _ = fold_honest(d, COV, "binary", C=1.0)
            ah, ac = roc_auc_score(y, d.hscore.values), roc_auc_score(y, c1)
            lo, hi, pp = paired_boot(y, d.hscore.values, c1)
            nodal.append({"assay": label, "coding": vname, "n": len(d), "AUC_model": round(ah, 3),
                          "AUC_clin_C1": round(ac, 3), "gain": round(ah - ac, 3),
                          "CI": ci(lo, hi), "p": round(pp, 4)})

        # 2 scaler inside the inner split of the tuned comparator
        t_old, c_old = fold_honest(d0, COV, "binary", tune=True)
        t_new, c_new = tuned_scaler_inside(d0, COV, "binary")
        ah = roc_auc_score(y0, d0.hscore.values)
        lo, hi, pp = paired_boot(y0, d0.hscore.values, t_new)
        scaler.append({"assay": label, "C_chosen_reported": " ".join(map(str, c_old)),
                       "C_chosen_scaler_inside": " ".join(map(str, c_new)),
                       "AUC_tuned_reported": round(roc_auc_score(y0, t_old), 3),
                       "AUC_tuned_scaler_inside": round(roc_auc_score(y0, t_new), 3),
                       "gain_scaler_inside": round(ah - roc_auc_score(y0, t_new), 3),
                       "CI": ci(lo, hi), "p": round(pp, 4)})

        # 3 subgroup CIs for the histology AUC
        for lbl, m in [("ALL", np.ones(len(d0), bool)), ("ER+", d0.ERpos.values == 1),
                       ("ER+/HER2-", (d0.ERpos.values == 1) & (d0.HER2.values == 0))]:
            yy, ss = y0[m], d0.hscore.values[m]
            if len(np.unique(yy)) < 2:
                subs.append({"assay": label, "cohort": lbl, "n": int(m.sum()), "pos": int(yy.sum()),
                             "AUC_model": np.nan, "CI": ""}); continue
            lo, hi = boot_ci(yy, ss)
            subs.append({"assay": label, "cohort": lbl, "n": int(m.sum()), "pos": int(yy.sum()),
                         "AUC_model": round(roc_auc_score(yy, ss), 3), "CI": ci(lo, hi)})

    N = pd.DataFrame(nodal)
    for v in N.coding.unique():
        m = N.coding == v
        N.loc[m, "p_holm"] = np.round(holm(N.loc[m, "p"].values), 4)
    S = pd.DataFrame(scaler)
    S["p_holm"] = np.round(holm(S.p.values), 4)
    N.to_csv(out / "sensitivity_nodal.csv", index=False)
    S.to_csv(out / "sensitivity_tuned_scaler.csv", index=False)
    pd.DataFrame(subs).to_csv(out / "sensitivity_subgroup_ci.csv", index=False)
    pd.set_option("display.width", 250)
    print(N.to_string(index=False)); print(); print(S.to_string(index=False)); print()
    print(pd.DataFrame(subs).to_string(index=False))


if __name__ == "__main__":
    main()
