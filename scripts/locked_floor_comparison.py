#!/usr/bin/env python3
"""The primary endpoint, computed on the locked run's held-out predictions.

For each assay: the imaging model's AUC against two floors, on exactly the same
patients and the same folds --

  * **ER status alone**, the single variable that drives most of the apparent
    performance in this cohort;
  * **a six-variable clinical model** (age, ER, PR, HER2, nodal status, stage),
    i.e. everything a pathology report already contains.

Both floors are fitted fold-honestly: for each outer fold the clinical model is
trained on the other folds and predicts the held-out one, so it faces the same
task as the imaging model rather than being fitted and scored on the same data.

Label orientation is fixed a priori from the sign of corr(continuous target,
binary label) -- a property of the label definition -- rather than by taking
max(AUC, 1-AUC) after the fact, which silently grants every model a free
half-decision.

HER2 follows the ASCO/CAP order: FISH result first, IHC only where FISH is
absent or uninformative.

    python scripts/locked_floor_comparison.py --results Results_locked_modal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]

NAMES = {"ghi": "GHI (Oncotype)", "mammaprint": "MammaPrint",
         "ror_p": "ROR-P", "ror_s": "ROR-S"}
COV = ["AGE", "ERpos", "PRpos", "HER2", "node_pos", "stage"]
STAGE = {"Stage I": 1, "Stage IA": 1, "Stage II": 2, "Stage IIA": 2,
         "Stage IIB": 2.5, "Stage IIIA": 3, "Stage IIIB": 3.2, "Stage IIIC": 3.5}


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


def fold_honest(d: pd.DataFrame, cols, ycol: str) -> np.ndarray:
    out = pd.Series(index=d.index, dtype=float)
    for f in d.outer_fold.unique():
        tr, te = d[d.outer_fold != f], d[d.outer_fold == f]
        if tr[ycol].nunique() < 2:
            out.loc[te.index] = 0.5
            continue
        sc = StandardScaler().fit(tr[cols].values.astype(float))
        m = LogisticRegression(max_iter=2000).fit(sc.transform(tr[cols].values.astype(float)),
                                                  tr[ycol].values)
        out.loc[te.index] = m.predict_proba(sc.transform(te[cols].values.astype(float)))[:, 1]
    return out.values


def paired_boot(y, a, b, n=4000, seed=20260906):
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in rng.integers(0, len(y), size=(n, len(y))):
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(diffs)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), max(p, 1 / len(d))


def boot_ci(y, s, n=4000, seed=20260906):
    rng = np.random.default_rng(seed)
    v = [roc_auc_score(y[i], s[i]) for i in rng.integers(0, len(y), size=(n, len(y)))
         if len(np.unique(y[i])) > 1]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="locked results directory")
    p.add_argument("--clinical",
                   default=str(REPO / "data/tcga_brca/clinical_91_from_vincenzo.csv"))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    res = Path(args.results)
    out = Path(args.out) if args.out else res
    clin = load_clinical(Path(args.clinical))

    rows, sub_rows = [], []
    for key, label in NAMES.items():
        f = res / key / "test_predictions.csv"
        if not f.exists():
            print(f"  (skipping {label}: no predictions yet)")
            continue
        d = pd.read_csv(f).merge(clin, on="CLID", how="left")
        d = d.dropna(subset=COV + ["binary"]).reset_index(drop=True)
        y = d["binary"].values.astype(int)
        if len(np.unique(y)) < 2:
            continue

        sign = np.sign(np.corrcoef(d["true"].values, y)[0, 1])
        score = sign * d["pred"].values

        er = fold_honest(d, ["ERpos"], "binary")
        cl = fold_honest(d, COV, "binary")

        a_m, a_e, a_c = roc_auc_score(y, score), roc_auc_score(y, er), roc_auc_score(y, cl)
        mlo, mhi = boot_ci(y, score)
        l1, h1, p1 = paired_boot(y, score, er)
        l2, h2, p2 = paired_boot(y, score, cl)
        rows.append({"assay": label, "n": len(d),
                     "AUC_model": round(a_m, 3), "model_CI": f"[{mlo:.3f}, {mhi:.3f}]",
                     "AUC_ER": round(a_e, 3), "AUC_clinical": round(a_c, 3),
                     "d_vs_ER": round(a_m - a_e, 3), "CI_ER": f"[{l1:.3f}, {h1:.3f}]",
                     "p_ER": round(p1, 4),
                     "d_vs_clinical": round(a_m - a_c, 3), "CI_clin": f"[{l2:.3f}, {h2:.3f}]",
                     "p_clinical": round(p2, 4)})

        for lbl, m in [("ALL", np.ones(len(d), bool)),
                       ("ER+", d.ERpos.values == 1),
                       ("ER+/HER2-", (d.ERpos.values == 1) & (d.HER2.values == 0))]:
            yy = y[m]
            sub_rows.append({"assay": label, "cohort": lbl, "n": int(m.sum()),
                             "pos": int(yy.sum()),
                             "AUC_model": round(roc_auc_score(yy, score[m]), 3)
                             if len(np.unique(yy)) > 1 else np.nan,
                             "AUC_clinical": round(roc_auc_score(yy, cl[m]), 3)
                             if len(np.unique(yy)) > 1 else np.nan})

    if not rows:
        print("no predictions found yet")
        return
    R, S = pd.DataFrame(rows), pd.DataFrame(sub_rows)
    R.to_csv(out / "locked_floor_comparison.csv", index=False)
    S.to_csv(out / "locked_subgroups.csv", index=False)
    print("=== LOCKED RUN: model vs ER alone, and vs a full clinical model ===\n")
    print(R.to_string(index=False))
    print("\n=== within the assay indication ===\n")
    print(S.to_string(index=False, na_rep="single-class"))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
