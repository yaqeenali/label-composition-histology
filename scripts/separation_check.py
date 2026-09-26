#!/usr/bin/env python3
"""Check the unpenalized adjusted model behind the odds ratios in Table 3.

``review_response_analyses.logit_or`` fits an unpenalized logistic model of the
label on the six clinical variables and the histology score, all standardized,
on the 81 patients. ER status separates some labels almost completely (every
ER-negative patient has the same label), so the ER coefficient diverges. This
script records that, and refits the same model with three optimizers to show
that the histology odds ratio does not depend on how the fit stops.

    python scripts/separation_check.py --results Results_locked/modal \\
        --clinical data/tcga_brca/clinical_91_from_vincenzo.csv

Writes <results>/review/separation_check.csv.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_response_analyses import COV, NAMES, load_clinical  # noqa: E402


def main(argv=None) -> None:
    import statsmodels.api as sm
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--clinical", required=True)
    a = p.parse_args(argv)
    res = Path(a.results)
    clin = load_clinical(Path(a.clinical))
    rows = []
    for key, label in NAMES.items():
        d = pd.read_csv(res / key / "test_predictions.csv").merge(clin, on="CLID", how="left")
        d = d.dropna(subset=COV + ["binary"]).reset_index(drop=True)
        y = d["binary"].values.astype(int)
        sign = np.sign(np.corrcoef(d["true"].values, y)[0, 1])
        h = sign * d["pred"].values
        X = np.column_stack([d[COV].values.astype(float), h])
        X = (X - X.mean(0)) / X.std(0, ddof=1)          # as in logit_or
        Xc = sm.add_constant(X)
        er = 1 + COV.index("ERpos")
        fits = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for meth in ("newton", "bfgs", "lbfgs"):
                fits[meth] = sm.Logit(y, Xc).fit(method=meth, disp=0, maxiter=5000)
        neg = d.ERpos.values == 0
        rows.append({
            "assay": label, "n": len(d), "ER_negative": int(neg.sum()),
            "ER_negative_label_1": int(y[neg].sum()), "ER_negative_label_0": int((1 - y[neg]).sum()),
            "ER_coef_newton": round(float(fits["newton"].params[er]), 2),
            "ER_SE_newton": round(float(fits["newton"].bse[er]), 1),
            **{f"OR_hist_{k}": round(float(np.exp(v.params[-1])), 3) for k, v in fits.items()},
        })
    out = pd.DataFrame(rows)
    out.to_csv(res / "review" / "separation_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
