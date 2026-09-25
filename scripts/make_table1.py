#!/usr/bin/env python3
"""Table 1: cohort characteristics, overall and by the primary endpoint.

Built from the 91-patient clinical extract, restricted to the patients who
actually enter the models.  HER2 follows the ASCO/CAP order -- FISH result
first, IHC only where FISH is absent or uninformative -- which is what makes
the ER+/HER2- subgroup usable (n = 56 rather than 35).

Continuous variables are reported as median [IQR], categorical as n (%).  No
p-values: with 82 patients and a target defined from expression data, a
significance test between the two label groups answers a question nobody asked
and invites the reader to treat the split as a comparison of populations.

    python scripts/make_table1.py --clinical <91-patient csv> --partition <shared_partition.csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def her2_ascocap(row) -> str:
    """FISH overrides IHC; 'Unknown' only when neither is informative."""
    fish = str(row.get("HER2_FISH_STATUS", "")).strip()
    ihc = str(row.get("IHC_HER2", "")).strip()
    if fish in ("Positive", "Negative"):
        return fish
    if ihc in ("Positive", "Negative"):
        return ihc
    return "Unknown"


def _num(d: pd.DataFrame, col: str, label: str) -> dict:
    v = pd.to_numeric(d[col], errors="coerce").dropna()
    if v.empty:
        return {"key": label, "variable": label, "value": "-"}
    return {"key": label, "variable": label,
            "value": f"{v.median():.0f} [{v.quantile(.25):.0f}\u2013{v.quantile(.75):.0f}]"}


def _cat(d: pd.DataFrame, col: str, label: str, order=None) -> list:
    s = d[col].astype(str).str.strip()
    s = s.replace({"[Not Available]": "Unknown", "nan": "Unknown",
                   "[Unknown]": "Unknown", "[Not Evaluated]": "Unknown"})
    vc = s.value_counts()
    keys = [k for k in (order or []) if k in vc.index] + \
           [k for k in vc.index if k not in (order or [])]
    # Keys must be unique across the whole table: "Negative" occurs under ER, PR
    # and HER2, and merging the stratified columns on a duplicated key produces a
    # cross join rather than a table.
    rows = [{"key": label, "variable": label, "value": ""}]
    for k in keys:
        rows.append({"key": f"{label}::{k}", "variable": f"    {k}",
                     "value": f"{vc[k]} ({100*vc[k]/len(d):.0f}%)"})
    return rows


def build(d: pd.DataFrame) -> pd.DataFrame:
    rows = [_num(d, "AGE", "Age at diagnosis, years")]
    rows += _cat(d, "MENOPAUSE_STATUS", "Menopausal status")
    rows += _cat(d, "ER_STATUS_BY_IHC", "ER status", ["Positive", "Negative"])
    rows += _cat(d, "PR_STATUS_BY_IHC", "PR status", ["Positive", "Negative"])
    rows += _cat(d, "HER2", "HER2 status (ASCO/CAP)", ["Negative", "Positive", "Unknown"])
    rows += _cat(d, "AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC pathologic stage",
                 ["Stage I", "Stage IA", "Stage II", "Stage IIA", "Stage IIB",
                  "Stage IIIA", "Stage IIIC"])
    rows += _cat(d, "node_status", "Nodal status", ["Node negative", "Node positive"])
    rows += _cat(d, "HISTOLOGICAL_DIAGNOSIS", "Histology")
    rows += _cat(d, "Pam50.Call", "PAM50 subtype", ["LumA", "LumB", "Her2", "Basal", "Normal"])
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clinical", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--perou", default=str(REPO / "data/tcga_brca/Perou-TCGA-BRCA-metadata.csv"))
    p.add_argument("--strat_col", default="ROR_P_bin",
                   help="binary column to stratify by; the primary endpoint")
    p.add_argument("--out", default=str(REPO / "Results_locked"))
    args = p.parse_args(argv)

    clin = pd.read_csv(args.clinical)
    clin["CLID"] = clin.PATIENT_ID.astype(str).str.strip()
    clin["HER2"] = clin.apply(her2_ascocap, axis=1)
    clin["node_status"] = np.where(
        clin.AJCC_NODES_PATHOLOGIC_PN.astype(str).str.startswith("N0"),
        "Node negative", "Node positive")

    part = pd.read_csv(args.partition)
    perou = pd.read_csv(args.perou)
    perou["CLID"] = perou.CLID.astype(str).str.strip().str[:12]

    d = part.merge(clin, on="CLID", how="left").merge(
        perou[["CLID", "Pam50.Call"]], on="CLID", how="left")
    print(f"cohort: {len(d)} patients\n")

    strat = args.strat_col
    overall = build(d).rename(columns={"value": f"All (n={len(d)})"})
    g0 = d[d[strat] == 0]
    g1 = d[d[strat] == 1]
    t = overall
    for lbl, sub in [(f"{strat}=0 (n={len(g0)})", g0), (f"{strat}=1 (n={len(g1)})", g1)]:
        t = t.merge(build(sub).drop(columns=["variable"]).rename(columns={"value": lbl}),
                    on="key", how="left")
    t = t.drop(columns=["key"]).fillna("0 (0%)")
    t.loc[t.iloc[:, 1] == "", t.columns[1:]] = ""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "table1.csv", index=False)
    (out / "table1.md").write_text(t.to_markdown(index=False))
    print(t.to_string(index=False))
    print(f"\nwritten to {out/'table1.csv'}")


if __name__ == "__main__":
    main()
