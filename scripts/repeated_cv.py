#!/usr/bin/env python3
"""Repeated cross-validation robustness check for the locked design.

The locked run (locked_rerun.py) is a single multi-label stratified 5-fold
partition. A reviewer's first question is whether the head-to-head result
survives a different split. This repeats the *entire* locked pipeline --
same fixed configuration BB, same model, same fixed 512-patch pools, same
fold-honest C = 1 clinical comparator -- under R independently drawn
partitions, and reports the distribution of the histology margin over the
clinicopathological model across them.

Nothing is tuned. The only thing that changes between repeats is the seed
that draws the outer partition and the inner (epoch-selection) split; repeat 0
uses the locked seed, so it reproduces the locked predictions exactly and
serves as a check. Nothing under Results_locked/ is read or written; output
goes to a separate directory.

    python scripts/repeated_cv.py --pt_dir <pool> --out Results_repeatedcv \\
        --clinical data/tcga_brca/clinical_91_from_vincenzo.csv --repeats 5
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the locked pipeline verbatim -- same cohort build, same stratifier,
# same model/config factory, same single train/val/test cycle.
from locked_rerun import (  # noqa: E402
    ASSAYS, SEED, N_OUTER, N_INNER, build_cohort, iterative_stratify,
    make_cfg, one_run,
)

FIXED_GRID = "BB"            # the pre-declared modal configuration
COV = ["AGE", "ERpos", "PRpos", "HER2", "node_pos", "stage"]
STAGE = {"Stage I": 1, "Stage IA": 1, "Stage II": 2, "Stage IIA": 2,
         "Stage IIB": 2.5, "Stage IIIA": 3, "Stage IIIB": 3.2, "Stage IIIC": 3.5}
BOOT_SEED = 20260906


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


def clinical_oof(d: pd.DataFrame, ycol: str, fold_col: str) -> np.ndarray:
    """Fold-honest out-of-fold probabilities from the pre-declared C = 1
    six-variable logistic model, on this repeat's own partition."""
    out = np.full(len(d), 0.5)
    idx = {i: p for p, i in enumerate(d.index)}
    for f in sorted(d[fold_col].unique()):
        tr, te = d[d[fold_col] != f], d[d[fold_col] == f]
        ytr = tr[ycol].values
        if len(np.unique(ytr)) < 2:
            continue
        sc = StandardScaler().fit(tr[COV].values.astype(float))
        m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[COV].values.astype(float)), ytr)
        pr = m.predict_proba(sc.transform(te[COV].values.astype(float)))[:, 1]
        for i, v in zip(te.index, pr):
            out[idx[i]] = v
    return out


def per_fold_auc(y, score, fold):
    v = []
    for f in np.unique(fold):
        m = fold == f
        if len(np.unique(y[m])) > 1:
            v.append(roc_auc_score(y[m], score[m]))
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def partition_for(df: pd.DataFrame, ps: int) -> pd.DataFrame:
    """This repeat's outer partition; deterministic in the seed, so it is safe
    to recompute per fold."""
    labels = df[[s["binary_target_col"] for s in ASSAYS.values()] + ["ER_pos"]]
    df = df.copy()
    df["outer_fold"] = iterative_stratify(labels.fillna(0).values.astype(int), N_OUTER, ps)
    return df


def run_one_fold(rep: int, akey: str, k: int, df: pd.DataFrame, num_patches) -> pd.DataFrame:
    """Train the fixed-config MIL model on one outer fold and return its
    held-out predictions. This is the checkpoint unit (~90-130 s)."""
    ps = SEED + rep                         # repeat 0 == the locked seed
    spec = ASSAYS[akey]
    dfp = partition_for(df, ps)
    work = dfp[dfp[spec["binary_target_col"]].notna() & dfp[spec["target_col"]].notna()].copy()
    test = work[work.outer_fold == k]
    dev = work[work.outer_fold != k].reset_index(drop=True)
    ilab = dev[[s["binary_target_col"] for s in ASSAYS.values()] + ["ER_pos"]]
    dev["inner_fold"] = iterative_stratify(
        ilab.fillna(0).values.astype(int), N_INNER, ps + 1000 + k)
    cfg = make_cfg(akey, FIXED_GRID, num_patches)
    tr, va = dev[dev.inner_fold != 0], dev[dev.inner_fold == 0]
    res = one_run(cfg, tr, va, test, ps + k)
    return pd.DataFrame({
        "CLID": test.CLID.values, "outer_fold": k,
        "true": res.test_targets, "pred": res.test_preds,
        "binary": res.test_binary if res.test_binary is not None else np.nan})


def assemble_cell(rep: int, akey: str, fold_dir: Path, clin: pd.DataFrame) -> dict | None:
    """Once all N_OUTER fold-prediction files for a (repeat, assay) exist,
    pool them and compute the histology AUC and the margin over the fold-honest
    clinical model. Returns None if any fold is missing."""
    files = [fold_dir / f"r{rep}_{akey}_f{k}.csv" for k in range(N_OUTER)]
    if not all(f.exists() for f in files):
        return None
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).merge(
        clin, on="CLID", how="left")
    p = p.dropna(subset=COV + ["binary"]).reset_index(drop=True)
    y = p["binary"].values.astype(int)
    sign = np.sign(np.corrcoef(p["true"].values, y)[0, 1])
    hist = sign * p["pred"].values
    clin_oof = clinical_oof(p, "binary", "outer_fold")
    a_h = roc_auc_score(y, hist)
    a_c = roc_auc_score(y, clin_oof)
    fh = per_fold_auc(y, hist, p.outer_fold.values)
    return {"repeat": rep, "partition_seed": SEED + rep, "assay": ASSAYS[akey]["name"],
            "n": len(p), "AUC_hist": round(a_h, 4),
            "AUC_hist_foldmean": round(fh[0], 4), "AUC_hist_foldsd": round(fh[1], 4),
            "AUC_clin": round(a_c, 4), "margin": round(a_h - a_c, 4)}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a in [s["name"] for s in ASSAYS.values()]:
        g = df[df.assay == a]
        out.append({
            "assay": a, "repeats": len(g),
            "AUC_hist_mean": round(g.AUC_hist.mean(), 3),
            "AUC_hist_sd": round(g.AUC_hist.std(ddof=1), 3),
            "AUC_hist_min": round(g.AUC_hist.min(), 3),
            "AUC_hist_max": round(g.AUC_hist.max(), 3),
            "margin_mean": round(g.margin.mean(), 3),
            "margin_sd": round(g.margin.std(ddof=1), 3),
            "margin_min": round(g.margin.min(), 3),
            "margin_max": round(g.margin.max(), 3),
            "repeats_margin_positive": int((g.margin > 0).sum()),
        })
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt_dir", required=True)
    ap.add_argument("--perou_csv", default=str(REPO / "data/tcga_brca/Perou-TCGA-BRCA-metadata.csv"))
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--out", default=str(REPO / "Results_repeatedcv"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--num_patches", type=int, default=0)
    ap.add_argument("--first_repeat", type=int, default=0)
    ap.add_argument("--max_seconds", type=float, default=0,
                    help="stop cleanly before starting a new (repeat, assay) once this "
                         "much wall time has passed; 0 = no limit. Lets each invocation "
                         "fit inside a tool-call timeout and be resumed.")
    a = ap.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.log", "a", buffering=1)
    def log(m): print(m, flush=True); logf.write(m + "\n")

    t0 = time.time()
    df = build_cohort(Path(a.perou_csv), Path(a.pt_dir),
                      REPO / "data/tcga_brca/tcga_clinical_pathology_clean.csv")
    df = df[df.excluded_reason.isna()].reset_index(drop=True)
    clin = load_clinical(Path(a.clinical))
    fold_dir = out / "foldpreds"; fold_dir.mkdir(exist_ok=True)

    # Resume-friendly at the FOLD grain: each fold's held-out predictions are
    # written the moment it finishes, so a container reclaim or a tool-call
    # timeout costs at most one fold (~2 min).
    cells = [(rep, k) for rep in range(a.first_repeat, a.first_repeat + a.repeats)
             for k in ASSAYS]
    fold_jobs = [(rep, k, f) for (rep, k) in cells for f in range(N_OUTER)]
    pending = [(rep, k, f) for rep, k, f in fold_jobs
               if not (fold_dir / f"r{rep}_{k}_f{f}.csv").exists()]
    log(f"cohort {len(df)} slides; fixed config {FIXED_GRID}; "
        f"{len(fold_jobs) - len(pending)}/{len(fold_jobs)} folds done, {len(pending)} to run")

    ran = 0
    for rep, k, f in pending:
        if a.max_seconds and (time.time() - t0) > a.max_seconds:
            log(f"time budget {a.max_seconds:.0f}s reached after {ran} folds; "
                f"stopping cleanly — rerun to resume")
            break
        fp = run_one_fold(rep, k, f, df, a.num_patches or None)
        fp.to_csv(fold_dir / f"r{rep}_{k}_f{f}.csv", index=False)
        ran += 1
        log(f"  fold done: repeat {rep} {ASSAYS[k]['name']:11s} fold {f} "
            f"({(time.time() - t0) / 60:.1f} min this run, {ran} folds)")

    # Assemble every complete (repeat, assay) cell into per_repeat.csv.
    rows = [r for (rep, k) in cells
            if (r := assemble_cell(rep, k, fold_dir, clin)) is not None]
    per_path = out / "per_repeat.csv"
    if rows:
        pd.DataFrame(rows).to_csv(per_path, index=False)
    n_cells_done = len(rows)
    if n_cells_done < len(cells):
        log(f"\n{n_cells_done}/{len(cells)} (repeat,assay) cells complete; "
            f"summary written once all folds are in")
        return
    per = pd.read_csv(per_path)
    summ = summarise(per)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "design": "repeated multi-label stratified 5-fold, fixed config BB",
        "seeds": [int(SEED + r) for r in sorted(per.repeat.unique())],
        "repeat_0_equals_locked": True,
        "pt_dir": str(a.pt_dir), "num_patches": a.num_patches or "whole pool",
        "clinical_comparator": "L2 logistic C=1, fold-honest, 6 variables",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "elapsed_min": round((time.time() - t0) / 60, 1),
    }, indent=2))
    log("\n=== summary across repeats ===")
    log(summ.to_string(index=False))
    log(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
