#!/usr/bin/env python3
"""Cross-fitted combined model: clinical variables + histology score, without leakage.

In the combined ("stacked") model of ``review_response_analyses.py`` the
histology scores of training-fold patients were their pooled out-of-fold
predictions. For outer test fold f, a training patient in fold g got its score
from the model trained on all folds except g -- which includes fold f -- so the
combined model's training features depended on the held-out fold.

This script removes that dependence by cross-fitting:

  train   for every outer fold f and every other fold g != f, the locked
          histology model (configuration BB, same code path as
          ``locked_rerun.py``) is trained on the three folds outside {f, g}, with
          its own inner split for epoch selection, and scores fold g. These
          models never see fold f. 5 x 4 = 20 models per target, 80 in total.
  analyse for outer fold f, the combined model (six clinical variables + the
          standardized histology score, L2 penalty chosen by inner three-fold
          cross-validation exactly as in ``fold_honest(tune=True)``) is fitted
          on training patients carrying their cross-fitted scores and applied
          to fold f carrying its locked out-of-fold predictions (the model
          trained on all folds except f, as in standard stacking). It is
          compared with the tuned clinical model by paired bootstrap on the
          same resamples as before; Holm across the four targets.

Every trained model is checkpointed, so an interrupted run resumes.

    python scripts/crossfit_combined.py train   --pt_dir <pool> --results Results_locked/modal
    python scripts/crossfit_combined.py analyse --results Results_locked/modal \\
        --clinical data/tcga_brca/clinical_91_from_vincenzo.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

KEYS = ["ghi", "mammaprint", "ror_p", "ror_s"]


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def train(pt_dir: Path, results: Path, max_seconds: float, check: bool) -> None:
    from locked_rerun import (ASSAYS, EXCLUDE, N_INNER, N_OUTER, SEED, build_cohort,
                              iterative_stratify, make_cfg, make_partition, one_run)
    t0 = time.time()
    df = build_cohort(REPO / "data/tcga_brca/Perou-TCGA-BRCA-metadata.csv", pt_dir,
                      REPO / "data/tcga_brca/tcga_clinical_pathology_clean.csv")
    df = df[df.excluded_reason.isna()].reset_index(drop=True)
    df["outer_fold"] = make_partition(df, N_OUTER, SEED)
    locked = pd.read_csv(results / "shared_partition.csv")
    same = df.set_index("CLID").outer_fold.reindex(locked.CLID).values == locked.outer_fold.values
    if not same.all():
        raise SystemExit("partition does not reproduce shared_partition.csv")
    out = results / "crossfit"; out.mkdir(parents=True, exist_ok=True)
    labels = [s["binary_target_col"] for s in ASSAYS.values()] + ["ER_pos"]

    if check:
        # harness check: retrain the locked model for one fold with this code
        # path and compare with the locked held-out predictions
        key, k = "ghi", 0
        spec = ASSAYS[key]
        work = df[df[spec["binary_target_col"]].notna() & df[spec["target_col"]].notna()].copy()
        test = work[work.outer_fold == k]
        dev = work[work.outer_fold != k].reset_index(drop=True)
        dev["inner_fold"] = iterative_stratify(dev[labels].fillna(0).values.astype(int),
                                               N_INNER, SEED + 1000 + k)
        res = one_run(make_cfg(key, "BB", None), dev[dev.inner_fold != 0],
                      dev[dev.inner_fold == 0], test, SEED + k)
        ref = pd.read_csv(results / key / "test_predictions.csv")
        ref = ref[ref.outer_fold == k].set_index("CLID").pred.reindex(test.CLID).values
        diff = float(np.max(np.abs(np.asarray(res.test_preds) - ref)))
        print(f"harness check (GHI, fold 0): max |pred - locked pred| = {diff:.2e}", flush=True)
        if diff > 1e-6:
            raise SystemExit("harness does not reproduce the locked run")

    for key in KEYS:
        spec = ASSAYS[key]
        work = df[df[spec["binary_target_col"]].notna() & df[spec["target_col"]].notna()].copy()
        cfg = make_cfg(key, "BB", None)
        for f in range(N_OUTER):
            for g in range(N_OUTER):
                if g == f:
                    continue
                ck = out / f"{key}_f{f}_g{g}.csv"
                if ck.exists():
                    continue
                if max_seconds and time.time() - t0 > max_seconds:
                    print("time budget reached; rerun to resume", flush=True)
                    return
                test = work[work.outer_fold == g]
                dev = work[~work.outer_fold.isin([f, g])].reset_index(drop=True)
                dev["inner_fold"] = iterative_stratify(dev[labels].fillna(0).values.astype(int),
                                                       N_INNER, SEED + 2000 + 10 * f + g)
                res = one_run(cfg, dev[dev.inner_fold != 0], dev[dev.inner_fold == 0], test,
                              SEED + 3000 + 10 * f + g)
                pd.DataFrame({"CLID": test.CLID.values, "excluded_fold": f, "scored_fold": g,
                              "true": res.test_targets, "pred": res.test_preds,
                              "n_train": int((dev.inner_fold != 0).sum()),
                              "n_val": int((dev.inner_fold == 0).sum())}).to_csv(ck, index=False)
                print(f"  {key} f{f} g{g}: trained on {len(dev)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print("all cross-fitted models done", flush=True)


# --------------------------------------------------------------------------- #
# analyse
# --------------------------------------------------------------------------- #

def analyse(results: Path, clinical: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from review_response_analyses import (C_GRID, COV, NAMES, SEED, ci, fold_honest, holm,
                                          load_clinical, paired_boot, per_fold_auc)
    clin = load_clinical(clinical)
    cf_dir = results / "crossfit"
    rows = []
    for key in KEYS:
        label = NAMES[key]
        d = pd.read_csv(results / key / "test_predictions.csv").merge(clin, on="CLID", how="left")
        d = d.dropna(subset=COV + ["binary"]).reset_index(drop=True)
        y = d["binary"].values.astype(int)
        sign = np.sign(np.corrcoef(d["true"].values, y)[0, 1])
        cf = pd.concat([pd.read_csv(p) for p in sorted(cf_dir.glob(f"{key}_f*_g*.csv"))])
        if len(cf.excluded_fold.unique()) != 5:
            raise SystemExit(f"{key}: cross-fitted models incomplete")

        clin_tuned, _ = fold_honest(d, COV, "binary", tune=True)
        combined = pd.Series(index=d.index, dtype=float)
        chosen = []
        for f in sorted(d.outer_fold.unique()):
            tr, te = d[d.outer_fold != f].copy(), d[d.outer_fold == f].copy()
            cmap = cf[cf.excluded_fold == f].set_index("CLID").pred
            tr["h"] = sign * cmap.reindex(tr.CLID).values
            te["h"] = sign * te.pred.values
            if tr.h.isna().any():
                raise SystemExit(f"{key} fold {f}: missing cross-fitted score")
            cols = COV + ["h"]
            Xtr, Xte = tr[cols].values.astype(float), te[cols].values.astype(float)
            ytr = tr["binary"].values.astype(int)
            sc = StandardScaler().fit(Xtr)
            best, best_s = None, -np.inf
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED + int(f))
            for cc in C_GRID:
                s = []
                for itr, iva in inner.split(Xtr, ytr):
                    if len(np.unique(ytr[iva])) < 2:
                        continue
                    m = LogisticRegression(C=cc, max_iter=5000).fit(sc.transform(Xtr[itr]), ytr[itr])
                    s.append(roc_auc_score(ytr[iva], m.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
                if s and np.mean(s) > best_s:
                    best_s, best = np.mean(s), cc
            chosen.append(best)
            m = LogisticRegression(C=best, max_iter=5000).fit(sc.transform(Xtr), ytr)
            combined.loc[te.index] = m.predict_proba(sc.transform(Xte))[:, 1]
        comb = combined.values
        a_c, a_t = roc_auc_score(y, comb), roc_auc_score(y, clin_tuned)
        fm = per_fold_auc(d, comb, "binary")
        lo, hi, p = paired_boot(y, comb, clin_tuned)
        r_cf = cf.groupby("excluded_fold").apply(
            lambda x: np.corrcoef(x.true, x.pred)[0, 1], include_groups=False)
        rows.append({"assay": label, "n": len(d),
                     "AUC_clin_tuned": round(a_t, 3),
                     "AUC_combined_cf": round(a_c, 3),
                     "AUC_combined_cf_foldmean": round(fm[0], 3),
                     "AUC_combined_cf_foldsd": round(fm[1], 3),
                     "d_incremental_cf": round(a_c - a_t, 3), "CI_incremental_cf": ci(lo, hi),
                     "p_incremental_cf": round(p, 4),
                     "C_chosen": " ".join(str(c) for c in chosen),
                     "crossfit_r_mean": round(float(r_cf.mean()), 3),
                     "crossfit_n_train_range": f"{int(cf.n_train.min())}-{int(cf.n_train.max())}"})
    R = pd.DataFrame(rows)
    R["p_incremental_cf_holm"] = np.round(holm(R.p_incremental_cf.values), 4)
    R.to_csv(results / "review" / "combined_crossfit.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["train", "analyse"])
    p.add_argument("--pt_dir", default=None)
    p.add_argument("--results", required=True)
    p.add_argument("--clinical", default=str(REPO / "data/tcga_brca/clinical_91_from_vincenzo.csv"))
    p.add_argument("--max_seconds", type=float, default=0)
    p.add_argument("--check", action="store_true", help="first reproduce one locked fold")
    a = p.parse_args(argv)
    if a.step == "train":
        if not a.pt_dir:
            raise SystemExit("--pt_dir is required for training")
        train(Path(a.pt_dir), Path(a.results), a.max_seconds, a.check)
    else:
        analyse(Path(a.results), Path(a.clinical))


if __name__ == "__main__":
    main()
