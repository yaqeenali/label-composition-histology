#!/usr/bin/env python3
"""The confirmatory run: one partition, one grid, nested selection.

Why this script exists
======================
Reading ``mmfusion.training.mil`` settles what the leak in the original results
is, and is not.  The test split is **not** used during training or model
selection: selection is on validation, the classification threshold is tuned on
validation, and test is touched once, after training ends.  There is no in-loop
leakage.

The contamination is a level up, and it has two parts.

**Configuration.**  Across 459 development runs the four assays acquired six
differing hyperparameters -- width, dropout, weight decay, patch count, loss and
selection rule -- chosen while test performance was visible.

**Partition.**  The four assays use *four different* 5-fold partitions of the
same 83 patients.  GHI's fold 0 test set is not MammaPrint's.  So the
cross-assay table, which is the centre of the paper, confounds assay against
split.

Both are fixed here:

1. **One shared partition.**  A single multi-label stratified 5-fold, balanced
   simultaneously on all four binary targets and on ER status, used identically
   by every assay.  Differences between assays are then differences between
   assays.
2. **One pre-specified grid, selected inside the training folds.**  The grid is
   not invented: it is the two configurations that actually appear in the
   original notebooks, plus their two crossings, so that neither the capacity
   choice nor the objective choice is inherited from test-informed tuning.
   Selection runs as an inner 3-fold CV over the outer fold's development set;
   the test fold is not read until the outer model is finished.
3. **A fixed patch pool.**  Every slide is represented by the same number of
   patches (see ``pool_from_coords.py`` and ``pool_npz_to_pt.py``), which removes both the GHI-vs-rest
   patch-count divergence and the per-access ``torch.randperm`` draw recorded as
   N-10.  Evaluation becomes deterministic.

Deliberate divergences from the published numbers are listed as D-entries in
``REFACTOR_NUMERICS.md``.  Nothing under ``Results/`` is read or written.

    python scripts/locked_rerun.py --pt_dir <pool> --out Results_locked
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mmfusion.config import AssayConfig, MILModelConfig, MILTrainConfig  # noqa: E402
from mmfusion.seeding import set_seed  # noqa: E402
from mmfusion.training.mil import run_fold  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-specification.  Everything in this block is fixed before any test split is
# read, and is written verbatim into the run manifest.
# ---------------------------------------------------------------------------

SEED = 20260906
N_OUTER = 5
N_INNER = 3
EPOCHS = 20

#: The four assays, as the notebooks defined them.  Only the *targets* differ
#: here -- every hyperparameter now comes from the grid below.
ASSAYS = {
    "ghi": dict(name="GHI", target_col="GHI_RS Score", group_col="GHI_RS_3Group",
                binary_target_col="GHI_bin",
                label_map={"High": 1, "Intermediate": 1, "Low": 0}),
    "mammaprint": dict(name="MammaPrint", target_col="NKI70", group_col="Mammaprint",
                       binary_target_col="Mammaprint_bin",
                       label_map={"NKI70_Good": 1, "NKI70_Bad": 0}),
    "ror_p": dict(name="ROR-P", target_col="ROR_P", group_col="ROR-P Group",
                  binary_target_col="ROR_P_bin",
                  label_map={"high": 1, "med": 1, "low": 0}),
    "ror_s": dict(name="ROR-S", target_col="UNC_ROR_S", group_col="ROR-S Group",
                  binary_target_col="ROR_S_bin",
                  label_map={"high": 1, "med": 1, "low": 0}),
}

#: Capacity and objective blocks, taken from the two configurations that the
#: original notebooks actually used.  "A" is the GHI notebook, "B" is the other
#: three.  The grid is the 2x2, so neither axis is inherited from tuning.
CAPACITY = {
    "A": dict(hidden_dim=256, dropout=0.50, weight_decay=1e-3),
    "B": dict(hidden_dim=1024, dropout=0.25, weight_decay=1e-4),
}
OBJECTIVE = {
    "A": dict(loss_type="quantile", selection_metric="val_loss"),
    "B": dict(loss_type="huber", selection_metric="val_pearson"),
}
GRID = [f"{c}{o}" for c in CAPACITY for o in OBJECTIVE]

#: Excluded before any split is drawn.  Pre-surgical chemotherapy alters H&E
#: morphology; this is a standard exclusion and it is applied to every assay.
EXCLUDE = {"TCGA-BH-A0B6": "neoadjuvant chemotherapy"}


def grid_config(key: str) -> Dict:
    cap, obj = CAPACITY[key[0]], OBJECTIVE[key[1]]
    return {**cap, **obj}


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------

def build_cohort(perou_csv: Path, pt_dir: Path, clinical_csv: Path | None) -> pd.DataFrame:
    """Perou scores joined to available feature bundles, with binary targets."""
    perou = pd.read_csv(perou_csv)
    perou["CLID"] = perou.CLID.astype(str).str.strip().str[:12]

    bundles = {p.name[:12]: p for p in sorted(pt_dir.glob("*.pt"))}
    perou["pt_path"] = perou.CLID.map(lambda c: str(bundles[c]) if c in bundles else None)
    df = perou[perou.pt_path.notna()].copy()

    for spec in ASSAYS.values():
        df[spec["binary_target_col"]] = df[spec["group_col"]].map(spec["label_map"])

    if clinical_csv is not None and Path(clinical_csv).exists():
        clin = pd.read_csv(clinical_csv)
        idcol = "bcr_patient_barcode" if "bcr_patient_barcode" in clin.columns else "CLID"
        clin = clin.rename(columns={idcol: "CLID"})
        ercol = next((c for c in clin.columns if "estrogen_receptor_status" in c), None)
        if ercol:
            df = df.merge(clin[["CLID", ercol]].rename(columns={ercol: "ER"}), on="CLID", how="left")
    if "ER" not in df.columns:
        df["ER"] = "Unknown"
    df["ER_pos"] = (df.ER == "Positive").astype(int)

    df["excluded_reason"] = df.CLID.map(EXCLUDE)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# One shared partition
# ---------------------------------------------------------------------------

def iterative_stratify(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Multi-label stratified k-fold assignment (Sechidis et al., 2011).

    ``labels`` is [n_samples, n_labels] binary.  Samples are placed one at a
    time, rarest label first, into whichever fold is furthest below its target
    count for that label -- so every fold carries a similar positive rate for
    *all* labels at once.  A single ordinary StratifiedKFold cannot do this,
    and with four targets plus ER status the alternative is folds where one
    assay has a single-class test set.
    """
    n, m = labels.shape
    rng = np.random.default_rng(seed)
    fold_of = np.full(n, -1, dtype=int)

    desired = np.full(k, n / k, dtype=float)
    desired_lab = np.array([[labels[:, j].sum() / k for j in range(m)] for _ in range(k)])
    remaining = set(range(n))

    while remaining:
        rem = np.array(sorted(remaining))
        counts = labels[rem].sum(0)
        positive = np.where(counts > 0)[0]
        if len(positive) == 0:                      # only all-zero rows left
            order = rng.permutation(rem)
            for i in order:
                f = int(np.argmax(desired))
                fold_of[i] = f
                desired[f] -= 1
                remaining.discard(int(i))
            break

        j = int(positive[np.argmin(counts[positive])])   # rarest remaining label
        members = rng.permutation(rem[labels[rem, j] == 1])
        for i in members:
            col = desired_lab[:, j]
            best = np.flatnonzero(col == col.max())
            if len(best) > 1:
                sub = desired[best]
                best = best[np.flatnonzero(sub == sub.max())]
            f = int(best[0] if len(best) == 1 else rng.choice(best))
            fold_of[i] = f
            desired[f] -= 1
            desired_lab[f] -= labels[i]
            remaining.discard(int(i))
    return fold_of


def make_partition(df: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    cols = [s["binary_target_col"] for s in ASSAYS.values()] + ["ER_pos"]
    labels = df[cols].fillna(0).values.astype(int)
    return iterative_stratify(labels, k, seed)


def partition_report(df: pd.DataFrame, fold_col: str = "outer_fold") -> pd.DataFrame:
    rows = []
    for f in sorted(df[fold_col].unique()):
        sub = df[df[fold_col] == f]
        row = {"fold": f, "n": len(sub)}
        for spec in ASSAYS.values():
            b = spec["binary_target_col"]
            row[f"{b}_pos"] = int(sub[b].sum())
        row["ER_pos"] = int(sub.ER_pos.sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def make_cfg(assay_key: str, grid_key: str, num_patches, results_dir: str = "") -> AssayConfig:
    spec = ASSAYS[assay_key]
    g = grid_config(grid_key)
    return AssayConfig(
        name=f"{spec['name']}__{grid_key}",
        target_col=spec["target_col"],
        group_col=spec["group_col"],
        binary_target_col=spec["binary_target_col"],
        label_map=spec["label_map"],
        binary_class=False,
        results_dir=results_dir,
        model=MILModelConfig(input_dim=1024, hidden_dim=g["hidden_dim"], dropout=g["dropout"]),
        train=MILTrainConfig(
            epochs=EPOCHS, lr=1e-4, weight_decay=g["weight_decay"], grad_clip=1.0,
            batch_size=1, shuffle_train=False, num_patches=num_patches,
            loss_type=g["loss_type"], quantile_tau=0.5, weight_bins=10,
            selection_metric=g["selection_metric"],
        ),
    )


def one_run(cfg: AssayConfig, train, val, test, seed: int, verbose=False):
    """A single train/val/test cycle at a fixed seed, no artifacts written."""
    set_seed(seed)
    return run_fold(cfg, "x", {"train": train, "val": val, "test": test},
                    fold_dir=None, write_artifacts=False, verbose=verbose)


def run_assay(assay_key: str, df: pd.DataFrame, out: Path, num_patches,
              verbose: bool = True, fixed_grid=None) -> Dict:
    spec = ASSAYS[assay_key]
    bcol = spec["binary_target_col"]
    work = df[df[bcol].notna() & df[spec["target_col"]].notna()].copy()

    inner_records, outer_records, preds = [], [], []
    for k in range(N_OUTER):
        test = work[work.outer_fold == k]
        dev = work[work.outer_fold != k].reset_index(drop=True)

        # ---- inner selection: the test fold is not read in this block -------
        inner_lab = dev[[s["binary_target_col"] for s in ASSAYS.values()] + ["ER_pos"]]
        dev["inner_fold"] = iterative_stratify(
            inner_lab.fillna(0).values.astype(int), N_INNER, SEED + 1000 + k)

        scores = {}
        for gkey in (GRID if fixed_grid is None else []):
            cfg = make_cfg(assay_key, gkey, num_patches)
            rs = []
            for j in range(N_INNER):
                itr, iva = dev[dev.inner_fold != j], dev[dev.inner_fold == j]
                if iva[bcol].nunique() < 2 or len(iva) < 3:
                    continue
                r = one_run(cfg, itr, iva, iva, SEED + 10 * k + j)
                rs.append(r.metrics.get("Pearson", np.nan))
                inner_records.append({"assay": spec["name"], "outer": k, "grid": gkey,
                                      "inner": j, "val_pearson": r.metrics.get("Pearson")})
            scores[gkey] = float(np.nanmean(rs)) if rs else -np.inf
            if verbose:
                print(f"  [{spec['name']}] outer {k} grid {gkey}: "
                      f"inner mean r = {scores[gkey]:.4f}", flush=True)

        # ``fixed_grid`` skips the search and uses one pre-declared setting.
        # The rule that picks it has to be test-independent: "BB" is the
        # configuration three of the four original notebooks shared, adopted as
        # the modal setting -- not as the best-performing one.
        if fixed_grid is not None:
            best = fixed_grid
            scores[best] = float("nan")
        else:
            best = max(scores, key=scores.get)

        # ---- refit at the chosen setting, then read test once ---------------
        cfg = make_cfg(assay_key, best, num_patches)
        tr = dev[dev.inner_fold != 0]
        va = dev[dev.inner_fold == 0]
        res = one_run(cfg, tr, va, test, SEED + k)

        outer_records.append({"assay": spec["name"], "outer_fold": k,
                              "selected": best, "inner_score": scores[best],
                              "n_train": len(tr), "n_val": len(va), "n_test": len(test),
                              **{m: res.metrics.get(m) for m in
                                 ("Pearson", "R2", "MAE", "Binary_AUC",
                                  "Binary_Accuracy", "Binary_F1")}})
        preds.append(pd.DataFrame({
            "CLID": test.CLID.values, "outer_fold": k, "assay": spec["name"],
            "selected": best, "true": res.test_targets, "pred": res.test_preds,
            "binary": res.test_binary if res.test_binary is not None else np.nan,
        }))
        if verbose:
            print(f"  [{spec['name']}] outer {k}: selected {best}  "
                  f"test r={res.metrics.get('Pearson'):.3f} "
                  f"AUC={res.metrics.get('Binary_AUC')}", flush=True)

    adir = out / assay_key
    adir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inner_records).to_csv(adir / "inner_selection.csv", index=False)
    pd.DataFrame(outer_records).to_csv(adir / "outer_folds.csv", index=False)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_csv(adir / "test_predictions.csv", index=False)
    return {"assay": spec["name"], "n": len(work),
            "selected": [r["selected"] for r in outer_records]}


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt_dir", required=True, help="directory of pooled *.pt bundles")
    p.add_argument("--perou_csv", default=str(REPO / "data/tcga_brca/Perou-TCGA-BRCA-metadata.csv"))
    p.add_argument("--clinical_csv",
                   default=str(REPO / "data/tcga_brca/tcga_clinical_pathology_clean.csv"))
    p.add_argument("--out", default=str(REPO / "Results_locked"))
    p.add_argument("--assays", nargs="*", default=list(ASSAYS))
    p.add_argument("--num_patches", type=int, default=0,
                   help="0 = use the whole fixed pool (recommended, deterministic)")
    p.add_argument("--keep_neoadjuvant", action="store_true")
    p.add_argument("--fixed_grid", default=None, choices=GRID,
                   help="Skip the inner search and use this pre-declared setting. "
                        "BB is the configuration 3 of the 4 notebooks shared.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = build_cohort(Path(args.perou_csv), Path(args.pt_dir), Path(args.clinical_csv))
    n_all = len(df)
    if not args.keep_neoadjuvant:
        df = df[df.excluded_reason.isna()].reset_index(drop=True)

    df["outer_fold"] = make_partition(df, N_OUTER, SEED)
    rep = partition_report(df)
    df[["CLID", "outer_fold", "ER_pos"] +
       [s["binary_target_col"] for s in ASSAYS.values()]].to_csv(
        out / "shared_partition.csv", index=False)
    rep.to_csv(out / "partition_balance.csv", index=False)

    print(f"cohort: {n_all} with scores and features -> {len(df)} after exclusions")
    print("\nshared partition balance (positives per fold):")
    print(rep.to_string(index=False))
    print()

    summaries = []
    for a in args.assays:
        print(f"===== {ASSAYS[a]['name']} =====", flush=True)
        summaries.append(run_assay(a, df, out, args.num_patches or None,
                                   verbose=not args.quiet, fixed_grid=args.fixed_grid))

    manifest = {
        "git_sha": git_sha(),
        "seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER, "epochs": EPOCHS,
        "grid": {g: grid_config(g) for g in GRID},
        "selection": ("pre-declared modal configuration " + args.fixed_grid
                      if args.fixed_grid else "nested inner CV"),
        "excluded": EXCLUDE if not args.keep_neoadjuvant else {},
        "cohort_n": len(df),
        "pt_dir": str(args.pt_dir),
        "num_patches": args.num_patches or "whole pool",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "selected_per_assay": {s["assay"]: s["selected"] for s in summaries},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwritten to {out}   ({manifest['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
