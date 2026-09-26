#!/usr/bin/env python3
"""Continuous-score agreement, per-fold stability and calibration (Table S1).

Reads the held-out predictions written by ``locked_rerun.py`` and writes
``continuous_and_calibration.csv`` next to them:

  Pearson_r, r_CI      Pearson correlation of predicted with observed score over
                       all 82 patients, with a 95% percentile bootstrap interval
                       (4,000 patient resamples, seed 20260906)
  r_perfold_*          mean and SD (ddof = 1) of the per-fold correlations
  AUC_perfold_*        mean and SD (ddof = 1) of the per-fold AUCs against the
                       binary label
  calib_slope/intercept least-squares regression of observed on predicted score
  MAE                  mean absolute error

    python scripts/continuous_metrics.py --results Results_locked/modal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

SEED = 20260906
N_BOOT = 4000
ASSAYS = {"ghi": "GHI (Oncotype)", "mammaprint": "MammaPrint",
          "ror_p": "ROR-P", "ror_s": "ROR-S"}


def summarise(t: pd.DataFrame, name: str) -> dict:
    x, y = t.pred.to_numpy(float), t.true.to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = [np.corrcoef(x[i], y[i])[0, 1] for i in rng.integers(0, len(x), (N_BOOT, len(x)))]
    r_fold = [pearsonr(g.pred, g.true)[0] for _, g in t.groupby("outer_fold")]
    auc_fold = [roc_auc_score(g.binary, g.pred) for _, g in t.groupby("outer_fold")]
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "assay": name, "n": len(t),
        "Pearson_r": round(pearsonr(x, y)[0], 3),
        "r_CI": f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]",
        "r_perfold_mean": round(float(np.mean(r_fold)), 3),
        "r_perfold_sd": round(float(np.std(r_fold, ddof=1)), 3),
        "AUC_perfold_mean": round(float(np.mean(auc_fold)), 3),
        "AUC_perfold_sd": round(float(np.std(auc_fold, ddof=1)), 3),
        "calib_slope": round(float(slope), 3),
        "calib_intercept": round(float(intercept), 2),
        "MAE": round(float(np.abs(x - y).mean()), 2),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="locked results directory")
    a = p.parse_args(argv)
    res = Path(a.results)
    rows = [summarise(pd.read_csv(res / k / "test_predictions.csv"), v)
            for k, v in ASSAYS.items()]
    out = pd.DataFrame(rows)
    out.to_csv(res / "continuous_and_calibration.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
