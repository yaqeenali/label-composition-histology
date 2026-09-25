# review/

Outputs of `scripts/review_response_analyses.py` on the locked predictions
(comparators.csv, subgroups.csv, binarisation_sensitivity.csv, mdd.csv).

`provisional_per_target.csv` is a verbatim copy of
`Results_indication/clinical_floor_comparison.csv`: the same pre-declared
comparator (L2 logistic regression, C = 1, ASCO/CAP HER2, 4000-resample paired
bootstrap, seed 20260906) applied to the *provisional* per-target pipelines --
per-target partitions, per-target tuned hyperparameters, a random patch
subsample at every access, and the neoadjuvant patient still included (n = 82
= 83 minus the HER2-unresolved patient). It is the "before" column of the
design-findings table in the manuscript.
