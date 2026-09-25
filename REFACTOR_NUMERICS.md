# Numerical change register

Every judgement call in the refactor that could touch a number, what was decided,
and the test that holds the decision in place.

**The contract:** no scientific calculation was changed. Known defects are
reproduced exactly and marked `BUG-PRESERVED` in the source; fixing any of them
is a separate change that must be asked for, because each one moves published
numbers.

Run the evidence with:

```bash
pip install -e ".[dev]"
pytest                 # 194 tests, ~70 s
pytest -m "not slow"   # skip the end-to-end pipeline runs, ~10 s
```

An independent audit was run against this refactor after the first suite was
green. It found two divergences I had introduced (N-19, D-07), six behaviours
that were preserved but *untested* — mutating them left the suite passing —
and five tests whose assertions were weaker than their names suggested. All are
fixed; `tests/test_parity_gaps.py` holds the tripwires that were missing.

---

## How parity is established

`legacy/_frozen/` holds the pre-refactor sources, plus verbatim extractions of
the `run_cv` cell from each pathology notebook. Nothing imports them except the
tests. Every parity test runs the frozen implementation and the packaged one over
the same input from the same seed and asserts **exact** equality — not
`approx`, except in the golden tests where the comparison is against values
already rounded into a CSV.

The strongest evidence is `tests/test_parity_mil.py`: it runs the whole
cross-validated MIL pipeline both ways and compares the metrics CSVs *and* the
per-case prediction CSVs. A divergence anywhere — target scaling, feature
standardisation, model init, dropout, patch subsampling, epoch order, model
selection, threshold tuning, resampling, metric assembly — would surface there.

The one edit made to the frozen sources: intra-project imports were rewritten
(`from config import` → `from legacy._frozen.config import`) so the frozen code
imports its own copies rather than the new shims. Nothing else was touched;
`transfer.py`'s byte-identity is asserted directly in
`test_parity_transfer.py::test_source_is_unmodified`.

---

## N — preserved behaviour (nothing changed; the test stops it changing)

### N-01 · `set_seed` stays session-level, not per-fold
`mmfusion/seeding.py`. The original seeded once per session, so fold *k*'s RNG
state depends on every fold before it and a single fold cannot be re-run in
isolation. Reseeding per fold would be an improvement and would change every
result, so it was not done. `seeded_rng()` exists for new code and is used by no
legacy path.
→ `test_parity_mil.py::test_run_is_reproducible_from_the_same_seed`

### N-02 · The 242/42 seed discrepancy is preserved
`run_training.py` called `set_seed(242)`; `make_splits.py` and every notebook use
42. `mmfusion-train` defaults to 242, `mmfusion-train-mil` to 42 — each matching
what its predecessor did. The value is now a `--seed` argument and is recorded,
rather than being a literal at the bottom of a module.

### N-03 · `bootstrap_ci` still draws from the unseeded global RNG
Passing an explicit `rng=` is possible but changes the numbers, so it defaults to
`None`. **Consequence:** the `Bootstrap_CI_*` values in `Results/` cannot be
reproduced by any implementation, including the original.
→ `test_parity_metrics.py::test_bootstrap_ci_parity`
→ `test_golden_results.py::test_resampled_metrics_are_not_reproducible_from_artifacts`

### N-04 · The `permutation_test` defect is preserved
`mmfusion/metrics/inference.py`. The null is built from
`pearson_r(y_true, permutation(y_true))` — the predictions never enter it. Every
`Permutation_p` in `Results/` came from this. A corrected
`permutation_test_paired` sits beside it and is deliberately **not** wired into
`regression_report`.
→ `test_parity_metrics.py::test_permutation_test_bug_is_preserved` reconstructs
the null from `y_true` alone and asserts the returned p-value matches it. Fixing
the bug fails this test by design.

### N-05 · Resampling call order is fixed
`regression_report` calls `permutation_test` (1000 global draws) **before**
`bootstrap_ci` (2000). Both read the same global stream, so swapping them changes
both results.

### N-06 · `auroc` and `f1_score_macro` still swallow every exception
A single-class fold becomes `NaN` rather than an error. Preserved, because
`Results/` contains such folds.
→ `test_parity_metrics.py::test_auroc_single_class_returns_nan_in_both`

### N-07 · `PredictionHead` keeps its final `nn.Tanh()`
`mmfusion/models/common.py`. Bounds every fusion-arm output to [-1, 1]: a 0-100
recurrence score is unreachable, classification logits are clamped so sigmoid
lands in ~[0.27, 0.73], Cox log-hazards are squashed. All 459 W&B runs and the
CHIMERA checkpoints were trained with it.
→ `test_parity_models.py::test_prediction_head_tanh_is_preserved`

### N-08 · Tabular column selection is unchanged
A tabular modality with empty `feature_cols` still selects every numeric column
except the case column — which, on this project's split CSVs, means the target
columns. The selection is identical; a `UserWarning` was added when a target is
among them. **The leak is still there; it is now loud.**
→ `test_parity_data.py::test_tabular_column_resolution_parity`
→ `test_parity_data.py::test_target_leak_warning_only_fires_when_a_target_is_selected`

### N-09 · `log_test_every_epoch=False` is **not** numerically neutral
This was my working assumption and it was wrong; the test caught it.
`DataLoader.__iter__` draws once from the global torch RNG **every time it is
called, shuffled or not**. Removing the per-epoch test pass removes a draw and
shifts the stream, changing trained weights from that epoch on.

The flag stays, defaulting to `True` (legacy). Use it for **new** runs where the
test split should stay unseen during development — never expecting to reproduce
an existing run.
→ `test_parity_training.py::test_dataloader_iteration_always_draws_from_the_global_rng`
→ `test_parity_training.py::test_test_logging_toggle_changes_results`

### N-10 · Patch subsampling still draws from the global RNG at every access
`WSIMILDataset.__getitem__` calls `torch.randperm` whenever `num_patches` is set
and the slide is larger — including during evaluation, so test predictions depend
on the draw. `deterministic_subsample=True` fixes this per item; it changes
results and is off by default.
→ `test_parity_data.py::test_wsi_mil_subsampling_consumes_one_draw_per_oversized_slide`
→ `test_parity_data.py::test_deterministic_subsample_is_opt_in_and_changes_results`

### N-11 · `compute_sample_weights` keeps its `np.zeros_like(y)` cast
An integer target truncates every inverse-frequency weight to an integer (0.4 →
0, 1.6 → 1). Only reachable via the `weighted_huber` path.
→ `test_parity_data.py::test_sample_weight_dtype_truncation_is_preserved`

### N-12 · Missing modalities are still zero-filled with no indicator
`None` → zeros in the collate function; nothing distinguishes "missing" from
"genuinely zero". The `print` became a `logger.warning`.
→ `test_parity_data.py::test_multimodal_dataset_item_parity`

### N-13 · The `val_pearson` selection rule is fragile at small *n* (found by a test)
When validation Pearson is `NaN` at every epoch, `r > best_r` never fires,
`best_state` stays `None`, and `load_state_dict(None)` raises. Reproduced
faithfully — both implementations fail identically. Surfaced while building the
fixtures with a 2-case validation split; the real folds have 9-10, so it did not
bite, but it is a live fragility of that rule.

### N-14 · Model submodule construction order is unchanged
Every `nn.Linear` draws at construction, so the order of submodule creation fixes
the initial weights of the whole model. The three fusion models' shared preamble
moved into a `_FusionBase`, and the order was preserved exactly.
→ `test_parity_models.py::test_initial_weights_identical` compares every tensor
in the `state_dict` after identical seeding, for 3 architectures × 4 task shapes.

### N-15 · The tabular CSV is read once instead of twice — column selection identical
The original read headers with `nrows=0` and then the whole file for dtypes. Only
column *names* came from the first read, so one read gives the same answer.
→ `test_parity_data.py::test_tabular_column_resolution_parity`

### N-16 · Duplicate attention-heatmap writes removed — same files on disk
The GHI cell rendered the aggregate heatmap twice to each of two paths (a
copy-paste duplication): same arrays, same filenames, second write overwrote the
first. The duplicates are gone; the files produced are unchanged. No RNG.
→ `test_parity_mil.py::test_artifacts_flag_is_numerically_neutral`

### N-17 · Fold iteration now skips non-directories
`for f in sorted(os.listdir(csv_path))` had no filter and would crash on a stray
`.DS_Store` — of which this project has several. Any run that previously
*succeeded* saw only fold directories, so the visit order is unchanged. Order
matters (it sets the cross-fold RNG sequence), which is why this is listed.

### N-18 · `train_one_epoch` dropped an unused parameter
It took `task_cfg` and never used it. The packaged function drops it; the
`train.py` shim wraps both it and `_extract_modalities` so the *original* call
signatures still work for un-migrated notebooks.

### N-19 · A custom `es_split` must not gain an evaluation pass — **a bug I introduced and reverted**
My first version appended `es_split` to the per-epoch evaluation list when early
stopping monitored a split outside `("val", "test")`, reasoning that silently not
firing early stopping was worse. That was wrong on the same mechanism as N-09:
the extra `DataLoader.__iter__` is an extra RNG draw, so trained weights diverged
from the original. With `es_split="train"` it was worse still — the eval-mode
loss overwrote `train_loss`, changing the selection signal itself.

Reverted to the original's exact `("val", "test")` iteration. When `es_split` is
outside that set the loop now logs a warning and, as before, selects no epoch.
→ `test_parity_gaps.py::test_custom_es_split_does_not_add_an_evaluation_pass`
compares against the frozen `fit` for `es_split` in `{"train", "holdout"}`.

### N-20 · `assign_weights` — the untested half of N-11
Its sibling `compute_sample_weights` was covered; this one was not, and mutating
its `np.zeros_like` left the suite green. It bites harder: normalised inverse
frequencies are all < 1, so an integer target zeroes *every* validation and test
weight rather than truncating some.
→ `test_parity_gaps.py::test_assign_weights_parity`,
`::test_assign_weights_integer_truncation_is_preserved`

### N-21 · `find_best_threshold` takes the FIRST tied maximum
`np.argmax` semantics. On 9-10 validation patients ties occur (~1% of folds in
simulation); the choice propagates into `Best_Threshold`, `Binary_Accuracy`,
`Binary_F1` and the pooled `Global_Best_Threshold`.
→ `test_parity_gaps.py::test_find_best_threshold_takes_the_first_maximum`

### N-22 · Early stopping skips NaN scores without consuming patience
A NaN metric neither improves the best nor counts toward patience — so it cannot
end training early. Reachable whenever validation Pearson collapses.
→ `test_parity_gaps.py::test_early_stopping_skips_nan_without_counting_patience`,
`::test_early_stopping_keeps_the_first_of_tied_scores`

### N-23 · `n_val = max(1, int(len(train) * val_fraction))` truncates, never rounds
Rounding instead would change fold membership at realistic cohort sizes (n=57 →
6 vs 7; n=66 → 7 vs 8) and therefore every downstream number. The original
fixture (n=60) gave the same answer either way, which is why this went untested.
→ `test_parity_gaps.py::test_val_fraction_truncates_rather_than_rounds`
(parametrised over n = 57, 64, 66, 83, 100)

### N-24 · `_FusionBase._tokens` iterates `self.modalities`, not the input dict
The stacking order fixes both the token sequence and the per-modality
`torch.rand` draw order in modality dropout. Every earlier test happened to build
its input dict in declaration order, so iterating the caller's dict instead would
have passed. It does not, once the caller reorders.
→ `test_parity_gaps.py::test_token_order_follows_declared_modalities_not_input_dict`

---

## D — deliberate divergences (behaviour differs; none changes a computed number)

### D-01 · Legacy `binary_class` path gained a single-class AUC guard
The `binary_class=True and binary_target_col is None` branch called
`roc_auc_score` with no guard and would raise on a single-class fold; it now
returns `NaN` like every other AUC call. **Unreachable in this project** — ROR-P
and ROR-S set both flags, so the branch never ran.

### D-02 · `ClinicalPreprocessor.get_feature_names()` now exists
The clinical notebook already called it inside a bare `try/except`. Because it
did not exist, every fold silently wrote the *raw* column names into
`feature_names.json` while the saved tensors were the wider one-hot expansion.
The method now exists, so the manifest describes the tensors.
**Metadata only — no tensor, no metric changes.**
→ `test_parity_transfer.py::test_get_feature_names_now_exists_and_matches_width`
asserts the transformed arrays remain identical.

### D-03 · `--es_higher_is_better` gained a negative form
It was `action="store_true", default=True`, so it could never be turned off —
early stopping on a loss-type metric selected the *worst* epoch. Now
`BooleanOptionalAction` with the same default: a strict superset, so every
existing command line behaves identically.

### D-04 · `*_bin.csv` is written to a separate directory by default
The notebooks wrote derived labels back into the split folders, mutating inputs
as a side effect of analysis. `mmfusion-binarise` writes to
`<splits_dir>_binary` unless `--in_place` is passed. Same rows, same values.

### D-06 · MIL `grad_clip: 0` now means "off", not "zero every gradient"
The MIL loop clipped unconditionally (the notebooks hard-coded 1.0, so the value
was unreachable). `grad_clip` is now a settable config field with the same name
and meaning as the fusion arm's, where a `> 0` guard always existed. The guard
was added to match. Unreachable in any legacy configuration.
→ `test_parity_gaps.py::test_mil_grad_clip_zero_disables_clipping_rather_than_zeroing_gradients`

### D-07 · `binary_splits_dir` — **a workflow bug I introduced and fixed**
D-04 made `mmfusion-binarise` write out of place, but `mmfusion-train-mil` still
read `splits_dir`. On the real data, where `*_bin.csv` already sits beside the
splits from the notebooks, training would silently consume the *stale* labels and
every binary metric would be computed from them. `AssayConfig` gained
`binary_splits_dir` (empty = `splits_dir`, i.e. legacy), `run_cv` reads it, and
the binarise CLI now refuses to finish quietly when the two disagree.

### D-08 · `write_binary_splits` warns on a missing split file
It skipped silently; the originals raised. Skipping without a word could produce
an incomplete set of derived labels.

### D-09 · `add_binary_label` sorts unmapped values by `str`
A group column mixing types raised `TypeError` from `sorted()` *before* the
informative message. Same values, same error path, now reachable.

### D-10 · `_pool_global_metrics` restored to two independent `if`s
I had turned the second into an `elif`. Mutually exclusive in practice, but with
zero folds it changed which branch ran. Control flow now matches the original.

### D-05 · Paths are resolved through `mmfusion.paths.rebase`
Identity unless `--rewrite_root` is given, so a rebased path resolves to the same
file. This is what makes the project runnable off the machine that produced it.
→ `test_parity_data.py::test_path_resolver_defaults_to_identity`

---

## The locked re-run (`scripts/locked_rerun.py`)

These divergences are **deliberate and large**. They define a new confirmatory
analysis rather than reproducing the old one, and they write only to
`Results_locked*/`. Nothing under `Results/` is read or written.

### D-11 · One shared partition replaces four different ones
The four assays used four *different* 5-fold partitions of the same 83 patients
— verified by comparing test-set membership, which agrees on no fold and on no
unordered partition either. Any cross-assay comparison therefore confounded
assay against split. The locked run draws **one** multi-label stratified 5-fold,
balanced simultaneously on all four binary targets and on ER status, and every
assay uses it. Fold membership is written to `shared_partition.csv` and the
per-fold positive counts to `partition_balance.csv`.

### D-12 · A fixed patch pool replaces per-access subsampling
Each slide is reduced once, with a recorded per-slide seed, to a fixed pool
(`scripts/extract_patch_pool.py`). This removes two things at once: the
GHI-uses-500-patches / everyone-else-uses-all divergence, and the per-access
`torch.randperm` draw recorded as N-10, which made a fold's test predictions
depend on how many times the loader had been called before them. Evaluation is
now deterministic. Cost: models see a sample of each slide rather than all of
it, so absolute performance is not comparable to the published numbers.

### D-13 · Hyperparameters come from a pre-declared grid, chosen inside training folds
Across 459 development runs the four assays acquired six differing
hyperparameters, selected while test metrics were visible. The locked run fixes
a 2×2 grid — the capacity block and the objective block that the original
notebooks actually used, plus their two crossings — and selects between them by
inner cross-validation **within each outer fold's development set**. A
`--fixed_grid BB` mode instead applies the configuration three of the four
notebooks shared, adopted by a test-independent rule (it is the modal setting,
not the best-performing one).

### D-14 · One patient excluded before any split is drawn
`TCGA-BH-A0B6` received neoadjuvant chemotherapy, which alters H&E morphology.
Standard exclusion, applied identically to every assay, cohort 83 → 82.
`--keep_neoadjuvant` restores it.

### D-15 · Radiomics selection rebuilt as nested CV — **closes audit B2**
`scripts/radiomics_nested_cv.py` fits imputation, scaling, univariate filtering
and the classifier's regularisation strength inside each training fold. It also
reproduces the notebook's flat procedure alongside, so the optimism is measured
rather than asserted: **+0.02 to +0.12 AUC**, largest for MammaPrint.

---

## Audit findings: current status

| Audit | Status |
|---|---|
| B1 test-set visibility | **Reclassified.** Reading `training/mil.py` shows the test split is *not* used during training or model selection — selection is on validation, the threshold is tuned on validation, test is read once after training. There is no in-loop leakage. The real contamination is config- and partition-level, closed by D-11 and D-13. `log_test_every_epoch` remains relevant to the *fusion* arm only (N-09). |
| B2 radiomics selection leakage | **Closed** by D-15. |
| C1 permutation test | Open — swap in `permutation_test_paired` and recompute every p-value (N-04) |
| C2 output tanh | Open — remove it and retrain the fusion arm (N-07) |
| C3 tabular target leak | Open — make `feature_cols` mandatory (N-08 warns; it does not block) |
| S1 pooled headline metrics | `CVResult.per_fold_summary()` is written alongside, not instead |

The structure is in place for each remaining item: one config field or one
function swap, with a test that will fail loudly when it happens.
