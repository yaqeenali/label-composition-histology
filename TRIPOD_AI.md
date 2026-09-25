# TRIPOD+AI reporting checklist

Against Collins et al., *BMJ* 2024;385:e078378. Three columns matter: what the
item asks, where this study answers it, and — the useful one — **whether this
cohort can answer it at all**. Items marked ✗ are not oversights to be filled in
later; they are consequences of the data and belong in the limitations.

Status is against the locked analysis (`scripts/locked_rerun.py`), not the
notebook results. Table and figure labels refer to the JCO Clinical Cancer
Informatics submission: Tables 1–4 and Figs 1–2 in the main text, Tables S1–S5
and Fig S1 in the Data Supplement.

| # | Item | Where | Can we? |
|---|---|---|---|
| 1 | Title identifies the study as developing a prediction model, names the target and the data source | Title/abstract | ✓ Must say *recomputed signature score*, never the brand name |
| 2 | Abstract: TRIPOD+AI for abstracts | Abstract | ✓ |
| 3 | Background and rationale | Introduction | ✓ |
| 4 | Objectives, including whether development or validation | Introduction | ✓ Development only. No external validation set exists |
| 5 | Data source and separation of development/evaluation | Methods · Data | ✓ TCGA-BRCA, the 100-patient radiogenomics subset of Li et al. 2016 |
| 6 | Eligibility, with dates | Methods · Cohort | ✓ 83 = signature scores ∩ extracted features; 82 after excluding one neoadjuvant case |
| 7 | Outcome definition, blinded assessment | Methods · Targets | ⚠ The target is a `genefu` recomputation from expression, not an assay result and not a patient outcome. This must be stated as a limitation of construct, not just measurement |
| 8 | Predictors: how and when measured | Methods · Slide processing; Radiomic feature arm | ✓ UNI (`uni_v1`) patch embeddings from diagnostic H&E, extracted beforehand with the Trident toolkit as preparatory work; 36 pre-extracted Li et al. radiomic features from DCE-MRI, used as published |
| 9 | Missing data handling | Methods | ✓ Complete-case for imaging; median imputation inside each training fold for radiomics. HER2 by ASCO/CAP order, one patient uninformative |
| 10 | Sample size, with justification | Methods · Statistical analysis; Limitations | ✓ **Not** a formal size calculation — a resolution statement. Minimum detectable AUC difference 0.135–0.154 at n = 81 (Hanley–McNeil, both AUCs 0.80, ρ = 0.5, α = 0.05, power 0.80), 0.165–0.221 in the ER+/HER2− subgroup. Events per variable reported for every fitted comparator |
| 11 | Data preparation, including any resampling | Methods | ✓ Fixed seeded patch pool per slide; feature standardisation fitted on training folds only |
| 12 | Model type and rationale | Methods · Model | ✓ Gated-attention MIL over patch embeddings |
| 13 | Model building: predictor selection, hyperparameters, all tuning | Methods · Partition and configuration | ✓ One configuration, fixed before the confirmatory run and identical for every target, chosen as the modal setting of the four earlier per-target pipelines. Its provenance (those pipelines had seen these data) is stated in the Methods and Limitations. **This is the item the original results failed** |
| 14 | Approach to fairness / subgroup performance | Results · Subgroups | ⚠ Reported for ER and HER2 subgroups. Race is recorded but too sparse to analyse |
| 15 | Model output and how it is converted to a decision | Methods | ✓ Continuous score; discrimination assessed by AUC against the published binary group call, so no decision threshold is set or tuned |
| 16 | Performance measures, with rationale | Methods · Comparators; Statistical analysis | ✓ AUC (pooled and per fold) and Pearson r; histology compared head-to-head with ER alone and with a pre-declared six-variable clinical model (penalty-tuned comparator as sensitivity); incremental value by a stacked model and by an adjusted likelihood-ratio test; Holm adjustment across the four targets |
| 17 | Model updating / recalibration | — | ✗ Not applicable; no external data |
| 18 | Participant flow, with a diagram | Figure S1 (Data Supplement) | ✓ 100 with scores, 83 of them with extracted slide features, 82 after excluding one neoadjuvant case; 7 slides with features but no score were never eligible |
| 19 | Participant characteristics | Table 1 | ✓ `Results_locked/table1.csv` |
| 20 | Number of outcome events | Table 1 | ✓ And this is where the study stops: **3 recurrences, 1 death, median 42 months** |
| 21 | Model specification: all coefficients or a means to reproduce | Repository | ✓ Weights are not interpretable coefficients; the run manifest records git SHA, seed, grid, package versions |
| 22 | Model performance with confidence intervals | Results | ✓ Bootstrap CIs throughout; paired bootstrap for every model-vs-comparator comparison, with Holm-adjusted p values |
| 23 | Calibration | Data Supplement · Table S1 | ✓ Slope and intercept of observed on predicted continuous scores reported for every target; at n = 82 a calibration plot would be descriptive only |
| 24 | Model updating results | — | ✗ Not applicable |
| 25 | Interpretation in the context of objectives and evidence | Discussion | ✓ |
| 26 | Limitations, including generalisability | Discussion | ✓ Single cohort, surrogate target, no events, no external validation, sample of each slide rather than the whole slide |
| 27 | Usability in context; implications | Discussion | ⚠ Must state plainly that no clinical use is supported by this evidence |
| 28 | Data availability | Statement | ✓ TCGA-BRCA is public; the signature scores derive from published expression data |
| 29 | Code availability | Statement | ✓ Repository, with the locked run reproducible from the manifest |
| 30 | Funding and conflicts | Statement | ✓ To complete |

## Items this cohort cannot satisfy, and what each forecloses

- **No time-to-event data worth analysing** (item 20). Rules out survival
  analysis, decision-curve analysis and any chemotherapy-benefit statement, at
  any sample size. This is the single largest gap against competing papers.
- **No external validation** (items 4, 17, 24). The strongest available
  substitute is a pre-registered locked analysis on one shared partition, which
  is what `locked_rerun.py` produces.
- **Surrogate construct** (item 7). The models predict a recomputation of a
  signature, not the commercial assay and not the patient's course. Every claim
  has to be phrased at that level.
- **Underpowered model comparison** (item 10). At a detectable-difference floor
  of 0.135–0.154 AUC, comparisons between model architectures cannot be
  reported as findings, and the incremental (clinical + histology) analysis is
  estimable only for the two targets with more than four events per variable.
