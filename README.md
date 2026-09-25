# Label composition and what histology adds over routine clinicopathological variables

Code, locked results and manuscript for:

> Ali Y, Gregori J. *Label composition explains whether histology outperforms
> routine clinicopathological variables: a controlled comparison of four breast
> cancer risk signatures.* Submitted to JCO Clinical Cancer Informatics, 2026.

Four recomputed genomic risk signatures (Oncotype DX, MammaPrint, PAM50 ROR-S
and ROR-P) are predicted from H&E whole-slide images — UNI patch embeddings
pooled by gated-attention multiple-instance learning — in 82 TCGA-BRCA
patients, under one shared multi-label stratified partition and one
configuration fixed in advance. Each model is compared on identical folds, by
paired bootstrap, against ER status alone and a six-variable
clinicopathological model. The margin over the clinical model runs from +0.280
AUC (Oncotype) to −0.042 (MammaPrint), in almost the reverse of the order by
raw AUC, and is governed by *label composition*: how much of each binary label
ER status already explains. That ordering follows from the construction of the
scores and is stable across five independent partition draws.

## Layout

```
scripts/                 the pipeline, in the order it runs (see below)
src/mmfusion/            model, training and data code. Only the gated-attention
                         MIL arm (mmfusion.training.mil, mmfusion.models) is
                         used by this study; the package is shared with our
                         group's broader multimodal-fusion codebase and the
                         fusion modules are unused here.
data/tcga_brca/          clinical extract, recomputed signature scores (Perou /
                         Li et al 2016), pre-extracted radiomic features
Results_locked/modal/    the locked run: per-fold held-out predictions per
                         signature, run manifest (seed, configuration, package
                         versions), pre-declared comparators, and review/
                         (incremental value, tuned comparator, alternative
                         cuts, MDD, repeated-CV summary)
Results_locked/radiomics/  pre-extracted radiomic feature arm, nested vs
                         non-nested cross-validation
Results_repeatedcv/      five-partition repeat of the whole locked pipeline
manuscript/              manuscript.json (single source of truth), submission/
                         (Word and PDF, Data Supplement, figures), overleaf/
                         (LaTeX)
TRIPOD_AI.md             TRIPOD+AI reporting checklist
REFACTOR_NUMERICS.md     numerical provenance of the pipeline
```

## Regenerating

Everything downstream of the per-slide predictions regenerates in minutes on
CPU. The two training steps need the UNI patch pools (not tracked; see Data)
and take about 25 min (locked run) and 1.8 h (five-partition repeat) on CPU.

```
# training (needs the 512-patch pools)
python scripts/locked_rerun.py --pt_dir <pool> --out Results_locked --fixed_grid BB
python scripts/repeated_cv.py --pt_dir <pool> --out Results_repeatedcv \
    --clinical data/tcga_brca/clinical_91_from_vincenzo.csv --repeats 5

# comparators and analyses (CPU, seconds to minutes)
python scripts/locked_floor_comparison.py --results Results_locked/modal
python scripts/radiomics_nested_cv.py
python scripts/make_table1.py
python scripts/review_response_analyses.py --results Results_locked/modal \
    --clinical data/tcga_brca/clinical_91_from_vincenzo.csv \
    --perou data/tcga_brca/Perou-TCGA-BRCA-metadata.csv

# manuscript: tables, figures, Word, LaTeX -- every number checked against the CSVs
python scripts/manuscript_content.py --results Results_locked/modal \
    --radiomics Results_locked/radiomics --out manuscript/manuscript.json
python scripts/validate_manuscript.py --json manuscript/manuscript.json \
    --results Results_locked/modal --radiomics Results_locked/radiomics
python scripts/make_submission_figures.py --results Results_locked/modal --out figs
node   scripts/build_docx.js manuscript/manuscript.json figs manuscript/submission
python scripts/build_latex.py --json manuscript/manuscript.json --figures figs \
    --out manuscript/overleaf
```

`validate_manuscript.py` checks every number in the prose, tables and captions
against the result files and enforces the journal's limits; it exits non-zero
on any mismatch. Citations are numbered by first appearance when
`manuscript.json` is emitted.

Determinism: the locked run uses a fixed 512-patch pool per slide with a
recorded seed, so evaluation is deterministic given the pools; repeat 0 of
`Results_repeatedcv/` reproduces `Results_locked/modal/` exactly. Analyses
from the predictions onward are bit-reproducible (`requirements.txt` pins the
versions used).

## Data

- Whole-slide images and clinical data: NCI Genomic Data Commons, TCGA-BRCA.
- MRI: The Cancer Imaging Archive.
- Recomputed signature scores and the 36 DCE-MRI radiomic features:
  Li H et al, *Radiology* 281:382-391, 2016.
- Patch embeddings: tiles at 20x (256 px), encoded by the UNI foundation model
  with the Trident toolkit (Zhang A et al, arXiv:2502.06750, 2025), then
  reduced once per slide to a fixed pool of 512 patches
  (`scripts/pool_npz_to_pt.py`). The embeddings are not tracked here because
  of their size; they can be regenerated from the public slides.

## Setup

```
python -m pip install -r requirements.txt
python -m pip install -e .          # installs the mmfusion package
npm install docx@9.6.1              # only for the Word build
```

## License

MIT — see `LICENSE`.
