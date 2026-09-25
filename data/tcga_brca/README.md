# Input data

All files are derived from public, de-identified resources: The Cancer Genome
Atlas breast cohort (TCGA-BRCA, NCI Genomic Data Commons), The Cancer Imaging
Archive (TCIA), and the published supplement of Li H, Zhu Y, Burnside ES, et
al: *Radiology* 281:382-391, 2016. Patients are identified by the 12-character
TCGA patient barcode (`TCGA-XX-XXXX`), called `CLID` throughout the code. No
file contains dates of birth, names or any identifier beyond the public
barcode.

The cohort is the TCGA/TCIA breast radiogenomics subset of Li et al: 100
patients with recomputed genomic signature scores. Ninety-one patients have
clinical data and MRI radiomic features; 90 of those have a diagnostic H&E
slide with extracted features; 83 have both scores and slide features; and 82
remain after excluding the one patient treated with
neoadjuvant chemotherapy (`TCGA-BH-A0B6`, excluded in
`scripts/locked_rerun.py` before any partition is drawn).

## `Perou-TCGA-BRCA-metadata.csv` (100 rows, 16 columns)

Research recomputations of the four signatures from TCGA expression data,
as published for the Li et al 2016 cohort (genefu-style implementations,
Gendoo DMA et al, *Bioinformatics* 32:1097-1099, 2016; the `UNC_` columns are
the PAM50 centroid correlations). These are the prediction targets. They are
research versions, not commercial assay results.

| Column | Used as | Meaning |
|---|---|---|
| `CLID` | patient key | TCGA patient barcode |
| `GHI_RS Score` | continuous target (Oncotype) | 21-gene recurrence score, 0-100 |
| `GHI_RS_3Group` | binary label (Oncotype) | `High`, `Intermediate` -> 1; `Low` -> 0 |
| `NKI70` | continuous target (MammaPrint) | correlation with the 70-gene good-prognosis template |
| `Mammaprint` | binary label (MammaPrint) | `NKI70_Good` -> 1; `NKI70_Bad` -> 0 |
| `ROR_P` | continuous target (ROR-P) | PAM50 risk of recurrence with proliferation |
| `ROR-P Group` | binary label (ROR-P) | `high`, `med` -> 1; `low` -> 0 |
| `UNC_ROR_S` | continuous target (ROR-S) | PAM50 risk of recurrence, subtype only |
| `ROR-S Group` | binary label (ROR-S) | `high`, `med` -> 1; `low` -> 0 |
| `Pam50.Call` | descriptive | PAM50 intrinsic subtype (LumA 61, LumB 13, Basal 13, Her2 8, Normal 5) |
| `UNC_Basal`, `UNC_Her2`, `UNC_LumA`, `UNC_LumB`, `UNC_Norm`, `UNC_Prolif` | not used | centroid correlations and the proliferation score entering ROR-P |

The label maps above are `ASSAYS` in `scripts/locked_rerun.py`. The
orientation of each label (which class the model score should rank higher) is
fixed afterwards from the sign of the correlation between the continuous
target and the binary label, not chosen by hand; see the Methods. Alternative
cuts (High vs Intermediate+Low for Oncotype; high vs med+low for ROR) are
evaluated in `scripts/review_response_analyses.py` from the same columns.

## `clinical_91_from_vincenzo.csv` (91 rows, 115 columns)

The TCGA-BRCA clinical file (GDC / cBioPortal export format, upper-case
column names) restricted to the 91 patients of the radiogenomics subset, with
two added bookkeeping columns: `MRI_AVAILABLE` (the barcode again, present
for every patient with a TCIA MRI study, i.e. all 91) and `WSI` (GDC file UUID
and file name of the diagnostic slide, `<uuid>/<barcode>-01Z-00-DX1.<slide-uuid>.svs`,
which is how each slide was located on the GDC; empty for one patient without
a slide). The file was curated for our group by Vincenzo
Della Mea (University of Udine).

Columns read by the code:

| Column | Used for |
|---|---|
| `PATIENT_ID` | patient key (`CLID`) |
| `AGE` | covariate 1, age at diagnosis (years) |
| `ER_STATUS_BY_IHC` | covariate 2, `Positive` -> 1 else 0; also the ER status used for partition balancing, label composition and the ER-alone comparator |
| `PR_STATUS_BY_IHC` | covariate 3, `Positive` -> 1 else 0 |
| `HER2_FISH_STATUS`, `IHC_HER2` | covariate 4, HER2 in ASCO/CAP order: FISH `Positive`/`Negative` where informative, otherwise IHC `Positive`/`Negative`; equivocal or unavailable on both -> missing (one patient), who is excluded from the comparator analyses (n = 81) |
| `AJCC_NODES_PATHOLOGIC_PN` | covariate 5, nodal status: any value not starting with `N0` -> 1 |
| `AJCC_PATHOLOGIC_TUMOR_STAGE` | covariate 6, ordinal stage: I/IA 1, II/IIA 2, IIB 2.5, IIIA 3, IIIB 3.2, IIIC 3.5 |
| `MENOPAUSE_STATUS`, `HISTOLOGICAL_DIAGNOSIS` | cohort table only (`scripts/make_table1.py`) |

The six covariates form the pre-declared clinicopathological comparator
(`COV` and `load_clinical()` in `scripts/locked_floor_comparison.py`, repeated
verbatim in `scripts/review_response_analyses.py` and `scripts/repeated_cv.py`).
Grade and Ki-67 are not in the public extract. All other columns are carried
unchanged from the source file and are not read.

## `tcga_clinical_pathology_clean.csv` (334 rows, 25 columns)

A wider TCGA-BRCA clinical extract (lower-case GDC column names). The locked
run reads only `bcr_patient_barcode` and
`breast_carcinoma_estrogen_receptor_status` from it, to attach ER status to the
100 scored patients before the multi-label stratified partition is drawn
(`build_cohort()` in `scripts/locked_rerun.py`). Its ER values agree with
`ER_STATUS_BY_IHC` above for every patient in the cohort.

## `radiomics.csv` (91 rows, 37 columns)

The 36 DCE-MRI radiomic features of Li et al 2016, exactly as published in
their supplement (kinetic K1-K7, enhancement-variance E1-E4, texture T1-T14,
geometry G1-G3, margin M1-M3, size S1-S5), keyed by `CLID`. No image analysis
was performed in this study; the features enter
`scripts/radiomics_nested_cv.py` unchanged, are median-imputed and
standardised inside each training fold, and are joined to the shared
partition by `CLID` (82 patients).

## `pool_manifest.jsonl` (90 lines)

One JSON object per diagnostic slide with extracted features: `slide` (GDC
slide name, `<barcode>-01Z-00-DX1.<uuid>`), `source_patches` (number of
256-pixel tissue tiles at 20x that Trident produced for the slide) and `kept`
(size of the first-stage pool, 1,024). This documents which slides were used
and how heavily each was subsampled; the embeddings themselves are not
tracked (see the main README, "Patch embeddings").

## Not tracked

`_pool/` (1,024-patch `.npz` archives) and `_pool_pt/` (512-patch `.pt`
bundles) are the per-slide embedding pools described in the main README. They
are regenerated from the GDC slides and are excluded by `.gitignore`.
