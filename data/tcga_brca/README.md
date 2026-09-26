# Input data

All files are derived from public, de-identified resources: The Cancer Genome
Atlas breast cohort (TCGA-BRCA, NCI Genomic Data Commons) and the TCIA
TCGA-Breast-Radiogenomics collection (Morris E, et al. The Cancer Imaging
Archive, 2014, https://doi.org/10.7937/K9/TCIA.2014.8SIPIY6G; CC BY 3.0),
from which the score, radiomic-feature and 334-subject clinical tables are
taken. Li H, Zhu Y, Burnside ES, et al: *Radiology* 281:382-391, 2016 describe
how the research versions of the signature scores were computed. Patients are identified by the 12-character
TCGA patient barcode (`TCGA-XX-XXXX`), called `CLID` throughout the code. No
file contains dates of birth, names or any identifier beyond the public
barcode.

The score file covers 100 TCGA-BRCA patients. Slide features were extracted
only for the 91 patients of the TCGA breast radiogenomics (MRI) subset, which
the clinical and radiomics files cover; 90 of them have a diagnostic slide.
84 of the 100 scored patients belong to the MRI subset, 83 of these have a
slide, and 82 remain after excluding the one patient recorded as having
received neoadjuvant treatment (`TCGA-BH-A0B6`, excluded in
`scripts/locked_rerun.py` before any partition is drawn). The 16 scored
patients outside the MRI subset were not processed.

## `Perou-TCGA-BRCA-metadata.csv` (100 rows, 16 columns)

From the TCIA file `Perou-TCGA-BRCA-MRIsPAM50GHI21NKI70-MAILED.xlsx`.
Research versions of the four signatures, computed at the University of North
Carolina by applying the published models to TCGA RNA-sequencing data, as
described by Li et al 2016 (the `UNC_` columns are the PAM50 centroid
correlations). These are the prediction targets. They are research versions,
not commercial assay results. The Oncotype groups follow cut points of 18 and
31 on the recomputed 0-100 scale; 74 of the 100 patients are in the High group
and 31 scores are at the ceiling of 100, so this label is not equivalent to
clinical recurrence-score categories.

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
| `HER2_FISH_STATUS`, `IHC_HER2` | covariate 4, HER2: FISH `Positive`/`Negative` where available, otherwise IHC `Positive`/`Negative`; equivocal or unavailable on both -> missing (one patient), who is excluded from the comparator analyses (n = 81). Five patients have both results; they disagree for one (IHC 2+ recorded as positive, FISH negative), who is coded negative |
| `AJCC_NODES_PATHOLOGIC_PN` | covariate 5, nodal status: any value not starting with `N0` -> 1 (this includes one `NX` patient) |
| `AJCC_PATHOLOGIC_TUMOR_STAGE` | covariate 6, numeric stage score: I/IA 1, II/IIA 2, IIB 2.5, IIIA 3, IIIB 3.2, IIIC 3.5 |
| `MENOPAUSE_STATUS`, `HISTOLOGICAL_DIAGNOSIS` | cohort table only (`scripts/make_table1.py`) |

The six covariates form the clinicopathological comparator (L2 penalty C = 1,
fixed before the final run)
(`COV` and `load_clinical()` in `scripts/locked_floor_comparison.py`, repeated
verbatim in `scripts/review_response_analyses.py` and `scripts/repeated_cv.py`).
Grade and Ki-67 are not in the public extract. All other columns are carried
unchanged from the source file and are not read.

## `tcga_clinical_pathology_clean.csv` (334 rows, 25 columns)

The TCIA clinical table `brca-clinicalforwiki.xls` (334 subjects; lower-case
GDC column names). The locked
run reads only `bcr_patient_barcode` and
`breast_carcinoma_estrogen_receptor_status` from it, to attach ER status to the
100 scored patients before the multi-label stratified partition is drawn
(`build_cohort()` in `scripts/locked_rerun.py`). Its ER values agree with
`ER_STATUS_BY_IHC` above for every patient in the cohort.

## `radiomics.csv` (91 rows, 37 columns)

36 pre-extracted DCE-MRI radiomic features for the 91 cases of the MRI
subset (kinetic K1-K7, enhancement-variance E1-E4, texture T1-T14, geometry
G1-G3, margin M1-M3, size S1-S5), keyed by `CLID`, from the TCIA file
`TCGA-Run-2014_91cases_features_UChicago-V2010-MRI-Workstation.xls`
(University of Chicago MRI Quantitative Radiomics workstation, V2010). These
are the computer-extracted image phenotypes analyzed by Li et al 2016, who
describe 38 phenotypes for 84 of these cases. The table used here has 36: all
size, shape, margin, texture and enhancement-variance features and seven of
the nine kinetic-curve features (total rate variation and normalized total
rate variation are absent). Li et al give the TCIA DOI above as the access
route for the data. No
image analysis was performed in this study; the features enter
`scripts/radiomics_nested_cv.py` unchanged (no values are missing), are
standardised inside each training fold, and are joined to the shared
partition by `CLID` (82 patients).

## `pool_manifest.jsonl` (90 lines)

One JSON object per diagnostic slide with extracted features: `slide` (GDC
slide name, `<barcode>-01Z-00-DX1.<uuid>`), `source_patches` (number of
256-pixel tissue tiles at 20x that Trident produced for the slide) and `kept`
(size of the first-stage pool, 1,024). This documents which slides were used
and how heavily each was subsampled; the embeddings themselves are not
tracked (see the main README, "Patch embeddings").

## `patch_coords_1024.csv.gz` (92,160 rows)

The patches used from each slide: one row per patch of the first-stage pool,
1,024 per slide for the 90 slides in `pool_manifest.jsonl`. Columns: `slide`
(GDC slide name), `pool_index` (0 to 1,023, the order of the patch in the
pool), `x`, `y` (level-0 pixel coordinates exactly as stored in the Trident
output) and `in_512` (1 for the 512 patches used in training and evaluation,
the subset that `scripts/pool_npz_to_pt.py --keep 512` selects). With the
Trident output of a slide, `scripts/pool_from_coords.py` rebuilds that
slide's pool from this table (see the main README, "Patch embeddings").

## Not tracked

`_pool/` (1,024-patch `.npz` archives) and `_pool_pt/` (512-patch `.pt`
bundles) are the per-slide embedding pools described in the main README. They
are rebuilt from the GDC slides with `scripts/pool_from_coords.py` and
`scripts/pool_npz_to_pt.py`, and are excluded by `.gitignore`.
