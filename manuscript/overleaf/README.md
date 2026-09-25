# Overleaf project

Upload `overleaf-manuscript.zip` with **New Project -> Upload Project**. It
compiles with pdfLaTeX; nothing beyond a standard TeX Live installation is
needed.

```
main.tex            manuscript (structured abstract, main text, declarations,
                    references, main tables and figures)
supplement.tex      Data Supplement (supplementary methods and results,
                    Tables S1-S5, Figure S1)
figures/            600 dpi figures: Fig1, Fig2, FigS1
```

## Before you edit

**The tables are generated, not typed.** `scripts/manuscript_content.py` reads
the locked result files and writes `manuscript.json`; this project and the Word
version are both rendered from it. A numeric correction belongs in the analysis,
not in the `.tex`, and re-running the two build scripts regenerates everything.
Citations are numbered by first appearance when the JSON is emitted.

**Line numbers and double spacing are on**, which is what reviewers want.
Comment out `\linenumbers` and change `\doublespacing` for the camera-ready
version.

## Journal

Prepared for JCO Clinical Cancer Informatics (ASCO): Original Report, structured
abstract of at most 275 words, body of at most 3,000 words, at most six tables
and figures in the main text, single-blind review, Data Supplement allowed.
