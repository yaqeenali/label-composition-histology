// Cover letter, rendered from manuscript.json so the title, journal and
// corresponding author cannot drift from the manuscript.
//
//   node scripts/build_cover_letter.js manuscript.json outDir
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun } = require("docx");

const [, , JSON_PATH, OUT_DIR] = process.argv;
const M = JSON.parse(fs.readFileSync(JSON_PATH, "utf8"));
const corr = M.authors.find(a => a.corresponding);
const FONT = "Times New Roman";

const P = (text, o = {}) => new Paragraph({
  spacing: { line: 276, after: 160 },
  children: [new TextRun({ text, font: FONT, size: 24, bold: o.bold, italics: o.italics })],
});

const body = [
  P(`To the Editors, ${M.journal}`),
  P("Re: submission of an Original Report", { italics: true }),
  P("Dear Editors,"),
  P(`Please consider the enclosed manuscript, “${M.title},” for publication as an Original Report.`),
  P("Deep learning on hematoxylin and eosin sections is now well established to predict the "
    + "Oncotype DX recurrence score beyond routine clinicopathological variables, in cohorts of "
    + "several thousand patients. Every one of those studies predicts the same signature. Our "
    + "manuscript asks whether the result generalizes across signatures, finds that it does not "
    + "generalize uniformly, and shows that the direction of the effect is predictable from the "
    + "labels alone — from how much of each binarized signature label estrogen receptor status "
    + "already explains, a quantity that follows from the construction of the scores and can be "
    + "computed before any model is trained."),
  P("Comparing four recomputed genomic risk signatures in one cohort under a single design — "
    + "one shared multi-label stratified partition, one configuration fixed in advance, the same "
    + "model for every target — we find that the margin over a six-variable clinicopathological "
    + "model ranges from +0.280 AUC to −0.042, in almost the reverse of the order by raw "
    + "discrimination, and in exactly the order predicted from the score definitions. Across "
    + "seven label definitions, including three alternative cuts of the same scores, the margin "
    + "falls monotonically as the label becomes more nearly a restatement of ER status, and the "
    + "result is stable across five independent partition draws. We distinguish throughout "
    + "between outperforming the clinical variables and adding to them, and report the "
    + "incremental analysis with the events-per-variable constraint that limits it at this "
    + "sample size."),
  P("Two methodological results are quantified rather than asserted: moving from per-target "
    + "partitions and tuning to one shared design reordered all four signatures, and nesting "
    + "the feature selection in a pre-extracted radiomic feature arm on the same folds removed "
    + "most of that arm's apparent performance, an optimism of up to 0.121 AUC."),
  P("We believe this fits the scope of JCO Clinical Cancer Informatics: it is a study of how "
    + "clinical prediction models built on routine pathology images should be designed, "
    + "benchmarked and reported, it follows TRIPOD+AI, it reports bounded null results alongside "
    + "a positive one, and its recommendations cost nothing to adopt. We are explicit that the "
    + "cohort is small (n = 82), that only one of the four margins survives adjustment for "
    + "multiple comparisons, that the targets are research recomputations rather than assay "
    + "results, and that no prognostic claim is made. All analysis code, a locked run manifest "
    + "and the derived result tables are available."),
  P("The manuscript is original, is not under consideration elsewhere, and all authors have "
    + "approved the submission. A Data Supplement with supplementary methods, results, five "
    + "tables and one figure accompanies it."),
  P("Thank you for your consideration."),
  P("Yours sincerely,"),
  P(`${corr.name}, on behalf of the authors`),
  P(M.affiliations[corr.aff[0] - 1]),
  P(corr.email),
];

(async () => {
  const doc = new Document({
    creator: corr.name, title: "Cover letter",
    sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
                 children: body }],
  });
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, "cover_letter.docx"), await Packer.toBuffer(doc));
  console.log(`cover_letter.docx written to ${OUT_DIR}`);
})();
