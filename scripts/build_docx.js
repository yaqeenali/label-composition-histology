// Renders manuscript.json to Word for JCO Clinical Cancer Informatics.
//
// Three files from one JSON, so they can never disagree:
//   manuscript.docx        title page, structured abstract, main text, declarations,
//                          references, then the main tables and figures
//   supplement.docx        the Data Supplement: supplementary methods and results,
//                          Tables S1-S5, Figure S1
//   reading_copy.docx      everything inline where it is first cited, supplement
//                          appended -- for co-authors, not for submission
//
// JCO CCI reviews single-blind, so no blinded copy is produced.
//
//   node scripts/build_docx.js manuscript.json figsDir outDir
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, Footer, PageNumber, LineRuleType, TableLayoutType,
  convertInchesToTwip,
} = require("docx");

const [, , JSON_PATH, FIG_DIR, OUT_DIR] = process.argv;
const M = JSON.parse(fs.readFileSync(JSON_PATH, "utf8"));

const FONT = "Times New Roman";
const SZ = 24;                               // 12 pt
const DOUBLE = { line: 480, after: 0 };
const SINGLE = { line: 240, after: 120 };
const RULE = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const NONE = { style: BorderStyle.NONE };

const P = (text, o = {}) => new Paragraph({
  spacing: o.spacing || DOUBLE, alignment: o.align, indent: o.indent,
  children: [new TextRun({ text, font: FONT, size: o.size || SZ,
                           bold: o.bold, italics: o.italics })],
});

const runs = (parts, o = {}) => new Paragraph({
  spacing: o.spacing || DOUBLE,
  children: parts.map(p => new TextRun({
    text: p.t, bold: p.b, italics: p.i, font: FONT, size: o.size || SZ })),
});

const H = (text, level) => new Paragraph({
  spacing: { before: 300, after: 140, line: 280 },
  heading: level,
  children: [new TextRun({
    text, font: FONT, bold: true, color: "000000",
    size: level === HeadingLevel.HEADING_1 ? 28 : 24,
    italics: level === HeadingLevel.HEADING_2,
  })],
});

function table(t) {
  const total = t.widths.reduce((a, b) => a + b, 0);
  const cell = (text, { w, head = false, align, indent = false } = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "EFEFEF", color: "auto" } : undefined,
    borders: { top: head ? RULE : NONE, bottom: RULE, left: NONE, right: NONE },
    margins: { top: 60, bottom: 60, left: indent ? 220 : 80, right: 80 },
    children: [new Paragraph({
      spacing: { line: 240, after: 0 }, alignment: align,
      children: [new TextRun({ text: String(text), bold: head, font: FONT, size: 19 })],
    })],
  });
  const body = (t.structured ? t.rows : t.rows.map(r => ({ indent: false, cells: r })));
  return new Table({
    width: { size: total, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    columnWidths: t.widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: t.headers.map((h, i) =>
          cell(h, { head: true, w: t.widths[i], align: i ? AlignmentType.RIGHT : undefined })),
      }),
      ...body.map(r => new TableRow({
        children: r.cells.map((c, i) => cell(c, {
          w: t.widths[i], indent: i === 0 && r.indent,
          align: i ? AlignmentType.RIGHT : undefined,
        })),
      })),
    ],
  });
}

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function figure(file, widthIn) {
  const data = fs.readFileSync(path.join(FIG_DIR, file));
  const { w, h } = pngSize(data);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 200, line: 240, lineRule: LineRuleType.AUTO },
    children: [new ImageRun({
      type: "png", data,
      transformation: { width: widthIn * 96, height: widthIn * 96 * (h / w) },
    })],
  });
}

function citedIn(text, pending) {
  // "Fig 1a", "Figs 1 and 2", "Table 2", "Table S1", "(Table 4, Fig 2a)".
  const labels = new Set();
  const scan = (re, word) => {
    for (const m of text.matchAll(re)) {
      for (const n of m[1].matchAll(/S?\d+/g)) labels.add(`${word} ${n[0]}`);
    }
  };
  scan(/Figs?\.?\s*((?:S?\d+[a-z]?)(?:\s*(?:,|and)\s*S?\d+[a-z]?)*)/g, "Fig");
  scan(/Tables?\s*((?:S?\d+)(?:\s*(?:,|and)\s*S?\d+)*)/g, "Table");
  return pending.filter(item => labels.has(item.label));
}

const tableBlock = t => [
  runs([{ t: `${t.label}. `, b: true }, { t: t.caption }], { spacing: SINGLE, size: 20 }),
  table(t),
  P("", { spacing: { after: 240, line: 240 } }),
];
const figureBlock = f => [
  runs([{ t: `${f.label}. `, b: true }, { t: f.caption }], { spacing: SINGLE, size: 20 }),
  figure(f.file, 6.3),
];

function titlePage(body) {
  body.push(new Paragraph({
    spacing: { after: 240, line: 320 },
    children: [new TextRun({ text: M.title, bold: true, font: FONT, size: 30 })],
  }));
  const authorRuns = [];
  M.authors.forEach((a, i) => {
    authorRuns.push(new TextRun({ text: a.name, font: FONT, size: SZ }));
    authorRuns.push(new TextRun({ text: a.aff.join(","), font: FONT, size: SZ, superScript: true }));
    if (i < M.authors.length - 1) authorRuns.push(new TextRun({ text: ", ", font: FONT, size: SZ }));
  });
  body.push(new Paragraph({ spacing: SINGLE, children: authorRuns }));
  M.affiliations.forEach((aff, i) =>
    body.push(P(`${i + 1} ${aff}`, { spacing: SINGLE, size: 20, italics: true })));
  const corr = M.authors.find(a => a.corresponding);
  body.push(P(`Corresponding author: ${corr.name}, ${M.affiliations[corr.aff[0] - 1]}; ${corr.email}`,
              { spacing: SINGLE, size: 20 }));
  body.push(P(`Submitted to ${M.journal} as an Original Report.`,
              { spacing: SINGLE, size: 20, italics: true }));
}

function abstractBlock(body) {
  body.push(H("Abstract", HeadingLevel.HEADING_1));
  M.abstract_structured.forEach(([h, t]) =>
    body.push(runs([{ t: h + " ", b: true }, { t }], { spacing: { line: 360, after: 140 } })));
  body.push(runs([{ t: "Keywords ", b: true }, { t: M.keywords.join(" · ") }],
                 { spacing: SINGLE }));
}

function sectionsBlock(body, sections, emit) {
  sections.forEach(sec => {
    body.push(H(sec.h, HeadingLevel.HEADING_1));
    (sec.paras || []).forEach(emit);
    (sec.subs || []).forEach(sub => {
      body.push(H(sub.h, HeadingLevel.HEADING_2));
      sub.paras.forEach(emit);
    });
  });
}

function declarationsBlock(body) {
  M.declarations.forEach(([k, v]) => {
    body.push(H(k, HeadingLevel.HEADING_2));
    body.push(P(v, { spacing: SINGLE }));
  });
}

function referencesBlock(body) {
  body.push(H("References", HeadingLevel.HEADING_1));
  M.references.forEach((r, i) => body.push(new Paragraph({
    spacing: { line: 300, after: 80 },
    indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.35) },
    children: [new TextRun({ text: `${i + 1}. ${r}`, font: FONT, size: 21 })],
  })));
}

function makeDoc(body, title) {
  return new Document({
    creator: M.authors[0].name, title,
    styles: {
      default: {
        document: { run: { font: FONT, size: SZ }, paragraph: { spacing: DOUBLE } },
        heading1: { run: { font: FONT, size: 28, bold: true, color: "000000" } },
        heading2: { run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" } },
      },
    },
    sections: [{
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })],
      })] }) },
      children: body,
    }],
  });
}

// ---- manuscript: floats after the references, as the journal wants -------
function buildMain() {
  const body = [];
  titlePage(body);
  abstractBlock(body);
  body.push(new Paragraph({ children: [new PageBreak()] }));
  sectionsBlock(body, M.sections, text => body.push(P(text)));
  declarationsBlock(body);
  referencesBlock(body);
  body.push(new Paragraph({ children: [new PageBreak()] }));
  M.tables.forEach(t => tableBlock(t).forEach(el => body.push(el)));
  body.push(new Paragraph({ children: [new PageBreak()] }));
  body.push(H("Figure legends", HeadingLevel.HEADING_1));
  M.figures.forEach(f => body.push(runs([{ t: `${f.label}. `, b: true }, { t: f.caption }],
                                         { spacing: SINGLE })));
  M.figures.forEach(f => { body.push(new Paragraph({ children: [new PageBreak()] }));
                           figureBlock(f).forEach(el => body.push(el)); });
  return makeDoc(body, M.title);
}

// ---- Data Supplement -------------------------------------------------------
function buildSupplement() {
  const S = M.supplement;
  const body = [];
  body.push(new Paragraph({
    spacing: { after: 240, line: 320 },
    children: [new TextRun({ text: `${S.title}: ${M.title}`, bold: true, font: FONT, size: 28 })],
  }));
  body.push(P(M.authors.map(a => a.name).join(", "), { spacing: SINGLE }));
  const pending = [...S.tables.map(t => ({ kind: "table", label: t.label, obj: t })),
                   ...S.figures.map(f => ({ kind: "figure", label: f.label, obj: f }))];
  const emit = text => {
    body.push(P(text));
    for (const item of citedIn(text, pending)) {
      pending.splice(pending.indexOf(item), 1);
      (item.kind === "table" ? tableBlock(item.obj) : figureBlock(item.obj)).forEach(el => body.push(el));
    }
  };
  sectionsBlock(body, S.sections, emit);
  if (pending.length) {
    body.push(H("Supplementary tables and figures", HeadingLevel.HEADING_1));
    pending.forEach(item =>
      (item.kind === "table" ? tableBlock(item.obj) : figureBlock(item.obj)).forEach(el => body.push(el)));
  }
  return makeDoc(body, `${S.title}: ${M.title}`);
}

// ---- reading copy: everything inline, supplement appended ------------------
function buildReadingCopy() {
  const S = M.supplement;
  const body = [];
  titlePage(body);
  body.push(P("Reading copy: tables and figures sit where they are first cited and the Data "
              + "Supplement is appended. Not the file to submit.",
              { spacing: SINGLE, size: 20, italics: true }));
  abstractBlock(body);
  body.push(new Paragraph({ children: [new PageBreak()] }));
  const pending = [...M.tables, ...S.tables].map(t => ({ kind: "table", label: t.label, obj: t }))
    .concat([...M.figures, ...S.figures].map(f => ({ kind: "figure", label: f.label, obj: f })));
  const emit = text => {
    body.push(P(text));
    for (const item of citedIn(text, pending)) {
      pending.splice(pending.indexOf(item), 1);
      (item.kind === "table" ? tableBlock(item.obj) : figureBlock(item.obj)).forEach(el => body.push(el));
    }
  };
  sectionsBlock(body, M.sections, emit);
  declarationsBlock(body);
  referencesBlock(body);
  body.push(new Paragraph({ children: [new PageBreak()] }));
  body.push(H(S.title, HeadingLevel.HEADING_1));
  sectionsBlock(body, S.sections, emit);
  if (pending.length) {
    body.push(H("Additional tables and figures", HeadingLevel.HEADING_1));
    pending.forEach(item =>
      (item.kind === "table" ? tableBlock(item.obj) : figureBlock(item.obj)).forEach(el => body.push(el)));
  }
  return makeDoc(body, M.title);
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const [name, build] of [["manuscript.docx", buildMain],
                               ["supplement.docx", buildSupplement],
                               ["reading_copy.docx", buildReadingCopy]]) {
    const buf = await Packer.toBuffer(build());
    fs.writeFileSync(path.join(OUT_DIR, name), buf);
    console.log(`${name}  ${(buf.length / 1024).toFixed(0)} KB`);
  }
})();
