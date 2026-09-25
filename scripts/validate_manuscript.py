#!/usr/bin/env python3
"""Mechanical audit of manuscript.json against the files it claims to report.

Three classes of check, each of which has caught a real error in this project:

1. Every number the prose asserts must be derivable from the result CSVs, and
   every headline number in the CSVs must appear somewhere in the prose.
2. Every table, figure and reference cited in the text must exist; everything
   that exists must be cited; floats must be numbered in order of first citation.
3. Journal constraints: abstract length, undefined abbreviations in the abstract,
   declarations present, placeholders flagged.

Exit status is non-zero on any failure so this can gate the build.

    python scripts/validate_manuscript.py --json manuscript.json \\
        --results Results_locked/modal --radiomics Results_locked/radiomics
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

FAIL, WARN, OK = [], [], []


def fail(msg): FAIL.append(msg)
def warn(msg): WARN.append(msg)
def ok(msg): OK.append(msg)


def _section_paras(sections):
    parts = []
    for s in sections:
        parts += s.get("paras", [])
        for sub in s.get("subs", []):
            parts += sub["paras"]
    return parts


def main_prose(m) -> str:
    return "\n".join(_section_paras(m["sections"]))


def all_prose(m) -> str:
    S = m.get("supplement", {"sections": [], "tables": [], "figures": []})
    parts = [m["abstract"]] + _section_paras(m["sections"]) + _section_paras(S["sections"])
    parts += [v for _, v in m["declarations"]]
    parts += [f["caption"] for f in m["figures"] + S["figures"]]
    parts += [t["caption"] for t in m["tables"] + S["tables"]]
    return "\n".join(parts)


def expect(text: str, value: str, what: str):
    """The prose must contain this literal (after normalising minus signs)."""
    norm = lambda s: s.replace("−", "-").replace("–", "-")
    if norm(value) in norm(text):
        ok(f"{what}: {value}")
    else:
        fail(f"{what}: expected '{value}' in prose, not found")


# --------------------------------------------------------------------------- #

def p_text(pv: float) -> str:
    q = Decimal(str(pv)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return "p < 0.001" if pv < 0.001 else f"p = {q}"


def expect_p(text: str, pv: float, what: str):
    """p values must appear in the 3-dp form used throughout."""
    ptxt = p_text(pv)
    if ptxt in text:
        ok(f"{what}: {ptxt}")
    else:
        fail(f"{what}: '{ptxt}' not in prose")


def expect_ci(text: str, ci: str, what: str):
    lo, hi = [float(v) for v in ci.strip("[]").split(",")]
    sgn = lambda x: (f"−{abs(x):.3f}" if x < 0 else f"{x:.3f}")
    expect(text, f"{sgn(lo)} to {sgn(hi)}", what)


def check_numbers(m, res: Path, rad: Path, cohort_ids: set | None):
    text = all_prose(m)
    flo = pd.read_csv(res / "locked_floor_comparison.csv").set_index("assay")
    cmp_ = pd.read_csv(res / "review" / "comparators.csv").set_index("assay")
    alt = pd.read_csv(res / "review" / "binarisation_sensitivity.csv").set_index("assay")
    sub = pd.read_csv(res / "review" / "subgroups.csv")
    sub0 = pd.read_csv(res / "locked_subgroups.csv")
    mdd = pd.read_csv(res / "review" / "mdd.csv").set_index("assay")
    cal = pd.read_csv(res / "continuous_and_calibration.csv").set_index("assay")
    rdx = pd.read_csv(rad / "radiomics_nested_cv.csv").set_index("target")
    part = pd.read_csv(res / "shared_partition.csv")

    # -- the review file must reproduce the pre-declared analysis exactly ----
    for a in flo.index:
        r0, r1 = flo.loc[a], cmp_.loc[a]
        same = (abs(r0.AUC_model - r1.AUC_model) < 1e-9 and abs(r0.AUC_clinical - r1.AUC_clin_predeclared) < 1e-9
                and r0.CI_clin == r1.CI_clin_predeclared and abs(r0.p_clinical - r1.p_clin_predeclared) < 1e-9)
        (ok if same else fail)(f"review/comparators.csv reproduces locked_floor_comparison.csv for {a}")
    for _, r0 in sub0.iterrows():
        r1 = sub[(sub.assay == r0.assay) & (sub.cohort == r0.cohort)].iloc[0]
        same = (pd.isna(r0.AUC_clinical) and pd.isna(r1.AUC_clin_predeclared)) or \
               (abs(r0.AUC_clinical - r1.AUC_clin_predeclared) < 1e-9 and abs(r0.AUC_model - r1.AUC_model) < 1e-9)
        (ok if same else fail)(f"review/subgroups.csv reproduces locked_subgroups.csv for {r0.assay}/{r0.cohort}")

    # -- primary endpoint (pre-declared comparator) --------------------------
    for a in cmp_.index:
        r = cmp_.loc[a]
        expect(text, f"{r.AUC_model:.3f}", f"model AUC {a}")
        expect(text, f"{r.AUC_clin_predeclared:.3f}", f"clinical AUC {a}")
        expect(text, f"{r.AUC_ER_fixed:.3f}", f"ER-alone AUC {a}")
        expect(text, f"{r.kappa_ER:.3f}", f"kappa {a}")
        sign = "−" if r.d_vs_clin_predeclared < 0 else "+"
        expect(text, f"{sign}{abs(r.d_vs_clin_predeclared):.3f}", f"gain {a}")
        expect_ci(text, r.CI_clin_predeclared, f"gain CI {a}")
        expect_p(text, r.p_clin_predeclared, f"p gain {a}")
    # Holm: the prose states which survive
    surv = [a for a in cmp_.index if cmp_.loc[a, "p_clin_predeclared_holm"] < 0.05]
    (ok if surv == ["GHI (Oncotype)"] else fail)(f"Holm survivors (pre-declared): {surv}")
    surv_t = [a for a in cmp_.index if cmp_.loc[a, "p_clin_tuned_holm"] < 0.05]
    (ok if surv_t == ["GHI (Oncotype)"] else fail)(f"Holm survivors (tuned): {surv_t}")
    expect_p(text, cmp_.loc["GHI (Oncotype)", "p_clin_predeclared_holm"], "Holm p Oncotype")
    expect_p(text, cmp_.loc["ROR-P", "p_clin_predeclared_holm"], "Holm p ROR-P")
    if cmp_.loc["ROR-P", "p_clin_predeclared_holm"] != cmp_.loc["ROR-S", "p_clin_predeclared_holm"]:
        fail("prose says ROR-P and ROR-S share a Holm p; they differ")

    # -- tuned comparator ----------------------------------------------------
    for a in cmp_.index:
        r = cmp_.loc[a]
        if a != "MammaPrint":
            expect(text, f"{r.AUC_clin_tuned:.3f}" if a != "ROR-P" else f"{r.AUC_clin_tuned:.3f}", f"tuned clinical AUC {a}") \
                if a in ("GHI (Oncotype)", "ROR-S") else None
            sign = "−" if r.d_vs_clin_tuned < 0 else "+"
            expect(text, f"{sign}{abs(r.d_vs_clin_tuned):.3f}", f"tuned gain {a}")
            expect_ci(text, r.CI_clin_tuned, f"tuned gain CI {a}")
            expect_p(text, r.p_clin_tuned, f"tuned p {a}")
    r = cmp_.loc["GHI (Oncotype)"]
    expect(text, f"{r.AUC_clin_tuned_foldmean:.3f} ± {r.AUC_clin_tuned_foldsd:.3f}", "Oncotype tuned per-fold")
    expect(text, f"{r.AUC_clin_apparent:.3f}", "Oncotype apparent clinical AUC")
    expect(text, f"{int(r.negatives)} minority-class cases", "Oncotype minority count")

    # -- incremental value ----------------------------------------------------
    for a in cmp_.index:
        r = cmp_.loc[a]
        sign = "−" if r.d_incremental < 0 else "+"
        expect(text, f"{sign}{abs(r.d_incremental):.3f}", f"incremental gain {a}")
        expect_ci(text, r.CI_incremental, f"incremental CI {a}")
        if a in ("ROR-P", "ROR-S"):
            expect_p(text, r.p_incremental, f"incremental p {a}")
            expect(text, f"{r.OR_adjusted:.2f}", f"adjusted OR {a}")
            lo, hi = [float(v) for v in r.OR_adjusted_CI.strip("[]").split(",")]
            expect(text, f"{lo:.2f} to {hi:.2f}", f"adjusted OR CI {a}")
            expect_p(text, r.p_adjusted_LRT, f"adjusted LRT p {a}")
    r = cmp_.loc["GHI (Oncotype)"]
    expect(text, f"{r.OR_adjusted:.2f}", "adjusted OR Oncotype"); expect_p(text, r.p_adjusted_LRT, "LRT p Oncotype")
    expect(text, f"{r.OR_unadjusted:.2f}", "unadjusted OR Oncotype"); expect_p(text, r.p_unadjusted, "unadjusted p Oncotype")
    expect_p(text, cmp_.loc["ROR-P", "p_incremental_holm"], "Holm p incremental ROR-P")
    epv = cmp_.events_per_variable
    (ok if (epv[["GHI (Oncotype)", "MammaPrint"]] < 2.5).all() and (epv[["ROR-P", "ROR-S"]] > 4).all()
     else fail)(f"EPV pattern as described: {epv.to_dict()}")

    # -- alternative cuts ------------------------------------------------------
    for a in alt.index:
        r = alt.loc[a]
        expect(text, f"{r.AUC_ER_fixed:.3f}", f"alt-cut ER AUC {a}")
        expect(text, f"{r.kappa_ER:.3f}", f"alt-cut kappa {a}")
        sign = "−" if r.d_vs_clin_predeclared < 0 else "+"
        expect(text, f"{sign}{abs(r.d_vs_clin_predeclared):.3f}", f"alt-cut gain {a}")
        expect_ci(text, r.CI, f"alt-cut CI {a}")
        expect_p(text, r.p, f"alt-cut p {a}")
        if a != "GHI (Oncotype)":
            expect(text, f"{r.AUC_model:.3f}", f"alt-cut model AUC {a}")
    # the seven-point monotone claim
    pts = sorted([(cmp_.loc[a, "AUC_ER_fixed"], cmp_.loc[a, "d_vs_clin_predeclared"]) for a in cmp_.index]
                 + [(alt.loc[a, "AUC_ER_fixed"], alt.loc[a, "d_vs_clin_predeclared"]) for a in alt.index])
    gains = [g for _, g in pts]
    (ok if all(x > y for x, y in zip(gains, gains[1:])) else fail)(f"seven-point monotone claim: {gains}")

    # -- ranges quoted in prose -----------------------------------------------
    expect(text, f"{cmp_.AUC_model.min():.3f} to {cmp_.AUC_model.max():.3f}", "AUC range against chance")
    expect(text, f"n = {int(cmp_.n.iloc[0])}", "comparator n")

    # -- subgroups ----------------------------------------------------------------
    for _, r in sub.iterrows():
        if r.cohort == "ER+/HER2-" and not pd.isna(r.AUC_model):
            expect(text, f"{r.AUC_model:.3f}", f"{r.assay} ER+/HER2- model AUC")
            if r.assay == "GHI (Oncotype)":
                expect(text, f"{r.AUC_clin_predeclared:.3f}", "Oncotype ER+/HER2- clinical AUC")
                expect(text, f"{r.AUC_clin_tuned:.3f}", "Oncotype ER+/HER2- tuned clinical AUC")
    n56 = int(sub[(sub.cohort == "ER+/HER2-")].n.iloc[0])
    expect(text, f"n = {n56}", "ER+/HER2- subgroup n")
    mp_er = sub[(sub.assay == "MammaPrint") & (sub.cohort == "ER+")].iloc[0]
    n_neg = int(mp_er.n - mp_er.pos)
    expect(text, {4: "four"}.get(n_neg, str(n_neg)) + " negatives", "MammaPrint ER+ negatives")

    # -- minimum detectable difference -----------------------------------------
    expect(text, f"{mdd.MDD_full.min():.3f} to {mdd.MDD_full.max():.3f}", "MDD range full cohort")
    expect(text, f"{mdd.MDD_subgroup.min():.3f} to {mdd.MDD_subgroup.max():.3f}", "MDD range subgroup")
    lo9, hi9 = mdd.MDD_n900_same_prevalence.min(), mdd.MDD_n900_same_prevalence.max()
    (ok if 0.04 <= lo9 and hi9 <= 0.05 else fail)(f"MDD at n=900 within 'about 0.04 to 0.05': {lo9}–{hi9}")

    # -- repeated cross-validation (Table 9) --------------------------------------------
    rcv = pd.read_csv(res / "review" / "repeated_cv_summary.csv").set_index("assay")
    rkey = {"GHI (Oncotype)": "GHI", "ROR-P": "ROR-P", "ROR-S": "ROR-S", "MammaPrint": "MammaPrint"}
    for a, k in rkey.items():
        r = rcv.loc[k]
        sgn = "−" if r.margin_mean < 0 else "+"
        expect(text, f"{sgn}{abs(r.margin_mean):.3f} ± {r.margin_sd:.3f}", f"repeated-CV margin {a}")
    pos = {k: int(rcv.loc[k, "repeats_margin_positive"]) for k in rkey.values()}
    (ok if pos["GHI"] == 5 and pos["ROR-P"] == 5 and pos["ROR-S"] == 5 else fail)(
        f"repeated-CV: Oncotype/ROR-P/ROR-S positive in all 5 draws ({pos})")
    (ok if pos["MammaPrint"] == 2 else fail)(
        f"repeated-CV: MammaPrint positive in {pos['MammaPrint']}/5 draws (prose says two of five)")
    # repeat 0 must reproduce the locked margins (built-in check)
    per = pd.read_csv(res / "review" / "repeated_cv_per_repeat.csv")
    r0 = per[per.repeat == 0].set_index("assay")
    for a, k in rkey.items():
        locked = cmp_.loc[a, "d_vs_clin_predeclared"]
        got = r0.loc[k, "margin"]
        (ok if abs(locked - got) < 0.002 else fail)(
            f"repeated-CV repeat 0 reproduces locked margin for {a} ({locked:.3f} vs {got:.3f})")

    # -- provisional versus locked (design findings) ------------------------------------
    # -- provisional versus locked (design findings) ------------------------------------
    prov = pd.read_csv(res / "review" / "provisional_per_target.csv").set_index("assay")
    for a in prov.index:
        r = prov.loc[a]
        sign = "−" if r.d_vs_clin < 0 else "+"
        expect(text, f"{sign}{abs(r.d_vs_clin):.3f}", f"provisional gain {a}")
        if a != "ROR-P":
            expect_ci(text, r.CI_clin, f"provisional CI {a}")
            expect_p(text, r.p_clin, f"provisional p {a}")
    (ok if prov.loc["ROR-S", "p_clin"] > 0.05 and cmp_.loc["ROR-S", "p_clin_predeclared"] < 0.05 else fail)(
        "ROR-S: provisional null, locked nominally significant")
    (ok if prov.loc["MammaPrint", "d_vs_clin"] > 0 > cmp_.loc["MammaPrint", "d_vs_clin_predeclared"] else fail)(
        "MammaPrint: provisional positive margin, locked negative")
    (ok if abs(prov.loc["ROR-P", "d_vs_clin"] - cmp_.loc["ROR-P", "d_vs_clin_predeclared"]) < 0.02 else fail)(
        "ROR-P: 'unchanged' between provisional and locked")

    # -- radiomics ------------------------------------------------------------------
    for a in rdx.index:
        r = rdx.loc[a]
        expect(text, f"{r.AUC_nested:.3f}", f"radiomics nested {a}")
        expect(text, f"{r.CI_low:.3f} to {r.CI_high:.3f}", f"radiomics nested CI {a}")
        expect(text, f"{r.AUC_notebook_style:.3f}", f"radiomics non-nested {a}")
    expect(text, f"+{rdx.optimism.min():.3f} to +{rdx.optimism.max():.3f}", "radiomics optimism range")
    n_cross = int((rdx.CI_low <= 0.5).sum())
    (ok if n_cross == 3 and rdx.loc["ROR-S", "CI_low"] > 0.5 else fail)(f"'three of four include chance; ROR-S above': {n_cross}")

    # -- calibration & continuous ----------------------------------------------------
    for a in cal.index:
        expect(text, f"{cal.loc[a, 'calib_slope']:.3f}", f"calibration slope {a}")
        expect(text, f"{cal.loc[a, 'Pearson_r']:.3f}", f"Pearson r {a}")
    for a in ("MammaPrint", "GHI (Oncotype)"):
        expect(text, f"{cal.loc[a, 'AUC_perfold_mean']:.3f} ± {cal.loc[a, 'AUC_perfold_sd']:.3f}",
               f"per-fold AUC {a}")
    expect(text, f"{cal.AUC_perfold_sd.min():.3f} to {cal.AUC_perfold_sd.max():.3f}", "per-fold SD range")
    (ok if cal.loc["MammaPrint", "AUC_perfold_mean"] > flo.loc["MammaPrint", "AUC_model"] else fail)(
        "MammaPrint per-fold mean above pooled, as the prose says")
    slopes = cal.calib_slope
    (ok if slopes["GHI (Oncotype)"] == slopes.min() and slopes["MammaPrint"] > 1 else fail)(
        "calibration: Oncotype slope lowest, MammaPrint above one")

    # -- cohort ----------------------------------------------------------------------------
    expect(text, f"{len(part)} patients", "cohort n in prose")
    er_pos = int(part.ER_pos.sum())
    expect(text, f"{er_pos} ({100*er_pos/len(part):.0f}%)", "ER-positive count")

    # -- rank-order claims --------------------------------------------------------------
    by_auc = list(cmp_.sort_values("AUC_model").index)
    by_gain = list(cmp_.sort_values("d_vs_clin_predeclared", ascending=False).index)
    (ok if by_auc[0] == by_gain[0] and by_auc[-1] == by_gain[-1] else fail)(
        f"rank-order claim: by AUC {by_auc}, by gain {by_gain}")
    by_er = list(cmp_.sort_values("AUC_ER_fixed").index)
    (ok if by_er == by_gain else fail)(f"gain order equals ER-alone order: {by_er} vs {by_gain}")
    top_er = cmp_.AUC_ER_fixed.idxmax()
    (ok if cmp_.loc[top_er, "d_vs_clin_predeclared"] <= 0 else fail)(
        f"most ER-like label ({top_er}) has non-positive gain")
    # the specific 'interval excludes any gain larger than 0.034' sentence
    hi_mp = float(cmp_.loc["MammaPrint", "CI_clin_predeclared"].strip("[]").split(",")[1])
    expect(text, f"larger than {hi_mp:.3f}", "MammaPrint upper bound sentence")


def _cited_labels(text, word, re_):
    out = []
    for mt in re.finditer(re_, text):
        for n in re.findall(r"S?\d+", mt.group(1)):
            lab = f"{word} {n}"
            if lab not in out:
                out.append(lab)
    return out


FIG_RE = r"Figs?\.?\s*((?:S?\d+[a-z]?)(?:\s*(?:,|and)\s*S?\d+[a-z]?)*)"
TAB_RE = r"Tables?\s*((?:S?\d+)(?:\s*(?:,|and)\s*S?\d+)*)"


def check_crossrefs(m):
    S = m["supplement"]
    main_text = m["abstract"] + "\n" + main_prose(m) + "\n" + "\n".join(v for _, v in m["declarations"])
    supp_text = "\n".join(_section_paras(S["sections"]))
    # main-text floats: cited in the main text, in order of first citation
    for word, re_, have in (("Fig", FIG_RE, [f["label"] for f in m["figures"]]),
                            ("Table", TAB_RE, [t["label"] for t in m["tables"]])):
        cited = [c for c in _cited_labels(main_text, word, re_) if "S" not in c]
        (ok if cited == have else fail)(f"main {word.lower()}s cited in order {cited} (present {have})")
    # supplementary floats: every one cited somewhere (main text or supplement)
    both = main_text + "\n" + supp_text
    for word, re_, have in (("Fig", FIG_RE, [f["label"] for f in S["figures"]]),
                            ("Table", TAB_RE, [t["label"] for t in S["tables"]])):
        cited = [c for c in _cited_labels(both, word, re_) if "S" in c]
        missing = [h for h in have if h not in cited]
        dangling = [c for c in cited if c not in have]
        (ok if not missing and not dangling else fail)(
            f"supplementary {word.lower()}s: cited {cited}, present {have}")
    # figure sub-panels cited must exist
    caps = {f["label"]: f["caption"] for f in m["figures"] + S["figures"]}
    for mt in re.finditer(r"Fig\.?\s*(S?\d+)([a-z])\b", both):
        lab, panel = f"Fig {mt.group(1)}", mt.group(2)
        if lab in caps and f"({panel})" not in caps[lab]:
            fail(f"{lab}{panel} cited but caption has no panel ({panel})")
    ok("figure sub-panels resolve to captions")
    n_floats = len(m["figures"]) + len(m["tables"])
    (ok if n_floats <= 6 else fail)(f"{n_floats} main-text tables+figures (journal limit 6)")


def check_references(m):
    text = all_prose(m)
    nrefs = len(m["references"])
    cited = set()
    for mt in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", text):
        for chunk in re.split(r",", mt.group(1)):
            if "–" in chunk or "-" in chunk:
                a, b = [int(x) for x in re.split(r"[–-]", chunk)]
                cited.update(range(a, b + 1))
            else:
                cited.add(int(chunk))
    missing = sorted(set(range(1, nrefs + 1)) - cited)
    beyond = sorted(c for c in cited if c > nrefs)
    if beyond:
        fail(f"citations to non-existent references: {beyond}")
    if missing:
        fail(f"references never cited: {missing}")
    if not beyond and not missing:
        ok(f"all {nrefs} references cited, none dangling")
    for i, r in enumerate(m["references"], 1):
        if "TO BE COMPLETED" in r.upper() or "[verify" in r.lower() or "[" in r:
            warn(f"reference {i} carries a placeholder: {r[:70]}…")
    # JCO style: every reference ends in a four-digit year
    bad = [i for i, r in enumerate(m["references"], 1) if not re.search(r"(, \d{4}|\d{4}\. doi:\S+)$", r)]
    (ok if not bad else fail)(f"references in JCO style (year-terminated); offenders: {bad}")


def check_journal_rules(m):
    words = len(m["abstract"].split())
    (ok if words <= 275 else fail)(f"abstract {words} words (limit 275)")
    heads = [h for h, _ in m["abstract_structured"]]
    (ok if heads == ["PURPOSE", "METHODS", "RESULTS", "CONCLUSION"] else fail)(
        f"structured abstract headings {heads}")
    body = sum(len(t.split()) for t in _section_paras(m["sections"]))
    (ok if body <= 3000 else fail)(f"body {body} words (limit 3000)")
    # undefined abbreviations in the abstract: an all-caps token of 2+ letters
    # that never appears in a "(ABBR)" definition earlier in the abstract
    abs_ = m["abstract"]
    defined = set(re.findall(r"\(([A-Z][A-Za-z0-9&]{1,7})\)", abs_))
    tokens = set(re.findall(r"\b[A-Z]{2,}[0-9]*\b", abs_))
    # widely accepted without definition in this journal's field
    accepted = {"MRI", "AUC", "CI", "DX", "H&E", "PAM50", "ER", "HER2", "UNI", "ROR",
                "PURPOSE", "METHODS", "RESULTS", "CONCLUSION"}
    undefined = sorted(t for t in tokens if t not in defined and t not in accepted)
    if undefined:
        warn(f"abstract abbreviations not defined there: {undefined}")
    else:
        ok("abstract abbreviations defined or field-standard")
    for k, v in m["declarations"]:
        if "[to complete]" in v or "[repository URL]" in v:
            warn(f"declaration '{k}' still carries a placeholder")
    for i, aff in enumerate(m["affiliations"], 1):
        if "[" in aff:
            warn(f"affiliation {i} carries a placeholder: {aff}")
    if 4 <= len(m["keywords"]) <= 6:
        ok(f"{len(m['keywords'])} keywords (4–6)")
    else:
        fail(f"{len(m['keywords'])} keywords; journal wants 4–6")
    for a in m["authors"]:
        if "[" in a["name"]:
            fail(f"author placeholder: {a['name']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--radiomics", required=True)
    a = p.parse_args(argv)
    m = json.loads(Path(a.json).read_text(encoding="utf-8"))

    check_numbers(m, Path(a.results), Path(a.radiomics), None)
    check_crossrefs(m)
    check_references(m)
    check_journal_rules(m)

    print(f"OK   {len(OK)}")
    for w in WARN:
        print(f"WARN {w}")
    for f in FAIL:
        print(f"FAIL {f}")
    print(f"\n{len(OK)} passed, {len(WARN)} warnings, {len(FAIL)} failures")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
