"""Multimedia Appendix 2: TRIPOD+AI and TRIPOD+AI for Abstracts checklists for the revised manuscript (v3.0).

Item wording is taken from the submitted checklist (MA2_v2.1_extracted.md, which reproduces TRIPOD+AI, Collins et al,
BMJ 2024;385:e078378); the location column is rewritten for manuscript_uric_v3.0.md (section headings, because page
numbers depend on the final layout) and checked against the built manuscript: the first section name cited after
'Methods,', 'Results,' or 'Discussion,' in an entry must be a heading of the manuscript or the start of one.
Supplementary tables are cited by key, {{T:key}} or {{T:key1,key2,...}}, with the keys of make_supplement.py (a table
or one of its panels): the label and the panel letter both come from supplement_index.csv (columns kind, key, label,
panel; written by make_supplement.py), so a citation follows any renumbering of Multimedia Appendix 1. When
../revision_notes/supplement_renumbering_v5.csv (also written by make_supplement.py) lists a key, its label and panel
must agree with the index, otherwise the run stops (one of the two files is stale). The renumbering file of 2026-10-12
(../revision_notes/supplement_renumbering.csv) is history and is not read. A key that supplement_index.csv does not
list stops the run (the citation must be refreshed); a key without a panel letter (a table, or the only panel of a
table) is cited as its whole table. Labels written out in full are still checked: every cited supplementary label must
be defined in supplement_index.csv and every cited panel letter must exist in that table, so a stale number stops the
run.
Content (decisions of 2026-10-16 and 2026-10-17): the models are the retrained ensembles, called "the models": 12
inputs (the 9 dipstick fields, age, sex and BMI; leukocyte esterase is not an input because KNHANES does not measure
it), hyperparameters tuned on the training persons only, the data, inputs, split, KNHANES coding and modelling
procedure fixed before the models were fitted or evaluated, serum urate and BMI of the urine-test day, one person-level
split shared by the thresholds, and the label-matched KNHANES dipstick coding X1, with the alternative codings as
sensitivity analyses. Item 18f cites the Data Availability statement only. The checklist has no wording about the
original (submitted) models, the audit, the reconciliation or the history of the submission; the run stops if such
wording appears. The manuscript section names were refreshed on 2026-10-17 for the rewritten manuscript (Methods
subsection 'Sensitivity Analyses for Coding, Linkage, and Time'; item 9b points to Development Cohort for when BMI and
serum urate were recorded).
Run: /opt/anaconda3/bin/python make_checklist.py
"""
import os, re, html
import csv
HERE = os.path.dirname(os.path.abspath(__file__))
# Manuscript folder (inputs and outputs): PROJ_U_REVISION_DIR overrides; original: parent of this folder.
REV = os.environ.get("PROJ_U_REVISION_DIR", os.path.dirname(HERE))
SRC = os.path.join(REV, "MA2_v2.1_extracted.md")
MS = os.path.join(REV, "manuscript_uric_v3.0.md")
OUT = os.path.join(REV, "supplement", "Multimedia_Appendix_2_TRIPOD_AI.md")
INDEX = os.path.join(HERE, "supplement_index.csv")
RENUMBER_V5 = os.path.join(REV, "revision_notes", "supplement_renumbering_v5.csv")   # cross-check only

txt = open(SRC, encoding="utf-8").read()
rows = re.findall(r"<tr[^>]*>(.*?)</tr>", txt, re.S)
items = []
section = ""
for r in rows:
    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh]>(.*?)</t[dh]>", r, re.S)]
    if len(cells) != 5 or cells[0] == "Section/Topic":
        continue
    if cells[1] == "" and cells[0].isupper():
        section = cells[0]
        continue
    items.append(dict(section=section, topic=cells[0], item=cells[1], de=cells[2], text=cells[3]))

LOC = {
 "1": "Title",
 "2": "Abstract (see TRIPOD+AI for Abstracts below)",
 "3a": "Introduction, paragraphs 1-2 and 4",
 "3b": "Introduction, paragraphs 2 and 4; Methods, Development Cohort (intended population: adults); Discussion, Principal Results and Implications for Implementation",
 "3c": "Results, Clinical Utility (operating points by sex); Table 3; Discussion, Principal Results",
 "4": "Introduction, paragraph 4",
 "5a": "Methods, Data Sources and Participants; Results, Study Population and Setting; {{T:setting_all,setting_dev,knflow}}",
 "5b": "Methods, Development Cohort and External Validation Cohort",
 "6a": "Methods, Development Cohort (single tertiary hospital; setting marker) and External Validation Cohort; Table 1; {{T:setting_all,setting_dev}}",
 "6b": "Methods, Development Cohort (adults with serum urate and BMI recorded on the day of the urine test), Training and Internal Test Sets, and External Validation Cohort; Figure 1",
 "6c": "Not applicable (treatments not recorded); Discussion, Limitations (first)",
 "7": "Methods, Input Features (dipstick coding; KNHANES codes mapped to the development grade with the same label, a mapping fixed before the models were fitted or evaluated); {{T:inputmap,inputdist}}",
 "8a": "Methods, Outcomes (serum urate measured on the day of the urine test); Discussion, Limitations (first)",
 "8b": "Not applicable (laboratory measurement)",
 "8c": "Not applicable (laboratory measurement)",
 "9a": "Methods, Input Features (12 inputs: the 9 dipstick fields measured in both cohorts, age, sex, and BMI; leukocyte esterase is not an input because KNHANES does not measure it)",
 "9b": "Methods, Development Cohort (serum urate and BMI recorded on the day of the urine test; BMI from a same-day health checkup or the nursing record) and Input Features; {{T:inputmap,codebook}}",
 "9c": "Not applicable (laboratory and anthropometric measurements)",
 "10": "Methods, External Validation Cohort (no sample size calculation; events against published minimums)",
 "11": "Methods, Missing Data; Results, Missing Data; {{T:missdata}}",
 "12a": "Methods, Training and Internal Test Sets (one person-level split, stratified on serum urate and shared by the thresholds); Added Value of the Dipstick",
 "12b": "Methods, Input Features",
 "12c": "Methods, Model Development (hyperparameters tuned on the training persons only; data, inputs, split, KNHANES coding, and modeling procedure fixed before the models were fitted or evaluated); {{T:hyperparams}}",
 "12d": "Not applicable (single development site); survey periods and years in Methods, Sensitivity Analyses for Coding, Linkage, and Time",
 "12e": "Methods, Statistical Analysis (Discrimination and Calibration; Confidence Intervals and Comparisons; Clinical Utility; Added Value)",
 "12f": "Methods, Discrimination and Calibration (external intercept update); Clinical Utility (cross-fitted update)",
 "12g": "Methods, Model Development and Software (predictions computed with the saved model files)",
 "13": "Not applicable (no class-imbalance methods)",
 "14": "Methods, Clinical Utility (operating points by sex, sex-specific thresholds) and Subgroups; Results, Clinical Utility and Subgroups",
 "15": "Methods, Clinical Utility (probability output; thresholds and how they were chosen)",
 "16": "Methods, Input Features (leukocyte esterase is not measured in KNHANES and is not an input; KNHANES dipstick coding fixed before the models were fitted or evaluated, alternative codings as sensitivity analyses) and External Validation Cohort; Results, Study Population and Setting; Table 1; {{T:setting_all,setting_dev,knflow}}; {{T:inputmap,inputdist}}",
 "17": "Methods, Ethical Considerations",
 "18a": "Funding Statement",
 "18b": "Conflicts of Interest",
 "18c": "Methods, Study Design (no protocol prepared)",
 "18d": "Methods, Study Design (not registered)",
 "18e": "Data Availability",
 "18f": "Data Availability",
 "19": "Methods, Study Design (no patient or public involvement)",
 "20a": "Results, Study Population and Setting; Figure 1; {{T:flow}}",
 "20b": "Results, Study Population and Setting; Table 1; {{T:setting_all,setting_dev,knflow}}; {{T:missing}}",
 "20c": "Table 1; {{T:shift}}",
 "21": "Methods, Training and Internal Test Sets (one split shared by the thresholds) and Added Value of the Dipstick; Tables 1-2 and 4",
 "22": "Data Availability (model files and their access terms); {{T:hyperparams}} (hyperparameters)",
 "23a": "Results; Tables 2-5; Figures 2-5",
 "23b": "Not applicable (single development site); survey years in Table 5 and {{T:knyear}}",
 "24": "Results, Discrimination and Calibration (intercept update); Table 2; Clinical Utility (updated decision curves); Figure 4; {{T:recal,recalop,nbref}}",
 "25": "Discussion, Principal Results and Comparison With Prior Work",
 "26": "Discussion, Limitations",
 "27a": "Discussion, Implications for Implementation (BMI availability, dipstick coding, input monitoring)",
 "27b": "Discussion, Implications for Implementation",
 "27c": "Discussion, Implications for Implementation, Limitations and Conclusions",
}
ABS = [
 ("Title", "1", "Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted", "Title"),
 ("Background", "2", "Brief explanation of the healthcare context and rationale", "Abstract, Background"),
 ("Objectives", "3", "Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both)", "Abstract, Objective"),
 ("Methods: data", "4", "Describe the sources of data", "Abstract, Methods"),
 ("Methods: participants", "5", "Describe the eligibility criteria and the setting", "Abstract, Methods"),
 ("Methods: outcome", "6", "Specify the outcome predicted by the model", "Abstract, Methods"),
 ("Methods: model", "7", "Specify the type of model and a summary of the model-building steps", "Abstract, Methods"),
 ("Methods: model evaluation", "8", "Specify the measures used to assess model performance", "Abstract, Methods"),
 ("Results: participants", "9", "Report the number of participants and outcome events", "Abstract, Methods"),
 ("Results: predictors", "10", "Summarise the predictors in the final model", "Abstract, Objective and Methods"),
 ("Results: model performance", "11", "Report model performance estimates with confidence intervals", "Abstract, Results"),
 ("Discussion: interpretation", "12", "Give an overall interpretation of the main results", "Abstract, Conclusions"),
 ("Registration", "13", "Registration number and name of registry, or state not registered", "Not registered (Methods, Study Design); not stated in the abstract"),
]


# ---------------------------------------------------------------- supplementary citations by key ({{T:key,...}})
def csv_rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


INDEX_ROWS = csv_rows(INDEX)
if not INDEX_ROWS or "panel" not in INDEX_ROWS[0]:
    raise SystemExit("supplement_index.csv has no 'panel' column: rerun make_supplement.py, which writes the panel letters")
LABEL_OF = {r["key"].strip(): r["label"].strip() for r in INDEX_ROWS if r["kind"].strip() == "table"}
# key -> panel letter; '' = a table key, or the only panel of a table (cited as the whole table)
PANEL_OF = {r["key"].strip(): (r["panel"] or "").strip() for r in INDEX_ROWS if r["kind"].strip() == "table"}
for k, p in PANEL_OF.items():
    assert p == "" or re.fullmatch(r"[A-H]", p), (k, p)
# cross-check: supplement_renumbering_v5.csv (new_label 'Table S12 panel C', 'Table S8', 'removed') must agree with the index
if os.path.exists(RENUMBER_V5):
    n_x = 0
    for r in csv_rows(RENUMBER_V5):
        if r["kind"].strip() != "table" or r["old_key"].strip() not in LABEL_OF:
            continue
        m = re.fullmatch(r"(Table S\d+)(?: panel ([A-H]))?", r["new_label"].strip())
        k = r["old_key"].strip()
        if not m or (m.group(1), m.group(2) or "") != (LABEL_OF[k], PANEL_OF[k]):
            raise SystemExit(f"supplement_index.csv ({LABEL_OF[k]} {PANEL_OF[k]}) and {os.path.basename(RENUMBER_V5)} "
                             f"({r['new_label']}) disagree for key '{k}': rerun make_supplement.py")
        n_x += 1
    print(f"{n_x} supplementary keys agree between supplement_index.csv and {os.path.basename(RENUMBER_V5)}")
else:
    print(f"note: {os.path.relpath(RENUMBER_V5, HERE)} not found; panel letters taken from supplement_index.csv unchecked")


def join(xs):
    return xs[0] if len(xs) == 1 else f"{xs[0]} and {xs[1]}" if len(xs) == 2 else ", ".join(xs[:-1]) + f", and {xs[-1]}"


def panel_words(letters):
    """['C', 'D', 'E'] -> 'panels C-E'; ['A', 'C'] -> 'panels A and C'; ['B'] -> 'panel B'."""
    ls = sorted(set(letters))
    runs = [[ls[0]]]
    for x in ls[1:]:
        if ord(x) == ord(runs[-1][-1]) + 1:
            runs[-1].append(x)
        else:
            runs.append([x])
    parts = [p for run in runs for p in ([f"{run[0]}-{run[-1]}"] if len(run) >= 3 else run)]
    return ("panel " if len(ls) == 1 else "panels ") + join(parts)


def cite(keys, item):
    """Keys of supplementary tables or panels -> 'Table S3 panels C-E', 'Tables S4 and S5', 'Table S4 and Table S5 panel
    A', ... (tables in ascending order)."""
    groups = {}         # label -> panel letters ('' = the whole table)
    for k in keys:
        if k not in LABEL_OF:
            raise SystemExit(f"item {item}: supplementary key '{k}' is not in supplement_index.csv; refresh the citation")
        lab = LABEL_OF[k]
        if not re.fullmatch(r"Table S\d+", lab):
            raise SystemExit(f"item {item}: key '{k}' resolves to '{lab}', not to a supplementary table")
        groups.setdefault(lab, []).append(PANEL_OF[k])
    out = [lab if "" in ls else f"{lab} {panel_words(ls)}"
           for lab, ls in sorted(groups.items(), key=lambda kv: int(kv[0].split("S")[1]))]
    if len(out) > 1 and all(" panel" not in x for x in out):
        return "Tables " + join([x.split(" ")[1] for x in out])
    return join(out)


TOKEN = re.compile(r"\{\{T:([\w,]+)\}\}")
LOC = {k: TOKEN.sub(lambda m, k=k: cite(m.group(1).split(","), k), v) for k, v in LOC.items()}
assert not any("{{" in v or "}}" in v for v in LOC.values()), "unresolved supplementary citation"

ms = open(MS, encoding="utf-8").read()
heads = set(h.strip() for h in re.findall(r"(?m)^#{2,4} (.+)$", ms))
def heading_at(rest):
    """True if `rest` starts with a whole heading of the manuscript followed by ' (', ';', ',', ' and ' or the end of the
    entry, or with the first words of a heading followed by ' (', ';', ',' or the end. Headings may contain commas and
    'and' ('Sensitivity Analyses for Coding, Linkage, and Time'), so a stale name such as 'Sensitivity Analyses for
    Linkage, Coding, and Time' or 'Training and Internal Test Sets of the Original Models' is not accepted."""
    for h in heads:
        if rest.startswith(h) and re.match(r"(?: \(|;|,|$| and )", rest[len(h):]):
            return True
        words = h.split(" ")
        for n in range(1, len(words)):
            p = " ".join(words[:n]).rstrip(",")
            if rest.startswith(p) and re.match(r"(?: \(|;|,|$)", rest[len(p):]):
                return True
    return False


for k, v in LOC.items():
    # the text after 'Methods, ', 'Results, ' or 'Discussion, ' must name a heading of the manuscript
    for m in re.finditer(r"(?:Methods|Results|Discussion), ", v):
        if not heading_at(v[m.end():]):
            raise SystemExit(f"item {k}: heading not in the manuscript near '{v[m.start():m.start() + 80]}'")
missing = [it["item"] for it in items if it["item"] not in LOC]
assert not missing, missing

# supplementary labels: defined in the current numbering, panels present in the cited table
defined = set()
for r in INDEX_ROWS:
    m = re.fullmatch(r"(Table|Figure) S(\d+)", r["label"].strip())
    if m:
        defined.add((m.group(1), int(m.group(2))))
panels = {}             # table number -> its panel letters (tables with one panel have none)
for k, lab in LABEL_OF.items():
    m = re.fullmatch(r"Table S(\d+)", lab)
    if m and PANEL_OF[k]:
        panels.setdefault(int(m.group(1)), set()).add(PANEL_OF[k])
PANEL = r"(?: panels? [A-Z](?!\w)(?:(?:, and |, | and |-)[A-Z](?!\w))*)?"
ITEM = rf"S\d+(?:-S\d+)?{PANEL}"
SEP = r"(?:, and |, | and )"
LIST = re.compile(rf"(?<![\w-])(Table|Figure)s? ({ITEM}(?:{SEP}(?:\1s? )?{ITEM})*)")
n_cited = 0
for k, v in list(LOC.items()) + [(a[1], a[3]) for a in ABS]:
    for m in LIST.finditer(v):
        for it in re.finditer(rf"S(\d+)(?:-S(\d+))?({PANEL})", m.group(2)):
            a, b = int(it.group(1)), int(it.group(2) or it.group(1))
            for n in range(a, b + 1):
                n_cited += 1
                if (m.group(1), n) not in defined:
                    raise SystemExit(f"item {k}: {m.group(1)} S{n} is not defined in supplement_index.csv ({v!r})")
            letters = set()
            for x, y in re.findall(r"(?<!\w)([A-Z])(?:-([A-Z]))?(?!\w)", it.group(3)):
                letters.update(chr(c) for c in range(ord(x), ord(y or x) + 1))
            if letters and panels and not letters <= panels.get(a, set()):
                raise SystemExit(f"item {k}: Table S{a} has no panel {sorted(letters - panels.get(a, set()))} ({v!r})")
print(f"{n_cited} supplementary table and figure citations checked against supplement_index.csv")

example = re.search(r"Table S\d+ panel [A-Z](?!\w)", " ".join(LOC.values()))
out = ["# Multimedia Appendix 2", "",
       "**TRIPOD+AI and TRIPOD+AI for Abstracts checklists for: Machine Learning Models Combining Routine Urine Dipstick Results "
       "With Age, Sex, and Body Mass Index to Triage Serum Urate Testing: Development and External Validation Study.**", "",
       "Checklist items from Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. Locations refer to "
       "section headings of the manuscript, tables (Table n) and figures (Figure n) of the main text, and supplementary "
       "tables and figures (Table Sn, Figure Sn) of Multimedia Appendix 1"
       + (f"; a panel letter (for example, {example.group(0)}) refers to one panel of a supplementary table" if example else "")
       + ". D: development; E: evaluation.", "",
       "## TRIPOD+AI", "",
       "| Section | Item | D/E | Checklist item | Where reported |", "|---|---|---|---|---|"]
sec = None
for it in items:
    if it["section"] != sec:
        sec = it["section"]
        out.append(f"| **{sec.title()}** |  |  |  |  |")
    out.append(f"| {it['topic']} | {it['item']} | {it['de']} | {it['text']} | {LOC[it['item']]} |")
out += ["", "## TRIPOD+AI for Abstracts", "", "| Section | Item | Checklist item (abbreviated) | Where reported |", "|---|---|---|---|"]
out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in ABS]
out += ["", "TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis with "
        "Artificial Intelligence."]
text = "\n".join(out) + "\n"
assert "–" not in text and "—" not in text
# terminology (decision of 2026-10-16): the checklist describes the retrained models only, with no history of the
# submission (2026-10-17: nor 'revised manuscript'); leukocyte esterase appears only in the design statement that it is
# not an input
bad = re.search(r"(?i)\b(original\w*|locked|deployed|submitted|submission|first version|reconcil\w*|audit\w*|pars(?:e|ed|ing)"
                r"|person-clean|retrain\w*|revis\w*|esterase constant|10F|13F)\b|0\.321", text)
if bad:
    raise SystemExit(f"terminology: '{bad.group(0)}' near {text[max(0, bad.start() - 100):bad.end() + 60]!r}")
for m in re.finditer(r"(?i)leukocyte", text):
    if not re.match(r"(?i)leukocyte esterase is not (?:an input|measured in KNHANES and is not an input)", text[m.start():]):
        raise SystemExit(f"terminology: leukocyte esterase outside the design statement near "
                         f"{text[max(0, m.start() - 100):m.end() + 80]!r}")
open(OUT, "w", encoding="utf-8").write(text)
print(f"wrote {OUT}: {len(items)} items")
