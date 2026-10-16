"""Assemble the revised manuscript (JMIR Medical Informatics #107515, R1) from its source text.

Input   ../manuscript_uric_v3.0_src.md   authored text with tokens:
          {{T:key}}   -> "Table Sn"   (supplementary table; key as in make_supplement.py)
          {{N:key}}   -> "Sn"         (number only, for "Tables S5 and S6")
          {{F:key}}   -> "Figure Sn"  (supplementary figure)
          <!-- TABLE k -->  on its own line -> main-text Table k from ../tables.md (make_tables.py)
          citations [n], [n,m], [n-m] use the source numbering of the reference list; they are renumbered
          in order of first citation in the body and the reference list is reordered to match (JMIR rule).
        ../tables.md                      Tables 1-5 (make_tables.py)
        supplement_index.csv              key -> label (and panel letter) map written by make_supplement.py: every
                                          table of Multimedia Appendix 1, the keys of its panels as aliases of its label
                                          (so several keys share one label), and the figures. It is the only source of
                                          labels: the renumbering files in ../revision_notes are history and are not
                                          read, so a key that the index does not list (a table or panel removed or
                                          renamed) stops the build, and the error lists every such key with the file
                                          that cites it. A label of the form "Table 2" (no S) would be printed as a
                                          main-text table (no such key is listed since 2026-10-16).
Output  ../manuscript_uric_v3.0.md

After token substitution (in the text and in the inserted main tables), lists of supplementary labels are tidied: a
table named twice in a list is named once ("Tables S7 and S7" -> "Table S7"; "Tables S7, S7, and S9" -> "Tables S7
and S9"), "Table" or "Tables" follows the number of different tables, labels are put in ascending order (unless
"respectively" follows in the same sentence), and in a rewritten list 3 or more consecutive tables become a range
(S1-S25). A list without repeats, in ascending order and with the right Table/Tables is left exactly as written (for
example "Tables S24-S25" or "Tables S3, S4, and S5"). A token that resolves to a main-text table is printed without
"in Multimedia Appendix 1" and, in a list, before the supplementary labels. Every change is printed.

Checks (the build fails on any of them): unresolved tokens or table markers; supplementary keys that
supplement_index.csv does not list (all of them are listed in one error, with the file that cites them); en or em
dashes; placeholder text ([[...]], [TOOL, [AUTHORS, [TAG); main tables and figures cited before first appearance or out of
order; a main-text table or figure followed by "in Multimedia Appendix 1"; a supplementary table or figure label that
supplement_index.csv does not define (stale numbering); a panel letter ("Table S12 panels A and C") that the cited
table does not have (panel column of supplement_index.csv); supplementary tables first cited without "Multimedia
Appendix 1" in the same sentence (warning only); a reference cited in the body but missing from the list, or listed but
never cited. Word counts are printed by section (JMIR counts title, abstract, body, tables, figure legends and back
matter; references are excluded).

Run: /opt/anaconda3/bin/python build_manuscript.py
"""
import os
import re
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Manuscript folder (inputs and outputs): PROJ_U_REVISION_DIR overrides; original: parent of this folder.
REV = os.environ.get("PROJ_U_REVISION_DIR", os.path.dirname(HERE))
SRC = os.path.join(REV, "manuscript_uric_v3.0_src.md")
OUT = os.path.join(REV, "manuscript_uric_v3.0.md")
TABLES = os.path.join(REV, "tables.md")
INDEX = os.path.join(HERE, "supplement_index.csv")
CIT = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")
TOKEN = re.compile(r"\{\{([TNF]):([^}]+)\}\}")

# ---------------------------------------------------------------- supplementary labels and lists of labels
LABEL = re.compile(r"(Table|Figure) (S?)(\d+)\b")   # "Table S7", "Figure S2"; "Table 2" is a main-text table
MAIN = ""   # wraps the number of a token that resolved to a main-text table; removed by tidy_lists()
PANELS = r"(?: panels? [A-Z](?!\w)(?:(?:, and |, | and |-)[A-Z](?!\w))*)?"
ITEM = rf"(?:S\d+(?:-S\d+)?{PANELS}|{MAIN}\d+{MAIN})"
SEP = r"(?:, and |, | and )"
# "Tables S7 and S9", "Table S3 and Table S3", "Tables S1-S25", "Table S3 panels A and C", each with an optional
# " in Multimedia Appendix 1" after it
LIST = re.compile(rf"(?<![\w-])(Table|Figure)s? ({ITEM}(?:{SEP}(?:\1s? )?{ITEM})*)( in Multimedia Appendix 1)?")
ITEM_PARTS = re.compile(rf"S(\d+)(?:-S(\d+))?({PANELS})|{MAIN}(\d+){MAIN}")
MAIN_IN_MA1 = re.compile(r"(?<![\w*])(?<!counterpart of )(?<!counterparts of )(?:Table|Figure)s? \d+"
                         r"(?:(?:, and |, | and |-)\d+)* in Multimedia Appendix 1")


def fail(msg):
    sys.exit(f"BUILD FAILED: {msg}")


def load_labels():
    """From supplement_index.csv only: ({(kind, key): (word, "S" or "", number)}, {table number: panel letters} or None
    when the index has no panel column). No renumbering file is consulted, so a key the index does not list is unknown."""
    lab, panels = {}, {}
    idx = pd.read_csv(INDEX, dtype=str, keep_default_na=False)
    has_panel = "panel" in idx.columns
    for r in idx.to_dict("records"):
        kind, key = r["kind"].strip(), r["key"].strip()
        m = LABEL.search(r["label"])   # "Table S7", "Figure S2"
        if not m or (m.group(1) == "Figure") != (kind == "figure"):
            fail(f"supplement_index.csv: label {r['label']!r} does not fit {kind}:{key}")
        if (kind, key) in lab:
            fail(f"supplement_index.csv lists {kind}:{key} twice")
        lab[(kind, key)] = m.groups()
        letter = r.get("panel", "").strip()
        if letter:
            if kind != "table" or not m.group(2) or not re.fullmatch(r"[A-H]", letter):
                fail(f"supplement_index.csv: panel {letter!r} does not fit {kind}:{key} ({r['label']})")
            panels.setdefault(int(m.group(3)), set()).add(letter)
    if not has_panel:
        print("note: supplement_index.csv has no panel column (written before 2026-10-17); panel letters not checked")
    return lab, (panels if has_panel else None)


def unknown_keys(lab, texts):
    """{token: {file: count}} for every {{T:}}, {{N:}} or {{F:}} token of `texts` ({file name: text}) whose key
    supplement_index.csv does not list."""
    out = {}
    for name, txt in texts.items():
        for kind, key in TOKEN.findall(txt):
            if ("figure" if kind == "F" else "table", key) not in lab:
                where = out.setdefault(f"{kind}:{key}", {})
                where[name] = where.get(name, 0) + 1
    return out


def join(xs):
    """'a', 'a and b', 'a, b, and c' (serial comma, as in the manuscript)."""
    return xs[0] if len(xs) == 1 else (" and ".join(xs) if len(xs) == 2 else ", ".join(xs[:-1]) + ", and " + xs[-1])


def panel_set(spec):
    """' panels A-C' -> {A, B, C}; ' panels A and C' -> {A, C}."""
    out = set()
    for a, b in re.findall(r"(?<!\w)([A-Z])(?:-([A-Z]))?(?!\w)", spec):
        out.update(chr(c) for c in range(ord(a), ord(b or a) + 1))
    return frozenset(out)


def labels_text(word, prefix, nums, panels, keep_order):
    """'Table S7', 'Tables S7 and S9', 'Tables S1-S25' (3 or more consecutive), 'Table S3 panels A and C'."""
    nums = list(nums) if keep_order else sorted(nums)
    parts, i = [], 0
    while i < len(nums):
        j = i
        while (not keep_order and j + 1 < len(nums) and nums[j + 1] == nums[j] + 1
               and panels.get(nums[i]) is None and panels.get(nums[j + 1]) is None):
            j += 1
        if j - i >= 2:
            parts.append(f"{prefix}{nums[i]}-{prefix}{nums[j]}")
            i = j + 1
            continue
        p = panels.get(nums[i])
        parts.append(f"{prefix}{nums[i]}" + (f" panel{'s' if len(p) > 1 else ''} {join(sorted(p))}" if p else ""))
        i += 1
    return f"{word}{'s' if len(nums) > 1 else ''} {join(parts)}"


def tidy_lists(text):
    """Name each table of a list once, in order, with Table/Tables agreeing; main-text aliases first and without
    'in Multimedia Appendix 1' (see the module docstring). Returns (text, [(before, after), ...])."""
    changes = []

    def one(m):
        word, items, tail = m.group(1), m.group(2), m.group(3) or ""
        supp, main, order, degenerate = {}, set(), [], False
        for it in ITEM_PARTS.finditer(items):
            if it.group(4):
                main.add(int(it.group(4)))
                continue
            a, b = int(it.group(1)), int(it.group(2) or it.group(1))
            if b < a:
                fail(f"reversed range in {m.group(0)!r}")
            degenerate |= bool(it.group(2)) and b == a
            p = panel_set(it.group(3)) if it.group(3) and a == b else None
            for n in range(a, b + 1):
                order.append(n)
                supp[n] = p if n not in supp else (None if supp[n] is None or p is None else supp[n] | p)
        distinct = list(dict.fromkeys(order))
        rest = re.split(r"\.\s|;", m.string[m.end():m.end() + 300], maxsplit=1)[0]
        keep_order = "respectively" in rest
        own_words = re.search(rf"{SEP}{word}s? ", items) is not None     # "Table S3 and Table S5"
        plural_ok = own_words or m.group(0).startswith(word + "s ") == (len(distinct) > 1)
        if (not main and len(order) == len(distinct) and not degenerate and plural_ok
                and (keep_order or distinct == sorted(distinct))):
            return m.group(0)
        parts = []
        if main:
            parts.append(labels_text(word, "", sorted(main), {}, False))
        if supp:
            parts.append(labels_text(word, "S", distinct, supp, keep_order))
        new = " and ".join(parts) + (tail if supp else "")
        changes.append((m.group(0), new))
        return new

    return LIST.sub(one, text), changes


def main_tables():
    txt = open(TABLES, encoding="utf-8").read()
    blocks = {}
    for m in re.finditer(r"<!-- (Table \d+): generated.*?-->\n\n(.*?)(?=\n\n<!-- Table \d+: generated|\Z)", txt, re.S):
        blocks[int(m.group(1).split()[1])] = f"<!-- {m.group(1)}: generated by working/make_tables.py; do not edit by hand -->\n\n" \
                                            + m.group(2).rstrip()
    return blocks


def expand(nums):
    out = []
    for part in re.split(r"\s*,\s*", nums):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def compress(nums):
    nums = sorted(set(nums))
    runs, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            runs.append((start, prev)); start = prev = n
    runs.append((start, prev))
    return ",".join(f"{a}" if a == b else (f"{a},{b}" if b == a + 1 else f"{a}-{b}") for a, b in runs)


def words(s):
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"[|*#^]", " ", s)
    s = re.sub(r"^-{3,}.*$", " ", s, flags=re.M)
    return len(re.findall(r"[A-Za-z0-9≥≤±%][^\s]*", s))


def main():
    src = open(SRC, encoding="utf-8").read()
    lab, panels = load_labels()
    # every supplementary key cited in the source or in the main tables must be in supplement_index.csv; all unknown keys
    # are listed at once (a removed or renamed table or panel: rewrite the citation)
    unknown = unknown_keys(lab, {os.path.basename(SRC): src,
                                 os.path.basename(TABLES): open(TABLES, encoding="utf-8").read()})
    if unknown:
        fail(f"{len(unknown)} supplementary key(s) not in supplement_index.csv (removed or renamed; rewrite these "
             "citations): " + "; ".join(f"{tok} ({', '.join(f'{n} x{c}' for n, c in where.items())})"
                                       for tok, where in sorted(unknown.items())))

    # ---------------------------------------------------------------- supplementary tokens
    def sub_tok(m):
        kind, key = m.group(1), m.group(2)
        k = "figure" if kind == "F" else "table"
        if (k, key) not in lab:
            fail(f"unknown supplementary key {kind}:{key}")
        word, s, num = lab[(k, key)]
        if not s:   # a label without 'S' is a main-text table
            return f"{MAIN}{num}{MAIN}" if kind == "N" else f"{word} {MAIN}{num}{MAIN}"
        return f"S{num}" if kind == "N" else f"{word} S{num}"
    text = TOKEN.sub(sub_tok, src)
    # the source-file notice must not reach the built file (it would be copied into the Word document)
    text = re.sub(r"<!-- SOURCE FILE.*?-->\n*", "", text, flags=re.S)
    tabs = main_tables()
    for k, block in tabs.items():
        tabs[k] = TOKEN.sub(sub_tok, block)

    def sub_table(m):
        k = int(m.group(1))
        if k not in tabs:
            fail(f"Table {k} not in tables.md")
        return f"<!-- BEGIN TABLE {k} -->\n\n{tabs[k]}\n\n<!-- END TABLE {k} -->"
    text = re.sub(r"(?m)^<!-- TABLE (\d+) -->$", sub_table, text)
    text, tidied = tidy_lists(text)
    for before, after in tidied:
        print(f"list tidied: {before!r} -> {after!r}")
    # "[[": placeholders such as [[DOI]] must be filled (or the sentence reworded) before the manuscript is built
    for bad in ("{{", "}}", "<!-- TABLE", "[[", "[TOOL", "[AUTHORS", "[TAG", "[DEVELOPER", "–", "—", MAIN):
        if bad in text:
            i = text.index(bad)
            fail(f"forbidden text {bad!r} near: {text[max(0, i - 80):i + 80]!r}")

    # ---------------------------------------------------------------- references
    head, sep, refs = text.partition("\n## References\n")
    if not sep:
        fail("no References section")
    refs_body, sep2, tail = refs.partition("\n## Figures")  # no separate figure section expected; keep any tail
    entries = re.findall(r"(?m)^(\d+)\. (.+)$", refs_body)
    ref_text = {int(n): t for n, t in entries}
    order = []
    for m in CIT.finditer(head):
        for n in expand(m.group(1)):
            if n not in ref_text:
                fail(f"citation [{n}] has no reference entry")
            if n not in order:
                order.append(n)
    unused = sorted(set(ref_text) - set(order))
    if unused:
        fail(f"references listed but never cited: {unused}")
    new = {old: i + 1 for i, old in enumerate(order)}
    head = CIT.sub(lambda m: "[" + compress([new[n] for n in expand(m.group(1))]) + "]", head)
    ref_list = "\n\n".join(f"{new[o]}. {ref_text[o]}" for o in order)
    text = head + "\n## References\n\n" + ref_list + "\n" + (sep2 + tail if sep2 else "")

    # ---------------------------------------------------------------- order checks
    body = text.split("\n## References\n")[0]
    for kind, n_max in (("Table", 5), ("Figure", 5)):
        first_cite, first_show = {}, {}
        for k in range(1, n_max + 1):
            c = re.search(rf"(?<![\w*]){kind} {k}(?![\d.])", body)
            s = re.search(rf"\*\*{kind} {k}\.\*\*", body)
            if not c or not s:
                fail(f"{kind} {k}: citation or display missing")
            first_cite[k], first_show[k] = c.start(), s.start()
            if c.start() > s.start():
                fail(f"{kind} {k} displayed before its first citation")
        if sorted(first_cite, key=first_cite.get) != list(range(1, n_max + 1)):
            fail(f"{kind}s not first cited in numerical order: {sorted(first_cite, key=first_cite.get)}")
    # main-text tables and figures are not in Multimedia Appendix 1 (an alias of the dropped former Table S56 is
    # printed as "Table 2" without it)
    m = MAIN_IN_MA1.search(body)
    if m:
        fail(f"main-text label followed by 'in Multimedia Appendix 1': {body[max(0, m.start() - 60):m.end()]!r}")
    # every supplementary label cited must exist in the current numbering (catches stale hand-typed labels), and every
    # panel letter cited must exist in that table (catches stale hand-typed panel letters)
    defined = {(w, int(n)) for w, s, n in lab.values() if s}
    n_panel_cites = 0
    for m in LIST.finditer(body):
        for it in ITEM_PARTS.finditer(m.group(2)):
            if it.group(4):   # cannot happen after tidy_lists (checked above); kept for safety
                continue
            a, b = int(it.group(1)), int(it.group(2) or it.group(1))
            for n in range(a, b + 1):
                if (m.group(1), n) not in defined:
                    fail(f"{m.group(1)} S{n} is cited but not defined in supplement_index.csv: {m.group(0)!r}")
            # (figure panels are drawn in the figure, not listed in the index; a range of tables takes no letters)
            if it.group(3) and panels is not None and m.group(1) == "Table" and a == b:
                letters = panel_set(it.group(3))
                n_panel_cites += 1
                if not letters <= panels.get(a, set()):
                    fail(f"Table S{a} has no panel {', '.join(sorted(letters - panels.get(a, set())))} "
                         f"(panels: {', '.join(sorted(panels.get(a, set()))) or 'none'}): {m.group(0)!r}")
    seen = set()
    warn = []
    for m in re.finditer(r"Tables? (S\d+)((?:(?:, and |, | and |-)S\d+)*)", body):
        labs = [m.group(1)] + re.findall(r"S\d+", m.group(2))
        new_labs = [x for x in labs if x not in seen]
        if new_labs:
            ctx = body[m.start(): m.start() + 160]
            if "Multimedia Appendix 1" not in ctx.split(")")[0] and "Multimedia Appendix 1" not in ctx.split(". ")[0]:
                warn.append(f"first citation of {', '.join(new_labs)} without 'Multimedia Appendix 1': {ctx[:90]!r}")
        seen.update(labs)
    for w in warn:
        print("WARNING:", w)

    # one paragraph per abbreviation: consecutive lines would be merged into a single paragraph by pandoc
    def space_abbr(m):
        lines = [l for l in m.group(2).split("\n") if l.strip()]
        return m.group(1) + "\n\n".join(lines) + "\n"
    text = re.sub(r"(## Abbreviations\n\n)(.*?)(?=\n## )", space_abbr, text, flags=re.S)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)

    # ---------------------------------------------------------------- word counts
    secs = re.split(r"(?m)^(## .+)$", body)
    counts = {}
    counts["Title block"] = words(secs[0].split("**Running title**")[0])
    for i in range(1, len(secs), 2):
        name, content = secs[i][3:].strip(), secs[i + 1]
        tab_words = sum(words(b) for b in re.findall(r"(?s)<!-- BEGIN TABLE \d+ -->(.*?)<!-- END TABLE \d+ -->", content))
        leg = sum(words(x) for x in re.findall(r"(?m)^\*\*Figure \d+\.\*\*.*$", content))
        counts[name] = (words(content), tab_words, leg)
    print(f"wrote {OUT}")
    print(f"references: {len(order)}; supplementary citations: {len(seen)}; lists tidied: {len(tidied)}; panel citations "
          f"checked: {n_panel_cites if panels is not None else 'not checked (no panel column)'}")
    tot = 0
    for k, v in counts.items():
        if isinstance(v, tuple):
            print(f"  {k:<32} {v[0]:>6} words (tables {v[1]}, figure legends {v[2]})")
            if k not in ("Abbreviations", "Multimedia Appendix 1", "Multimedia Appendix 2"):
                tot += v[0]
        else:
            print(f"  {k:<32} {v:>6} words")
            tot += v
    print(f"  total counted by the JMIR definition (approximate): {tot}")


if __name__ == "__main__":
    main()
