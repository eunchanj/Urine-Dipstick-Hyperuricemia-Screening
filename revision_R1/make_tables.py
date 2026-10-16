"""Main tables of the revised manuscript (JMIR Medical Informatics #107515, revision R1).

Writes
  ../tables.md                 Tables 1-5 (main text; >=7.0 primary, >=9.0 secondary)
  ../tables_appendix_ge10.md   >=10.0 mg/dL (exploratory) counterparts of Tables 2-4 for the Multimedia Appendix
  baseline_urate.csv           serum urate distribution of the two cohorts (see below)

The tables describe the models of the revision: the calibrated LightGBM + XGBoost + CatBoost ensembles (bestmodel_v5;
12 inputs: the 9 dipstick fields, age, sex and BMI), trained on the training persons of the development cohort
(cohort.main_cohort: adults at their first urine test with serum urate and BMI of the urine-test day, one record per
person, one person-level split shared by the three thresholds) and validated on the internal test persons and on
KNHANES adults (dipstick results coded by the label-matched mapping X1). Cohort counts come from cohort_info_v2.csv
(build_cohort_v2.py).

Every number is read from the analysis CSVs in this folder and formatted here; nothing is typed by hand.
The one quantity no analysis CSV holds is the serum urate distribution of the two cohorts (mean/SD and median/IQR).
It is computed from the cohort cache with cohort.main_cohort() into baseline_urate.csv (rebuilt with --rebuild-urate,
and automatically whenever the file does not match cohort_info_v2.csv) and cross-checked against cohort_info_v2.csv
and encoding.csv before use.

Row labels that an analysis script may rename (the ensemble strategies of fu_sex_operating.csv, the ensemble arms of
ensemble_vs_simple.csv, the refitted ensemble of temporal.csv) are found by pattern with one_label(); a missing or
ambiguous label stops the script and lists the labels found.

Pandoc markdown conventions (as in v2.0/working/build_docx.py): pipe tables, **bold** first-column rows
for sections (the docx builder indents the other rows), ^a^ superscript footnote markers, an HTML comment
naming the source CSVs above each table (dropped in docx). Ranges use an ASCII hyphen (0.834-0.844);
an interval with a negative bound uses "to" (-0.52 to -0.45). Supplementary tables are cited with the
{{T:key}} / {{N:key}} tokens of supplement_index.csv (resolved by build_manuscript.py). A token resolves to the table
only, so the panel letters that follow a token are written by supp_panels() from make_supplement.STRUCTURE (the
structure that make_supplement.py checks against the appendix it writes), never typed by hand.

Table 5 (sensitivity analyses) includes the row 'Logistic instead of isotonic calibration' (platt_vs_isotonic.csv,
method logistic_logodds, KNHANES adults) and gives the IPW rows the weighted O:E of verification_bias.csv.

Rounding (results_digest.md 0.5): AUROC/AUPRC 3 decimals; AUROC differences 3 decimals, 4 where the
difference is <0.005; O:E, calibration intercept and slope 2 decimals; Brier 4 decimals; scaled Brier
3 decimals; percentages 1 decimal, 2 where <1% (or >99%).

Usage: python make_tables.py [--rebuild-urate] [--no-render-check]
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Output folder for the table files: PROJ_U_REVISION_DIR overrides; original: parent of this folder.
REV = os.environ.get("PROJ_U_REVISION_DIR", os.path.normpath(os.path.join(HERE, "..")))
OUT_MAIN = os.path.join(REV, "tables.md")
OUT_APPX = os.path.join(REV, "tables_appendix_ge10.md")
URATE_CSV = os.path.join(HERE, "baseline_urate.csv")

MAIN_TARGETS = (">=7.0", ">=9.0")
APPX_TARGETS = (">=10.0",)
ALL_TARGETS = MAIN_TARGETS + APPX_TARGETS
TLAB = {">=7.0": "≥7.0 mg/dL", ">=9.0": "≥9.0 mg/dL", ">=10.0": "≥10.0 mg/dL"}
ROLE = {">=7.0": "primary", ">=9.0": "secondary", ">=10.0": "exploratory"}
THR = {">=7.0": 7.0, ">=9.0": 9.0, ">=10.0": 10.0}
NA = "N/A"

# cohort_info_v2.csv items (build_cohort_v2.py); per-threshold items are f"{prefix} events {target}" with the prefixes
# "cohort", "training", "test", "KNHANES" and "KNHANES female"
CI_DEV = "analytic cohort persons"
CI_EXT = "KNHANES adults with 12 inputs and urate"
CI_TRAIN, CI_TEST = "training persons", "test persons"
CI_BELOW = "serum urate '<1.0' persons (counted as 0.5)"
CI_BMI_CHECKUP, CI_BMI_NURSING = "BMI source checkup (persons)", "BMI source nursing (persons)"
CI_MINOR = "cohort persons younger than 19 by exact completed years (atlas)"

# model labels of feature_tiers.csv / tuned_per_set.csv / ensemble_vs_simple.csv (calc_feature_tiers.py); the ensemble
# and its LightGBM-member arm are resolved with ens_labels()
LGB, LR, SPL, GRID = "LightGBM (fixed)", "Logistic regression", "LR (age/BMI splines + age x male)", "LightGBM (GRID-tuned)"
TIERS = ("2F", "3F", "9F", "11F", "12F")
TL = {"2F": "Age, sex", "3F": "Age, sex, BMI", "9F": "Dipstick only", "11F": "Dipstick, age, sex",
      "12F": "Dipstick, age, sex, BMI"}
TIER_LAB = {k: f"{k}: {v[0].lower()}{v[1:]}" for k, v in TL.items()}   # "12F: dipstick, age, sex, BMI"
# bmi_window.csv, section "performance" (calc_missing_bias.py): internal test persons by the source of BMI
BMI_SRC = (("internal test: source checkup record", "BMI from a same-day health checkup"),
           ("internal test: source nursing record", "BMI from the nursing record"))


# ============================================================================ reading helpers
def R(name):
    return pd.read_csv(os.path.join(HERE, name), low_memory=False)


def pick(df, **kw):
    """Exactly one row matching every key=value (floats compared with isclose)."""
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        col = df[k]
        if isinstance(v, float):
            m &= np.isclose(pd.to_numeric(col, errors="coerce"), v).astype(bool)
        else:
            m &= (col.astype(str) == str(v)).values
    sub = df[m]
    if len(sub) != 1:
        raise KeyError(f"pick({kw}) matched {len(sub)} rows")
    return sub.iloc[0]


def one_label(values, what, include, exclude=()):
    """The one distinct label among `values` whose lower-case text contains every string of `include` and none of
    `exclude` (labels written by analysis scripts that may rename them). Fails with the labels found."""
    vals = sorted({str(v) for v in pd.Series(list(values)).dropna()})
    hit = [v for v in vals if all(s in v.lower() for s in include) and not any(s in v.lower() for s in exclude)]
    if len(hit) != 1:
        raise KeyError(f"{what}: expected one label containing {list(include)}"
                       + (f" and none of {list(exclude)}" if exclude else "") + f"; found {hit} among {vals}")
    return hit[0]


def info():
    """cohort_info_v2.csv (build_cohort_v2.py) as {item: value}."""
    ci = R("cohort_info_v2.csv")
    assert ci["item"].is_unique, "cohort_info_v2.csv: repeated items"
    return dict(zip(ci["item"], ci["value"]))


def cnum(ci, item):
    """Integer count of a cohort_info_v2.csv item."""
    if item not in ci:
        raise KeyError(f"cohort_info_v2.csv has no item {item!r}")
    return int(round(float(ci[item])))


# ============================================================================ formatting helpers
def fint(x):
    return f"{int(round(float(x))):,}"


def fnum(x, d):
    return f"{float(x):.{d}f}"


def _nz(x, d):
    """Format x with d decimals; a nonzero value that would print as (-)0.000... gets more decimals."""
    s = fnum(x, d)
    k = d
    while float(x) != 0 and re.fullmatch(r"-?0\.0+", s) and k < d + 3:
        k += 1
        s = fnum(x, k)
    return s


def frange(lo, hi, d):
    a, b = _nz(lo, d), _nz(hi, d)
    return f"{a} to {b}" if (a.startswith("-") or b.startswith("-")) else f"{a}-{b}"


def fci(e, lo, hi, d, sign=False):
    s = fnum(e, d)
    if sign and not s.startswith("-"):
        s = "+" + s
    return f"{s} ({frange(lo, hi, d)})"


def pdec(v):
    """Decimals for a percentage: 2 where <1% or >99%, else 1."""
    v = abs(float(v))
    return 2 if (v < 1 or v > 99) else 1


def fpct(p, d=None):
    """p as a proportion (0-1) -> percentage string."""
    v = 100 * float(p)
    d = pdec(v) if d is None else d
    return fnum(v, d)


def fpct_ci(p, lo, hi, d=None):
    v = 100 * float(p)
    d = pdec(v) if d is None else d
    return f"{fnum(v, d)} ({frange(100 * float(lo), 100 * float(hi), d)})"


def fpct_small(p):
    """Proportion -> percentage with 2 decimals below 10%, else 1 (prevalence-type quantities)."""
    v = 100 * float(p)
    return fnum(v, 2 if v < 10 else 1)


def fnpct(n, total):
    n = int(round(float(n)))
    if n == 0:
        return "0"
    v = 100 * n / float(total)
    return f"{n:,} (<0.01)" if v < 0.005 else f"{n:,} ({fnum(v, pdec(v))})"


def fdiff(e, lo, hi):
    """Signed AUROC difference: 3 decimals, 4 where |difference| < 0.005 (more where it would print as 0.0000)."""
    d = 4 if abs(float(e)) < 0.005 else 3
    s = _nz(e, d)
    return f"{s if s.startswith('-') else '+' + s} ({frange(lo, hi, d)})"


def fp(p):
    """JMIR style P value: *P*=.046, *P*<.001."""
    p = float(p)
    return "*P*<.001" if p < 0.001 else "*P*=" + fnum(p, 3).lstrip("0")


def ref(table, appendix):
    return f"the counterpart of {table}" if appendix else table


def supp_panels(*keys):
    """'{{T:key}} panels A and C': a token for the supplementary table that holds the panels `keys` (the first key is
    the token), followed by their letters. Letters come from make_supplement.STRUCTURE (static_labels), which
    make_supplement.py asserts against the appendix it writes; all keys must be lettered panels of one table."""
    sys.path.insert(0, HERE)
    import make_supplement  # noqa: E402  (static structure only; nothing is read or written on import)
    tables, _, _ = make_supplement.static_labels()
    unknown = [k for k in keys if k not in tables]
    assert not unknown, f"supplementary keys not in make_supplement.STRUCTURE: {unknown}"
    labs = {tables[k][0] for k in keys}
    letters = [tables[k][1] for k in keys]
    assert len(labs) == 1, f"panels of different supplementary tables in one citation: {keys} -> {sorted(labs)}"
    assert all(letters), f"not a lettered panel (a table key, or the only panel of a table): {keys}"
    return "{{T:" + keys[0] + "}}" + make_supplement.panel_list(set(letters))


def fthr(x):
    """Probability threshold to 3 significant digits, keeping trailing zeros (0.200, 0.0141)."""
    x = float(x)
    d = max(3, 2 - int(math.floor(math.log10(abs(x))))) if x > 0 else 3
    return fnum(x, d)


def sets_abbrev(tiers):
    """'2F, 3F, 9F, 11F, 12F: input sets with 2, 3, 9, 11 and 12 inputs' (counts taken from the labels)."""
    n = [str(int(t[:-1])) for t in tiers]
    return f"{', '.join(tiers)}: input sets with {', '.join(n[:-1])} and {n[-1]} inputs"


class Notes:
    """Footnote letters assigned in order of first use (JMIR: left to right, top to bottom);
    general notes (no marker) are printed after the lettered ones."""

    def __init__(self):
        self.order, self.text, self.general = [], {}, []

    def __call__(self, key, text=None):
        if key not in self.text:
            if text is None:
                raise KeyError(key)
            self.order.append(key)
            self.text[key] = text
        return f"^{chr(ord('a') + self.order.index(key))}^"

    def note(self, text):
        self.general.append(text)

    def multi(self, *keys):
        """One superscript for several registered notes, letters in order: ^b,c^."""
        return "^" + ",".join(sorted(self(k)[1:-1] for k in keys)) + "^"

    def render(self):
        return [f"^{chr(ord('a') + i)}^{self.text[k]}" for i, k in enumerate(self.order)] + self.general


def md_table(header, rows, budget=94, label_cap=45, label_min=14, pad=3):
    """Pipe table. The separator dash counts set pandoc's relative column widths (used because the rows are longer
    than pandoc's 72-column limit). `budget` is about the number of 10-pt Times New Roman digits across the 6.5-inch
    text width of the JMIR docx build; `pad` allows for cell padding. Priorities: (1) no number or word in a body
    cell is split (body-token floor); (2) header words are not split (header-token floor), dropped only if the
    table cannot otherwise fit; (3) cells are kept on one line where the budget allows; the label column takes
    what is left. Section rows (text in the first cell only) are ignored."""
    for r in rows:
        assert len(r) == len(header), (header, r)
    plain = lambda c: re.sub(r"\^[^^]*\^|\*\*", "", str(c))  # noqa: E731
    body = [r for r in rows if any(str(c) for c in r[1:])]
    n = len(header)
    btok = [pad + max([len(w) for r in body for w in plain(r[j]).split()] + [1]) for j in range(n)]
    htok = [pad + max(len(w) for w in plain(header[j]).split()) for j in range(n)]
    full = [pad + max(len(plain(r[j])) for r in body) for j in range(n)]
    floor = [max(x, y) for x, y in zip(btok, htok)]
    w = [min(max(full[0], floor[0]), label_cap)] + [max(full[j], floor[j]) for j in range(1, n)]
    if sum(w) > budget:  # shrink value columns towards their floors
        for f in np.linspace(1, 0, 201):
            w = [w[0]] + [max(floor[j], math.ceil(full[j] * f)) for j in range(1, n)]
            if sum(w) <= budget:
                break
    if sum(w) > budget:  # then the label column
        w[0] = max(btok[0], label_min, budget - sum(w[1:]))
    if sum(w) > budget:  # last resort: let header words wrap, never body tokens
        slack = {j: w[j] - btok[j] for j in range(1, n) if w[j] > btok[j]}
        excess = sum(w) - budget
        tot = sum(slack.values())
        for j, sl in slack.items():
            w[j] = max(btok[j], w[j] - math.ceil(excess * sl / tot))
    sep = "|" + "|".join("-" * max(3, x) for x in w) + "|"
    out = ["| " + " | ".join(header) + " |", sep]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def block(number, caption, sources, header, rows, notes, abbrev):
    parts = [f"<!-- {number}: generated by working/make_tables.py from {', '.join(sources)}; do not edit by hand -->",
             f"**{number}.** {caption}", md_table(header, rows)]
    parts += notes.render()
    parts.append(abbrev)
    return "\n\n".join(parts)


def blank(n):
    return [""] * n


# ============================================================================ serum urate (cohort cache)
def build_urate_csv():
    """Serum urate distribution of the development cohort (dev_v2) and the KNHANES adults, both from
    cohort.main_cohort() -> baseline_urate.csv."""
    sys.path.insert(0, HERE)
    import cohort  # noqa: E402  (heavy import only when rebuilding)
    d, _, _, ext = cohort.main_cohort()
    assert d[cohort.ID].is_unique and ext["id"].is_unique
    rows = []
    for name, f, below in (("development", d, int(d["urate_day_below"].astype(bool).sum())),
                           ("knhanes_adults", ext, np.nan)):
        u = f["urate"].astype(float)
        rows.append(dict(cohort=name, n=len(f), age_min=float(f["age"].min()), age_max=float(f["age"].max()),
                         n_urate=int(u.notna().sum()), urate_mean=u.mean(),
                         urate_sd=u.std(ddof=1), urate_median=u.median(), urate_q1=u.quantile(0.25),
                         urate_q3=u.quantile(0.75), urate_min=u.min(), urate_max=u.max(),
                         n_below_detection_coded_05=below,
                         **{f"n_ge{int(t)}": int((u >= t).sum()) for t in THR.values()}))
    out = pd.DataFrame(rows)
    out["note"] = ("serum urate in mg/dL; development = cohort.main_cohort() d (dev_v2: serum urate of the urine-test "
                   "day, '<1.0' results counted as 0.5 mg/dL, n_below_detection_coded_05 in persons); knhanes_adults = "
                   "cohort.main_cohort() ext (KNHANES adults aged >= 19 with the 12 inputs); quantiles by pandas "
                   "linear interpolation")
    out.to_csv(URATE_CSV, index=False)
    print(f"wrote {URATE_CSV}")


def urate_csv_current(ci):
    """True when baseline_urate.csv describes the cohorts of cohort_info_v2.csv."""
    if not os.path.exists(URATE_CSV):
        return False
    u = R("baseline_urate.csv")
    if sorted(u["cohort"].astype(str)) != ["development", "knhanes_adults"]:
        return False
    u = u.set_index("cohort")
    return (int(u.loc["development", "n"]) == cnum(ci, CI_DEV) and int(u.loc["knhanes_adults", "n"]) == cnum(ci, CI_EXT)
            and int(u.loc["development", "n_below_detection_coded_05"]) == cnum(ci, CI_BELOW))


def cohort_constants():
    """AGE_MAX and ADULT_AGE as defined in cohort.py (parsed, not imported)."""
    src = open(os.path.join(HERE, "cohort.py"), encoding="utf-8").read()
    return {k: int(re.search(rf"(?m)^{k}\s*=\s*(\d+)", src).group(1)) for k in ("AGE_MAX", "ADULT_AGE")}


def load_urate(ci):
    """baseline_urate.csv rows (development, knhanes_adults), cross-checked against cohort_info_v2.csv and
    encoding.csv."""
    u = R("baseline_urate.csv").set_index("cohort")
    dev, ext = u.loc["development"], u.loc["knhanes_adults"]
    adult = cohort_constants()["ADULT_AGE"]
    assert int(dev.n) == cnum(ci, CI_DEV) and int(ext.n) == cnum(ci, CI_EXT), "baseline_urate.csv is stale"
    assert int(dev.n_urate) == int(dev.n) and int(ext.n_urate) == int(ext.n)
    assert float(dev.age_min) >= adult and float(ext.age_min) >= adult
    enc = R("encoding.csv")
    sm = enc[enc.section == "summary"].set_index("feature")
    assert float(ext.age_min) == float(sm.loc["age", "min_ext"]) and float(ext.age_max) == float(sm.loc["age", "max_ext"])
    assert float(dev.age_min) == float(sm.loc["age", "min_dev"]) and float(dev.age_max) == float(sm.loc["age", "max_dev"])
    for t, v in THR.items():
        assert int(dev[f"n_ge{int(v)}"]) == cnum(ci, f"cohort events {t}"), t
        assert int(ext[f"n_ge{int(v)}"]) == cnum(ci, f"KNHANES events {t}"), t
    assert int(dev.n_below_detection_coded_05) == cnum(ci, CI_BELOW)
    return dev, ext


def design():
    """Design descriptors taken from code/CSVs: adult age limit (cohort.py) and the KNHANES survey years."""
    cc = cohort_constants()
    u = R("baseline_urate.csv").set_index("cohort")
    dby = R("dipstick_by_year.csv")
    yrs = dby[(dby.period_type == "year") & dby.cohort.str.startswith("KNHANES")].period.astype(int)
    assert float(u.loc["knhanes_adults", "age_min"]) >= cc["ADULT_AGE"]
    return dict(adult=cc["ADULT_AGE"], ext_years=f"{yrs.min()}-{yrs.max()}")


# ============================================================================ Table 1
AGE_BANDS = ((0, 19, "Age <19 years"), (19, 40, "Age 19-39 years"), (40, 60, "Age 40-59 years"),
             (60, 80, "Age 60-79 years"), (80, math.inf, "Age ≥80 years"))


def _bin_bounds(label):
    lo, hi = re.match(r"\[\s*([\d.]+)\s*,\s*([\d.]+|inf)\s*\)", label).groups()
    return float(lo), (math.inf if hi == "inf" else float(hi))


def table1():
    ci = info()
    enc = R("encoding.csv")
    cb = enc[enc.section == "codebook"]
    raw = enc[enc.section == "raw_codes"]
    sm = enc[enc.section == "summary"].set_index("feature")
    bw = R("bmi_window.csv")
    sw = R("survey_weighted.csv")
    dby = R("dipstick_by_year.csv")
    sg = R("subgroups.csv")
    dev_u, ext_u = load_urate(ci)
    dz = design()

    N_DEV, N_EXT = cnum(ci, CI_DEV), cnum(ci, CI_EXT)
    n_inputs = len(sm)                              # the model inputs, one summary row each
    assert "ua_wbc(s)" not in set(cb.feature) | set(sm.index), "encoding.csv still lists leukocyte esterase"
    for f in sm.index:
        assert int(sm.loc[f, "n_dev"]) == N_DEV and int(sm.loc[f, "n_ext"]) == N_EXT, f

    def cnt(feature, cond, col):
        s = cb[cb.feature == feature]
        return int(s.loc[cond(s.value_num.astype(float)), col].sum())

    def pos_row(label, feature, cond, ext=True):
        return [label, fnpct(cnt(feature, cond, "n_dev"), N_DEV),
                fnpct(cnt(feature, cond, "n_ext"), N_EXT) if ext else None]

    def msd(feature, d, suffix=""):
        return (f"{fnum(sm.loc[feature, 'mean_dev'], d)} ({fnum(sm.loc[feature, 'sd_dev'], d)})",
                f"{fnum(sm.loc[feature, 'mean_ext'], d)} ({fnum(sm.loc[feature, 'sd_ext'], d)}){suffix}")

    fn = Notes()
    # ---------------------------------------------------------------- periods
    dy = dby[dby.period_type == "year"].copy()
    dy["yr"] = dy.period.astype(int)
    is_dev, is_ext = dy.cohort.str.startswith("development"), dy.cohort.str.startswith("KNHANES")
    dev_years, ext_years = dy[is_dev].yr, dy[is_ext].yr
    assert int(dy[is_dev].n.sum()) == N_DEV and int(dy[is_ext].n.sum()) == N_EXT
    # ---------------------------------------------------------------- age bands (codebook bins aggregated)
    age = cb[cb.feature == "age"].copy()
    age[["lo", "hi"]] = [list(_bin_bounds(v)) for v in age.value]
    bands = []
    for lo, hi, lab in AGE_BANDS:
        s = age[(age.lo >= lo) & (age.hi <= hi)]
        bands.append((lab, int(s.n_dev.sum()), int(s.n_ext.sum())))
    assert sum(b[1] for b in bands) == N_DEV and sum(b[2] for b in bands) == N_EXT
    assert bands[0][1:] == (0, 0) and cnum(ci, CI_MINOR) == 0           # adults only in both cohorts
    ext_sg = sg[sg.cohort.str.startswith("External") & (sg.target == ">=7.0")]
    for lab, sgname in zip([b[0] for b in bands[1:]], ("Age 19-39", "Age 40-59", "Age 60-79", "Age >=80")):
        assert int(pick(ext_sg, subgroup=sgname).n) == dict((b[0], b[2]) for b in bands)[lab], lab
    # ---------------------------------------------------------------- BMI and its source (urine-test day)
    bmi = cb[cb.feature == "bmi_unified"].copy()
    bmi[["lo", "hi"]] = [list(_bin_bounds(v)) for v in bmi.value]
    assert 25.0 in set(bmi.lo)
    bmi25 = bmi[bmi.lo >= 25]
    n_ck, n_nu = cnum(ci, CI_BMI_CHECKUP), cnum(ci, CI_BMI_NURSING)
    assert n_ck + n_nu == N_DEV
    cmix = bw[bw.section == "case_mix_by_source"]
    assert int(pick(cmix, population="analytic cohort: source checkup record").n) == n_ck
    assert int(pick(cmix, population="analytic cohort: source nursing record").n) == n_nu
    # ---------------------------------------------------------------- footnote material
    age_max_ext = int(sm.loc["age", "max_ext"])
    rc = raw.set_index(["feature", "value_num"])

    def kn_code(feature, raw_code, mapping="X1"):
        """Model-scale value that a KNHANES coding gives a raw KNHANES code (encoding.csv raw_codes)."""
        return float(rc.loc[(feature, float(raw_code)), f"map_{mapping}"])

    # bilirubin: no trace level in the development data; KNHANES "+" (raw code 2) coded as the development 1+ (1.0)
    bil = cb[cb.feature == "ua_bilirubin"]
    assert int(bil.loc[np.isclose(bil.value_num.astype(float), 0.5), "n_dev"].sum()) == 0
    assert kn_code("ua_bilirubin", 2) == 1.0
    # urobilinogen: KNHANES normal (raw 0) and trace (raw 1) coded as the development normal (its modal code)
    uro = cb[cb.feature == "ua_urobilinogen"]
    uro_norm = float(uro.loc[uro.n_dev.astype(float).idxmax(), "value_num"])
    assert kn_code("ua_urobilinogen", 0) == kn_code("ua_urobilinogen", 1) == uro_norm < kn_code("ua_urobilinogen", 2)
    uro_ext_above = cnt("ua_urobilinogen", lambda v: v > uro_norm, "n_ext")
    assert abs(uro_ext_above / N_EXT - float(sm.loc["ua_urobilinogen", "abn_ext"])) < 1e-9
    sg_dev_max = float(sm.loc["ua_sg", "max_dev"])
    sg_ext_max = float(sm.loc["ua_sg", "max_ext"])
    sg_ext_hi = cnt("ua_sg", lambda v: np.round(v, 4) > round(sg_dev_max, 4), "n_ext")
    sgc = cb[cb.feature == "ua_sg"]
    step_dev = float(np.round(np.diff(np.sort(sgc[sgc.n_dev >= 100].value_num.astype(float))).min(), 3))
    step_ext = float(np.round(np.diff(np.sort(sgc[sgc.n_ext > 0].value_num.astype(float))).min(), 3))
    # development reporting change (late 2021): ketone and bilirubin positivity before and after
    pooled = dby[(dby.period_type == "pooled") & dby.cohort.str.startswith("development")].set_index("period")
    early, late = pooled.loc["2010-2021"], pooled.loc["2022-2025"]
    assert late.pos_ketone < early.pos_ketone and late.pos_bilirubin < early.pos_bilirubin

    # ---------------------------------------------------------------- rows
    c_mark = fn("cohorts",
                f"Development: persons aged ≥{dz['adult']} years at their first urine test, with serum urate and BMI "
                f"recorded on the day of that test and all {n_inputs} inputs (one record per person). KNHANES: "
                f"participants aged ≥{dz['adult']} years with the {n_inputs} inputs and serum urate; dipstick results "
                "coded by the label-matched mapping X1 ({{T:mapping}} in Multimedia Appendix 1).")
    H = ["Characteristic", f"Development cohort (n={N_DEV:,}){c_mark}", f"KNHANES adults (n={N_EXT:,}){c_mark}"]
    rows = [["**Demographics**", "", ""],
            ["Years of testing", f"{dev_years.min()}-{dev_years.max()}",
             f"{ext_years.min()}-{ext_years.max()} (survey years)"]]
    a_mean = msd("age", 1)
    rows.append(["Age, years, mean (SD)", a_mean[0], a_mean[1]])
    for lab, nd, ne in bands:
        if nd == 0 and ne == 0:   # adults only: the band <19 years is empty in both cohorts
            continue
        mark = ""
        if lab.startswith("Age ≥80"):
            mark = fn("age80", f"KNHANES age is top-coded at {age_max_ext} years.")
        rows.append([f"{lab}, n (%){mark}", fnpct(nd, N_DEV), fnpct(ne, N_EXT)])
    male = cb[(cb.feature == "male")]
    rows.append(["Male sex, n (%)", fnpct(pick(male, value_num=1.0).n_dev, N_DEV),
                 fnpct(pick(male, value_num=1.0).n_ext, N_EXT)])
    b_mean = msd("bmi_unified", 1)
    rows.append(["BMI, kg/m², mean (SD)", b_mean[0], b_mean[1]])
    rows.append(["BMI ≥25 kg/m², n (%)", fnpct(bmi25.n_dev.sum(), N_DEV), fnpct(bmi25.n_ext.sum(), N_EXT)])
    src_mark = fn("bmisrc",
                  "BMI measured at a health checkup on the day of the urine test, the only marker of care setting in "
                  "the data; for the other persons, BMI was calculated from height and weight in the nursing record "
                  "of that day.")
    rows.append([f"BMI from a same-day health checkup, n (%){src_mark}", fnpct(n_ck, N_DEV), "Not applicable"])

    rows.append(["**Urine dipstick**", "", ""])
    rows.append(pos_row("Glucose, trace or above, n (%)", "ua_glucose", lambda v: v > 0))
    rows.append(pos_row("Protein, trace or above, n (%)", "ua_protein", lambda v: v > 0))
    k_mark = fn("lab2021",
                "Development positivity fell with a laboratory reporting change in late 2021 (Tables {{N:posyear}} "
                "and {{N:levels}} in Multimedia Appendix 1).")
    rows.append(pos_row(f"Ketone, trace or above, n (%){k_mark}", "ua_ketone", lambda v: v > 0))
    fn("bil", "The development laboratory reports no trace level; KNHANES \"+\" was coded as 1+ ({{T:mapping}} in "
              "Multimedia Appendix 1).")
    rows.append(pos_row(f"Bilirubin, 1+ or above, n (%){fn.multi('lab2021', 'bil')}", "ua_bilirubin", lambda v: v > 0))
    rows.append(pos_row("Occult blood, trace or above, n (%)", "ua_blood", lambda v: v > 0))
    u_mark = fn("uro",
                "Above normal: 1+ or above in the development data and \"+\" or above in KNHANES, where normal and "
                "trace results were both coded as the development normal ({{T:mapping}} in Multimedia Appendix 1).")
    rows.append(pos_row(f"Urobilinogen, above normal, n (%){u_mark}", "ua_urobilinogen", lambda v: v > uro_norm))
    rows.append(pos_row("Nitrite, positive, n (%)", "ua_nitrite", lambda v: np.isclose(v, 1.0)))
    p_mean = msd("ua_pH", 2)
    rows.append(["pH, mean (SD)", p_mean[0], p_mean[1]])
    s_mark = fn("sg",
                f"Steps of {step_dev:.3f} up to {sg_dev_max:.3f} in the development data; steps of {step_ext:.3f} up to "
                f"{sg_ext_max:.3f} in KNHANES ({fpct(sg_ext_hi / N_EXT)}% above {sg_dev_max:.3f}).")
    s_ms = msd("ua_sg", 3)
    rows.append([f"Specific gravity, mean (SD){s_mark}", s_ms[0], s_ms[1]])

    rows.append(["**Serum urate**", "", ""])
    ur_mark = fn("urate",
                 f"\"<1.0 mg/dL\" results were counted as 0.5 mg/dL ({int(dev_u.n_below_detection_coded_05):,} persons "
                 "in the development cohort).")
    rows.append([f"Serum urate, mg/dL, mean (SD){ur_mark}",
                 f"{fnum(dev_u.urate_mean, 2)} ({fnum(dev_u.urate_sd, 2)})",
                 f"{fnum(ext_u.urate_mean, 2)} ({fnum(ext_u.urate_sd, 2)})"])
    rows.append(["Serum urate, mg/dL, median (IQR)",
                 f"{fnum(dev_u.urate_median, 2)} ({frange(dev_u.urate_q1, dev_u.urate_q3, 2)})",
                 f"{fnum(ext_u.urate_median, 2)} ({frange(ext_u.urate_q1, ext_u.urate_q3, 2)})"])
    w = sw.set_index("target")
    wt_mark = fn("wt",
                 "Unweighted. With the KNHANES examination weights, prevalence was "
                 + "; ".join(f"{fpct_small(w.loc[t, 'prevalence_weighted'])}% ({TLAB[t]})" for t in ALL_TARGETS)
                 + ".")
    lab = {">=7.0": "Hyperuricemia, ≥7.0 mg/dL (primary)", ">=9.0": "≥9.0 mg/dL (secondary)",
           ">=10.0": "≥10.0 mg/dL (exploratory)"}
    for i, t in enumerate(ALL_TARGETS):
        rows.append([f"{lab[t]}, n (%)", fnpct(cnum(ci, f"cohort events {t}"), N_DEV),
                     fnpct(cnum(ci, f"KNHANES events {t}"), N_EXT) + (wt_mark if i == 0 else "")])

    return block("Table 1",
                 "Baseline characteristics of the development cohort and the KNHANES external validation cohort.",
                 ["cohort_info_v2.csv", "encoding.csv", "baseline_urate.csv", "bmi_window.csv", "dipstick_by_year.csv",
                  "survey_weighted.csv", "subgroups.csv"],
                 H, rows, fn,
                 "BMI: body mass index; IQR: interquartile range; KNHANES: Korea National Health and Nutrition "
                 "Examination Survey; SD: standard deviation.")


# ============================================================================ Table 2
def table2(targets, number, appendix=False):
    ci = info()
    dis, cal, rc = R("discrimination.csv"), R("calibration.csv"), R("recalibration.csv")
    mp, ip = R("main_performance.csv"), R("fu_int_persons.csv").set_index("target")
    dz = design()
    fn = Notes()
    cols = [(t, c) for t in targets for c in ("internal", "external")]
    N = {"internal": cnum(ci, CI_TEST), "external": cnum(ci, CI_EXT)}
    EVENTS = {"internal": "test events", "external": "KNHANES events"}
    MP_COHORT = {"internal": "internal test", "external": "KNHANES adults"}

    def D(t, c):
        return pick(dis, target=t, cohort=c)

    def C(t, c):
        return pick(cal, target=t, cohort=c)

    for t, c in cols:   # the files describe the same evaluations of the models (main_performance.csv: calc_main.py)
        d_, c_, m_ = D(t, c), C(t, c), pick(mp, target=t, cohort=MP_COHORT[c])
        assert int(d_.n) == int(c_.n) == int(m_.n) == N[c], (t, c)
        assert int(d_.positives) == int(c_.events) == int(m_.events) == cnum(ci, f"{EVENTS[c]} {t}"), (t, c)
        assert abs(float(d_.auroc) - float(m_.auroc)) < 1e-9 and abs(float(c_.oe) - float(m_.oe)) < 1e-9, (t, c)
    for t in targets:   # one record per person; the same internal test persons for every threshold
        r = ip.loc[t]
        assert int(r.records) == int(r.persons) == N["internal"] and int(r.persons_with_2plus_records) == 0, t
        assert int(r.training_persons) == cnum(ci, CI_TRAIN), t
    int_note = (f"Internal test set: {N['internal']:,} persons of the development cohort held out from training (one "
                "record per person; the same persons for every threshold).")
    ext_note = (f"KNHANES {dz['ext_years']} participants aged ≥{dz['adult']} years; dipstick results coded by the "
                "label-matched mapping X1 ({{T:mapping}} in Multimedia Appendix 1).")
    head = ["Measure"]
    for t, c in cols:
        head.append(f"{TLAB[t]}, {c}" + (fn("int", int_note) if c == "internal" else fn("ext", ext_note)))

    nb = int(C(targets[0], "internal").n_boot_used)
    assert all(int(C(t, c).n_boot_used) == nb for t, c in cols)

    rows = [["**Cohort**"] + blank(len(cols)),
            ["Persons, n"] + [fint(D(t, c).n) for t, c in cols],
            ["Events, n (%)"] + [f"{fint(D(t, c).positives)} ({fpct(D(t, c).prevalence)})" for t, c in cols],
            ["**Discrimination**"] + blank(len(cols)),
            ["AUROC (95% CI)"] + [fci(D(t, c).auroc, D(t, c).auroc_lo, D(t, c).auroc_hi, 3) for t, c in cols],
            ["AUPRC (95% CI)"] + [fci(D(t, c).auprc, D(t, c).auprc_lo, D(t, c).auprc_hi, 3) for t, c in cols],
            ["AUPRC relative to prevalence (95% CI)"]
            + [fci(D(t, c).auprc_lift, D(t, c).auprc_lift_lo, D(t, c).auprc_lift_hi, 1) for t, c in cols],
            ["**Calibration**"] + blank(len(cols)),
            ["Mean predicted risk, %"] + [fpct_small(C(t, c).mean_pred) for t, c in cols],
            ["Calibration intercept (95% CI)"]
            + [fci(C(t, c).intercept, C(t, c).intercept_lo, C(t, c).intercept_hi, 2) for t, c in cols],
            ["Calibration slope (95% CI)"]
            + [fci(C(t, c).slope, C(t, c).slope_lo, C(t, c).slope_hi, 2) for t, c in cols],
            ["O:E ratio (95% CI)"] + [fci(C(t, c).oe, C(t, c).oe_lo, C(t, c).oe_hi, 2) for t, c in cols],
            ["Brier score (95% CI)"] + [fci(C(t, c).brier, C(t, c).brier_lo, C(t, c).brier_hi, 4) for t, c in cols],
            ["Scaled Brier score (95% CI)"]
            + [fci(C(t, c).scaled_brier, C(t, c).scaled_brier_lo, C(t, c).scaled_brier_hi, 3) for t, c in cols]]
    p0_mark = fn("p0",
                 "Isotonic calibration assigns a predicted risk of exactly 0 to some persons ({{T:isofloor}} in "
                 "Multimedia Appendix 1).")
    rows.append([f"Predicted risk exactly 0, n (%); events{p0_mark}"]
                + [f"{fnpct(C(t, c).n_p0, C(t, c).n)}; {int(C(t, c).events_p0)}" for t, c in cols])
    # recalibration (external only)
    rr = {t: pick(rc, target=t) for t in targets}
    for t in targets:
        assert int(rr[t].n) == N["external"] and int(rr[t].events) == cnum(ci, f"KNHANES events {t}"), t
    nsplit = int(rr[targets[0]].n_splits)
    rec_mark = fn("recal",
                  f"Intercept update fitted on one random half of KNHANES and evaluated on the other ({nsplit} splits "
                  f"stratified on the outcome; mean and 2.5th-97.5th percentiles over splits).")

    def ext_only(f):
        return [NA if c == "internal" else f(rr[t]) for t, c in cols]

    rows += [[f"**External, after an intercept-only update**{rec_mark}"] + blank(len(cols)),
             ["Intercept update, log-odds (95% range)"]
             + ext_only(lambda r: fci(r.intercept_fit, r.intercept_fit_lo, r.intercept_fit_hi, 2)),
             ["O:E ratio (95% range)"]
             + ext_only(lambda r: fci(r.oe_after_intercept, r.oe_after_intercept_lo, r.oe_after_intercept_hi, 2)),
             ["Scaled Brier score (95% range)"]
             + ext_only(lambda r: fci(r.sbrier_intercept, r.sbrier_intercept_lo, r.sbrier_intercept_hi, 3)),
             ["Calibration slope if refitted (95% range)"]
             + ext_only(lambda r: fci(r.slope_fit, r.slope_fit_lo, r.slope_fit_hi, 2))]
    fn.note(f"95% CIs are percentile bootstrap intervals ({nb:,} resamples). Scaled Brier score: 1 minus the Brier score "
            "divided by that of a constant prediction at the observed prevalence; values near 0 mean no better than that "
            "constant prediction, and negative values mean worse.")
    cap = ("Discrimination and calibration of the models in the internal test set and in KNHANES adults, with "
           "external calibration after an intercept-only update" + (" (≥10.0 mg/dL, exploratory)." if appendix else "."))
    return block(number, cap, ["discrimination.csv", "calibration.csv", "recalibration.csv", "main_performance.csv",
                               "fu_int_persons.csv", "cohort_info_v2.csv"], head, rows, fn,
                 "AUPRC: area under the precision-recall curve; AUROC: area under the receiver operating "
                 "characteristic curve; KNHANES: Korea National Health and Nutrition Examination Survey; N/A: not "
                 "applicable; O:E: observed-to-expected.")


# ============================================================================ Table 3 (appendix counterpart)
RULES = (("youden", "Youden index"), ("sens80", "Sensitivity 80%"), ("sens90", "Sensitivity 90%"))
TOPK = (("top_5%", "Top 5% referred"), ("top_10%", "Top 10% referred"), ("top_20%", "Top 20% referred"))


def _fmt_ranges(s, step):
    d = len(f"{step:g}".split(".")[1]) if "." in f"{step:g}" else 0
    parts = []
    for seg in str(s).split(";"):
        nums = [fnum(float(x), d) for x in seg.strip().split("-")]
        parts.append("-".join(nums))
    return " and ".join(parts)


def table3(targets, number, appendix=False):
    op, sh, dcr = R("operating_points.csv"), R("splithalf_operating.csv"), R("decision_curve_ranges.csv")
    fn = Notes()
    head = ["Rule, cohort (threshold)", "Sensitivity (95% CI)", "Specificity", "PPV (95% CI)", "NPV", "Flagged",
            "Referrals per case (95% CI)", "Cases per 1,000"]
    nc = len(head)
    ci_w = {"spec": [], "npv": []}  # widths of the CIs not shown (percentage points)

    def op_row(label, r):
        for k in ci_w:
            ci_w[k].append(100 * (float(r[f"{k}_hi"]) - float(r[f"{k}_lo"])))
        return [f"{label} ({fthr(r.threshold)})", fpct_ci(r.sens, r.sens_lo, r.sens_hi), fpct(r.spec),
                fpct_ci(r.ppv, r.ppv_lo, r.ppv_hi), fpct(r.npv), fpct(r.pos_rate),
                fci(r.referrals_per_case, r.referrals_per_case_lo, r.referrals_per_case_hi, 1),
                fnum(r.detected_per_1000, 1)]

    rows = []
    n_rep = int(sh.n_repeats.iloc[0])
    assert (sh.n_repeats == n_rep).all()
    for t in targets:
        rows.append([f"**{TLAB[t]} ({ROLE[t]} model)**"] + blank(nc - 1))
        for pol, lab in RULES:
            ri, re_ = pick(op, target=t, cohort="internal", policy=pol), pick(op, target=t, cohort="external", policy=pol)
            # thresholds of the internal test persons (labels kept by calc_utility.py), applied unchanged externally
            assert ri.threshold == re_.threshold and "apparent" in str(ri.threshold_source)
            assert str(re_.threshold_source).startswith("internal") and "apparent" not in str(re_.threshold_source)
            s = pick(sh, target=t, rule=pol)
            m_rule = fn("rule",
                        "Thresholds chosen on the internal test set (so internal values are apparent performance) and "
                        "applied unchanged to KNHANES. Youden index: threshold maximising sensitivity + specificity - 1; "
                        "sensitivity 80% or 90%: highest threshold reaching that internal sensitivity.")
            rows.append([f"**{lab}**{m_rule}"] + blank(nc - 1))
            rows.append(op_row("Internal", ri))
            rows.append(op_row("External", re_))
            m_loc = fn("local",
                       f"External, local: threshold re-chosen in KNHANES by the same rule, selected on a random half "
                       f"and applied to "
                       f"the other half ({n_rep} repeats stratified on the outcome); mean threshold and mean "
                       f"performance with 2.5th-97.5th percentiles over repeats. Referrals per case (1/PPV) and cases "
                       f"detected per 1,000 (sensitivity x prevalence x 1,000) are computed from the mean values.")
            prev = float(re_.prevalence)
            rows.append([f"External, local ({fthr(s.threshold_mean)}){m_loc}",
                         fpct_ci(s.sens, s.sens_lo, s.sens_hi), fpct(s.spec), fpct_ci(s.ppv, s.ppv_lo, s.ppv_hi),
                         fpct(s.npv), fpct(s.pos_rate), fnum(1 / float(s.ppv), 1),
                         fnum(float(s.sens) * prev * 1000, 1)])
        for pol, lab in TOPK:
            ri, re_ = pick(op, target=t, cohort="internal", policy=pol), pick(op, target=t, cohort="external", policy=pol)
            m_top = fn("top", "Participants with the highest k% of predicted risk within each cohort are referred; "
                              "the threshold is therefore cohort-specific.")
            rows.append([f"**{lab}**{m_top}"] + blank(nc - 1))
            rows.append(op_row("Internal", ri))
            rows.append(op_row("External", re_))
    fn.note(f"95% CIs (percentile bootstrap) are shown for sensitivity, PPV and referrals per case; the 95% CIs of "
            f"specificity and NPV spanned at most {fnum(max(ci_w['spec']), 1)} and {fnum(max(ci_w['npv']), 1)} "
            f"percentage points, respectively, in the internal and external rows.")
    # transfer summary + decision curves
    s80 = {t: (pick(op, target=t, cohort="internal", policy="sens80"),
               pick(op, target=t, cohort="external", policy="sens80")) for t in targets}
    fn.note(
       "External predictions were not recalibrated, so the transferred thresholds inherit the external "
       f"over-prediction (see {ref('Table 2', appendix)}). With the sensitivity-80% threshold, "
       + "; ".join(f"{TLAB[t]}: external sensitivity {fpct(s80[t][1].sens)}%, {fpct(s80[t][0].pos_rate)}% of the "
                   f"internal and {fpct(s80[t][1].pos_rate)}% of the external cohort flagged" for t in targets) + ".")
    dc_txt = []
    for t in targets:
        a = pick(dcr, target=t, cohort="external", criterion="point estimate > max(treat-all, treat-none)")
        b = pick(dcr, target=t, cohort="external", criterion="95% CI lower bound > 0")
        dc_txt.append(f"{_fmt_ranges(a.ranges, a.grid_step)} for {TLAB[t]} (lower 95% CI bound of the gain "
                      f"above 0 at {_fmt_ranges(b.ranges, b.grid_step)}; threshold grid "
                      f"{_fmt_ranges(f'{a.grid_min}-{a.grid_max}', a.grid_step)})")
    fn.note("Decision curves in KNHANES (unrecalibrated predictions): net benefit exceeded both treat-all and "
            "treat-none at threshold probabilities of " + " and ".join(dc_txt) + ".")
    cap = (f"Operating points of the {'model' if len(targets) == 1 else 'models'}: internally derived thresholds "
           "applied to KNHANES adults, thresholds re-chosen locally, and fixed-fraction (top-k) referral"
           + (" (≥10.0 mg/dL, exploratory)" if appendix else "")
           + ". Sensitivity, specificity, PPV, NPV and the proportion flagged are percentages; thresholds (in "
             "parentheses) are predicted probabilities; cases are those detected per 1,000 screened.")
    return block(number, cap, ["operating_points.csv", "splithalf_operating.csv", "decision_curve_ranges.csv"],
                 head, rows, fn,
                 "KNHANES: Korea National Health and Nutrition Examination Survey; NPV: negative predictive value; "
                 "PPV: positive predictive value.")


# ============================================================================ Table 4 helpers
def ens_labels(evm):
    """(ensemble, ensemble-member LightGBM arm) model labels of ensemble_vs_simple.csv (calc_feature_tiers.py): the
    two isotonic-calibrated arms; every other arm is uncalibrated."""
    ens = one_label(evm.model, "ensemble_vs_simple.csv ensemble arm", ("ensemble",))
    gbi = one_label(evm.model, "ensemble_vs_simple.csv ensemble-member LightGBM arm", ("lightgbm", "isotonic"))
    iso = evm.groupby("model")["isotonic_calibrated"].agg(lambda s: bool(s.astype(str).str.lower().eq("true").all()))
    assert set(iso[iso].index) == {ens, gbi}, iso.to_dict()
    assert {LR, SPL, LGB, GRID} <= set(iso.index), sorted(iso.index)
    return ens, gbi


def refit_sizes(tiers, ci, targets):
    """(training persons, internal test persons) of the refits of feature_tiers.csv, checked against
    cohort_info_v2.csv: the training and internal test persons of the models."""
    r = tiers[(tiers.setting == "internal") & tiers.target.isin(targets)]
    ntr, nte = {int(x) for x in r.n_train}, {int(x) for x in r.n}
    assert ntr == {cnum(ci, CI_TRAIN)} and nte == {cnum(ci, CI_TEST)}, (ntr, nte)
    return ntr.pop(), nte.pop()


def coding_change(cs, targets):
    """Largest |external AUROC under the alternative KNHANES codings M1 or M2 minus that under X1| over the arms with
    dipstick inputs (feature_tiers_coding_sensitivity.csv)."""
    s = cs[cs.target.isin(targets)]
    assert len(s) > 0
    return float(max((s[f"auroc_{m}"] - s["auroc_X1"]).abs().max() for m in ("M1", "M2")))


def check_tiers(ft, targets):
    """feature_tiers.csv: tiers 2F-12F, one external setting (X1); returns the tier rows and the tier contrasts."""
    assert set(ft.setting.dropna()) == {"internal", "external"}, sorted(set(ft.setting.dropna()))
    tiers = ft[(ft.row_type == "tier") & ft.target.isin(targets)]
    assert set(tiers.tier) == set(TIERS), sorted(set(tiers.tier))
    return tiers, ft[(ft.row_type == "contrast_tier") & ft.target.isin(targets)]


def n_features(tiers, tier):
    v = {int(x) for x in tiers[tiers.tier == tier].features}
    assert len(v) == 1 and v.pop() == int(tier[:-1]), tier
    return int(tier[:-1])


# ============================================================================ Table 4 (appendix counterpart)
def table4(targets, number, appendix=False):
    ft, tu, ev = R("feature_tiers.csv"), R("tuned_per_set.csv"), R("ensemble_vs_simple.csv")
    cs, dis = R("feature_tiers_coding_sensitivity.csv"), R("discrimination.csv")
    ci = info()
    fn = Notes()
    cols = [(t, c) for t in targets for c in ("internal", "external")]

    # ---------------------------------------------------------------- consistency checks
    tiers, ctr = check_tiers(ft, targets)
    ntr, nte = refit_sizes(tiers, ci, targets)
    n_in, n_dip = n_features(tiers, "12F"), n_features(tiers, "12F") - n_features(tiers, "3F")
    evm, evc = ev[ev.row_type == "model"], ev[ev.row_type == "contrast"]
    ENS, GBI = ens_labels(evm)
    assert {int(x) for x in evm[evm.target.isin(targets)].n_train} == {ntr}
    for t, c in cols:   # the ensemble arm is the models of Table 2
        assert abs(float(pick(evm, target=t, setting=c, model=ENS).auroc) - float(pick(dis, target=t, cohort=c).auroc)) \
            < 1e-9, (t, c)
    n_grid = int(len(tu[(tu.row_type == "cv_grid") & (tu.target == targets[0]) & (tu.tier == "12F")]))
    assert n_grid > 0
    chg = coding_change(cs, targets)
    ext_mark = fn("ext",
                  f"KNHANES adults, dipstick results coded by the label-matched mapping X1 for every model. With the "
                  f"alternative codings M1 and M2, the external AUROCs of the models with dipstick inputs changed by "
                  f"at most {_nz(chg, 4)} " + "({{T:coding}} in Multimedia Appendix 1).")
    head = ["Model and input set"] + [f"{TLAB[t]}, {c}" + (ext_mark if c == "external" else "") for t, c in cols]

    def tier_cell(t, c, model, tier):
        r = pick(tiers, target=t, model=model, setting=c, tier=tier)
        return fci(r.auroc, r.auroc_lo, r.auroc_hi, 3)

    def tier_delta(t, c, model, a="12F", b="3F"):
        r = pick(ctr, target=t, setting=c, model=model, tier=f"{a}-{b}")
        return fdiff(r.delta_auroc, r.delta_lo, r.delta_hi)

    evtxt = ", ".join(f"{fint(pick(tiers, target=t, model=LGB, setting='internal', tier='12F').positives)} events "
                      f"at {TLAB[t]}" for t in targets)
    fn("pa",
       f"Comparison models refitted for this analysis: each input set fitted with one fixed LightGBM configuration and "
       f"with logistic regression on the training persons of the development cohort (n={ntr:,}) and evaluated on the "
       f"internal test persons (n={nte:,}, {evtxt}) and KNHANES adults.")
    nb = int(pick(R("calibration.csv"), target=targets[0], cohort="internal").n_boot_used)
    fn("boot", f"95% CIs: percentile bootstrap, {nb:,} resamples; differences use paired resamples.")
    rows = [[f"**A. What the dipstick adds: equal-budget refits by input set**{fn.multi('pa', 'boot')}"]
            + blank(len(cols))]
    d_mark = fn("dinc", f"12F minus 3F: the AUROC gained by adding the {n_dip} dipstick fields to age, sex and BMI.")
    for model, title, tiers_ in ((LGB, "LightGBM, fixed configuration", ("3F", "9F", "11F", "12F")),
                                 (LR, "Logistic regression, main effects", ("3F", "9F", "11F", "12F")),
                                 (SPL, "Logistic regression, age and BMI splines plus age-by-sex term",
                                  ("3F", "11F", "12F"))):
        rows.append([f"**{title}**"] + blank(len(cols)))
        for tier in tiers_:
            rows.append([TIER_LAB[tier]] + [tier_cell(t, c, model, tier) for t, c in cols])
        rows.append([f"Dipstick increment, 12F minus 3F{d_mark}"] + [tier_delta(t, c, model) for t, c in cols])
    # tuned per set
    tuned = tu[tu.row_type == "tuned"]
    tmark = fn("tuned",
               f"LightGBM tuned separately for each input set ({n_grid} configurations compared by 3-fold "
               f"cross-validated AUROC on the training persons, then refitted).")
    rows.append([f"**LightGBM, tuned for each input set**{tmark}"] + blank(len(cols)))
    for tier in ("3F", "12F"):
        cells = []
        for t, c in cols:
            r = pick(tuned, target=t, tier=tier, setting=c)
            cells.append(fci(r.auroc, r.auroc_lo, r.auroc_hi, 3))
        rows.append([TIER_LAB[tier]] + cells)
    tc = tu[tu.row_type == "contrast"]
    rows.append(["Dipstick increment, 12F minus 3F"]
                + [fdiff(*pick(tc, target=t, setting=c, tier="12F-3F")[["delta_auroc", "delta_lo", "delta_hi"]])
                   for t, c in cols])

    # ---------------------------------------------------------------- panel B
    panel_b = fn("pb",
                 f"Comparison models with the same {n_in} inputs, fitted on the same {ntr:,} training persons as the "
                 f"ensemble; internal evaluation on the internal test set (see {ref('Table 2', appendix)}).")
    rows.append([f"**B. Ensemble (LightGBM + XGBoost + CatBoost, isotonic calibration) vs simpler models fitted on "
                 f"the same training persons**{panel_b}"] + blank(len(cols)))
    models = ((ENS, "Ensemble"),
              (LR, "Logistic regression, main effects"),
              (SPL, "Logistic regression, splines plus age-by-sex term"),
              (LGB, "LightGBM, fixed configuration"),
              (GRID, "LightGBM, grid-tuned"),
              (GBI, "LightGBM, ensemble member configuration with isotonic calibration"))
    diff_lab = {LR: "main-effects logistic regression", SPL: "spline logistic regression",
                LGB: "LightGBM, fixed configuration", GRID: "LightGBM, grid-tuned",
                GBI: "LightGBM, ensemble member configuration"}
    gmark = fn("member", "The LightGBM member of the ensemble with its own hyperparameters and the same 5-fold isotonic "
                         "calibration (its fold learners are identical to the ensemble's LightGBM members); the "
                         "difference from the ensemble is therefore the contribution of adding XGBoost and CatBoost.")
    rows.append(["**AUROC (95% CI)**"] + blank(len(cols)))
    for key, lab in models:
        cells = []
        for t, c in cols:
            r = pick(evm, target=t, setting=c, model=key)
            cells.append(fci(r.auroc, r.auroc_lo, r.auroc_hi, 3))
        rows.append([lab + (gmark if key == GBI else "")] + cells)
    rows.append(["**Difference, ensemble minus comparator (95% CI)**"] + blank(len(cols)))
    dis_mark = None
    for key, _ in models[1:]:
        cells = []
        for t, c in cols:
            r = pick(evc, target=t, setting=c, comparison=f"{ENS} vs {key}")
            cell = fdiff(r.delta_auroc, r.delta_lo, r.delta_hi)
            if str(r.delong_ci_disagree).lower() == "true":
                dis_mark = fn("delong", "The paired bootstrap CI and the DeLong test disagree at the 5% level; the "
                                        "bootstrap CI is taken as primary. DeLong test: ")
                fn.text["delong"] += f"{TLAB[t]} {c}, minus {diff_lab[key]}, {fp(r.delong_p)}; "
                cell += dis_mark
            cells.append(cell)
        rows.append([f"Minus {diff_lab[key]}" + (gmark if key == GBI else "")] + cells)
    if dis_mark:
        fn.text["delong"] = fn.text["delong"].rstrip("; ") + "."
    cap = ("Added value of the urine dipstick over age, sex and BMI, and comparison of the ensemble with simpler "
           "models (AUROC)" + (" (≥10.0 mg/dL, exploratory)." if appendix else "."))
    return block(number, cap, ["feature_tiers.csv", "tuned_per_set.csv", "ensemble_vs_simple.csv",
                               "feature_tiers_coding_sensitivity.csv", "discrimination.csv", "calibration.csv",
                               "cohort_info_v2.csv"],
                 head, rows, fn,
                 f"{sets_abbrev(('3F', '9F', '11F', '12F'))}; AUROC: area under the receiver operating "
                 "characteristic curve; BMI: body mass index; KNHANES: Korea National Health and Nutrition "
                 "Examination Survey.")


# ============================================================================ main-text Table 3 (by sex)
def table3m(targets, number):
    so, op = R("fu_sex_operating.csv"), R("operating_points.csv")
    ext = so[so.cohort == "external"]
    ci = info()
    fn = Notes()
    # strategy labels of calc_followup.py (resolved by pattern)
    S_ENS = one_label(ext.strategy, "fu_sex_operating.csv ensemble strategy", ("ensemble",), ("sex-specific",))
    S_SEX = one_label(ext.strategy, "fu_sex_operating.csv sex-specific strategy", ("ensemble", "sex-specific"))
    S_MEN = one_label(ext.strategy, "fu_sex_operating.csv 'test all men' strategy", ("all men",))
    S_3F = one_label(ext.strategy, "fu_sex_operating.csv age-sex-BMI spline strategy", ("3f", "spline"))
    for t in targets:   # the ensemble thresholds are those of operating_points.csv (chosen on the internal test set)
        for rule, _ in RULES:
            r = pick(ext, target=t, strategy=S_ENS, rule=rule, group="all")
            assert abs(float(r.threshold) - float(pick(op, target=t, cohort="internal", policy=rule).threshold)) < 1e-12, \
                (t, rule)
            assert "internal" in str(r.threshold_source).lower(), (t, rule, r.threshold_source)
        for rule in ("sens80", "top10%"):
            r = pick(ext, target=t, strategy=S_3F, rule=rule, group="all")
            assert "external" not in str(r.threshold_source).lower() and "knhanes" not in str(r.threshold_source).lower()
    head = ["Strategy (threshold)", "Flagged, % (men; women)", "Men among flagged, %", "Sensitivity, % (95% CI)",
            "Sensitivity, men; women, %", "PPV, % (95% CI)", "PPV, men; women, %"]
    nc = len(head)

    def cell_pct(v):
        if isnan(v):
            return "N/A"
        v = float(v)
        if v == 0 or v == 1:
            return f"{int(100 * v)}"
        return fnum(100 * v, pdec(100 * v))

    def row(label, strategy, rule, t, ci_from=None):
        """ci_from: the operating_points.csv row whose 95% CIs replace those of fu_sex_operating.csv (top-k referral:
        the CIs of Table S12 panel A, which re-select the top fraction in each resample; the point estimates must agree)."""
        g = {k: pick(ext, target=t, strategy=strategy, rule=rule, group=k) for k in ("all", "men", "women")}
        a = g["all"]
        c = a if ci_from is None else ci_from
        if ci_from is not None:
            assert abs(float(a.sens) - float(c.sens)) < 1e-9 and abs(float(a.ppv) - float(c.ppv)) < 1e-9, (t, rule)
        return [label, f"{cell_pct(a.flagged)} ({cell_pct(g['men'].flagged)}; {cell_pct(g['women'].flagged)})",
                cell_pct(a.share_flagged_men), fpct_ci(a.sens, c.sens_lo, c.sens_hi, 1),
                f"{cell_pct(g['men'].sens)}; {cell_pct(g['women'].sens)}", fpct_ci(a.ppv, c.ppv_lo, c.ppv_hi, 1),
                f"{cell_pct(g['men'].ppv)}; {cell_pct(g['women'].ppv)}"]

    rows = []
    for t in targets:
        rows.append([f"**{TLAB[t]} ({ROLE[t]} model)**"] + blank(nc - 1))
        m_int = fn("int", "Threshold chosen on the internal test set (apparent performance there, "
                          f"{supp_panels('sexop', 'oppoints')} in Multimedia Appendix 1) and applied unchanged to KNHANES.")
        rows.append([f"**Ensemble model, internally derived thresholds**{m_int}"] + blank(nc - 1))
        for rule, lab in RULES:
            thr = pick(ext, target=t, strategy=S_ENS, rule=rule, group="all").threshold
            rows.append(row(f"{lab} ({fthr(thr)})", S_ENS, rule, t))
        m_top = fn("top", "The 10% of adults with the highest predicted risk; participants tied at the cut-off are "
                          "flagged fractionally. 95% CIs as in " + supp_panels("oppoints") + " in Multimedia Appendix 1, "
                          "with the top 10% re-selected in each resample (other rows: thresholds fixed).")
        thr = pick(ext, target=t, strategy=S_ENS, rule="top10%", group="all").threshold
        op_top = pick(op, target=t, cohort="external", policy="top_10%")
        assert abs(float(op_top.threshold) - float(thr)) < 1e-9, (t, op_top.threshold, thr)
        rows.append(row(f"Top 10% referred ({fthr(thr)}){m_top}", S_ENS, "top10%", t, ci_from=op_top))
        sx = pick(ext, target=t, strategy=S_SEX, rule="sens80_sexspecific", group="all")
        thm, thf = re.findall(r"[\d.]+", str(sx.threshold))
        m_sx = fn("sex", "Separate thresholds reaching 80% sensitivity within men and within women in the internal "
                         "test set.")
        rows.append(row(f"Sex-specific, sensitivity 80% (men {fthr(thm)}; women {fthr(thf)}){m_sx}",
                        S_SEX, "sens80_sexspecific", t))
        rows.append(["**Reference strategies**"] + blank(nc - 1))
        rows.append(row("Test all men", S_MEN, "all men", t))
        m_3f = fn("3f", "Logistic regression with splines for age and BMI and an age-by-sex term, fitted on the "
                        "training persons (Table 4); thresholds chosen on the internal test set.")
        thr = pick(ext, target=t, strategy=S_3F, rule="sens80", group="all").threshold
        rows.append(row(f"Age, sex, BMI model, sensitivity 80% ({fthr(thr)}){m_3f}", S_3F, "sens80", t))
        thr = pick(ext, target=t, strategy=S_3F, rule="top10%", group="all").threshold
        rows.append(row(f"Age, sex, BMI model, top 10% ({fthr(thr)})", S_3F, "top10%", t))
    n_ext = cnum(ci, CI_EXT)
    men = {t: pick(ext, target=t, strategy=S_MEN, rule="all men", group="men") for t in targets}
    women = {t: pick(ext, target=t, strategy=S_MEN, rule="all men", group="women") for t in targets}
    for t in targets:
        assert int(men[t].n) + int(women[t].n) == n_ext, t
        assert int(men[t].events) + int(women[t].events) == cnum(ci, f"KNHANES events {t}"), t
        assert int(women[t].events) == cnum(ci, f"KNHANES female events {t}"), t
    t0 = targets[0]
    fn.note(f"KNHANES adults (n={n_ext:,}): {fint(men[t0].n)} men and {fint(women[t0].n)} women, "
            + "; ".join(f"{fint(men[t].events)} and {fint(women[t].events)} events at {TLAB[t]}" for t in targets)
            + f". Predictions not recalibrated (after an intercept update: {supp_panels('recalop')} in Multimedia Appendix "
              "1); thresholds fixed.")
    cap = ("External operating points of the models and of reference strategies in KNHANES adults, overall and by "
           "sex. Thresholds (in parentheses) are predicted probabilities.")
    return block(number, cap, ["fu_sex_operating.csv", "operating_points.csv", "cohort_info_v2.csv"], head, rows, fn,
                 "KNHANES: Korea National Health and Nutrition Examination Survey; N/A: not applicable (no one "
                 "flagged); PPV: positive predictive value.")


# ============================================================================ main-text Table 4
def table4m(targets, number):
    ft, tu, ev = R("feature_tiers.csv"), R("tuned_per_set.csv"), R("ensemble_vs_simple.csv")
    cs, dis = R("feature_tiers_coding_sensitivity.csv"), R("discrimination.csv")
    ci = info()
    fn = Notes()
    cols = [(t, c) for t in targets for c in ("internal", "external")]

    tiers, ctr = check_tiers(ft, targets)
    ntr, nte = refit_sizes(tiers, ci, targets)
    n_in = n_features(tiers, "12F")
    chg = coding_change(cs, targets)
    ext_mark = fn("ext", "KNHANES adults, dipstick results coded by the label-matched mapping X1 for every model; with "
                         f"the alternative codings M1 and M2, the external AUROCs of the models with dipstick inputs "
                         f"changed by at most {_nz(chg, 4)} " + "({{T:coding}} in Multimedia Appendix 1).")
    head = ["Model and input set"] + [f"{TLAB[t]}, {c}" + (ext_mark if c == "external" else "") for t, c in cols]
    pa = fn("pa", f"Comparison models refitted for this analysis on the training persons of the development cohort "
                  f"(n={ntr:,}) and evaluated on the internal test persons (n={nte:,}) and KNHANES adults; differences "
                  f"use paired bootstrap resamples.")
    rows = [[f"**A. AUROC by input set (95% CI)**{pa}"] + blank(len(cols))]

    def tcell(t, c, model, tier):
        r = pick(tiers, target=t, model=model, setting=c, tier=tier)
        return fci(r.auroc, r.auroc_lo, r.auroc_hi, 3)

    for model, title, tl in ((LGB, "LightGBM, fixed configuration", ("2F", "3F", "9F", "11F", "12F")),
                             (SPL, "Logistic regression, age and BMI splines, age-by-sex term",
                              ("2F", "3F", "11F", "12F"))):
        rows.append([f"**{title}**"] + blank(len(cols)))
        for tier in tl:
            rows.append([TL[tier]] + [tcell(t, c, model, tier) for t, c in cols])
    rows.append(["**B. Increments (AUROC difference, 95% CI)**"] + blank(len(cols)))

    def ft_delta(t, c, model, a, b):
        r = pick(ctr, target=t, setting=c, model=model, tier=f"{a}-{b}")
        return fdiff(r.delta_auroc, r.delta_lo, r.delta_hi)

    n_grid = int(len(tu[(tu.row_type == "cv_grid") & (tu.target == targets[0]) & (tu.tier == "12F")]))
    assert n_grid > 0
    tmark = fn("tuned", f"LightGBM tuned separately for each input set ({n_grid} configurations compared by 3-fold "
                        "cross-validation on the training persons).")
    tc = tu[tu.row_type == "contrast"]
    rows.append(["**Dipstick added to age, sex, BMI (12F minus 3F)**"] + blank(len(cols)))
    rows.append(["LightGBM, fixed"] + [ft_delta(t, c, LGB, "12F", "3F") for t, c in cols])
    rows.append([f"LightGBM, tuned{tmark}"]
                + [fdiff(*pick(tc, target=t, setting=c, tier="12F-3F")[["delta_auroc", "delta_lo", "delta_hi"]])
                   for t, c in cols])
    rows.append(["Logistic regression, main effects"] + [ft_delta(t, c, LR, "12F", "3F") for t, c in cols])
    rows.append(["Logistic regression, splines"] + [ft_delta(t, c, SPL, "12F", "3F") for t, c in cols])
    rows.append(["**Dipstick added to age and sex (11F minus 2F)**"] + blank(len(cols)))
    for model, lab in ((LGB, "LightGBM, fixed"), (LR, "Logistic regression, main effects"),
                       (SPL, "Logistic regression, splines")):
        rows.append([lab] + [ft_delta(t, c, model, "11F", "2F") for t, c in cols])
    rows.append(["**BMI added to age and sex (3F minus 2F)**"] + blank(len(cols)))
    for model, lab in ((LGB, "LightGBM, fixed"), (SPL, "Logistic regression, splines")):
        rows.append([lab] + [ft_delta(t, c, model, "3F", "2F") for t, c in cols])
    # ---------------------------------------------------------------- panel C
    evm, evc = ev[ev.row_type == "model"], ev[ev.row_type == "contrast"]
    ENS, GBI = ens_labels(evm)
    assert {int(x) for x in evm[evm.target.isin(targets)].n_train} == {ntr}
    for t, c in cols:   # the ensemble arm is the models of Table 2
        assert abs(float(pick(evm, target=t, setting=c, model=ENS).auroc) - float(pick(dis, target=t, cohort=c).auroc)) \
            < 1e-9, (t, c)
    pc = fn("pc", f"Comparison models with the same {n_in} inputs, fitted on the same {ntr:,} training persons as the "
                  "ensemble and evaluated on the same persons; differences are the ensemble minus the comparison model.")
    rows.append([f"**C. Ensemble versus simpler {n_in}-input models**{pc}"] + blank(len(cols)))
    rows.append(["Ensemble, AUROC (95% CI)"]
                + [fci(*pick(evm, target=t, setting=c, model=ENS)[["auroc", "auroc_lo", "auroc_hi"]], 3)
                   for t, c in cols])
    gmark = fn("member", "The LightGBM member of the ensemble with the same hyperparameters and the same 5-fold "
                         "isotonic calibration (its fold learners are identical to the ensemble's LightGBM members), so "
                         "the difference is the contribution of adding XGBoost and CatBoost.")
    dis_mark = None
    for key, lab in ((LR, "Minus main-effects logistic regression"), (SPL, "Minus spline logistic regression"),
                     (GRID, "Minus LightGBM, grid-tuned"), (GBI, f"Minus LightGBM, ensemble member configuration{gmark}")):
        cells = []
        for t, c in cols:
            r = pick(evc, target=t, setting=c, comparison=f"{ENS} vs {key}")
            cell = fdiff(r.delta_auroc, r.delta_lo, r.delta_hi)
            if str(r.delong_ci_disagree).lower() == "true":
                dis_mark = fn("delong", "Paired bootstrap CI and DeLong test disagree at the 5% level; the bootstrap "
                                        "CI is primary (DeLong ")
                fn.text["delong"] += f"{fp(r.delong_p)}, {TLAB[t]} {c}; "
                cell += dis_mark
            cells.append(cell)
        rows.append([lab] + cells)
    if dis_mark:
        fn.text["delong"] = fn.text["delong"].rstrip("; ") + ")."
    cap = ("Added value of the urine dipstick over age and sex and over age, sex and BMI, and comparison of the "
           "ensemble with simpler models (AUROC).")
    return block(number, cap, ["feature_tiers.csv", "tuned_per_set.csv", "ensemble_vs_simple.csv",
                               "feature_tiers_coding_sensitivity.csv", "discrimination.csv", "cohort_info_v2.csv"],
                 head, rows, fn,
                 f"{sets_abbrev(TIERS)}; AUROC: area under the receiver operating characteristic curve; BMI: body "
                 "mass index; KNHANES: Korea National Health and Nutrition Examination Survey.")


# ============================================================================ main-text Table 5 (sensitivity analyses)
def coding_labels(enc, ms):
    """Row labels of the alternative KNHANES codings (mapping_sensitivity.csv), each checked against the code tables
    of encoding.csv (raw_codes: model-scale value per raw KNHANES code under each coding) and described relative to
    the primary coding X1."""
    rc = enc[enc.section == "raw_codes"]

    def codes(mapping, feature):
        s = rc[rc.feature == feature].sort_values("value_num")
        return s[f"map_{mapping}"].astype(float).to_numpy()

    def same(a, b, feature):
        return np.array_equal(codes(a, feature), codes(b, feature))

    other = [f for f in rc.feature.unique() if f not in ("ua_bilirubin", "ua_urobilinogen")]
    for m in ("M1", "M2", "M3", "X2"):          # the fields other than bilirubin and urobilinogen are coded alike
        assert all(same(m, "X1", f) for f in other), m
    b1, bx = codes("M1", "ua_bilirubin"), codes("X1", "ua_bilirubin")
    u1, ux = codes("M1", "ua_urobilinogen"), codes("X1", "ua_urobilinogen")
    assert (b1 >= bx).all() and (b1 > bx).any() and (u1 <= ux).all() and (u1 < ux).any()          # M1
    u2 = codes("M2", "ua_urobilinogen")
    assert same("M2", "X1", "ua_bilirubin") and (u2[1:] == ux[1:]).all() and u2[0] != ux[0]         # M2
    u3 = codes("M3", "ua_urobilinogen")
    assert same("M3", "M1", "ua_bilirubin") and (u3[1:] == u1[1:]).all() and u3[0] == ux[0]         # M3
    assert same("X2", "X1", "ua_bilirubin") and same("X2", "X1", "ua_urobilinogen")                # X2: SG only
    x2 = ms[ms.mapping == "X2"]
    assert len(x2) and not x2.in_spec.astype(str).str.lower().eq("true").any()
    assert x2.description.astype(str).str.contains("SG").all()
    return {"M1": "Bilirubin coded higher, urobilinogen lower (M1)",
            "M2": f"Urobilinogen normal coded {u2[0]:g} (M2)",
            "M3": f"As M1, urobilinogen normal coded {u3[0]:g} (M3)",
            "X2": "Specific gravity rounded to the development grid (X2)"}


def table5(targets, number):
    fn = Notes()
    ci = info()
    dis, cal = R("discrimination.csv"), R("calibration.csv")
    bw, mi, vb = R("bmi_window.csv"), R("mice_bmi.csv"), R("verification_bias.csv")
    fp_, tp_ = R("fu_internal_period.csv"), R("temporal.csv")
    fy, sw, ms = R("fu_knhanes_year.csv"), R("survey_weighted.csv"), R("mapping_sensitivity.csv")
    sgr, kt, enc = R("fu_sg_restrict.csv"), R("fu_ketone.csv"), R("encoding.csv")
    co, pv = R("continuous.csv"), R("platt_vs_isotonic.csv")
    head = ["Analysis", f"n (events at {TLAB[targets[0]]})"] + [f"{TLAB[t]}, {m}" for t in targets for m in ("AUROC", "O:E")]
    nc = len(head)
    rows = []

    def au(x):
        return fnum(x, 3)

    def oe(x):
        return "N/A" if x is None or isnan(x) else fnum(x, 2)

    def add(label, n, ev, vals):
        """vals: {t: (auroc, oe)}"""
        rows.append([label, "" if n is None else (f"{fint(n)} ({fint(ev)})" if ev is not None else fint(n))]
                    + [c for t in targets for c in (au(vals[t][0]) if not isinstance(vals[t][0], str) else vals[t][0],
                                                     oe(vals[t][1]) if not isinstance(vals[t][1], str) else vals[t][1])])

    def sec(label, mark=""):
        rows.append([f"**{label}**{mark}"] + blank(nc - 1))

    n_test = cnum(ci, CI_TEST)
    sec("Reference")
    d0 = pick(dis, target=targets[0], cohort="internal")
    add("Internal test set", d0.n, d0.positives,
        {t: (pick(dis, target=t, cohort="internal").auroc, pick(cal, target=t, cohort="internal").oe) for t in targets})
    d1 = pick(dis, target=targets[0], cohort="external")
    add("KNHANES adults", d1.n, d1.positives,
        {t: (pick(dis, target=t, cohort="external").auroc, pick(cal, target=t, cohort="external").oe) for t in targets})
    assert int(d0.n) == n_test and int(d1.n) == cnum(ci, CI_EXT)

    sec("Setting and selection (internal test set unless stated)")
    perf = bw[bw.section == "performance"]
    src_mark = fn("src", "Internal test persons by the source of BMI on the urine-test day: a health checkup on that day "
                         "or, otherwise, height and weight in the nursing record ({{T:bmiperf}} in Multimedia Appendix 1).")
    n_src = 0
    for i, (pop, lab) in enumerate(BMI_SRC):
        r0 = pick(perf, population=pop, target=targets[0])
        n_src += int(r0.n)
        add(lab + (src_mark if i == 0 else ""), r0.n, r0.positives,
            {t: (pick(perf, population=pop, target=t).auroc, pick(perf, population=pop, target=t).oe) for t in targets})
    assert n_src == n_test
    ipw_mark = fn("ipw", "AUROC and O:E with each internal test person weighted by the inverse probability that serum "
                         "urate was measured on the urine-test day, with the measurement model fitted among all adults "
                         "with complete dipstick results, age and sex (primary) or among those who also had BMI "
                         "({{T:ipw}} in Multimedia Appendix 1).")
    ipw = vb[(vb.section == "auroc") & vb["item"].astype(str).str.startswith("IPW, 1/P(measured) truncated at 99th")]
    for key, lab in (("spec", f"Weighted for urate measurement{ipw_mark}"),
                     ("bmi_available", "Same, measurement model among persons with BMI")):
        # verification_bias.csv: estimate = weighted AUROC, oe = weighted O:E (sum w*y / sum w*p)
        add(lab, None, None, {t: (float(pick(ipw, target=t, propensity_model=key).estimate),
                                  float(pick(ipw, target=t, propensity_model=key).oe)) for t in targets})
    pooled = mi[mi.section == "pooled"]
    m0 = pick(pooled, imputation_model="full", target=targets[0])
    mi_mark = fn("mi", f"The models applied to persons excluded because BMI was missing on the urine-test day, with BMI "
                       f"multiply imputed ({int(m0.m)} imputations) from age, sex, dipstick fields and serum urate (full "
                       "model) or from age and sex only.")
    add(f"Excluded for missing BMI, BMI imputed (full model){mi_mark}", m0.n, m0.positives,
        {t: (pick(pooled, imputation_model="full", target=t).auroc,
             pick(pooled, imputation_model="full", target=t).oe) for t in targets})
    add("Same, BMI imputed from age and sex", None, None,
        {t: (pick(pooled, imputation_model="age_sex", target=t).auroc,
             pick(pooled, imputation_model="age_sex", target=t).oe) for t in targets})

    sec("Time")
    # the internal test persons by period of the urine test (fu_internal_period.csv) use the periods of the temporal
    # holdout (temporal.csv: refitted on the early period, tested on the late one)
    tm = tp_[(tp_.design == "temporal") & (tp_.evaluation == "internal test")]
    TM = one_label(tm.model, "temporal.csv refitted ensemble", ("ensemble",))
    r_tm = pick(tm, target=targets[0], model=TM)
    n_per = 0
    for per in (str(r_tm.train_period), str(r_tm.test_period)):
        r0 = pick(fp_, target=targets[0], period=per)
        n_per += int(r0.n)
        add(f"Urine tests in {per}", r0.n, r0.events,
            {t: (pick(fp_, target=t, period=per).auroc, pick(fp_, target=t, period=per).oe) for t in targets})
    assert n_per == n_test, "fu_internal_period.csv: the two periods do not cover the internal test set"
    tmp_mark = fn("temp", f"The ensemble configuration (hyperparameters unchanged) refitted on the persons tested in "
                          f"{r_tm.train_period} and evaluated on those tested in {r_tm.test_period}.")
    add(f"Temporal holdout, refitted{tmp_mark}", r_tm.n, r_tm.positives,
        {t: (pick(tm, target=t, model=TM).auroc, pick(tm, target=t, model=TM).oe) for t in targets})

    def rng(df, t, col, d):
        v = df[df.target == t][col].astype(float)
        return f"{fnum(v.min(), d)}-{fnum(v.max(), d)}"

    n_yr = int(fy[fy.target == targets[0]].year.nunique())
    add(f"KNHANES single survey years ({n_yr}), range", None, None,
        {t: (rng(fy, t, "auroc", 3), rng(fy, t, "oe", 2)) for t in targets})
    w = sw.set_index("target")
    wmark = fn("wt", "KNHANES examination weights, not re-adjusted for complete-case exclusion.")
    add(f"KNHANES with survey weights{wmark}", None, None,
        {t: (w.loc[t, "auroc_weighted"], w.loc[t, "oe_weighted"]) for t in targets})

    code_mark = fn("coding", "Alternative codings of the KNHANES dipstick results, each compared with the primary "
                             "label-matched coding X1 (Tables {{N:mapping}} and {{N:mapsens}} in Multimedia Appendix 1).")
    sec("KNHANES input coding", code_mark)
    clab = coding_labels(enc, ms)
    for mp_ in ("M1", "M2", "M3", "X2"):
        add(clab[mp_], None, None, {t: (pick(ms, target=t, mapping=mp_).auroc, pick(ms, target=t, mapping=mp_).oe_ratio)
                                    for t in targets})
    sm = enc[enc.section == "summary"].set_index("feature")
    g_sg = one_label(sgr.group, "fu_sg_restrict.csv development-range group", ("development range",))
    r0 = pick(sgr, target=targets[0], group=g_sg)
    add(f"Adults with specific gravity ≤{float(sm.loc['ua_sg', 'max_dev']):.3f}", r0.n, r0.events,
        {t: (pick(sgr, target=t, group=g_sg).auroc, pick(sgr, target=t, group=g_sg).oe) for t in targets})
    a_ket = one_label(kt.analysis, "fu_ketone.csv external analysis with all ketone set to 0",
                      ("external", "all ketone", "set to 0"))
    kx = kt[(kt.analysis == a_ket) & (kt.group == "all")]
    add("Ketone set to 0", None, None, {t: (pick(kx, target=t).auroc_after, pick(kx, target=t).oe_after) for t in targets})

    sec("Model form (refitted; KNHANES adults)")
    cc = co[(co.row_type == "classification") & co.cohort.str.startswith("External KNHANES >=19 y (primary")
            & co.model.str.startswith("Simple homoscedastic")]
    add("Continuous urate model, constant variance", None, None,
        {t: (pick(cc, target=t).auroc, pick(cc, target=t).oe_ratio) for t in targets})
    # the variant that also models the variance (fu_continuous.csv) is reported in Multimedia Appendix 1 only (2026-10-17)
    lg = pv[(pv.cohort == "external") & (pv.method == "logistic_logodds")]
    cal_mark = fn("calib", "The fold learners of the ensemble with a logistic calibrator on the log-odds of their "
                           "soft-vote probability, fitted on the same calibration folds, instead of isotonic regression "
                           "({{T:platt}} in Multimedia Appendix 1).")
    add(f"Logistic instead of isotonic calibration{cal_mark}", None, None,
        {t: (pick(lg, target=t).auroc, pick(lg, target=t).oe) for t in targets})
    fn.note("Point estimates (95% CIs in Multimedia Appendix 1); O:E below 1 indicates over-prediction. Rows without "
            "n use the full internal test set or all KNHANES adults.")
    cap = "Sensitivity analyses of the models: discrimination (AUROC) and calibration (O:E ratio)."
    return block(number, cap, ["discrimination.csv", "calibration.csv", "bmi_window.csv", "verification_bias.csv",
                               "mice_bmi.csv", "fu_internal_period.csv", "temporal.csv", "fu_knhanes_year.csv",
                               "survey_weighted.csv", "mapping_sensitivity.csv", "encoding.csv", "fu_sg_restrict.csv",
                               "fu_ketone.csv", "continuous.csv", "platt_vs_isotonic.csv", "cohort_info_v2.csv"],
                 head, rows, fn,
                 "AUROC: area under the receiver operating characteristic curve; BMI: body mass index; KNHANES: Korea "
                 "National Health and Nutrition Examination Survey; N/A: not applicable; O:E: observed-to-expected.")


def isnan(x):
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
    except (TypeError, ValueError):
        return False


# ============================================================================ render check
# wording that belongs to the history of the submission, not to the tables of the retrained models
STALE = re.compile(r"(?i)\b(?:original\w*|deployed|locked|submitted|submission|reconcil\w*|person-clean|leukocyte|"
                   r"esterase|pars(?:e|ed|ing)|harmoni[sz]\w*|retrain\w*|audit\w*|10F|13F)\b|0\.321")


def md_table_rows(txt):
    """Rows (header + body, separator excluded) of each pipe table, in order."""
    out, cur = [], 0
    for line in txt.splitlines() + [""]:
        if line.startswith("|"):
            cur += 0 if re.match(r"^\|-", line) else 1
        elif cur:
            out.append(cur)
            cur = 0
    return out


def check_text(path, n_tables):
    """Style checks: no en/em dashes, no '--' outside separators/comments, no '~' (pandoc subscript), no NaN; no
    wording of the submission history; supplementary-table tokens with double braces."""
    txt = open(path, encoding="utf-8").read()
    body = [ln for ln in txt.splitlines() if not ln.startswith("|-") and not ln.startswith("<!--")]
    joined = "\n".join(body)
    bad = [s for s in ("–", "—", "--", "~") if s in joined]
    bad += re.findall(r"\b(?:nan|NaN|None|inf)\b", joined)
    bad += STALE.findall(joined)
    bad += re.findall(r"(?<!\{)\{[TNF]:[^}]*\}(?!\})", joined)
    assert not bad, f"{path}: forbidden text {bad}"
    assert len(md_table_rows(txt)) == n_tables, (path, md_table_rows(txt))
    return txt


def render_check(path, n_tables):
    """pandoc -> HTML (count tables and rows, no stray pipes) and -> docx (python-docx table count)."""
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("pandoc not found: render check skipped")
        return
    md = open(path, encoding="utf-8").read()
    html = subprocess.run([pandoc, "-f", "markdown", "-t", "html", path], capture_output=True, text=True,
                          check=True).stdout
    assert html.count("<table") == n_tables, (path, html.count("<table"))
    assert "|" not in re.sub(r"(?s)<table.*?</table>", "", html), "unparsed pipe-table text outside tables"
    html_rows = [t.count("<tr") for t in html.split("<table")[1:]]
    assert md_table_rows(md) == html_rows, (md_table_rows(md), html_rows)
    assert "<sup>a</sup>" in html and "^" not in re.sub(r"<[^>]+>", "", html), "unrendered superscript"
    try:
        import docx  # python-docx
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), os.path.basename(path) + ".check.docx")
        subprocess.run([pandoc, path, "-o", tmp], check=True)
        d = docx.Document(tmp)
        assert len(d.tables) == n_tables
        print(f"render check OK: {os.path.basename(path)}: {n_tables} tables, rows {html_rows} (HTML and docx)")
    except ImportError:
        print(f"render check OK (HTML only): {os.path.basename(path)}: {n_tables} tables, rows {html_rows}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-urate", action="store_true")
    ap.add_argument("--no-render-check", action="store_true")
    a = ap.parse_args()
    ci = info()
    if a.rebuild_urate or not urate_csv_current(ci):
        build_urate_csv()
        assert urate_csv_current(ci), "baseline_urate.csv does not match cohort_info_v2.csv after the rebuild"
    stamp = f"<!-- Generated {date.today().isoformat()} by revision_R1/working/make_tables.py. Do not edit by hand; edit the script and rerun. -->"
    main_md = "\n\n".join([stamp, table1(), table2(MAIN_TARGETS, "Table 2"), table3m(MAIN_TARGETS, "Table 3"),
                           table4m(MAIN_TARGETS, "Table 4"), table5(MAIN_TARGETS, "Table 5")]) + "\n"
    appx_md = "\n\n".join([stamp.replace("-->", "Exploratory >=10.0 mg/dL counterparts of Tables 2-4 for the Multimedia Appendix. -->"),
                           table2(APPX_TARGETS, "Appendix Table (counterpart of Table 2)", appendix=True),
                           table3(APPX_TARGETS, "Appendix Table (counterpart of Table 3)", appendix=True),
                           table4(APPX_TARGETS, "Appendix Table (counterpart of Table 4)", appendix=True)]) + "\n"
    for path, txt in ((OUT_MAIN, main_md), (OUT_APPX, appx_md)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        check_text(path, 5 if path == OUT_MAIN else 3)
        print(f"wrote {path}")
        if not a.no_render_check:
            render_check(path, 5 if path == OUT_MAIN else 3)


if __name__ == "__main__":
    main()
