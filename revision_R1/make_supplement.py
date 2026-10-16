"""Generate Multimedia Appendix 1 (supplementary methods, tables and figures) for the R1 revision of
JMIR Medical Informatics manuscript #107515 (urine dipstick models for serum urate thresholds).

Every result in the appendix is read from the analysis outputs in this folder (CSV files written by the calc_*.py
scripts, build_cohort_v2.py and calc_main.py); none is typed by hand. Data facts are read as well: the study years
(dipstick_by_year.csv), the KNHANES age top-code and the specific-gravity ranges and steps (encoding.csv). One input is
not a CSV: Table S1 panel A (and the base of the age exclusion in Table S5 panel A) reads the first-visit exclusions by
age from one line of calc_followup_log.txt (the line make_figures.py reads for main-text Figure 1). Every input must be
newer than cohort_info_v2.csv (check_inputs; INPUTS), so that no output of an analysis not rerun for the retrained
models can enter the appendix. Methods constants (number of bootstrap resamples, seeds, grid definitions, tuning
budget) are quoted from the analysis scripts.

Models (author decision of 2026-10-16): the three calibrated soft-voting ensembles (LightGBM, XGBoost and CatBoost,
isotonic CalibratedClassifierCV with 5 folds), one per threshold, with 12 inputs (9 dipstick fields, age, sex and BMI),
trained on the training persons of ONE person-level 80/20 split of the development cohort (cohort.py, section "v2:
corrected cohort") and evaluated on its internal test persons (the same 25,907 persons for every threshold) and on
KNHANES adults with the dipstick fields coded by the label-matched mapping X1. The appendix describes these models only,
as a self-contained development and validation study: it does not mention earlier versions of the models, the
submitted analysis or the corrections made to it (they are in the response letter). The analysis outputs keep a few
former keys that the generator selects on but never prints (the method key 'isotonic_deployed' of
calibration_curve.csv and platt_vs_isotonic.csv, the cohort label 'internal person-clean test' of the importance
outputs); main() fails if a term of the former framing reaches the text (FORBIDDEN), and CSV notes pass through
clean_note(), which drops any clause that contains such a term. Table S5 panel A (missing data) shows the rows of
missing_table.csv without the nitrite parsing note, the row on checkup BMI of another day and the wording '(plausibility
rule unchanged)' (tab_missing: MISSING_DROP, MISSING_NOTE), and states the base of the age exclusion (window persons of
all ages, as in the reference rows of Table S1 panel A). Rows of sg_univariate.csv whose model did not converge are
printed as 'not estimable', never as estimates.

Structure: 22 tables, most of them with lettered panels (A, B, ...), in 12 sections; STRUCTURE lists the panels of
every table and SECTIONS the sections. Each panel is written by one tab_* function.
  S1  Study flow and setting: Table S1
  S2  Inputs, encoding and the KNHANES mapping: Tables S2-S4
  S3  Missing data, selection and verification bias: Tables S5-S7
  S4  Tuned hyperparameters: Table S8
  S5  What the dipstick adds, and the ensemble versus simpler models: Tables S9-S10
  S6  Discrimination and calibration: Table S11
  S7  Clinical utility: Tables S12-S13
  S8  Transportability over time and population: Tables S14-S16, Figure S1
  S9  Subgroups: Table S17
  S10 Nested-threshold coherence and the continuous-urate model: Table S18
  S11 Interpretation of the ensembles: Tables S19-S20, Figures S2-S3
  S12 Exploratory threshold >=10.0 mg/dL: Tables S21-S22, Figure S4
Removed on 2026-10-16 with the retraining (they concerned the submitted models): the reconciliation with the submitted
version (former Tables S1-S2: cohorts, performance, corrections, repository), the urine-blood interval (serum urate is
taken from the urine-test day by definition), the leukocyte esterase sensitivity analysis, the BMI linkage interval
(BMI is taken from the urine-test day) and the nitrite parsing error (former Section S13). The discrimination panel
moved to the calibration table. The exploratory >=10.0 mg/dL section is Section S12 (EXPL_SECTION; the manuscript and
the response letter cite it by that number, so check_structure() stops the run if it moves).

Outputs
  ../supplement/Multimedia_Appendix_1.md
  ../supplement/figures/FigS*.png and .pdf
  supplement_index.csv (columns kind, key, label, panel: every table (panel ''), the keys of its panels as aliases with
      their panel letter ('' in a table with one panel), the figures; read by build_manuscript.py (labels) and by
      make_checklist.py (labels and panel letters))
  ../revision_notes/supplement_renumbering_v5.csv (every table, panel and figure label of the appendix of 2026-10-12, as
      listed in _snapshot_original_models_261016/working/supplement_index.csv, -> its label here or 'removed'; plus the
      sections). ../revision_notes/supplement_renumbering.csv (the 72 -> 25 renumbering of 2026-10-12) is history: it
      is read, never written. check_structure() checks, before anything is written, that every key of 2026-10-12 is
      either kept or listed in REMOVED.
  make_supplement_log.txt

Figures use figstyle.py from this folder when it exists (shared visual language with the main figures);
otherwise a local style with the same palette and font sizes.

Run from the working folder:  /opt/anaconda3/bin/python make_supplement.py
"""
import os
import re
import sys
import contextlib
import datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Manuscript folder (inputs and outputs): PROJ_U_REVISION_DIR overrides; original: parent of this folder.
REV = os.environ.get("PROJ_U_REVISION_DIR", os.path.dirname(HERE))
SUPP = os.path.join(REV, "supplement")
FIGDIR = os.path.join(SUPP, "figures")
OUT_MD = os.path.join(SUPP, "Multimedia_Appendix_1.md")
LOG = os.path.join(HERE, "make_supplement_log.txt")
# appendix of 2026-10-12 (before the retraining): read only, to write the renumbering file
SNAP_INDEX = os.path.join(HERE, "_snapshot_original_models_261016", "working", "supplement_index.csv")
RENUMBER_HISTORY = os.path.join(REV, "revision_notes", "supplement_renumbering.csv")    # history; never written
RENUMBER_V5 = os.path.join(REV, "revision_notes", "supplement_renumbering_v5.csv")

MAIN = [">=7.0", ">=9.0"]
TITLE = ("Machine Learning Models Combining Routine Urine Dipstick Results With Age, Sex, and Body Mass Index to Triage Serum "
         "Urate Testing: Development and External Validation Study")
EXPL = [">=10.0"]
ALLT = MAIN + EXPL
TLAB = {">=7.0": "≥7.0 mg/dL", ">=9.0": "≥9.0 mg/dL", ">=10.0": "≥10.0 mg/dL"}
TROLE = {">=7.0": "primary", ">=9.0": "secondary", ">=10.0": "exploratory"}
TKEY = {"uric_hyper_70": ">=7.0", "uric_severe_9": ">=9.0", "uric_severe_10": ">=10.0",
        7.0: ">=7.0", 9.0: ">=9.0", 10.0: ">=10.0"}
# the 12 model inputs (cohort.FMODEL order)
FEAT = {"ua_glucose": "Glucose", "ua_protein": "Protein", "ua_ketone": "Ketone", "ua_bilirubin": "Bilirubin",
        "ua_blood": "Occult blood", "ua_urobilinogen": "Urobilinogen", "ua_nitrite": "Nitrite",
        "ua_pH": "pH", "ua_sg": "Specific gravity", "age": "Age", "male": "Male sex", "bmi_unified": "BMI"}
FEAT_ORDER = list(FEAT)
SEMI = ["ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen", "ua_nitrite"]
ISO = "isotonic_deployed"      # method key of the isotonic models in calibration_curve.csv / platt_vs_isotonic.csv
CODINGS = ["X1", "M1", "M2", "M3", "X2"]

LOGLINES = []


def log(msg):
    LOGLINES.append(msg)
    print(msg)


# ============================================================ reading
_cache = {}
# Every CSV the appendix reads. cohort_info_v2.csv (build_cohort_v2.py) defines the cohort of the retrained models and was
# written before they were trained, so every other input must be newer: an older file (for example an fu_*.csv of
# calc_followup.py from before the retraining, which describes the earlier models) would put results of other models
# under the labels of this appendix. check_inputs() fails on any such file.
COHORT_FILE = "cohort_info_v2"
INPUTS = ["main_performance", "supplement_derived", "missing_table", "encoding", "mapping_sensitivity", "feature_tiers",
          "feature_tiers_coding_sensitivity", "tuned_per_set", "ensemble_vs_simple", "mcar", "mice_bmi", "bmi_window",
          "verification_bias", "hyperparameters", "discrimination", "calibration", "calibration_curve", "platt_vs_isotonic",
          "recalibration", "operating_points", "splithalf_operating", "decision_curve", "decision_curve_ranges",
          "utility_12F_vs_3F", "temporal", "knhanes_by_year", "dipstick_by_year", "survey_weighted", "subgroups",
          "sex_difference", "nested_coherence", "continuous", "permutation_importance", "shap_ensemble", "shap_by_sg",
          "sg_univariate", "pdp_calibrated",
          # calc_followup.py
          "fu_setting", "fu_knhanes_flow", "fu_bmi_source", "fu_within_sex", "fu_isotonic_floor", "fu_sex_operating",
          "fu_oof_thresholds", "fu_recal_operating", "fu_net_benefit", "fu_net_benefit_ranges", "fu_internal_period",
          "fu_knhanes_year", "fu_ketone", "fu_sg_restrict", "fu_continuous"]


def check_inputs():
    """Every input exists and is newer than cohort_info_v2.csv (see INPUTS)."""
    path = lambda n: os.path.join(HERE, n + ".csv")  # noqa: E731
    assert os.path.exists(path(COHORT_FILE)), f"{COHORT_FILE}.csv missing"
    t0 = os.path.getmtime(path(COHORT_FILE))
    missing = [n for n in INPUTS if not os.path.exists(path(n))]
    stale = [f"{n}.csv ({datetime.datetime.fromtimestamp(os.path.getmtime(path(n))):%Y-%m-%d %H:%M})" for n in INPUTS
             if n not in missing and os.path.getmtime(path(n)) < t0]
    if missing or stale:
        raise RuntimeError(f"inputs missing: {missing}; inputs older than {COHORT_FILE}.csv "
                           f"({datetime.datetime.fromtimestamp(t0):%Y-%m-%d %H:%M}), i.e. not rerun for the models of this "
                           f"appendix: {stale}")
    log(f"inputs: {len(INPUTS)} CSV files, all newer than {COHORT_FILE}.csv")


def rd(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(os.path.join(HERE, name + ".csv"), low_memory=False)
    return _cache[name].copy()


def tnorm(s):
    if isinstance(s, (int, float, np.floating)) and not isinstance(s, bool):
        return TKEY.get(float(s), str(s))
    s = str(s)
    return TKEY.get(s, s)


def with_t(df, col="target"):
    df = df.copy()
    df["_t"] = df[col].map(tnorm)
    return df


def one(df, **kw):
    """Exactly one row matching the keyword filters (strings compared exactly, floats with isclose)."""
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        if isinstance(v, float) and df[k].dtype != object:
            m &= np.isclose(df[k].astype(float), v)
        else:
            m &= df[k] == v
    sub = df[m]
    if len(sub) != 1:
        raise KeyError(f"expected 1 row for {kw}, found {len(sub)}")
    return sub.iloc[0]


def ci_items(name):
    d = rd(name)
    return dict(zip(d["item"].astype(str), d["value"]))


def cohort_side(s):
    """'external' / 'internal' from a cohort label of the analysis outputs ('External KNHANES >=19 y (primary)',
    'internal person-clean test', 'Internal test (shared person-level split; secondary)', ...)."""
    s = str(s).strip().lower()
    if s.startswith("external"):
        return "external"
    if s.startswith("internal"):
        return "internal"
    raise KeyError(f"cohort label neither external nor internal: {s!r}")


# ------------------------------------------------------------ data facts quoted in notes (read, not typed)
def study_years(which):
    """'2010-2025' for 'development' (years of the urine tests of the development cohort) or '2016-2024' for 'knhanes'
    (survey years of the external validation cohort): first and last year with persons (dipstick_by_year.csv)."""
    dy = rd("dipstick_by_year")
    dy = dy[dy.period_type == "year"]
    dev = dy.cohort.astype(str).str.lower().str.startswith("development")
    sel = dy[dev if which == "development" else ~dev]
    assert len(sel) and (sel["n"] > 0).all(), which
    yrs = sel["period"].astype(int)
    return f"{yrs.min()}-{yrs.max()}"


def sg_facts():
    """Specific gravity in the two cohorts (encoding.csv): largest value and recording step, as 3-decimal strings. The
    development step ignores levels with fewer than 100 persons (isolated off-grid values)."""
    enc = rd("encoding")
    sm = enc[enc.section == "summary"].set_index("feature")
    cb = enc[(enc.section == "codebook") & (enc.feature == "ua_sg")]

    def step(col, least):
        v = np.sort(cb[cb[col] >= least]["value_num"].astype(float).unique())
        return float(np.round(np.diff(v).min(), 3))
    return dict(dev_max=f"{float(sm.loc['ua_sg', 'max_dev']):.3f}", ext_max=f"{float(sm.loc['ua_sg', 'max_ext']):.3f}",
                dev_step=f"{step('n_dev', 100):.3f}", ext_step=f"{step('n_ext', 1):.3f}")


def knhanes_age_topcode():
    """Highest KNHANES age (encoding.csv summary): KNHANES records age 80 and older as 80."""
    enc = rd("encoding")
    return int(float(enc[enc.section == "summary"].set_index("feature").loc["age", "max_ext"]))


# ============================================================ formatting
def isnan(x):
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
    except (TypeError, ValueError):
        return False


def f(x, d=3):
    if isnan(x):
        return "NA"
    s = f"{float(x):.{d}f}"
    return s


def n_(x):
    if isnan(x):
        return "NA"
    return f"{int(round(float(x))):,}"


def pc(x, d=1):
    """proportion -> percent string (no % sign)."""
    if isnan(x):
        return "NA"
    return f"{100 * float(x):.{d}f}"


def pct_r(v):
    """A percentage (0-100) with the rounding rule of the manuscript: 1 decimal, 2 where it lies strictly between 0 and
    1% or between 99% and 100% (0.15, 99.47), so that a share near 0 or 100 keeps its information and the appendix prints
    the value the main text cites. 0 and 100 keep 1 decimal."""
    if isnan(v):
        return "NA"
    v = float(v)
    return f"{v:.{2 if (0 < abs(v) < 1 or 99 < v < 100) else 1}f}"


def pcr(x):
    """proportion -> percent string with the rule of pct_r (no % sign)."""
    if isnan(x):
        return "NA"
    return pct_r(100 * float(x))


def ci(lo, hi, d=3):
    if isnan(lo) or isnan(hi):
        return "NA"
    a, b = f(lo, d), f(hi, d)
    sep = " to " if (a.startswith("-") or b.startswith("-")) else "-"
    return f"{a}{sep}{b}"


def est(x, lo, hi, d=3):
    if isnan(lo) or isnan(hi):
        return f(x, d)
    return f"{f(x, d)} ({ci(lo, hi, d)})"


def pest(x, lo, hi, d=1):
    """proportions -> percent with CI."""
    if isnan(lo) or isnan(hi):
        return pc(x, d)
    return f"{pc(x, d)} ({ci(100 * lo, 100 * hi, d)})"


def sg(x, d=3):
    if isnan(x):
        return "NA"
    return f"{float(x):+.{d}f}"


def sgci(x, lo, hi, d=3):
    if isnan(lo) or isnan(hi):
        return sg(x, d)
    return f"{sg(x, d)} ({ci(lo, hi, d)})"


def pv(p):
    """JMIR style P value (no leading zero)."""
    if isnan(p):
        return "NA"
    p = float(p)
    if p < 0.001:
        return "<.001"
    if p < 0.01:
        return f"{p:.3f}".lstrip("0")
    s = f"{p:.2f}"
    if s == "1.00":
        return ">.99"
    return s.lstrip("0")


def pe(p):
    """P value with its comparator in JMIR style: P<.001, P=.04, P>.99."""
    s = pv(p)
    if s == "NA":
        return "P not computed"
    return "P" + (s if s[0] in "<>" else "=" + s)


def num(x, sig=4):
    """hyperparameter values: integers as integers, floats to 4 significant digits (trailing zeros kept)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(v):
        return "NA"
    if v == int(v) and abs(v) < 1e9:
        return f"{int(v)}"
    return format(v, f"#.{sig}g")


def val(x):
    """A dipstick code or a model-scale value: 0, 0.5, 1, 5.5, 1.005 (no padding zeros)."""
    if isnan(x):
        return "NA"
    return f"{float(x):g}"


_TIDY = [("  minus  ", " minus "), ("+/-", "±"), (" >= ", " ≥"), (">= ", "≥"), (" <= ", " ≤"), ("<= ", "≤"),
         (">=", "≥"), ("<=", "≤"), ("> 365", ">365"), ("> 5 y", ">5 y"), ("< 19", "<19"), ("> 110", ">110"),
         ("(< 20 non-zero)", "(<20 non-zero)"), ("BMI_demo", "health-checkup BMI"), ("bmi_unified", "BMI"),
         ("ua_glucose", "glucose"), ("ua_protein", "protein"), ("ua_ketone", "ketone"), ("ua_bilirubin", "bilirubin"),
         ("ua_blood", "occult blood"), ("ua_urobilinogen", "urobilinogen"), ("ua_nitrite", "nitrite"),
         ("ua_wbc(s)", "leukocyte esterase"), ("ua_pH", "pH"), ("ua_sg", "specific gravity"), ("height_weight", "height/weight"),
         ("weight/height^2", "weight/height²"), ("(dev_v2)", "(development cohort)"), ("dev_v2", "development cohort"),
         ("(ext_v2, KNHANES ≥19)", "(KNHANES adults)"), ("ext_v2", "external cohort"), (" -> ", " → "),
         ("analytic cohort", "development cohort")]   # the appendix and main-text Figure 1 say 'development cohort'


def tidy(s, cap=True):
    """Make a CSV label readable: feature codes to names, comparison signs, capitalised first letter."""
    if isnan(s):
        return ""
    s = str(s)
    for a, b in _TIDY:
        s = s.replace(a, b)
    s = s.strip()
    if cap and s and "a" <= s[0] <= "z" and not s.startswith("pH"):
        s = s[0].upper() + s[1:]
    return s


def step_text(s):
    """'largest step 1.015 -> 1.02 (-17%)' -> 'largest step 1.015 to 1.02: -17%'."""
    m = re.match(r"largest step ([\d.]+) -> ([\d.]+) \(([-+]?\d+)%\)", str(s))
    if not m:
        return tidy(s, cap=False)
    return f"largest step {m.group(1)} to {m.group(2)}: {m.group(3)}%"


def shape_text(s):
    """'U-shaped (minimum at 70)' -> 'U-shaped, minimum at 70'."""
    return re.sub(r"\s*\((minimum|maximum) at ([\d.]+)\)", r", \1 at \2", str(s))


# ------------------------------------------------------------ terminology of the appendix
# The appendix reports the models as a self-contained development and validation study. These terms belong to the
# history of the submission (response letter only) or are raw analysis keys; main() fails when one reaches the text.
FORBIDDEN = [r"\bdeployed\b", r"isotonic_deployed", r"\boriginal", r"\bsubmitted\b", r"\bsubmission", r"\blocked\b",
             r"reconcil", r"\baudit", r"retrain", r"parsing error", r"person-clean", r"row-level", r"bestmodel",
             r"\bv[45]\b", r"0\.321", r"\b1[03]F\b", r"\b13[- ]input", r"\b10 dipstick", r"\bcorrected\b",
             r"_corrected\b", r"\b(?:dev|ext)_v2\b", r"noLE", r"LE=0", r"leukocyte esterase (?:constant|set to)",
             r"DEPLOYED_WORD",
             # history of the nitrite parsing and of the cohort definition (response letter only); 'harmonised' was the
             # name of a removed analysis setting (the KNHANES analysis file is called 'pooled' here)
             r"\bpars(?:e|ed|es|ing)\b", r"\bstill missing\b", r"\bharmoni[sz]",
             # 2026-10-17: no trace of the revision itself ('revised', 'revision working folder')
             r"\brevis"]


def forbidden_hit(s):
    for pat in FORBIDDEN:
        m = re.search(pat, str(s), flags=re.I)
        if m:
            return m
    return None


def clean_note(s):
    """A note from an analysis CSV without the clauses ('; '-separated) that use a term of FORBIDDEN."""
    if isnan(s):
        return ""
    parts = [p for p in re.split(r";\s+", str(s)) if not forbidden_hit(p)]
    return "; ".join(parts)


_NEUTRAL = [(r"(?i)\b(?:deployed|retrained|original|locked)\s+(?=(?:calibrated\s+)?(?:soft-voting\s+)?ensembles?\b)", ""),
            (r"\s*\(v[45]\)", ""), (r"(?i)\bv[45] config(?:uration)?\b", "ensemble member configuration"),
            (r"(?i)\bisotonic as deployed\b", "isotonic as fitted"), (r"(?i)\bas deployed\b", "as fitted"),
            (r"(?i)\bisotonic \((?:deployed|retrained|original)\)", "isotonic (as fitted)"),
            (r"(?i)person-clean\s+", ""), (r"\(development test split\)", "(internal test persons)"),
            (r"(?i)\bisotonic CV5\b", "isotonic")]


def neutral(s):
    """A model or analysis label of the analysis outputs in the terms of the appendix ('Retrained ensemble (v5)' ->
    'Ensemble'). Anything left that FORBIDDEN matches is caught by the final check in main()."""
    if isnan(s):
        return ""
    s = str(s)
    for a, b in _NEUTRAL:
        s = re.sub(a, b, s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s[:1].upper() + s[1:] if s[:1].islower() and not s.startswith("pH") else s


def neutral_lc(s):
    """neutral() for use inside a sentence: first letter in lower case (unless the label starts with an acronym)."""
    s = neutral(s)
    return s[:1].lower() + s[1:] if s[:2] != s[:2].upper() else s


MODEL_LAB = {"LightGBM (fixed)": "LightGBM (fixed configuration)", "LightGBM (GRID-tuned)": "LightGBM (grid-tuned)",
             "Logistic regression": "Logistic regression", "LR (age/BMI splines + age x male)": "Spline logistic regression"}


def model_label(m):
    """Display label of a model of feature_tiers.csv, ensemble_vs_simple.csv and feature_tiers_coding_sensitivity.csv.
    The ensemble arm and the arm with the ensemble's LightGBM configuration are recognised by their wording, so that the
    analysis script may rename them ('Retrained ensemble (v5)', 'LightGBM (v5 config, isotonic CV5)')."""
    s = str(m)
    low = s.lower()
    if "ensemble" in low and "lightgbm" not in low:
        return "Ensemble (tuned, isotonic)"
    if "lightgbm" in low and "isotonic" in low:
        return "LightGBM (ensemble member configuration, isotonic)"
    return MODEL_LAB.get(s, neutral(s))


def is_ensemble(m):
    return model_label(m) == "Ensemble (tuned, isotonic)"


# ============================================================ structure
# Author decision of 2026-10-16 (retrained models): 22 tables with lettered panels; each panel is written by one tab_*
# function. Entries: (key of the table, title, keys of its panels in order). The table and panel keys are those of the
# appendix of 2026-10-12 (build_manuscript.py resolves them through supplement_index.csv).
STRUCTURE = [
    ("flowset", "Study flow, setting and the KNHANES analysis file", ["flow", "setting_all", "setting_dev", "knflow"]),
    ("inputmap", "Input features, their coding and the KNHANES mapping", ["inputs", "mapping"]),
    ("inputdist", "Input distributions and distribution shift", ["codebook", "shift"]),
    ("codingsens", "Sensitivity to the KNHANES coding", ["mapsens", "coding"]),
    ("missdata", "Missing data, the MCAR test and multiple imputation of BMI", ["missing", "included", "mcar", "mice"]),
    ("bmilink", "BMI source and performance by BMI source", ["bmisource", "bmiperf"]),
    ("verifbias", "Verification bias", ["verif", "ipw"]),
    ("hyperparams", "Tuned hyperparameters of the base learners", ["hyper"]),
    ("addedvalue", "Added value of the dipstick over demographic models", ["tiers", "tiercon", "tuned", "tiers2f", "withinsex"]),
    ("ensvssimple", "The ensemble versus simpler 12-input models", ["ens"]),
    ("calibration", "Discrimination, calibration, calibration methods and recalibration",
     ["discrim", "calib", "calbins", "platt", "isofloor", "recal"]),
    ("operating", "Operating points and the robustness of the threshold choice",
     ["oppoints", "splithalf", "sexop", "oof", "recalop"]),
    ("netbenefit", "Decision curves and net benefit", ["dca", "nbref", "utility"]),
    ("overtime", "Performance over time", ["temporal", "intperiod", "periods", "knyear"]),
    ("byyear", "Dipstick positivity and reporting by year", ["posyear", "levels"]),
    ("othersens", "Other sensitivity analyses: ketone, specific gravity range and survey weights",
     ["ketone", "sgrestrict", "survey"]),
    ("subgroupsex", "Subgroups and the female-male difference", ["subgroups", "sexdiff"]),
    ("nestedcont", "Nested-threshold coherence and continuous-urate models", ["nested", "continuous", "contfu"]),
    ("importance", "Importance of the inputs: permutation importance and SHAP", ["perm", "shap"]),
    ("sgage", "Specific gravity and age effects", ["sglevel", "sgconf", "agepdp"]),
    ("expl10a", "Exploratory ≥10.0 mg/dL model: performance, calibration and operating points", ["expl_perf", "expl_util"]),
    ("expl10b", "Exploratory ≥10.0 mg/dL model: added value, subgroups and sensitivity analyses", ["expl_value", "expl_sens"]),
]
# sections in order: (key, title); a section is written with doc.section(key) and cited as SEC(key) ('S12')
SECTIONS = [
    ("flow", "Study flow and setting"),
    ("inputs", "Inputs, encoding and the KNHANES mapping"),
    ("missing", "Missing data, selection and verification bias"),
    ("hyper", "Tuned hyperparameters"),
    ("value", "What the dipstick adds, and the ensemble versus simpler models"),
    ("calib", "Discrimination and calibration"),
    ("utility", "Clinical utility"),
    ("transport", "Transportability over time and population"),
    ("subgroups", "Subgroups"),
    ("nested", "Nested-threshold coherence and the continuous-urate model"),
    ("interp", "Interpretation of the ensembles"),
    ("expl", "Exploratory threshold: serum urate ≥10.0 mg/dL"),
]
# the manuscript, Multimedia Appendix 2 and the response letter cite the exploratory section by its number (no token)
EXPL_SECTION = ("expl", "S12")
# figures in order: (key, file name)
FIGURES = [("years", "FigS1_dipstick_by_year"), ("perm", "FigS2_permutation_importance"), ("sg", "FigS3_specific_gravity"),
           ("expl", "FigS4_exploratory_10")]
# keys of the appendix of 2026-10-12 that no longer exist (the former Table S56 'adult' was already a main-text alias)
REMOVED = {"recon", "recon_cohort", "recon_perf", "audit", "corrections", "repo", "nitrite", "nitrec", "niteval", "gap",
           "le", "bmigap", "adult"}
# sections of the appendix of 2026-10-12 -> key of the section that holds their tables now (None: removed)
OLD_SECTIONS = [("S1", "Reconciliation with the submitted version", None),
                ("S2", "Study flow, setting and the urine-blood interval", "flow"),
                ("S3", "Inputs, encoding and the KNHANES mapping", "inputs"),
                ("S4", "Missing data, selection and verification bias", "missing"),
                ("S5", "Tuned hyperparameters", "hyper"),
                ("S6", "What the dipstick adds, and the ensemble versus simpler models", "value"),
                ("S7", "Calibration", "calib"),
                ("S8", "Clinical utility", "utility"),
                ("S9", "Transportability over time and population", "transport"),
                ("S10", "Subgroups", "subgroups"),
                ("S11", "Nested-threshold coherence and the continuous-urate model", "nested"),
                ("S12", "Interpretation of the deployed ensembles", "interp"),
                ("S13", "Nitrite parsing error and recovery analysis", None),
                ("S14", "Exploratory threshold: serum urate >=10.0 mg/dL", "expl")]
PANEL_LETTERS = "ABCDEFGH"


def static_labels():
    """Labels implied by STRUCTURE, SECTIONS and FIGURES: ({table or panel key: (label, panel letter)}, {section key:
    'Sn'}, {figure key: label}). main() checks that the written appendix has exactly these labels."""
    tables = {}
    for i, (key, _, keys) in enumerate(STRUCTURE, 1):
        tables[key] = (f"Table S{i}", "")
        for j, k in enumerate(keys):
            tables[k] = (f"Table S{i}", PANEL_LETTERS[j] if len(keys) > 1 else "")
    sections = {k: f"S{i}" for i, (k, _) in enumerate(SECTIONS, 1)}
    figs = {k: f"Figure S{i}" for i, (k, _) in enumerate(FIGURES, 1)}
    return tables, sections, figs


def check_structure():
    keys = [k for _, _, ks in STRUCTURE for k in ks]
    new = [k for k, _, _ in STRUCTURE]
    assert len(keys) == len(set(keys)), "a panel key is used twice"
    assert len(new) == len(set(new)) == 22 and not set(new) & set(keys), "table keys must be unique and differ from panel keys"
    assert not (set(new) | set(keys)) & REMOVED, f"removed keys in STRUCTURE: {(set(new) | set(keys)) & REMOVED}"
    assert max(len(ks) for _, _, ks in STRUCTURE) <= len(PANEL_LETTERS)
    assert len({k for k, _ in SECTIONS}) == len(SECTIONS) and len({k for k, _ in FIGURES}) == len(FIGURES)
    assert {k for _, _, k in OLD_SECTIONS if k} == {k for k, _ in SECTIONS}, "every section must have a former section"
    tables, sections, figs = static_labels()
    assert sections[EXPL_SECTION[0]] == EXPL_SECTION[1], \
        (f"the exploratory section is {sections[EXPL_SECTION[0]]}, but the manuscript, Multimedia Appendix 2 and the "
         f"response letter cite it as Section {EXPL_SECTION[1]}: update them and EXPL_SECTION")
    # every key of the appendix of 2026-10-12 is either kept or removed on purpose (checked before anything is written;
    # write_renumbering_v5() relies on it)
    if os.path.exists(SNAP_INDEX):
        old = pd.read_csv(SNAP_INDEX)
        lost = sorted(str(r.key) for r in old.itertuples()
                      if (figs if r.kind == "figure" else tables).get(str(r.key)) is None and str(r.key) not in REMOVED)
        assert not lost, f"keys of the appendix of 2026-10-12 neither kept nor listed in REMOVED: {lost}"
    else:
        log(f"note: {os.path.relpath(SNAP_INDEX, HERE)} not found; coverage of the former keys not checked")
    log(f"structure: {len(STRUCTURE)} tables, {len(keys)} panels, {len(SECTIONS)} sections, {len(FIGURES)} figures; "
        f"exploratory threshold in Section {sections[EXPL_SECTION[0]]}")


def panel_list(letters):
    """' panel A', ' panels A and C', ' panels A-C'; '' when the whole table is cited."""
    if "" in letters:
        return ""
    ls = sorted(letters, key=PANEL_LETTERS.index)
    if len(ls) == 1:
        return f" panel {ls[0]}"
    pos = [PANEL_LETTERS.index(x) for x in ls]
    if len(ls) >= 3 and pos == list(range(pos[0], pos[0] + len(ls))):
        return f" panels {ls[0]}-{ls[-1]}"
    return " panels " + ", ".join(ls[:-1]) + " and " + ls[-1]


# ============================================================ document
class Doc:
    def __init__(self):
        self.parts = []
        self.nt = 0            # tables
        self.np = 0            # panels
        self.nf = 0
        self.tables = {}       # key of a table or of one of its panels -> "Table Sn"
        self.panel = {}        # key of a panel -> letter ("" in a table with a single panel)
        self.figs = {}
        self.sections = {}     # key of a section -> "Sn"
        self.toc = []
        self.open = None       # (key, label, panel keys, panel keys emitted) of the table being written

    def h(self, level, text):
        self.parts.append("#" * level + " " + text + "\n")
        if level == 2:
            self.toc.append(("section", text))

    def section(self, key):
        """Heading of the next section of SECTIONS ('## S3. Missing data, ...')."""
        i = len(self.sections)
        assert i < len(SECTIONS) and SECTIONS[i][0] == key, f"section {key} written out of order"
        lab = f"S{i + 1}"
        self.sections[key] = lab
        self.h(2, f"{lab}. {SECTIONS[i][1]}")

    def p(self, text):
        self.parts.append(text.strip() + "\n")

    def bullets(self, items):
        self.parts.append("\n".join(f"- {it}" for it in items) + "\n")

    def begin(self, key):
        """Write the caption of the next table of STRUCTURE; the tab_* calls that follow write its panels."""
        assert self.open is None, f"table {self.open[0]} is still open"
        new_key, title, keys = STRUCTURE[self.nt]
        assert key == new_key, f"table {key} written out of order (expected {new_key})"
        self.nt += 1
        lab = f"Table S{self.nt}"
        self.tables[key] = lab
        self.toc.append(("table", f"{lab}. {title}."))
        self.parts.append(f"**{lab}. {title}.**\n")
        self.open = (key, lab, keys, [])

    def end(self):
        assert self.open is not None, "no table is open"
        key, _, keys, done = self.open
        assert done == keys, f"table {key}: panels {done} written, {keys} expected"
        self.open = None

    @contextlib.contextmanager
    def group(self, key):
        self.begin(key)
        yield
        self.end()

    def table(self, key, title, header, rows, notes=None, source=None):
        """One panel of the open table: a bold line '(A) <title of the panel>', then its pipe table, notes and source
        line. A table with one panel has no panel line."""
        assert self.open is not None, f"panel {key} written outside a table"
        tkey, lab, keys, done = self.open
        i = len(done)
        assert i < len(keys) and keys[i] == key, f"panel {key} out of order in table {tkey} (expected {keys[i:]})"
        done.append(key)
        letter = PANEL_LETTERS[i] if len(keys) > 1 else ""
        self.tables[key] = lab
        self.panel[key] = letter
        self.np += 1
        texts = [str(c) for r in rows for c in r] + ([notes] if isinstance(notes, str) else list(notes or [])) + [source or ""]
        for t in texts:   # '{{P:key}}' ('panel B') may only name another panel of the same table
            for k in re.findall(r"\{\{P:([^}]+)\}\}", t):
                assert k in keys and k != key, f"P:{k} in panel {key} is not another panel of {tkey}"
        out = [f"**({letter}) {title}**", ""] if letter else []
        out.append("| " + " | ".join(str(h).replace("|", "/") for h in header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            cells = [str(c).replace("|", "/") for c in r]
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        if notes:
            for nline in ([notes] if isinstance(notes, str) else notes):
                out.append(f"*{nline.strip()}*")
                out.append("")
        if source:
            out.append(f"*Source: {source}.*")
            out.append("")
        self.parts.append("\n".join(out))
        return lab

    def figure(self, key, title, relpath, notes=None, source=None):
        assert self.open is None, f"figure {key} written inside table {self.open[0]}"
        assert FIGURES[self.nf][0] == key, f"figure {key} written out of order"
        self.nf += 1
        lab = f"Figure S{self.nf}"
        self.figs[key] = lab
        self.toc.append(("figure", f"{lab}. {title}"))
        out = [f"**{lab}. {title}**", "", f"![{lab}]({relpath})", ""]
        if notes:
            for nline in ([notes] if isinstance(notes, str) else notes):
                out.append(f"*{nline.strip()}*")
                out.append("")
        if source:
            out.append(f"*Source: {source}.*")
            out.append("")
        self.parts.append("\n".join(out))
        return lab

    def ref(self, key):
        """(label, panel letter) of a table or panel key."""
        if key not in self.tables:
            raise KeyError(f"unresolved reference T:{key}")
        return self.tables[key], self.panel.get(key, "")

    def render(self, head):
        assert self.open is None, f"table {self.open[0]} was not closed"
        body = "\n".join(self.parts)

        def refs(m):
            """A run of references ('{{T:a}}, {{T:b}}') names each table once, with its cited panels in order:
            'Table S1 panels A and B, Table S3 panel A'."""
            order, cited = [], {}
            for key in re.findall(r"\{\{T:([^}]+)\}\}", m.group(0)):
                lab, letter = self.ref(key)
                if lab not in cited:
                    order.append(lab)
                    cited[lab] = set()
                cited[lab].add(letter)
            return ", ".join(lab + panel_list(cited[lab]) for lab in order)

        def fig(m):
            if m.group(1) not in self.figs:
                raise KeyError(f"unresolved reference F:{m.group(1)}")
            return self.figs[m.group(1)]

        def sec(m):
            if m.group(1) not in self.sections:
                raise KeyError(f"unresolved reference S:{m.group(1)}")
            return self.sections[m.group(1)]

        toc = ["## Contents", ""]
        for kind, text in self.toc:
            if kind == "section":
                toc.append(f"- **{text}**")
            else:
                toc.append(f"    - {text}")
        text = head + "\n" + "\n".join(toc) + "\n\n" + body
        text = re.sub(r"\{\{T:[^}]+\}\}(?:, \{\{T:[^}]+\}\})*", refs, text)
        text = re.sub(r"\{\{P:([^}]+)\}\}", lambda m: "panel " + self.panel[m.group(1)], text)
        text = re.sub(r"\{\{F:([^}]+)\}\}", fig, text)
        text = re.sub(r"\{\{S:([^}]+)\}\}", sec, text)
        return text


def T(key):
    """Reference to a panel or table: rendered as its table and panel ('Table S7 panel C')."""
    return "{{T:" + key + "}}"


def P(key):
    """Reference to another panel of the same table: rendered as 'panel B'."""
    return "{{P:" + key + "}}"


def F(key):
    return "{{F:" + key + "}}"


def SEC(key):
    """Reference to a section: rendered as its number ('S12')."""
    return "{{S:" + key + "}}"


def sfx(targets):
    """Key suffix and title suffix for the exploratory copies of a table."""
    if targets == EXPL:
        return "@10", ", ≥10.0 mg/dL model (exploratory)"
    return "", ""


# ============================================================ S1 flow and setting
def kn_step(s):
    """KNHANES flow step of supplement_derived.csv in the terms of the appendix (the concatenated annual survey files are
    the 'pooled' KNHANES file)."""
    s = re.sub(r"\s*=\s*submitted external cohort", "", str(s))
    s = re.sub(r"(?i)\bcorrected external cohort\b", "external validation cohort", s)
    s = re.sub(r"(?i)\bharmoni[sz]ed\b", "pooled", s)
    # KNHANES holds one record per participant: count participants, as the manuscript and Figure 1 do (2026-10-17)
    s = re.sub(r"(?i)\brecords in the pooled\b", "participants in the pooled", s)
    return tidy(s)


FOLLOWUP_LOG = os.path.join(HERE, "calc_followup_log.txt")


def first_visit_age_exclusion():
    """First urine dipstick test of every person in the extraction (all ages) and its exclusion by age: one line of
    calc_followup_log.txt (section a1; the line make_figures.py reads for main-text Figure 1, whose wording
    calc_followup.py keeps), because fu_setting.csv holds adults only. The log must be newer than cohort_info_v2.csv
    (see INPUTS)."""
    assert os.path.exists(FOLLOWUP_LOG), "calc_followup_log.txt missing"
    assert os.path.getmtime(FOLLOWUP_LOG) >= os.path.getmtime(os.path.join(HERE, COHORT_FILE + ".csv")), \
        "calc_followup_log.txt is older than cohort_info_v2.csv: rerun calc_followup.py"
    with open(FOLLOWUP_LOG, encoding="utf-8") as fh:
        txt = fh.read()
    m = re.search(r"first-visit persons ([\d,]+) \(all ages\); adult \((\d+)-(\d+) y\) ([\d,]+) \(([\d,]+) unique\); "
                  r"excluded: aged <(\d+) ([\d,]+), age <0 or >\d+ ([\d,]+), age not determinable[^;]*? ([\d,]+);", txt)
    assert m, "first-visit age exclusion not found in calc_followup_log.txt"
    num_ = lambda x: int(x.replace(",", ""))  # noqa: E731
    out = dict(all=num_(m.group(1)), age_min=int(m.group(2)), age_max=int(m.group(3)), adult=num_(m.group(4)),
               minor=num_(m.group(7)), implausible=num_(m.group(8)), unknown=num_(m.group(9)))
    assert num_(m.group(5)) == out["adult"] and int(m.group(6)) == out["age_min"], m.group(0)
    assert out["all"] == out["adult"] + out["minor"] + out["implausible"] + out["unknown"], out
    return out


def tab_flow(doc):
    # Development flow in persons, as main-text Figure 1: the first urine test of every person (all ages; step 0), the age
    # restriction at that level (step 1; calc_followup_log.txt), the linkage to a blood test within +/-14 days (step 2 =
    # the adult window persons: fu_setting.csv a1, = supplement_derived.csv dev_flow order 2 = missing_table.csv
    # attrition step 0), then the attrition among the adult window persons (steps 3-5 = dev_flow orders 3-5). The
    # window-level counts of cohort_info_v2.csv (dev_flow orders 0-1: window persons of all ages, then aged >=19) and the
    # dev_flow reference rows (order >= 90) are printed as reference rows. Then the split and the KNHANES flow.
    der = rd("supplement_derived")
    c = ci_items("cohort_info_v2")
    dfl = der[der.section == "dev_flow"].sort_values("order").reset_index(drop=True)
    by = {int(o): r for o, r in zip(dfl["order"], dfl.to_dict("records"))}
    assert {0, 1, 2, 3, 4, 5} <= set(by), sorted(by)
    persons = [int(by[o]["n_persons"]) for o in sorted(by) if o < 90]
    steps_v2 = [int(v) for k, v in c.items() if k.startswith("step: ")]          # cohort.cohort_steps()
    assert steps_v2 and set(steps_v2) <= set(persons) and steps_v2[-1] == persons[-1] == int(c["analytic cohort persons"])
    mt = rd("missing_table")
    att = [int(x) for x in mt[mt.section == "attrition"]["n_persons"]]
    assert att == [int(by[o]["n_persons"]) for o in (2, 3, 4, 5)], (att, persons)
    fv = first_visit_age_exclusion()
    fs = rd("fu_setting")
    a1 = fu_section(fs, "a1").set_index("group")["n"]
    no_blood = int(a1.loc["no blood within ±14 d (urine-only encounter)"])
    adult_window = int(by[2]["n_persons"])
    assert fv["adult"] == int(a1.loc["all"]) and fv["adult"] - no_blood == int(a1.loc["blood within ±14 d"]) == adult_window, \
        (fv, dict(a1), adult_window)
    n_tr, n_te = int(c["training persons"]), int(c["test persons"])
    assert n_tr + n_te == persons[-1]
    fl = der[der.section == "knhanes_flow"].sort_values("order")
    fl_steps = fl[fl.order < 90]
    assert int(fl_steps.iloc[-1]["n"]) == int(c["KNHANES adults with 12 inputs and urate"])
    adult = n_(fv["age_min"])
    exc1 = f"{n_(fv['minor'])} (aged <{adult} years); {n_(fv['implausible'])} (implausible age)"
    if fv["unknown"]:
        exc1 += f"; {n_(fv['unknown'])} (age not determinable)"
    rows = [["Development", "0. first urine dipstick test of each person in the extraction (one test per person, all ages)",
             n_(fv["all"]), ""],
            ["Development", f"1. adults: aged {adult}-{fv['age_max']} years at the urine test (younger persons are outside the "
             "adult target population)", n_(fv["adult"]), exc1],
            ["Development", "2. blood test within ±14 days of the urine test = adult window persons", n_(adult_window),
             f"{n_(no_blood)} (no blood test within ±14 days)"]]
    for o in (3, 4, 5):
        r = by[o]
        rows.append(["Development", tidy(r["step"]), n_(r["n_persons"]),
                     "" if isnan(r["n_persons_excluded"]) else n_(r["n_persons_excluded"])])
    # reference rows: the same cohort reached from the window of all ages (cohort_info_v2.csv), and the dev_flow references
    unnum = lambda s: tidy(re.sub(r"^\d+\.\s*", "", str(s)))  # noqa: E731
    r0 = by[0]
    win_rows = next((int(r["n"]) for o, r in by.items() if o >= 90 and int(r["n_persons"]) == int(r0["n_persons"])
                     and int(r["n"]) != int(r0["n_persons"])), None)
    rows.append(["Development", "Reference: " + unnum(r0["step"]),
                 n_(r0["n_persons"]) + (f" ({n_(win_rows)} window records)" if win_rows else ""), ""])
    for o in (1, 2):
        lab_ = unnum(by[o]["step"])
        rows.append(["Development", "Reference: of these, " + lab_[:1].lower() + lab_[1:], n_(by[o]["n_persons"]),
                     "" if isnan(by[o]["n_persons_excluded"]) else n_(by[o]["n_persons_excluded"])])
    for o, r in sorted(by.items()):
        if o >= 90 and int(r["n"]) == int(r["n_persons"]):
            rows.append(["Development", tidy(r["step"]), n_(r["n_persons"]), ""])
    for lab, key in (("Training persons (80%)", "training"), ("Internal test persons (20%)", "test")):
        ev = "; ".join(f"{TLAB[t]}: {n_(c[f'{key} events {t}'])}" for t in ALLT)
        rows.append(["Development, person-level split", f"{lab}; events {ev}", n_(c[f"{key} persons"]), ""])
    kn_lab = f"External (KNHANES {study_years('knhanes')})"
    for _, r in fl_steps.iterrows():
        rows.append([kn_lab, kn_step(r["step"]), n_(r["n"]), "" if isnan(r["n_excluded"]) else n_(r["n_excluded"])])
    last = fl_steps.iloc[-1]
    m = re.match(r"(\d+)\.", str(last["step"]))
    ext_step = f"step {m.group(1)}" if m else "the last step"
    doc.table("flow", "Study flow counts for the development and external validation cohorts.",
              ["Cohort", "Step", "n", "Excluded"], rows,
              notes=["Development: every count is persons, each with one first urine dipstick test. Steps 0 and 1 count first "
                     f"urine tests; step 1 excludes {n_(fv['minor'])} persons younger than {adult} years at the urine test, "
                     f"because the models are intended for adults aged ≥{adult} years (the age range of the external validation "
                     f"cohort), and {n_(fv['implausible'])} persons with an implausible age. Step 2 excludes {n_(no_blood)} persons "
                     "without a blood test within ±14 days; the adult first-visit urine tests with and without a blood test are "
                     f"described in {P('setting_all')}. Steps 3-5 are sequential; serum urate and BMI are those recorded on the "
                     "day of the urine test. The reference rows reach the same adult window persons from the window of all ages "
                     "(a urine test enters the window once for each blood test within 14 days) and are not steps.",
                     "Training and internal test persons: one person-level 80/20 split of the development cohort, stratified on "
                     "serum urate bands (<7.0, 7.0 to <9.0, 9.0 to <10.0 and ≥10.0 mg/dL; seed 42) and shared by the three "
                     "thresholds.",
                     "KNHANES: the steps are sequential in the pooled analysis file of the annual surveys (one record per "
                     f"participant); the external validation cohort is {ext_step}."],
              source="calc_followup_log.txt (first urine tests by age, as in main-text Figure 1), fu_setting.csv (a1), "
                     "supplement_derived.csv (dev_flow, knhanes_flow), cohort_info_v2.csv (split, events and the window-level "
                     "steps), missing_table.csv (attrition; cross-check)")


def fu_section(fs, prefix):
    """Rows of the section of fu_setting.csv whose name starts with prefix ('a1', 'a2', ...)."""
    names = [s for s in fs["section"].unique() if str(s).split()[0] == prefix]
    assert len(names) == 1, (prefix, list(fs["section"].unique()))
    return fs[fs.section == names[0]]


def tab_setting_all(doc):
    fs = rd("fu_setting")
    a1, a3 = fu_section(fs, "a1"), fu_section(fs, "a3")
    if "pct_age_lt19" in fs.columns:   # adults only: no column for persons younger than 19 years
        assert (pd.concat([a1, a3])["pct_age_lt19"].fillna(0) == 0).all()
    rows = []
    for _, r in a1.iterrows():
        rows.append(["First-visit urine tests", tidy(neutral(r["group"])), n_(r["n"]), f"{f(r['age_mean'], 1)} ({f(r['age_sd'], 1)})",
                     pct_r(r["pct_male"]), pct_r(r["pct_checkup_same_day"]), pct_r(r["pct_bmi_available"]), ""])
    for _, r in a3.iterrows():
        grp = re.sub(r"(?i)^window (?:records|persons),\s*", "", str(r["group"]))    # a3 rows are adult window persons
        rows.append(["Urine tests with a blood test within ±14 days", tidy(neutral(grp)), n_(r["n"]), f(r["age_mean"], 1),
                     pct_r(r["pct_male"]), "", pct_r(r["pct_bmi_available"]), pct_r(r["pct_urate_measured"])])
    doc.table("setting_all", "Urine tests with and without a blood test within ±14 days, and BMI and serum urate availability by "
              "health-checkup status.",
              ["Population", "Group", "n", "Age, mean (SD)", "Men, %", "Same-day health checkup, %", "BMI available, %",
               "Serum urate measured, %"], rows,
              notes=[f"Adults aged ≥19 years at the urine test only (the target population of the models; {P('flow')}). "
                     "First-visit urine tests: every adult's first urine dipstick test, one per person (the study uses each "
                     "person's first urine test). Blood status from the linked laboratory file; 'no blood within ±14 d' are "
                     "urine-only encounters (persons with a blood test 15-365 days away or none within 365 days). Age and sex "
                     "from the nursing (encounter) and health-checkup files. BMI available: BMI recorded on the day of the "
                     "urine test (a health-checkup record of that day, or height and weight in the nursing record of that "
                     "day), as in the development cohort. The urine tests with a blood test within ±14 days are grouped by "
                     "whether a health-checkup record was dated on the urine-test day; serum urate measured: on the "
                     "urine-test day."],
              source="fu_setting.csv (a1, a3; calc_followup.py)")


def tab_setting_dev(doc):
    fs = rd("fu_setting")
    a2 = fu_section(fs, "a2")
    if "pct_age_lt19" in fs.columns:
        assert (a2["pct_age_lt19"].fillna(0) == 0).all()
    rows = [[tidy(neutral(r["group"])), n_(r["n"]), f"{f(r['age_mean'], 1)} ({f(r['age_sd'], 1)})",
             f(r["pct_age_ge65"], 1), f(r["pct_male"], 1), f(r["prev_ge7"], 1), f(r["prev_ge9"], 2), f(r["pct_2022_2025"], 1)]
            for _, r in a2.iterrows()]
    a4 = with_t(fu_section(fs, "a4"))
    perf = "; ".join(f"{TLAB[r['_t']]}, {tidy(neutral_lc(r['group']), cap=False)}: n = {n_(r['n'])} ({n_(r['positives'])} events), "
                     f"AUROC {est(r['auroc'], r['auroc_lo'], r['auroc_hi'])}, O:E {est(r['oe'], r['oe_lo'], r['oe_hi'], 2)}"
                     for _, r in a4.iterrows())
    doc.table("setting_dev", "Development cohort by whether a health checkup took place on the day of the urine test, and "
              "performance of the models in the internal test set by the same split.",
              ["Group", "n", "Age, mean (SD)", "Age ≥65 y, %", "Men, %", "Serum urate ≥7.0 mg/dL, %",
               "Serum urate ≥9.0 mg/dL, %", "Tested 2022-2025, %"], rows,
              notes=["Development cohort and internal test persons (adults aged ≥19 years at the urine test). The data contain "
                     "no encounter type; persons without a same-day health checkup include outpatient, inpatient and emergency "
                     "encounters, which cannot be distinguished.", f"Internal test set: {perf}."],
              source="fu_setting.csv (a2, a4; calc_followup.py)")


# section labels of fu_knhanes_flow.csv as printed: KNHANES holds one record per participant, so the appendix counts
# participants (as the manuscript and Figure 1 do)
KN_SECTION = {"records per survey year": "participants per survey year"}


def tab_knflow(doc):
    d = rd("fu_knhanes_flow")
    rows = []
    for _, r in d.iterrows():
        extra = ""
        if not isnan(r.get("n_with_urate", np.nan)):
            extra = f"with serum urate {n_(r['n_with_urate'])} ({f(r['pct_with_urate'], 1)}%)"
        elif not isnan(r.get("age_mean", np.nan)):
            extra = (f"mean age {f(r['age_mean'], 1)}; aged ≥65 {f(r['pct_age_ge65'], 1)}%; men {f(r['pct_male'], 1)}%")
        sec_lab = KN_SECTION.get(str(r["section"]), str(r["section"]))
        rows.append([sec_lab.capitalize(), tidy(neutral(r["group"])), n_(r["n"]), extra])
    # the annual survey files of the pooled file: the survey years with records (KNHANES file names HNyy_ALL)
    yrs = sorted(int(g) for g in d[d.section.astype(str) == "records per survey year"]["group"])
    assert yrs and yrs == list(range(yrs[0], yrs[-1] + 1)), yrs
    assert f"{yrs[0]}-{yrs[-1]}" == study_years("knhanes"), (yrs, study_years("knhanes"))
    doc.table("knflow", "KNHANES analysis file: participants per survey year, serum urate availability by age, and adult "
              "attrition.",
              ["Section", "Group", "n", "Detail"], rows,
              notes=[f"The pooled KNHANES analysis file concatenates the {NUM_WORD.get(len(yrs), len(yrs))} annual survey files "
                     f"(HN{yrs[0] % 100:02d}_ALL to HN{yrs[-1] % 100:02d}_ALL), one record per participant of any age, including "
                     "participants who took part only in the interview. Serum urate is part of the survey's laboratory tests from "
                     "2016 and is measured in examined participants; children younger than 10 years do not give blood samples."],
              source="fu_knhanes_flow.csv (calc_followup.py)")


# ============================================================ S2 inputs
KN_VAR = {"ua_glucose": "HE_Uglu", "ua_protein": "HE_Upro", "ua_ketone": "HE_Uket", "ua_bilirubin": "HE_Ubil",
          "ua_blood": "HE_Ubld", "ua_urobilinogen": "HE_Uro", "ua_nitrite": "HE_Unitr", "ua_pH": "HE_Uph", "ua_sg": "HE_Usg",
          "age": "age", "male": "sex", "bmi_unified": "HE_wt, HE_ht (measured)"}


def raw_map(enc, feat, code, coding="X1"):
    """Model-scale value that a KNHANES coding gives to a KNHANES raw code (encoding.csv raw_codes)."""
    raw = enc[enc.section == "raw_codes"]
    return one(raw, feature=feat, value_num=float(code))[f"map_{coding}"]


def map_desc(enc):
    """Each alternative coding (M1, M2, M3) described by the fields where it differs from X1: {coding: text}."""
    raw = enc[enc.section == "raw_codes"]
    out = {}
    for m in ("M1", "M2", "M3"):
        parts = []
        for ft in list(dict.fromkeys(raw["feature"])):
            g = raw[raw.feature == ft].sort_values("value_num")
            if not np.allclose(g[f"map_{m}"].astype(float), g["map_X1"].astype(float)):
                parts.append(f"{FEAT[ft].lower()} codes {int(g.value_num.min())}-{int(g.value_num.max())} → "
                             + ", ".join(val(v) for v in g[f"map_{m}"]))
        out[m] = ("as X1 except " + " and ".join(parts)) if parts else "identical to X1"
    return out


def tab_inputs(doc):
    enc = rd("encoding")
    cb = enc[enc.section == "codebook"]
    sm = enc[enc.section == "summary"].set_index("feature")
    assert list(sm.index) == FEAT_ORDER, list(sm.index)
    bil_dev = cb[(cb.feature == "ua_bilirubin") & np.isclose(cb.value_num.astype(float), 0.5)]
    bil_trace_dev = int(bil_dev["n_dev"].sum()) if len(bil_dev) else 0
    sgf = sg_facts()
    rows = []
    for ft in FEAT_ORDER:
        s = sm.loc[ft]
        if ft in ("age", "bmi_unified"):
            dd = 0 if ft == "age" else 1
            dev = f"Continuous; range {f(s['min_dev'], dd)}-{f(s['max_dev'], dd)}"
            ext = f"Continuous; range {f(s['min_ext'], dd)}-{f(s['max_ext'], dd)}"
        elif ft == "ua_sg":
            dev = f"{n_(s['n_levels_dev'])} levels, {f(s['min_dev'], 3)}-{f(s['max_dev'], 3)} (steps of {sgf['dev_step']})"
            ext = f"{n_(s['n_levels_ext'])} levels, {f(s['min_ext'], 3)}-{f(s['max_ext'], 3)} (steps of {sgf['ext_step']})"
        else:
            g = cb[cb.feature == ft]
            dv = [val(v) for v, n in zip(g["value_num"], g["n_dev"]) if n > 0]
            ev = [val(v) for v, n in zip(g["value_num"], g["n_ext"]) if n > 0]
            dev = "Values " + ", ".join(dv)
            ext = "Values " + ", ".join(ev)
        note = ""
        if ft == "ua_urobilinogen":
            u0, u1 = val(raw_map(enc, ft, 0)), val(raw_map(enc, ft, 1))
            note = ("Normal coded 0.5 (the normal grade of the development strips); KNHANES normal and trace codes "
                    + (f"both coded {u0}" if u0 == u1 else f"coded {u0} and {u1}") + " under X1")
        elif ft == "ua_bilirubin":
            note = (("No trace level in development; " if bil_trace_dev == 0 else "") +
                    f"KNHANES '+' (code 2) coded {val(raw_map(enc, ft, 2))} under X1")
        elif ft == "male":
            note = "1 = male, 0 = female"
        elif ft == "bmi_unified":
            note = ("Development: BMI recorded on the day of the urine test: the health-checkup BMI (height and weight measured "
                    "at the checkup) when the checkup was on that day, otherwise height and weight in the nursing record of that "
                    "day; values outside 10-70 set to missing")
        elif ft == "age":
            note = f"KNHANES age is top-coded at {knhanes_age_topcode()} years"
        rows.append([FEAT[ft], KN_VAR[ft], dev, ext, note])
    doc.table("inputs", "Input features and their coding in the development and external cohorts.",
              ["Feature", "KNHANES source variable", "Development cohort", "KNHANES adults (as fed to the models)", "Note"], rows,
              notes=["Semi-quantitative dipstick fields are on the model scale 0 = negative, 0.5 = trace, 1 = 1+, 2 = 2+, 3 = 3+ or "
                     f"more (urobilinogen: 0.5 = normal). The KNHANES coding is given in {P('mapping')}. The development dipstick "
                     "also reports leukocyte esterase, which is not an input because KNHANES does not measure it."],
              source="encoding.csv (codebook, summary, raw_codes)")


def tab_mapping(doc):
    enc = rd("encoding")
    raw = enc[enc.section == "raw_codes"]
    desc = map_desc(enc)
    sgf = sg_facts()
    rows = []
    for ft in SEMI:
        g = raw[raw.feature == ft].sort_values("value_num")
        for _, r in g.iterrows():
            rows.append([FEAT[ft], r["value"], f"{n_(r['n_ext'])} ({pc(r['pct_ext'] / 100, 2)})"] +
                        [val(r[f"map_{m}"]) for m in ("X1", "M1", "M2", "M3")])
    doc.table("mapping", "KNHANES dipstick codes and their model-scale values under the label-matched coding X1 (primary) and "
              "three alternative codings.",
              ["Field", "KNHANES raw code", "KNHANES adults, n (%)", "X1 (primary): label-matched", "M1", "M2", "M3"], rows,
              notes=["X1, the primary coding, was fixed before the models were fitted or evaluated: each KNHANES code takes the "
                     "development grade with the same label, with 4+ merged into 3+; bilirubin '+' is 1+, urobilinogen normal and "
                     "trace are 0.5 (the development normal grade), and nitrite ≥1 is positive; pH and specific gravity are passed "
                     "as recorded. " + "; ".join(f"{m}: {desc[m]}" for m in ("M1", "M2", "M3")) + ". X2 (not shown) is X1 with "
                     f"specific gravity snapped to the development grid (steps of {sgf['dev_step']}, capped at {sgf['dev_max']}). "
                     f"M1-M3 and X2 are sensitivity analyses ({T('mapsens')}). No crosswalk or paired-strip study validates any of "
                     "these codings. Age and sex are passed through; leukocyte esterase is not measured in KNHANES and is not an "
                     "input."],
              source="encoding.csv (raw_codes, summary, codebook)")


def tab_codebook(doc):
    c = ci_items("cohort_info_v2")
    enc = rd("encoding")
    cb = enc[enc.section == "codebook"]
    rows = []
    for ft in SEMI + ["ua_pH"]:
        g = cb[cb.feature == ft].sort_values("value_num")
        for _, r in g.iterrows():
            if r["n_dev"] == 0 and r["n_ext"] == 0:
                continue
            rows.append([FEAT[ft], val(r["value_num"]), f"{n_(r['n_dev'])} ({pc(r['pct_dev'] / 100, 2)})",
                         f"{n_(r['n_ext'])} ({pc(r['pct_ext'] / 100, 2)})"])
    for ft in ["age", "bmi_unified"]:
        g = cb[cb.feature == ft]
        for _, r in g.iterrows():
            if r["n_dev"] == 0 and r["n_ext"] == 0:   # e.g. the age band [0, 19), empty in both adult cohorts
                continue
            rows.append([FEAT[ft], r["value"], f"{n_(r['n_dev'])} ({pc(r['pct_dev'] / 100, 2)})",
                         f"{n_(r['n_ext'])} ({pc(r['pct_ext'] / 100, 2)})"])
    n_dev = int(c["analytic cohort persons"])
    assert int(cb[cb.feature == "male"]["n_dev"].sum()) == n_dev, "codebook counts differ from the development cohort"
    doc.table("codebook", "Distribution of every input level in the development cohort and in KNHANES adults.",
              ["Input", "Value (model scale) or band", "Development, n (%)", "KNHANES adults, n (%)"], rows,
              notes=[f"Development: development cohort of adults aged ≥19 years, one record per person (n = {n_(n_dev)}). KNHANES: "
                     "adults aged ≥19 years, inputs as fed to the models (coding X1). Specific gravity (development 0.005 steps, "
                     f"KNHANES 0.001 steps) is summarised in {P('shift')}. Bands are left-closed ([a, b))."],
              source="encoding.csv (codebook), cohort_info_v2.csv")


def tab_shift(doc):
    enc = rd("encoding")
    sm = enc[enc.section == "summary"]
    rows = []
    for ft in FEAT_ORDER:
        r = one(sm, feature=ft)
        pos_d = "" if isnan(r["pos_dev"]) else pc(r["pos_dev"])
        pos_e = "" if isnan(r["pos_ext"]) else pc(r["pos_ext"])
        abn_d = "" if isnan(r["abn_dev"]) else pc(r["abn_dev"], 2)
        abn_e = "" if isnan(r["abn_ext"]) else pc(r["abn_ext"], 2)
        psi_alt = "" if isnan(r["psi_alt"]) else f(r["psi_alt"], 3)
        rows.append([FEAT[ft], f"{f(r['mean_dev'], 3)} ({f(r['sd_dev'], 3)})", f"{f(r['mean_ext'], 3)} ({f(r['sd_ext'], 3)})",
                     f(r["psi"], 3), psi_alt, pos_d, pos_e, abn_d, abn_e])
    big = sm[sm.psi.astype(float) > 0.25].sort_values("psi", ascending=False)
    big_txt = ("Inputs with PSI >0.25: " + ", ".join(f"{FEAT[r['feature']].lower() if r['feature'] != 'ua_pH' else 'pH'} "
                                                    f"({f(r['psi'], 2)})" for _, r in big.iterrows())
               + f"; positivity by year is in {T('posyear')}.") if len(big) else "No input has PSI >0.25."
    doc.table("shift", "Input distribution shift between the development cohort and KNHANES adults.",
              ["Input", "Development mean (SD)", "KNHANES mean (SD)", "PSI", "PSI, alternative binning", "Development >0, %",
               "KNHANES >0, %", "Development above normal, %", "KNHANES above normal, %"], rows,
              notes=["PSI: population stability index with the development cohort as reference (values >0.25 are conventionally read "
                     "as a large shift). Alternative binning: specific gravity snapped to the development grid; age and BMI with open "
                     "outer deciles. Above normal: urobilinogen >0.5 on the model scale in both cohorts (normal = 0.5, so nearly "
                     "every urobilinogen value is >0); for the other fields the same as >0. " + big_txt],
              source="encoding.csv (summary)")


def tab_mapsens(doc, targets):
    k, ts = sfx(targets)
    ms = rd("mapping_sensitivity")
    md = map_desc(rd("encoding"))
    sgf = sg_facts()
    desc = {"X1": "Label-matched (primary)", **{m: md[m][0].upper() + md[m][1:] for m in ("M1", "M2", "M3")},
            "X2": f"As X1 except specific gravity snapped to the development grid (steps of {sgf['dev_step']}, capped at "
                  f"{sgf['dev_max']})"}
    rows = []
    for t in targets:
        ref = one(ms, target=t, mapping="X1")
        assert int(ref["n_participants_inputs_changed"]) == 0 and int(ref["n_pred_changed"]) == 0 and \
            abs(float(ref["delta_auroc"])) < 1e-12, "X1 must reproduce the inputs and predictions of the main analysis"
        for mp in CODINGS:
            r = one(ms, target=t, mapping=mp)
            d = "reference" if mp == "X1" else sgci(r["delta_auroc"], r["delta_lo"], r["delta_hi"], 4)
            rows.append([TLAB[t], mp, desc[mp], n_(r["n_participants_inputs_changed"]),
                         est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), d,
                         "" if mp == "X1" else pv(r["delong_p"]), f(r["oe_ratio"], 2), f(r["cal_intercept"], 2),
                         f(r["cal_slope"], 2), f(r["mean_pred"], 4)])
    doc.table("mapsens" + k, "Performance of the models in KNHANES adults under alternative dipstick codings" + ts + ".",
              ["Threshold", "Coding", "Description", "Participants with changed inputs", "AUROC (95% CI)", "ΔAUROC vs X1 (95% CI)",
               "DeLong P", "O:E", "Calibration intercept", "Calibration slope", "Mean predicted risk"], rows,
              notes=["KNHANES dipstick inputs were rebuilt from the raw survey files under each coding and scored with the models; "
                     "X1 reproduces the inputs and predictions of the main analysis. The codings are defined in "
                     f"{T('mapping')}. ΔAUROC: paired bootstrap 95% CI (1000 resamples) and DeLong P. Calibration intercept: "
                     "calibration-in-the-large (slope fixed at 1); calibration slope: logistic regression of the outcome on the "
                     "logit of the prediction."],
              source="mapping_sensitivity.csv, encoding.csv (raw_codes)")


def tab_coding(doc, targets):
    k, ts = sfx(targets)
    cs = rd("feature_tiers_coding_sensitivity")
    rows = []
    for t in targets:
        for _, r in cs[cs.target == t].iterrows():
            rows.append([TLAB[t], f"{model_label(r['model'])}, {r['tier']}",
                         f(r["auroc_X1"]), f(r["auroc_M1"]), f(r["auroc_M2"]), f(r["auroc_X1_sgclip"]),
                         f(r["mean_pred_X1"], 4), f(r["mean_pred_M1"], 4), f(r["mean_pred_M2"], 4), f(r["observed"], 4),
                         f(r["nitrite_pos_mean_pred_X1"], 4), f(r["nitrite_pos_mean_pred_M1"], 4),
                         f(r["nitrite_pos_mean_pred_M2"], 4), f(r["nitrite_pos_observed"], 4)])
    r0 = cs[cs.target == targets[0]].iloc[0]
    sgf = sg_facts()
    doc.table("coding" + k, "External results of the ensemble and of the refitted comparison models under the codings X1, M1 "
              "and M2" + ts + ".",
              ["Threshold", "Model, inputs", "AUROC, X1", "AUROC, M1", "AUROC, M2", "AUROC, X1 with SG clipped",
               "Mean predicted, X1", "Mean predicted, M1", "Mean predicted, M2", "Observed risk",
               f"Nitrite-positive adults (n = {n_(r0['nitrite_pos_n'])}): mean predicted, X1", "Same, M1", "Same, M2",
               "Same, observed risk"], rows,
              notes=[f"Codings as in {T('mapping')}; X1 with SG clipped: X1 with specific gravity clipped to the range of the "
                     f"training persons (KNHANES records values up to {sgf['ext_max']}). Every model was fitted on the training "
                     f"persons: the tier models are those of {T('tiers')}, {T('tuned')}, and the 12-input models those of "
                     f"{T('ens')}. Point estimates. SG: specific gravity."],
              source="feature_tiers_coding_sensitivity.csv, encoding.csv (summary)")


# ============================================================ S3 missing data
# Table S5 panel A shows the rows of missing_table.csv (section 'missing') except these. The CSV keeps them for the analysis
# log; the appendix describes the cohort definition only, not the history of the data processing (response letter).
MISSING_DROP = {
    # a subset of the row above it: checkup BMI of a day other than the urine-test day is not used, and the row's note
    # describes the BMI linkage of the earlier analysis
    "BMI: checkup BMI only from another day (not used) and no nursing-record height/weight",
    # 0 persons; its note repeats the age exclusion, which the Age row states with its base
    "age implausible (> 110 y)",
}
# notes replaced: the nitrite note describes how the raw result text was read (nitrite is a dipstick field like the others)
MISSING_NOTE = {"ua_nitrite": ""}


def tab_missing(doc):
    mt = rd("missing_table")
    m = mt[mt.section == "missing"]
    absent = MISSING_DROP - set(m["item"])
    assert not absent, f"missing_table.csv: rows to leave out not found (relabelled?): {absent}"
    # base of the age exclusion before this table: the window persons of all ages, as in the reference rows of Table S1
    # panel A (supplement_derived.csv dev_flow orders 0-2); the CSV note of the Age row must state the same counts
    der = rd("supplement_derived")
    dfl = der[der.section == "dev_flow"]
    by = {int(o): r for o, r in zip(dfl["order"], dfl.to_dict("records"))}
    n_win_all, n_minor, n_implaus = (int(by[0]["n_persons"]), int(by[1]["n_persons_excluded"]),
                                     int(by[2]["n_persons_excluded"]))
    n_tot = int(m["n_total"].iloc[0])
    assert n_win_all - n_minor - n_implaus == n_tot == int(by[2]["n_persons"]), (n_win_all, n_minor, n_implaus, n_tot)
    age_note = str(one(m, item="age")["note"])
    assert f"{n_minor:,} persons aged" in age_note and f"{n_implaus} persons with age" in age_note, age_note
    fv = first_visit_age_exclusion()
    rows = []
    for _, r in m.iterrows():
        it = r["item"]
        if it in MISSING_DROP:
            continue
        item = "Serum urate" if it == "urate" else FEAT.get(it, it)
        if it in MISSING_NOTE:
            note = MISSING_NOTE[it]
        elif it == "age":
            note = (f"Age at the urine test; adults aged {fv['age_min']}-{fv['age_max']} y. Excluded before this table, among "
                    f"the {n_(n_win_all)} window persons of all ages: {n_(n_minor)} aged <{fv['age_min']} y and "
                    f"{n_(n_implaus)} with an implausible age (>{fv['age_max']} y)")
        else:
            note = "" if isnan(r["note"]) or r["note"] == "dipstick field" else tidy(clean_note(r["note"]))
            note = note.replace(" (plausibility rule unchanged)", "")
            note = re.sub(r"\bthe ([\d,]+) '<1\.0' results count as measured",
                          r"the \1 persons with a '<1.0' result count as measured", note)
            note = re.sub(r"^Count of persons WITH BMI from this source$", "Persons with BMI from this source (not missing)",
                          note)
        rows.append([tidy(item), f"{n_(r['n_missing'])} ({f(r['pct_missing'], 2)})",
                     f"{n_(r['n_missing_urate_measured'])} ({f(r['pct_missing_urate_measured'], 2)})", note])
    doc.table("missing", "Missing data among the adult window persons.",
              ["Variable", f"Missing, n (%) of all {n_(n_tot)} adult window persons", "Missing, n (%) among persons with serum urate",
               "Note"],
              rows, notes=[f"Adult window persons: adults aged {fv['age_min']}-{fv['age_max']} years with a first-visit urine test "
                           "and a blood test within ±14 days (one urine test per person; the target population of the models). "
                           "Serum urate and BMI count as recorded only when they come from the day of the urine test. The rows "
                           "'BMI available from ...' count persons with BMI from that source, not missing persons.",
                           f"Window persons of all ages: persons of any age with a first-visit urine test and a blood test within "
                           f"±14 days ({n_(n_win_all)}; reference rows of {T('flow')}). Step 1 of {T('flow')} counts the same age "
                           f"exclusion among all first-visit urine tests, before the blood-test criterion ({n_(fv['minor'])} aged "
                           f"<{fv['age_min']} years and {n_(fv['implausible'])} with an implausible age"
                           + (f"; {n_(fv['unknown'])} with age not determinable" if fv["unknown"] else "") + ")."],
              source="missing_table.csv (missing), supplement_derived.csv (dev_flow), calc_followup_log.txt (first urine tests "
                     "by age)")


def tab_included(doc):
    mt = rd("missing_table")
    m = mt[mt.section == "bmi_missing_comparison"]
    rows = []
    for _, r in m.iterrows():
        it = r["item"]
        if it.endswith(", n"):
            a, b = n_(r["included_bmi_available"]), n_(r["excluded_bmi_missing"])
        elif "year" in it:
            a, b = str(int(r["included_bmi_available"])), str(int(r["excluded_bmi_missing"]))
        elif "%" in it:
            dd = 3 if r["included_bmi_available"] < 0.1 else 1
            a, b = f(r["included_bmi_available"], dd), f(r["excluded_bmi_missing"], dd)
        elif "ua_sg" in it:
            a, b = f(r["included_bmi_available"], 4), f(r["excluded_bmi_missing"], 4)
        else:
            a, b = f(r["included_bmi_available"], 2), f(r["excluded_bmi_missing"], 2)
        note = "" if isnan(r["note"]) or r["note"] == "SMD" else tidy(clean_note(r["note"]))
        rows.append([tidy(it), a, b, "" if isnan(r["smd"]) else f(r["smd"], 3), note])
    doc.table("included", "Persons included (BMI recorded on the urine-test day) versus persons excluded only because BMI was "
              "missing.",
              ["Characteristic", "Included", "Excluded (BMI missing only)", "SMD", "Note"], rows,
              notes=["Adult window persons complete on every input except BMI (and, for the last row, serum urate). "
                     "SMD: standardised mean difference "
                     "(included minus excluded); |SMD| >0.1 is conventionally read as a meaningful imbalance."],
              source="missing_table.csv (bmi_missing_comparison)")


def tab_mcar(doc):
    mc = rd("mcar")
    rows = []
    r = mc[mc.section == "little_full"].iloc[0]
    rows.append(["All adult window persons", n_(r["n"]), n_(r["n_vars"]), n_(r["n_patterns"]), f(r["chi2"], 1), n_(r["df"]),
                 f(r["log10_p"], 1), f(r["chi2_over_df"], 1), ""])
    for _, r in mc[mc.section == "little_full_sensitivity"].iterrows():
        rows.append([tidy(r["sample"]), n_(r["n"]), n_(r["n_vars"]),
                     n_(r["n_patterns"]), f(r["chi2"], 1), n_(r["df"]), f(r["log10_p"], 1), f(r["chi2_over_df"], 1), ""])
    for sec in ["little_size_series_summary", "little_subsample_summary"]:
        for _, r in mc[mc.section == sec].iterrows():
            lg = np.log10(r["p"]) if r["p"] > 0 else np.nan
            rows.append([f"Random {r['sample']} (medians)", n_(r["n"]), "", "", f(r["chi2"], 1), f(r["df"], 1),
                         "<-300" if isnan(lg) else f(lg, 1), f(r["chi2_over_df"], 2), f"{n_(r['n_rejected_at_005'])}/10"])
    pat = mc[mc.section == "patterns"].head(8)
    note0 = clean_note(mc[mc.section == "little_full"]["note"].iloc[0])
    notes = [tidy(note0.rstrip(".").replace("(see little_full_pattern_contribution)",
                                           "(mcar.csv, section little_full_pattern_contribution)")) + ".",
             "Largest missingness patterns (all adult window persons): " +
             "; ".join(f"{tidy(r['pattern'], cap=False).replace('+', ' + ')} {n_(r['count'])} ({f(r['pct'], 1)}%)"
                       for _, r in pat.iterrows()) + ".",
             "log10 P below -300 is beyond double precision."]
    doc.table("mcar", "Little's test of missing completely at random (MCAR) for the 12 inputs and serum urate.",
              ["Sample", "n", "Variables", "Patterns", "χ²", "df", "log10 P", "χ²/df", "Rejected at .05"], rows, notes=notes,
              source="mcar.csv")


def tab_mice(doc, targets):
    k, ts = sfx(targets)
    mb = rd("mice_bmi")
    dg = mb[mb.section == "imputation_diagnostics"]
    lab = {r["imputation_model"]: tidy(clean_note(r["note"])) for _, r in dg.iterrows()}
    assert set(lab) == {"full", "age_sex"} and all(lab.values()), lab
    rows = []
    for t in targets:
        for im in ["full", "age_sex"]:
            r = one(mb[mb.section == "pooled"], imputation_model=im, target=t)
            rows.append([TLAB[t], lab[im], n_(r["n"]), n_(r["positives"]), est(r["auroc"], r["auroc_lo"], r["auroc_hi"]),
                         est(r["oe"], r["oe_lo"], r["oe_hi"], 2), f(r["cal_intercept"], 2), f(r["cal_slope"], 2),
                         f(r["fmi"], 2), n_(r["m"])])
        r = one(mb[mb.section == "complete_case_internal"], target=t)
        rows.append([TLAB[t], "Reference: internal test persons (BMI recorded)", n_(r["n"]), n_(r["positives"]),
                     est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), f(r["oe"], 2), f(r["cal_intercept"], 2),
                     f(r["cal_slope"], 2), "", ""])
    dn = "; ".join(f"{lab[r['imputation_model']].split(' (')[0]}: imputation R² {f(r['imputation_r2'], 3)}, imputed BMI mean "
                   f"{f(r['imputed_bmi_mean'], 2)} (SD {f(r['imputed_bmi_sd'], 2)})" for _, r in dg.iterrows())
    obs = dg.iloc[0]
    mt = rd("missing_table")
    cmp_ = mt[mt.section == "bmi_missing_comparison"].set_index("item")
    older = float(cmp_.loc["age, mean", "excluded_bmi_missing"]) > float(cmp_.loc["age, mean", "included_bmi_available"])
    doc.table("mice" + k, "The models on persons excluded because BMI was missing, with BMI multiply imputed" + ts + ".",
              ["Threshold", "Imputation model", "n", "Events", "AUROC (95% CI)", "O:E (95% CI)", "Calibration intercept",
               "Calibration slope", "FMI", "Imputations"], rows,
              notes=["Adult window persons with a serum urate of the urine-test day but no BMI of that day, never used for "
                     "training or testing; BMI imputed with IterativeImputer (Bayesian ridge, posterior sampling), estimates "
                     "pooled with Rubin's rules (AUROC on the logit scale). FMI: fraction of missing information.",
                     f"Imputation diagnostics: {dn}; observed BMI mean {f(obs['observed_bmi_mean'], 2)} "
                     f"(SD {f(obs['observed_bmi_sd'], 2)}).",
                     "The full imputation model uses serum urate (the outcome), as recommended for multiple imputation; imputed BMI "
                     "then carries outcome information, so its AUROC is optimistic. The age-sex model is a pessimistic bound."
                     + (f" The BMI-missing population is older ({P('included')}), so this is also a transportability check."
                        if older else "")],
              source="mice_bmi.csv, missing_table.csv (bmi_missing_comparison)")


def tab_bmisource(doc):
    b = rd("fu_bmi_source")
    rows = []
    p_weight = None
    for _, r in b.iterrows():
        v = r["value"]
        try:
            fv = float(v)
            v = n_(fv) if fv == int(fv) and "%" not in str(r["item"]) else pct_r(fv)
        except (TypeError, ValueError):
            pm = re.search(r"P=([0-9.]+(?:e-?[0-9]+)?)", str(v))
            if pm and str(r["item"]).startswith("checkup weight") and "chi-square" in str(r["item"]):
                p_weight = float(pm.group(1))
            v = re.sub(r"P=([0-9.]+(?:e-?[0-9]+)?)", lambda m: pe(float(m.group(1))), str(v))
        rows.append([tidy(neutral(r["item"])), v])
    bw = rd("bmi_window")
    for _, r in bw[bw.section == "bmi_demo_vs_height_weight"].iterrows():
        rows.append([f"Agreement of the health-checkup BMI with the nursing-record BMI of the same day ({tidy(r['population'], cap=False)})",
                     f"n = {n_(r['n'])}; absolute difference ≤0.5 kg/m² in {pct_r(r['pct_abs_diff_le05'])}%; "
                     f"r = {f(r['pearson_r'], 3)}"])
    digit = ("The checkup weight, recorded to 0.1 kg, shows no digit preference (test of uniform terminal digits "
             f"{pe(p_weight)}), consistent with measurement at the checkup, which the Health Promotion Center confirmed."
             if p_weight is not None and p_weight >= 0.05 else
             "The Health Promotion Center confirmed that height and weight are measured at the checkup.")
    doc.table("bmisource", "How BMI was recorded: the health-checkup BMI recomputed from the checkup height and weight, "
              "terminal-digit analysis, and agreement with the nursing record.", ["Item", "Value"], rows,
              notes=["Development cohort. Self-reported height and weight typically show digit preference (excess of terminal "
                     "digits 0 and 5, integer weights); instrument measurements do not. " + digit],
              source="fu_bmi_source.csv (calc_followup.py), bmi_window.csv (bmi_demo_vs_height_weight)")


def tab_bmiperf(doc, targets):
    k, ts = sfx(targets)
    bw = rd("bmi_window")

    def poplab(s):
        return tidy(re.sub(r"^internal test(?:, ([^:]+))?: ", lambda m: (m.group(1) + ": ") if m.group(1) else "", str(s)))
    rows = []
    for t in targets:
        for _, r in bw[(bw.section == "performance") & (bw.target == t)].iterrows():
            rows.append([TLAB[t], poplab(r["population"]), n_(r["n"]), n_(r["positives"]),
                         est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), f(r["oe"], 2),
                         "" if isnan(r["cal_slope"]) else f(r["cal_slope"], 2), tidy(clean_note(r.get("note", np.nan)))])
        for sec in ("performance_difference", "performance_difference_by_source_age"):
            for _, r in bw[(bw.section == sec) & (bw.target == t)].iterrows():
                rows.append([TLAB[t], "Difference: " + tidy(r["population"], cap=False), f"{n_(r['n_a'])} vs {n_(r['n_b'])}",
                             f"{n_(r['positives_a'])} vs {n_(r['positives_b'])}",
                             "not estimated" if isnan(r["auroc_diff"]) or isnan(r["auroc_diff_lo"]) else
                             sgci(r["auroc_diff"], r["auroc_diff_lo"], r["auroc_diff_hi"]),
                             "", "", ("DeLong " + pe(r["p_delong"])) if not isnan(r["p_delong"]) else tidy(clean_note(r.get("note", np.nan)))])
    cm = bw[bw.section == "case_mix_by_source"]
    assert (cm["pct_age_lt19"].fillna(0) == 0).all()
    cmn = "; ".join(f"{tidy(r['population'], cap=False)}: mean age {f(r['age_mean'], 1)}, aged ≥65 {f(r['pct_age_ge65'], 1)}%, men "
                    f"{f(r['pct_male'], 1)}%" for _, r in cm.iterrows())
    doc.table("bmiperf" + k, "Performance of the models on the internal test persons by BMI source" + ts + ".",
              ["Threshold", "Subset or contrast", "n", "Events", "AUROC (95% CI) or ΔAUROC (95% CI)", "O:E", "Calibration slope", "Note"],
              rows, notes=["BMI source: the health-checkup record of the urine-test day, or height and weight in the nursing record "
                           "of that day when there was no health checkup on that day. Differences: DeLong 95% CI and P for "
                           "independent subsets; with fewer than 10 events in a group the CI and test are not reported. The "
                           "BMI-source contrast is confounded by case mix (contrasts within age bands are shown separately): "
                           f"{cmn}."],
              source="bmi_window.csv (performance, performance_difference, performance_difference_by_source_age, case_mix_by_source)")


def tab_verif_model(doc):
    vb = rd("verification_bias")
    mm = vb[vb.section == "measurement_model"]
    rows = []
    auc_full, lr_p = np.nan, np.nan
    for _, r in mm.iterrows():
        v = "" if isnan(r["value"]) else clean_note(r["value"])
        if not isnan(r["estimate"]):
            if "likelihood" in r["item"]:
                v = f"χ² {f(r['estimate'], 1)}, df {n_(r['df'])}, log10 P {f(r['log10_p'], 1)}"
                lr_p = r["p"]
            else:
                v = (v + "; " if v else "") + f"AUROC {f(r['estimate'], 3)}"
                if str(r["item"]).startswith("AUROC full model ("):
                    auc_full = float(r["estimate"])
        rows.append(["Measurement model", tidy(r["item"]), tidy(v)])
    for _, r in vb[vb.section == "odds_ratio"].iterrows():
        it = r["item"]
        if it.startswith("age_"):
            lab = f"Age {it[4:]} years (vs 40-49)"
        elif it.startswith("year_"):
            lab = f"Calendar year {it[5:]} (vs 2010)"
        else:
            unit = "per grade" if str(r["unit"]).startswith("per grade") else r["unit"]
            lab = f"{FEAT.get(it, it)} ({unit})"
        rows.append(["Odds ratio of urate being measured", lab, f"{est(r['estimate'], r['lo'], r['hi'], 3)}, {pe(r['p'])}"])
    claims = []
    if not isnan(lr_p) and float(lr_p) < 0.05:
        claims.append("Testing was not random with respect to the dipstick result, age, sex or year")
    if not isnan(auc_full) and auc_full < 0.7:
        claims.append("the model discriminates only weakly (AUROC 0.5 = no information)")
    doc.table("verif", "Verification bias: who had serum urate measured on the day of the urine test.",
              ["Part", "Item", "Estimate"], rows,
              notes=["Logistic model for P(serum urate measured on the urine-test day) among adult window persons (aged 19-110 "
                     "years) with complete dipstick fields and sex, with terms for the 9 dipstick fields, 10-year age bands (age "
                     "19 forms its own band), sex and calendar year." + (" " + "; ".join(claims) + "." if claims else "")],
              source="verification_bias.csv (measurement_model, odds_ratio)")


def ipw_lab(s):
    s = str(s)
    s = s.replace(" [P(measured) model on records with complete dipstick, age and sex (BMI ignored)]", " (primary propensity model)")
    s = re.sub(r" \[P\(measured\) model on records with all (\d+) inputs \(sensitivity\)\]",
               r" (propensity model among persons with all \1 inputs; sensitivity)", s)
    s = re.sub(r"(?i)^unweighted \((?:person-clean )?internal test\)", "Unweighted", s)
    s = s.replace("internal test IPW weights", "Weights on the internal test persons")
    return tidy(neutral(s))


def tab_ipw(doc, targets):
    k, ts = sfx(targets)
    vb = rd("verification_bias")
    rows = []
    for t in targets:
        for _, r in vb[(vb.section == "auroc") & (vb.target == t)].iterrows():
            d = "" if isnan(r["diff_vs_unweighted"]) else sgci(r["diff_vs_unweighted"], r["diff_lo"], r["diff_hi"], 4)
            rows.append([TLAB[t], ipw_lab(r["item"]), est(r["estimate"], r["lo"], r["hi"]), d, f(r["oe"], 2)])
        w = vb[(vb.section == "weights") & (vb.target == t)]
        for _, r in w.iterrows():
            rows.append([TLAB[t], ipw_lab(r["item"]), f"weights median {f(r['w_median'], 2)}, 99th-percentile cap {f(r['w_p99_cap'], 2)}, "
                         f"maximum {f(r['w_max_untruncated'], 2)}; effective n {n_(r['effective_n'])}", "", ""])
        for _, r in vb[(vb.section == "auroc_by_measurement_propensity") & (vb.target == t)].iterrows():
            rows.append([TLAB[t], f"{r['item']} (P {r['ps_range']}; mean age {f(r['age_mean'], 1)}, men {f(r['pct_male'], 1)}%)",
                         est(r["estimate"], r["lo"], r["hi"]), "", f(r["oe"], 2)])
    doc.table("ipw" + k, "Internal AUROC reweighted for the probability that serum urate was measured" + ts + ".",
              ["Threshold", "Analysis", "AUROC (95% CI)", "Difference vs unweighted (95% CI)", "O:E"], rows,
              notes=["Inverse-probability weighting (IPW) of the internal test persons with weights 1/P(measured) from the "
                     f"measurement model ({P('verif')}); the bootstrap treats the propensity model as fixed. IPW corrects only for "
                     "the observed determinants of testing. The AUROC gradient across fifths of P(measured) mostly reflects case "
                     "mix (the fifths differ in age and sex)."],
              source="verification_bias.csv (auroc, weights, auroc_by_measurement_propensity)")


# ============================================================ S4 hyperparameters
LEARN = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}


def tab_hyper(doc, targets):
    k, ts = sfx(targets)
    hp = with_t(rd("hyperparameters"))
    rows = []
    keep = hp[(hp.tuned == True) | (hp.param == "n_trees_fitted")]  # noqa: E712
    for le in ["lgb", "xgb", "cat"]:
        params = list(dict.fromkeys(keep[keep.learner == le]["param"]))
        for pnm in params:
            vals = []
            for t in targets:
                s = keep[(keep._t == t) & (keep.learner == le) & (keep.param == pnm)]
                vals.append(num(s["value"].iloc[0]) if len(s) else "")
            rows.append([LEARN[le], pnm] + vals)
    ident = bool(hp[hp._t.isin(targets)]["identical_across_folds"].all())
    doc.table("hyper" + k, "Tuned hyperparameters of the base learners" + ts + ".",
              ["Learner", "Parameter"] + [TLAB[t] for t in targets], rows,
              notes=[f"Read from the saved models (5 calibration folds; values identical across folds: {'yes' if ident else 'no'}). "
                     "Tuning: Optuna tree-structured Parzen estimator (seed 42), 25 trials per learner, mean AUROC of 3-fold "
                     "shuffled stratified cross-validation on the training persons only. n_trees_fitted is the number of trees in "
                     "the fitted model. All other parameters are library defaults (listed in hyperparameters.csv). Soft voting with "
                     "equal weights; isotonic CalibratedClassifierCV with 5 folds."],
              source="hyperparameters.csv")


# ============================================================ S5 added value
TIER_LAB = {"2F": "2F (age, sex)", "3F": "3F (age, sex, BMI)", "9F": "9F (dipstick only)", "11F": "11F (dipstick, age, sex)",
            "12F": "12F (all inputs)"}
TIER_MODELS = ["LightGBM (fixed)", "Logistic regression", "LR (age/BMI splines + age x male)"]   # GRID-tuned: tuned_per_set
CONTRAST_MEANING = {"12F-3F": "Dipstick added to age, sex and BMI", "12F-11F": "BMI added to dipstick, age and sex",
                    "11F-9F": "Age and sex added to dipstick", "11F-2F": "Dipstick added to age and sex",
                    "3F-2F": "BMI added to age and sex"}


def lgbm_fixed_text():
    from common import LGBM_FIXED as c
    return (f"{c['n_estimators']} trees, learning rate {c['learning_rate']}, {c['num_leaves']} leaves, minimum "
            f"{c['min_child_samples']} samples per leaf, subsample {c['subsample']}, column sample {c['colsample_bytree']}, "
            f"L2 {c['reg_lambda']}")


def grid_text():
    from common import GRID
    vals = {k: sorted({g[k] for g in GRID}) for k in GRID[0]}
    return (f"Grid of {len(GRID)} configurations (leaves {'/'.join(map(str, vals['num_leaves']))} x learning rate "
            f"{'/'.join(map(str, vals['learning_rate']))} x minimum child samples {'/'.join(map(str, vals['min_child_samples']))})")


def has_2f(tier):
    return "2F" in str(tier).split("-")


def tab_tiers(doc, targets):
    k, ts = sfx(targets)
    ft = rd("feature_tiers")
    tr = ft[ft.row_type == "tier"]
    rows = []
    for t in targets:
        for mod in TIER_MODELS:
            for tier in ["3F", "9F", "11F", "12F"]:
                s = tr[(tr.target == t) & (tr.model == mod) & (tr.tier == tier)]
                if not len(s):
                    continue
                ri = one(s, setting="internal")
                re_ = one(s, setting="external")
                rows.append([TLAB[t], model_label(mod), TIER_LAB[tier], est(ri["auroc"], ri["auroc_lo"], ri["auroc_hi"]),
                             f(ri["auprc"]), est(re_["auroc"], re_["auroc_lo"], re_["auroc_hi"]), f(re_["auprc"]),
                             f(re_["sbrier"])])
    r0 = tr[(tr.target == targets[0]) & (tr.setting == "internal")].iloc[0]
    doc.table("tiers" + k, "Nested feature-tier ablation with an equal modelling budget" + ts + ".",
              ["Threshold", "Model", "Inputs", "Internal AUROC (95% CI)", "Internal AUPRC", "External AUROC (95% CI)",
               "External AUPRC", "External scaled Brier"], rows,
              notes=[f"Refitted comparison models, fitted on the training persons (n = {n_(r0['n_train'])}) and evaluated on the "
                     f"internal test persons (n = {n_(r0['n'])}) and on KNHANES adults (coding X1). LightGBM (fixed "
                     f"configuration): one configuration for every tier ({lgbm_fixed_text()}). Logistic regression: standardised "
                     "main effects, C = 1. Spline logistic regression: cubic B-splines for age and BMI (5 quantile knots) plus an "
                     f"age x sex term (tiers containing age). The age-and-sex tier (2F) is in {P('tiers2f')}.",
                     "Scaled Brier: 1 - Brier/Brier of the prevalence; the refits are not calibrated, so external values are low "
                     "or negative."],
              source="feature_tiers.csv (row_type = tier)")


def contrast_rows(ct, t, settings, keep):
    """Rows of the tier and model-class contrasts of feature_tiers.csv for one target: comparison, meaning, one cell
    per setting ('Δ (95% CI); P', † when DeLong and the bootstrap CI disagree)."""
    g = ct[(ct.target == t) & keep(ct)]
    rows = []
    for comp in list(dict.fromkeys(g["comparison"])):
        s = g[g.comparison == comp]
        cells = []
        for st in settings:
            r = s[s.setting == st]
            if not len(r):
                cells.append("")
                continue
            r = r.iloc[0]
            flag = " †" if bool(r["delong_ci_disagree"]) else ""
            cells.append(f"{sgci(r['delta_auroc'], r['delta_lo'], r['delta_hi'])}; {pe(r['delong_p'])}{flag}")
        r0 = s.iloc[0]
        if r0["row_type"] == "contrast_tier":
            meaning = CONTRAST_MEANING.get(r0["tier"], tidy(r0.get("meaning", "")))
        else:
            meaning = ("Boosting minus spline logistic regression, same inputs" if "spline" in comp
                       else "Boosting minus logistic regression, same inputs")
        comp_l = comp.replace("LR (age/BMI splines + age x male)", "Spline LR").replace("Logistic regression", "LR")
        rows.append([comp_l, meaning] + cells)
    return rows


def tab_tier_contrasts(doc, targets):
    k, ts = sfx(targets)
    ft = rd("feature_tiers")
    ct = ft[ft.row_type.isin(["contrast_tier", "contrast_model"])]
    keep = lambda d: d.model.isin(TIER_MODELS + ["LightGBM - Logistic regression",  # noqa: E731
                                                  "LightGBM - LR (age/BMI splines + age x male)"]) & ~d.tier.map(has_2f)
    rows = []
    for t in targets:
        rows += [[TLAB[t]] + r for r in contrast_rows(ct, t, ["internal", "external"], keep)]
    doc.table("tiercon" + k, "AUROC differences between feature tiers and between model classes" + ts + ".",
              ["Threshold", "Comparison", "Meaning", "Internal Δ (95% CI); DeLong P", "External Δ (95% CI); DeLong P"], rows,
              notes=["Δ: AUROC of the first minus the second arm on the same persons; paired bootstrap 95% CI (1000 resamples) is "
                     "primary; DeLong P for correlated curves. † DeLong P and bootstrap CI disagree about significance at .05 "
                     f"(read the CI). LR: logistic regression. Increments over the age-and-sex tier are in {P('tiers2f')}."],
              source="feature_tiers.csv (row_type = contrast_tier, contrast_model)")


def tab_tuned(doc, targets):
    k, ts = sfx(targets)
    tp = rd("tuned_per_set")
    tu = tp[tp.row_type == "tuned"]
    cn = tp[tp.row_type == "contrast"]
    rows = []
    for t in targets:
        for tier in ["3F", "12F"]:
            ri = one(tu, target=t, tier=tier, setting="internal")
            re_ = one(tu, target=t, tier=tier, setting="external")
            rows.append([TLAB[t], TIER_LAB[tier],
                         f"leaves {num(ri['best_num_leaves'])}, learning rate {num(ri['best_learning_rate'])}, min child samples "
                         f"{num(ri['best_min_child_samples'])}", f(ri["cv_auroc"]),
                         est(ri["auroc"], ri["auroc_lo"], ri["auroc_hi"]), sg(ri["tuning_gain"], 4),
                         est(re_["auroc"], re_["auroc_lo"], re_["auroc_hi"]), sg(re_["tuning_gain"], 4)])
        ci_ = one(cn, target=t, setting="internal", tier="12F-3F")
        ce = one(cn, target=t, setting="external", tier="12F-3F")
        rows.append([TLAB[t], "Tuned 12F minus tuned 3F", "", "", sgci(ci_["delta_auroc"], ci_["delta_lo"], ci_["delta_hi"]),
                     sg(ci_["increment_change_vs_fixed"], 4), sgci(ce["delta_auroc"], ce["delta_lo"], ce["delta_hi"]),
                     sg(ce["increment_change_vs_fixed"], 4)])
    doc.table("tuned" + k, "Dipstick increment with LightGBM tuned separately for each input set" + ts + ".",
              ["Threshold", "Inputs", "Selected configuration", "3-fold CV AUROC", "Internal AUROC (95% CI)",
               "Gain vs fixed configuration, internal", "External AUROC (95% CI)", "Gain vs fixed configuration, external"], rows,
              notes=[f"{grid_text()}; 300 trees; selected by 3-fold stratified cross-validated AUROC on the training persons and "
                     "refitted on all training persons. For the contrast rows the 'gain' columns give the change in the 12F-3F "
                     "increment relative to the fixed configuration. External: KNHANES adults (coding X1)."],
              source="tuned_per_set.csv")


def tab_tiers2f(doc):
    ft = rd("feature_tiers")
    rows = []
    for t in MAIN:
        for st in ["internal", "external"]:
            for mod in TIER_MODELS:
                g = ft[(ft.target == t) & (ft.setting == st) & (ft.model == mod)]
                r = one(g[g.row_type == "tier"], tier="2F")
                rows.append([TLAB[t], st, model_label(mod), TIER_LAB["2F"], "",
                             est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), ""])
                for tier in ["11F-2F", "3F-2F"]:
                    r = one(g[g.row_type == "contrast_tier"], tier=tier)
                    rows.append([TLAB[t], st, model_label(mod), tier.replace("-", " minus "), CONTRAST_MEANING[tier],
                                 sgci(r["delta_auroc"], r["delta_lo"], r["delta_hi"], 4 if abs(r["delta_auroc"]) < 0.005 else 3),
                                 pv(r["delong_p"]) + (" †" if bool(r["delong_ci_disagree"]) else "")])
    doc.table("tiers2f", "Age-and-sex-only models (2F) and the increments over them.",
              ["Threshold", "Setting", "Model", "Input set or contrast", "Meaning", "AUROC or ΔAUROC (95% CI)", "DeLong P"], rows,
              notes=["2F: age and sex; 3F: age, sex and BMI; 11F: dipstick, age and sex. Refitted comparison models of the "
                     f"feature-tier analysis ({P('tiers')}); external: KNHANES adults (coding X1). ΔAUROC: paired bootstrap 95% CI "
                     "(1000 resamples); † DeLong P and bootstrap CI disagree."],
              source="feature_tiers.csv (tier 2F and the contrasts over it)")


def tab_withinsex(doc):
    w = with_t(rd("fu_within_sex"))
    rows = [[TLAB[r["_t"]], r["group"], neutral(r["comparison"]).replace("Spline LR", "spline LR"),
             f"{n_(r['n'])} ({n_(r['events'])})", f"{f(r['auroc_a'])} vs {f(r['auroc_b'])}",
             sgci(r["diff"], r["diff_lo"], r["diff_hi"], 4 if abs(r["diff"]) < 0.005 else 3), pv(r["delong_p"])]
            for _, r in w.iterrows()]
    # interpretive sentence only when the data show it: the within-sex increment of the fixed LightGBM 12F minus 3F
    # exceeds the pooled increment in both sexes at every threshold shown
    larger = None
    lg = w[w.comparison.astype(str).str.startswith("LightGBM (fixed)") & w.comparison.astype(str).str.contains("12F minus 3F")]
    if len(lg) and {"men", "women", "all"} <= set(lg.group):
        larger = all(float(one(lg, _t=t, group=g)["diff"]) > float(one(lg, _t=t, group="all")["diff"])
                     for t in sorted(set(lg._t)) for g in ("men", "women"))
    lead = ("Within a sex, the baseline models discriminate less well, so the same dipstick information gives a larger AUROC "
            "increment than in the pooled population. " if larger else "")
    doc.table("withinsex", "Dipstick increment within each sex in KNHANES adults.",
              ["Threshold", "Group", "Comparison", "n (events)", "AUROC, first vs second", "Difference (95% CI)", "DeLong P"], rows,
              notes=[lead + f"Refitted comparison models as in {P('tiers')}; the ensemble and the refits use the KNHANES coding X1. "
                     "95% CIs: paired bootstrap (1,000)."],
              source="fu_within_sex.csv (calc_followup.py)")


def tab_ens(doc, targets):
    k, ts = sfx(targets)
    ev = rd("ensemble_vs_simple")
    md = ev[ev.row_type == "model"]
    cn = ev[ev.row_type == "contrast"]
    arms = list(dict.fromkeys(md["model"]))
    assert sum(is_ensemble(m) for m in arms) == 1, arms
    rows = []
    for t in targets:
        for mod in arms:
            ri = one(md, target=t, setting="internal", model=mod)
            re_ = one(md, target=t, setting="external", model=mod)
            rows.append([TLAB[t], model_label(mod), est(ri["auroc"], ri["auroc_lo"], ri["auroc_hi"]), f(ri["auprc"]), f(ri["sbrier"]),
                         est(re_["auroc"], re_["auroc_lo"], re_["auroc_hi"]), f(re_["auprc"]), f(re_["sbrier"])])
        for comp in list(dict.fromkeys(cn[cn.target == t]["comparison"])):
            ri = one(cn, target=t, setting="internal", comparison=comp)
            re_ = one(cn, target=t, setting="external", comparison=comp)
            fl_i = " †" if bool(ri["delong_ci_disagree"]) else ""
            fl_e = " †" if bool(re_["delong_ci_disagree"]) else ""
            a, b = comp.split(" vs ")
            rows.append([TLAB[t], f"Δ {model_label(a)} minus {model_label(b)}",
                         f"{sgci(ri['delta_auroc'], ri['delta_lo'], ri['delta_hi'], 4)}; {pe(ri['delong_p'])}{fl_i}", "", "",
                         f"{sgci(re_['delta_auroc'], re_['delta_lo'], re_['delta_hi'], 4)}; {pe(re_['delong_p'])}{fl_e}", "", ""])
    n_tr = {int(x) for x in md[md.target.isin(targets)]["n_train"]}
    c = ci_items("cohort_info_v2")
    assert n_tr == {int(c["training persons"])}, (n_tr, c["training persons"])
    from common import GRID
    doc.table("ens" + k, "The ensemble versus simpler 12-input models fitted on the same training persons" + ts + ".",
              ["Threshold", "Model or contrast", "Internal AUROC (95% CI) or Δ; P", "Internal AUPRC", "Internal scaled Brier",
               "External AUROC (95% CI) or Δ; P", "External AUPRC", "External scaled Brier"], rows,
              notes=[f"All comparators were fitted on the {n_(n_tr.pop())} training persons of the ensemble and evaluated on the "
                     "internal test persons and on KNHANES adults (coding X1). 'LightGBM (ensemble member configuration, "
                     "isotonic)' uses the ensemble's tuned LightGBM hyperparameters in the same isotonic calibration on the "
                     "ensemble's 5 calibration folds; its fold learners reproduce the ensemble's LightGBM members, so the ensemble "
                     "minus this arm is the contribution of adding XGBoost and CatBoost. Grid-tuned LightGBM: best of "
                     f"{len(GRID)} configurations by 3-fold CV ({T('tuned')}). Δ: paired bootstrap 95% CI (primary), DeLong P; "
                     "† DeLong and bootstrap disagree. Scaled Brier is like-for-like only between isotonic-calibrated arms."],
              source="ensemble_vs_simple.csv")


# ============================================================ S6 discrimination and calibration
def tab_discrimination(doc, targets):
    k, ts = sfx(targets)
    d = rd("discrimination")
    rows = []
    for t in targets:
        for co, lab in (("internal", "Internal test persons"), ("external", "KNHANES adults")):
            r = one(d, target=t, cohort=co)
            rows.append([TLAB[t], lab, n_(r["n"]), n_(r["positives"]), pc(r["prevalence"], 2),
                         est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), est(r["auprc"], r["auprc_lo"], r["auprc_hi"]),
                         est(r["auprc_lift"], r["auprc_lift_lo"], r["auprc_lift_hi"], 1)])
    doc.table("discrim" + k, "Discrimination of the models in the internal test set and in KNHANES adults" + ts + ".",
              ["Threshold", "Cohort", "n", "Events", "Prevalence, %", "AUROC (95% CI)", "AUPRC (95% CI)", "AUPRC / prevalence (95% CI)"],
              rows, notes=["AUPRC: area under the precision-recall curve (average precision). Its value under random ranking equals the "
                           "prevalence, so AUPRC/prevalence is the gain over chance. 95% CIs: percentile bootstrap, 1000 resamples."],
              source="discrimination.csv")


def tab_calib(doc, targets):
    k, ts = sfx(targets)
    cal = rd("calibration")
    rows = []
    for t in targets:
        for co, lab in (("internal", "Internal"), ("external", "External")):
            r = one(cal, target=t, cohort=co)
            rows.append([TLAB[t], lab, pc(r["prevalence"], 2), pc(r["mean_pred"], 2), est(r["oe"], r["oe_lo"], r["oe_hi"], 2),
                         est(r["intercept"], r["intercept_lo"], r["intercept_hi"], 2), est(r["slope"], r["slope_lo"], r["slope_hi"], 2),
                         est(r["brier"], r["brier_lo"], r["brier_hi"], 4),
                         est(r["scaled_brier"], r["scaled_brier_lo"], r["scaled_brier_hi"], 3), f(r["ece_quantile10"], 4),
                         f"{n_(r['n_p0'])} ({pcr(r['share_p0'])})", n_(r["events_p0"]),
                         "" if isnan(r["obs_risk_p0_upper95"]) else pc(r["obs_risk_p0_upper95"], 2), f(r["slope_excl_p0"], 2)])
    doc.table("calib" + k, "Calibration of the isotonic-calibrated ensembles" + ts + ".",
              ["Threshold", "Cohort", "Prevalence, %", "Mean predicted, %", "O:E (95% CI)", "Calibration intercept (95% CI)",
               "Calibration slope (95% CI)", "Brier (95% CI)", "Scaled Brier (95% CI)", "ECE, 10 bins", "Predicted risk exactly 0, n (%)",
               "Events at predicted 0", "Upper 95% bound of observed risk at predicted 0, %", "Slope excluding predicted 0"], rows,
              notes=["O:E: observed-to-expected ratio (mean observed risk divided by mean predicted risk; values <1 indicate "
                     "over-prediction). Calibration intercept: maximum-likelihood offset with the slope fixed at 1 (0 = calibrated "
                     "in the large; negative = over-prediction). Calibration slope: unpenalised logistic regression of the outcome "
                     "on logit(p) (1 = ideal); p = 0 is clipped to 1e-9. Scaled Brier: 1 - Brier/[prevalence x (1 - prevalence)]; "
                     "values near 0 mean the predictions are no better than assigning everyone the observed prevalence, and "
                     "negative values mean they are worse. ECE: expected "
                     "calibration error over 10 quantile bins. 95% CIs: percentile bootstrap, 1000 resamples. Isotonic regression "
                     "returns exactly 0 for some persons; a predicted risk of 0 is not a valid individual probability (a case at "
                     "p = 0 is missed at every threshold). Internal: internal test persons; external: KNHANES adults, unweighted."],
              source="calibration.csv")


def tab_calbins(doc, targets):
    k, ts = sfx(targets)
    cc = rd("calibration_curve")
    cc = cc[(cc.model == ISO) & (cc.bins_requested == 10)]
    rows = []
    for t in targets:
        for co, lab in (("internal", "Internal"), ("external", "External")):
            for _, r in cc[(cc.target == t) & (cc.cohort == co)].sort_values("bin").iterrows():
                p2o = "no events" if r["events"] == 0 else f(r["pred_to_obs"], 2)
                rows.append([TLAB[t], lab, n_(r["bin"]), n_(r["n"]), pc(r["mean_pred"], 2),
                             pest(r["observed"], max(0.0, r["observed_lo"]), r["observed_hi"], 2), n_(r["events"]), p2o])
    doc.table("calbins" + k, "Predicted and observed risk by quantile bin of predicted risk" + ts + ".",
              ["Threshold", "Cohort", "Bin", "n", "Mean predicted, %", "Observed, % (95% CI)", "Events", "Predicted/observed"], rows,
              notes=["Ten quantile bins of the models' predictions; ties in the isotonic output can merge bins, so fewer than 10 bins "
                     "may appear. Wilson 95% CI for the observed risk."],
              source="calibration_curve.csv (isotonic models, 10 bins)")


CAL_LAB = {ISO: "Isotonic (as fitted)", "sigmoid_refit": "Sigmoid (Platt) refit", "logistic_logodds": "Logistic on log-odds"}


def tab_platt(doc, targets):
    k, ts = sfx(targets)
    pvi = rd("platt_vs_isotonic")
    rows = []
    for t in targets:
        for co in ["internal", "external"]:
            for m in [ISO, "sigmoid_refit", "logistic_logodds"]:
                r = one(pvi, target=t, cohort=co, method=m)
                ref = m == ISO
                rows.append([TLAB[t], co.capitalize(), CAL_LAB[m], est(r["auroc"], r["auroc_lo"], r["auroc_hi"]),
                             "reference" if ref else sgci(r["d_auroc_vs_isotonic"], r["d_auroc_lo"], r["d_auroc_hi"], 4),
                             f(r["brier"], 5), "reference" if ref else sgci(r["d_brier_vs_isotonic"], r["d_brier_lo"], r["d_brier_hi"], 5),
                             f(r["oe"], 2), f(r["intercept"], 2), f(r["slope"], 2), f(r["p_min"], 4), n_(r["n_distinct_p"])])
        r = one(pvi, target=t, cohort="internal", method=ISO)
        rows.append([TLAB[t], "Calibration folds", "", f"events per fold {n_(r['calfold_events_min'])}-{n_(r['calfold_events_max'])}",
                     f"isotonic levels per fold {n_(r['iso_levels_per_fold_min'])}-{n_(r['iso_levels_per_fold_max'])}",
                     f"sigmoid floor {f(r['sigmoid_floor'], 4)}", "", "", "", "", "", ""])
    doc.table("platt" + k, "Isotonic versus parametric calibration of the same fold learners" + ts + ".",
              ["Threshold", "Cohort", "Calibrator", "AUROC (95% CI)", "ΔAUROC vs isotonic (95% CI)", "Brier", "ΔBrier vs isotonic (95% CI)",
               "O:E", "Calibration intercept", "Calibration slope", "Minimum predicted risk", "Distinct predicted values"], rows,
              notes=["Sigmoid (Platt) refit: the ensembles' own 5 fold learners, with each fold's isotonic calibrator replaced by a "
                     "sigmoid calibrator fitted on the same calibration fold. scikit-learn applies the sigmoid to the soft-vote "
                     "probability, so its output cannot fall below expit(-b) (the 'sigmoid floor'). Logistic on log-odds: an "
                     "unpenalised logistic calibrator on the logit of the soft-vote probability of the same fold learners, fitted on "
                     "the same calibration folds and averaged over folds. Paired bootstrap 95% CIs (1000 resamples)."],
              source="platt_vs_isotonic.csv")


def tab_isofloor(doc):
    d = with_t(rd("fu_isotonic_floor"))
    rows = [[TLAB[r["_t"]], r["cohort"], neutral(r["variant"]), f"{n_(r['n_zero'])} ({n_(r['events_at_zero'])})", f(r["min_pred"], 4),
             f(r["auroc"], 4), f(r["brier"], 5), f(r["log_loss"], 4), f(r["oe"], 3), f(r["cal_slope"], 2)] for _, r in d.iterrows()]
    doc.table("isofloor", "Isotonic outputs of exactly 0: flooring, replacement and the logistic calibrator.",
              ["Threshold", "Cohort", "Variant", "Predictions of 0 (events)", "Minimum prediction", "AUROC", "Brier score",
               "Log loss", "O:E", "Calibration slope"], rows,
              notes=[f"The logistic calibrator uses the same fold learners and calibration folds as the isotonic models ({P('platt')}). "
                     "Log loss is computed with predictions clipped at 1e-15, so an isotonic 0 assigned to an event is heavily "
                     "penalised. Calibration slopes use predictions clipped at 1e-9 on the logit scale, which is why exact zeros "
                     "move the slope."],
              source="fu_isotonic_floor.csv (calc_followup.py), preds_calibration_methods.csv")


def tab_recal(doc, targets):
    k, ts = sfx(targets)
    rc = rd("recalibration")
    rows = []
    for t in targets:
        r = one(rc, target=t)
        rows.append([TLAB[t], n_(r["n"]), n_(r["events"]), est(r["intercept_fit"], r["intercept_fit_lo"], r["intercept_fit_hi"], 2),
                     est(r["slope_fit"], r["slope_fit_lo"], r["slope_fit_hi"], 2), est(r["oe_raw"], r["oe_raw_lo"], r["oe_raw_hi"], 2),
                     est(r["oe_after_intercept"], r["oe_after_intercept_lo"], r["oe_after_intercept_hi"], 2),
                     est(r["oe_after_logistic"], r["oe_after_logistic_lo"], r["oe_after_logistic_hi"], 2),
                     est(r["sbrier_raw"], r["sbrier_raw_lo"], r["sbrier_raw_hi"]),
                     est(r["sbrier_intercept"], r["sbrier_intercept_lo"], r["sbrier_intercept_hi"]),
                     est(r["sbrier_slope"], r["sbrier_slope_lo"], r["sbrier_slope_hi"])])
    doc.table("recal" + k, "Recalibration of the models on KNHANES adults (split-sample)" + ts + ".",
              ["Threshold", "n", "Events", "Intercept offset", "Slope if refitted", "O:E before", "O:E after intercept update",
               "O:E after logistic recalibration", "Scaled Brier before", "Scaled Brier after intercept update",
               "Scaled Brier after logistic recalibration"], rows,
              notes=["100 stratified half-splits: the recalibration is fitted on one half and evaluated on the other; values are means "
                     "with 2.5th-97.5th percentiles over the splits. Intercept update: logit(p') = a + logit(p); logistic "
                     "recalibration: logit(p') = a + b·logit(p). A slope near 1 with a large negative intercept means the external "
                     "miscalibration is a difference in the overall level of risk, which an intercept update fitted on local data "
                     "removes."],
              source="recalibration.csv")


# ============================================================ S7 utility
POL = {"youden": "Youden index (internal)", "sens80": "Sensitivity 80% (internal)", "sens90": "Sensitivity 90% (internal)",
       "top_1%": "Top 1% of predicted risk", "top_2%": "Top 2%", "top_5%": "Top 5%", "top_10%": "Top 10%", "top_20%": "Top 20%"}


def tab_oppoints(doc, targets):
    k, ts = sfx(targets)
    op = rd("operating_points")
    rows = []
    for t in targets:
        for co in ["internal", "external"]:
            for pol in POL:
                r = one(op, target=t, cohort=co, policy=pol)
                thr = f(r["threshold"], 4)
                rows.append([TLAB[t], co.capitalize(), POL[pol], thr, pest(r["sens"], r["sens_lo"], r["sens_hi"]),
                             pest(r["spec"], r["spec_lo"], r["spec_hi"]), pest(r["ppv"], r["ppv_lo"], r["ppv_hi"]),
                             pest(r["npv"], r["npv_lo"], r["npv_hi"], 2), pc(r["pos_rate"]),
                             est(r["referrals_per_case"], r["referrals_per_case_lo"], r["referrals_per_case_hi"], 1),
                             est(r["detected_per_1000"], r["detected_per_1000_lo"], r["detected_per_1000_hi"], 1)])
    doc.table("oppoints" + k, "Operating points of the models" + ts + ".",
              ["Threshold", "Cohort", "Rule", "Risk cut-off", "Sensitivity, % (95% CI)", "Specificity, % (95% CI)", "PPV, % (95% CI)",
               "NPV, % (95% CI)", "Flagged, %", "Referrals per case (95% CI)", "Cases detected per 1,000 screened (95% CI)"], rows,
              notes=["Rule-based cut-offs (Youden, 80% and 90% sensitivity) were chosen on the internal test persons (apparent "
                     "performance internally) and applied unchanged to KNHANES. Top-k rules flag the stated fraction of each cohort "
                     "by predicted risk (ties split fractionally), so their cut-off is cohort-specific and the value shown is the "
                     "cut-off in that cohort. 95% CIs: bootstrap with the cut-off held fixed. PPV/NPV: positive/negative predictive "
                     "value. Predictions are not recalibrated, so the external over-prediction shifts sensitivity up and PPV down at "
                     "fixed cut-offs."],
              source="operating_points.csv")


def tab_splithalf(doc, targets):
    k, ts = sfx(targets)
    sh = rd("splithalf_operating")
    rows = []
    for t in targets:
        for rule in ["youden", "sens80", "sens90"]:
            r = one(sh, target=t, rule=rule)
            rows.append([TLAB[t], POL[rule].replace(" (internal)", ""), est(r["threshold_mean"], r["threshold_lo"], r["threshold_hi"], 4),
                         pest(r["sens"], r["sens_lo"], r["sens_hi"]), pest(r["spec"], r["spec_lo"], r["spec_hi"]),
                         pest(r["ppv"], r["ppv_lo"], r["ppv_hi"]), pest(r["npv"], r["npv_lo"], r["npv_hi"], 2),
                         pest(r["pos_rate"], r["pos_rate_lo"], r["pos_rate_hi"])])
    doc.table("splithalf" + k, "Operating points with the cut-off chosen on KNHANES (split-half)" + ts + ".",
              ["Threshold", "Rule", "Cut-off, mean (2.5-97.5 percentiles)", "Sensitivity, %", "Specificity, %", "PPV, %", "NPV, %",
               "Flagged, %"], rows,
              notes=["The cut-off is chosen on one random half of KNHANES adults and applied to the other half; 100 stratified repeats; "
                     "mean and 2.5th-97.5th percentiles. This shows the operating characteristics after local threshold selection, "
                     "which absorbs the external calibration offset."],
              source="splithalf_operating.csv")


def strategy_label(s):
    s = neutral(s)
    return (s.replace("3F Spline LR (refit)", "Age, sex, BMI spline model (refit)")
             .replace("3F LightGBM (fixed) (refit)", "Age, sex, BMI LightGBM (refit)"))


def tab_sexop(doc):
    so = with_t(rd("fu_sex_operating"))
    rows = []
    for _, r in so.iterrows():
        thr = r["threshold"] if isinstance(r["threshold"], str) else ("none (all men)" if isnan(r["threshold"]) else f(r["threshold"], 4))
        rows.append([TLAB[r["_t"]], strategy_label(r["strategy"]), tidy(r["rule"]), thr, r["cohort"], r["group"],
                     f"{n_(r['n'])} ({n_(r['events'])})", pcr(r["flagged"]), pest(r["sens"], r["sens_lo"], r["sens_hi"]),
                     "N/A" if isnan(r["ppv"]) else pest(r["ppv"], r["ppv_lo"], r["ppv_hi"]), pcr(r["spec"]),
                     "" if isnan(r["share_flagged_men"]) else pcr(r["share_flagged_men"])])
    doc.table("sexop", "Operating points overall and by sex, including reference strategies.",
              ["Threshold", "Strategy", "Rule", "Probability threshold", "Cohort", "Group", "n (events)", "Flagged, %",
               "Sensitivity, % (95% CI)", "PPV, % (95% CI)", "Specificity, %", "Men among flagged, %"], rows,
              notes=["Thresholds for the ensemble were chosen on the internal test persons (internal rows are therefore apparent "
                     "performance) and applied unchanged to KNHANES; top-k thresholds are cohort-specific with fractional flagging "
                     "of ties; sex-specific thresholds reach the stated internal sensitivity within each sex. Reference models were "
                     "refitted on the training persons and their thresholds chosen on the internal test persons. 95% CIs: bootstrap "
                     "within each group (1,000), thresholds fixed. Predictions not recalibrated."],
              source="fu_sex_operating.csv (calc_followup.py)")


def tab_oof(doc):
    o = with_t(rd("fu_oof_thresholds"))
    coh = {"training OOF": "Training persons, out-of-fold"}
    rows = [[TLAB[r["_t"]], tidy(r["rule"]), f"{f(r['threshold_oof'], 4)} ({f(r['threshold_testset'], 4)})",
             coh.get(r["cohort"], neutral(r["cohort"])), r["group"], f"{n_(r['n'])} ({n_(r['events'])})",
             pest(r["sens"], r["sens_lo"], r["sens_hi"]), pc(r["spec"]),
             "N/A" if isnan(r["ppv"]) else pest(r["ppv"], r["ppv_lo"], r["ppv_hi"]), pc(r["flagged"])] for _, r in o.iterrows()]
    doc.table("oof", "Thresholds chosen on out-of-fold training predictions of the ensembles.",
              ["Threshold", "Rule", "Probability threshold, out-of-fold (test-set choice)", "Cohort", "Group", "n (events)",
               "Sensitivity, % (95% CI)", "Specificity, %", "PPV, % (95% CI)", "Flagged, %"], rows,
              notes=["Each training person was scored by the model of the calibration fold whose base learners had not been fitted "
                     "on that person (the isotonic map of that fold had been fitted on these persons, so the predictions are not "
                     "fully out of sample). Thresholds were chosen on these out-of-fold predictions and applied to the internal test "
                     "persons and to KNHANES adults; the internal values are therefore not apparent performance. 95% CIs: bootstrap "
                     "(1,000; 200 for the training persons)."],
              source="fu_oof_thresholds.csv (calc_followup.py)")


def tab_recalop(doc):
    r_ = with_t(rd("fu_recal_operating"))
    rows = [[TLAB[r["_t"]], f(r["recalibrated_threshold"], 3), f(r["model_scale_threshold"], 4), r["group"], pc(r["sens"]),
             pc(r["ppv"]), pc(r["flagged"])] for _, r in r_.iterrows()]
    doc.table("recalop", "Operating points in KNHANES adults at fixed thresholds on the recalibrated risk scale.",
              ["Threshold", "Recalibrated risk threshold", "Equivalent threshold on the model scale", "Group", "Sensitivity, %",
               "PPV, %", "Flagged, %"], rows,
              notes=[f"Predictions after the cross-fitted intercept update of {T('nbref')} (mean over 100 random splits). The "
                     "model-scale equivalent uses the full-sample intercept."],
              source="fu_recal_operating.csv (calc_followup.py)")


NB_SHOW = {">=7.0": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40], ">=9.0": [0.004, 0.01, 0.02, 0.03, 0.05, 0.10],
           ">=10.0": [0.002, 0.004, 0.01, 0.02, 0.05]}


def tab_dca(doc, targets):
    k, ts = sfx(targets)
    rg = rd("decision_curve_ranges")
    dc = rd("decision_curve")
    rows = []
    for t in targets:
        for co in ["internal", "external"]:
            g = rg[(rg.target == t) & (rg.cohort == co)]
            a = one(g, criterion="point estimate > max(treat-all, treat-none)")
            b = one(g, criterion="95% CI lower bound > 0")
            rows.append([TLAB[t], co.capitalize(), f"{f(a['grid_min'], 3)}-{f(a['grid_max'], 3)} (step {f(a['grid_step'], 3)})",
                         f"{a['ranges']} ({n_(a['n_grid_beating'])}/{n_(a['n_grid'])})",
                         f"{b['ranges']} ({n_(b['n_grid_beating'])}/{n_(b['n_grid'])})", "", "", "", ""])
            for thr in NB_SHOW[t]:
                s = dc[(dc.target == t) & (dc.cohort == co) & np.isclose(dc.threshold, thr)]
                if not len(s):
                    continue
                r = s.iloc[0]
                rows.append([TLAB[t], co.capitalize(), f"threshold probability {f(thr, 2 if t == '>=7.0' else 3)}", "", "",
                             est(r["nb_model"], r["nb_model_lo"], r["nb_model_hi"], 4), f(r["nb_all"], 4),
                             sgci(r["delta_vs_best_default"], r["delta_lo"], r["delta_hi"], 4), f(r["interventions_avoided_per_100"], 1)])
    doc.table("dca" + k, "Decision-curve analysis of the models" + ts + ".",
              ["Threshold", "Cohort", "Grid or threshold probability", "Range where net benefit exceeds both defaults (points)",
               "Range where the 95% CI of the excess is above 0 (points)", "Net benefit of the model (95% CI)", "Net benefit, treat all",
               "Excess over the better default (95% CI)", "Interventions avoided per 100"], rows,
              notes=["Net benefit = TP/n - FP/n x pt/(1 - pt) at threshold probability pt; defaults are treat-all and treat-none (net "
                     "benefit 0). Grids suit each outcome's prevalence. Predictions not recalibrated; bootstrap 95% CIs (1000 "
                     "resamples). Ranges are listed as grid points where the criterion holds."],
              source="decision_curve_ranges.csv, decision_curve.csv")


def tab_nbref(doc):
    nb = with_t(rd("fu_net_benefit"))
    keep = {">=7.0": [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], ">=9.0": [0.002, 0.004, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]}
    rows = []
    for t in MAIN:
        g = nb[nb._t == t]
        for x in keep[t]:
            r = g.iloc[(g.threshold - x).abs().argmin()]
            rows.append([TLAB[t], f(r["threshold"], 3), f(1000 * r["ensemble_unrecalibrated"], 1),
                         f(1000 * r["ensemble_recalibrated_crossfit"], 1), f(1000 * r["3F_splineLR_unrecalibrated"], 1),
                         f(1000 * r["3F_splineLR_recalibrated_crossfit"], 1), f(1000 * r["test_all_men"], 1),
                         f(1000 * r["treat_all"], 1),
                         f"{sg(1000 * (r['ensemble_recalibrated_crossfit'] - r['test_all_men']), 1)} "
                         f"({ci(1000 * r['recal_minus_men_lo'], 1000 * r['recal_minus_men_hi'], 1)})"])
    rg = with_t(rd("fu_net_benefit_ranges"))
    rtxt = "; ".join(f"{TLAB[r['_t']]}, {neutral_lc(r['criterion'])}: {r['ranges']}" for _, r in rg.iterrows())
    grid = "; ".join(f"{val(g_['grid_min'].min())}-{val(g_['grid_max'].max())} at {TLAB[t]}"
                     for t, g_ in rg.groupby("_t", sort=False))
    doc.table("nbref", "External net benefit per 1,000 adults: the ensemble as validated and after a cross-fitted intercept "
              "update, the age-sex-BMI model, and 'test all men'.",
              ["Threshold", "Threshold probability", "Ensemble, as validated", "Ensemble, intercept updated",
               "Age, sex, BMI model, as validated", "Age, sex, BMI model, intercept updated", "Test all men", "Treat all",
               "Updated ensemble minus test all men (95% CI)"], rows,
              notes=["Intercept update cross-fitted in KNHANES: fitted on one random half and applied to the other, both ways, over "
                     "100 random splits (mean net benefit shown); the CI of the difference is a person-level bootstrap (1,000) of "
                     "one cross-fitted split. The age, sex and BMI model is the refitted spline logistic model. Threshold ranges "
                     f"(grid {grid}): {rtxt}."],
              source="fu_net_benefit.csv, fu_net_benefit_ranges.csv (calc_followup.py), decision_curve.csv")


def tab_utility(doc, targets):
    k, ts = sfx(targets)
    u = rd("utility_12F_vs_3F")
    assert set(u["scenario"].astype(str)) == {"primary"}, sorted(set(u["scenario"].astype(str)))
    rows = []
    for t in targets:
        g = u[u.target == t]
        for co in ["internal", "external"]:
            gc = g[g.cohort.astype(str).str.startswith(co)]
            s = gc[(gc.analysis == "discrimination") & (gc.metric == "auroc")]
            if not len(s):
                continue
            r = s.iloc[0]
            pr = gc[(gc.analysis == "discrimination") & (gc.metric == "auprc")].iloc[0]
            oe = gc[gc.analysis == "calibration"]
            oes = f"{f(oe.iloc[0]['value_12F'], 2)} vs {f(oe.iloc[0]['value_3F'], 2)}" if len(oe) else ""
            rows.append([TLAB[t], "Internal test persons" if co == "internal" else "External", "AUROC",
                         f"{f(r['value_12F'])} vs {f(r['value_3F'])}", sgci(r["diff"], r["diff_lo"], r["diff_hi"]),
                         f"AUPRC {f(pr['value_12F'])} vs {f(pr['value_3F'])}; O:E {oes}"])
        tk = g[(g.analysis == "topk_referral") & (g.metric == "detected_per_1000")]
        for _, r in tk.iterrows():
            rows.append([TLAB[t], "External", f"Cases detected per 1,000, top {pc(r['x'], 0)}%",
                         f"{f(r['value_12F'], 1)} vs {f(r['value_3F'], 1)}", sgci(r["diff"], r["diff_lo"], r["diff_hi"], 1), ""])
        st = g[(g.analysis == "net_benefit_summary") & (g.metric == "statement")]
        if len(st):
            stx = st.iloc[0]["ranges"].replace(" -> ", ": ").replace("at pt ", "at threshold probability ")
            rows.append([TLAB[t], "External", "Net benefit (summary)", "", "", stx])
    doc.table("utility" + k, "What the dipstick adds in clinical-utility terms: LightGBM 12F versus 3F" + ts + ".",
              ["Threshold", "Cohort", "Measure", "12F vs 3F", "Difference (95% CI)", "Other"], rows,
              notes=["Both models are the equal-budget LightGBM (fixed configuration) fitted on the training persons "
                     f"({T('tiers' + k)}). Top-k: the stated fraction of KNHANES adults with the highest predicted risk "
                     "is referred. The cases-per-1,000 differences and the net-benefit statements use 400 paired bootstrap resamples; "
                     "the AUROC differences use 1,000. Net-benefit statements: threshold ranges on the decision-curve grid where 12F "
                     "beats both defaults and its advantage over 3F has a paired 95% CI excluding 0."],
              source="utility_12F_vs_3F.csv")


# ============================================================ S8 transportability
TMOD = {"Ensemble-12F (tuned, refit)": "Ensemble, 12 inputs (tuned hyperparameters, refit)", "LGBM-12F": "LightGBM 12F (fixed)",
        "LGBM-3F": "LightGBM 3F (fixed)", "LR-12F": "Logistic regression 12F"}
TCON = "LGBM 12F minus 3F (AUROC difference)"


def temporal_periods(tp):
    """(training period, test period) of the temporal design (temporal.csv, internal test rows)."""
    r = tp[(tp.design == "temporal") & (tp.evaluation == "internal test")].iloc[0]
    return str(r["train_period"]), str(r["test_period"])


def tab_temporal(doc, targets):
    k, ts = sfx(targets)
    tp = rd("temporal")
    unknown = set(tp["model"]) - set(TMOD) - {TCON}
    assert not unknown, f"temporal.csv models without a label: {unknown}"
    p_tr, p_te = temporal_periods(tp)
    rows = []
    for t in targets:
        for design in ["temporal", "random (reference)"]:
            g = tp[(tp.target == t) & (tp.design == design)]
            dl = f"Temporal: {p_tr} → {p_te}" if design == "temporal" else "Random person-level 80/20 (reference)"
            for mod in list(TMOD) + [TCON]:
                for evn in ["internal test", "external KNHANES >=19"]:
                    s = g[(g.model == mod) & (g.evaluation == evn)]
                    if not len(s):
                        continue
                    r = s.iloc[0]
                    lab_e = "Internal test" if evn == "internal test" else "External"
                    if mod == TCON:
                        rows.append([TLAB[t], dl, "LightGBM 12F minus 3F", lab_e, n_(r["n"]), n_(r["positives"]),
                                     sgci(r["auroc"], r["auroc_lo"], r["auroc_hi"]) + f"; {pe(r['delong_p'])}", "", "", "", "", ""])
                        continue
                    pr = r["prevalence"] / r["prevalence_train"]
                    mp = r["mean_pred"] / r["prevalence_train"]
                    rows.append([TLAB[t], dl, TMOD[mod], lab_e, n_(r["n"]), n_(r["positives"]),
                                 est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), f(r["cal_intercept"], 2), f(r["cal_slope"], 2),
                                 f(r["oe"], 2), f(pr, 3), f(mp, 3)])
    g0 = tp[(tp.target == targets[0]) & (tp.design == "temporal")].iloc[0]
    late = p_te.split("-")[0]
    doc.table("temporal" + k, "Temporal holdout of refitted models in the development cohort" + ts + ".",
              ["Threshold", "Design", "Model", "Evaluation", "n", "Events", "AUROC (95% CI) or Δ (95% CI); P", "Calibration intercept",
               "Calibration slope", "O:E", "Prevalence / training prevalence", "Mean predicted / training prevalence"], rows,
              notes=[f"Development cohort split by the year of the urine test: training {p_tr} (n = {n_(g0['n_train'])}), test "
                     f"{p_te}; the random person-level 80/20 split of the same cohort (the split of the models) is the reference with "
                     "a similar training size. The ensemble keeps the tuned hyperparameters of the models (tuned on the training "
                     f"persons of the random split, which include persons tested in {late} or later) and is refitted, not re-tuned; "
                     "on the random split it is refitted on the models' own calibration folds and so re-creates the models. O:E = "
                     "(prevalence ratio) / (mean-predicted ratio): the first factor is the change in outcome frequency, the second "
                     "the change in predicted risk (covariate shift through the inputs). The test period begins just after a "
                     f"laboratory reporting change ({T('levels')}), so this design measures time plus measurement change, not "
                     "temporal drift alone."],
              source="temporal.csv")


def tab_intperiod(doc):
    d = with_t(rd("fu_internal_period"))
    rows = [[TLAB[r["_t"]], r["period"], f"{n_(r['n'])} ({n_(r['events'])})", pc(r["prevalence"], 2), pc(r["mean_pred"], 2),
             est(r["oe"], r["oe_lo"], r["oe_hi"], 2), est(r["auroc"], r["auroc_lo"], r["auroc_hi"])] for _, r in d.iterrows()]
    doc.table("intperiod", "The models in the internal test set by period and year of the urine test.",
              ["Threshold", "Period", "n (events)", "Prevalence, %", "Mean predicted risk, %", "O:E (95% CI)", "AUROC (95% CI)"],
              rows, notes=[f"The models were trained on persons tested in the whole period {study_years('development')} (random "
                           "person-level split); 95% CIs (bootstrap, 1,000) for the two periods and the total only."],
              source="fu_internal_period.csv (calc_followup.py), dipstick_by_year.csv (years)")


def tab_periods(doc, targets):
    k, ts = sfx(targets)
    ky = rd("knhanes_by_year")
    rows = []
    for t in targets:
        for _, r in ky[ky.target == t].iterrows():
            d = "" if isnan(r["auroc_diff_vs_2016_2018"]) else \
                f"{sgci(r['auroc_diff_vs_2016_2018'], r['diff_lo'], r['diff_hi'])}; {pe(r['diff_p'])}"
            rows.append([TLAB[t], r["period"], n_(r["n"]), n_(r["positives"]), pc(r["prevalence"], 2),
                         est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), d, est(r["oe"], r["oe_lo"], r["oe_hi"], 2),
                         f(r["cal_intercept"], 2), f(r["cal_slope"], 2)])
    doc.table("periods" + k, "Performance of the models in KNHANES adults by survey period" + ts + ".",
              ["Threshold", "Survey years", "n", "Events", "Prevalence, %", "AUROC (95% CI)", "ΔAUROC vs 2016-2018 (95% CI); P",
               "O:E (95% CI)", "Calibration intercept", "Calibration slope"], rows,
              notes=["Differences between independent periods: DeLong for independent curves. Unweighted."],
              source="knhanes_by_year.csv")


def tab_knyear(doc):
    d = with_t(rd("fu_knhanes_year"))
    rows = [[TLAB[r["_t"]], str(int(r["year"])), f"{n_(r['n'])} ({n_(r['events'])})", pc(r["prevalence"], 2),
             pc(r["mean_pred"], 2), est(r["oe"], r["oe_lo"], r["oe_hi"], 2), f(r["auroc"])] for _, r in d.iterrows()]
    doc.table("knyear", "Calibration and discrimination of the models in KNHANES adults by survey year.",
              ["Threshold", "Survey year", "n (events)", "Prevalence, %", "Mean predicted risk, %", "O:E (95% CI)", "AUROC"], rows,
              notes=["Unweighted. Dipstick positivity by survey year is given with the reporting-change results "
                     f"({T('posyear')}, {T('levels')})."],
              source="fu_knhanes_year.csv (calc_followup.py)")


def tab_posyear(doc):
    dy = rd("dipstick_by_year")
    rows = []
    for _, r in dy.iterrows():
        coh = "Development" if str(r["cohort"]).startswith("development") else "KNHANES adults"
        rows.append([coh, r["period"], n_(r["n"]), pc(r["pos_glucose"]), pc(r["pos_protein"]), pc(r["pos_ketone"]),
                     pc(r["pos_bilirubin"], 2), pc(r["pos_blood"]), pc(r["pos_urobilinogen_above_normal"], 2), pc(r["pos_nitrite"], 2),
                     f(r["mean_sg"], 4), f(r["mean_age"], 1), pc(r["male_frac"]), f(r["mean_bmi"], 1), pc(r["prev_urate_ge7"]),
                     pc(r["prev_urate_ge9"], 2), pc(r["prev_urate_ge10"], 2)])
    doc.table("posyear", "Dipstick positivity, demographics and urate prevalence by calendar year (development) and survey year "
              "(KNHANES).",
              ["Cohort", "Year(s)", "n", "Glucose >0, %", "Protein >0, %", "Ketone >0, %", "Bilirubin >0, %", "Blood >0, %",
               "Urobilinogen above normal, %", "Nitrite positive, %", "Mean SG", "Mean age", "Men, %",
               "Mean BMI", "Urate ≥7.0, %", "Urate ≥9.0, %", "Urate ≥10.0, %"], rows,
              notes=["Development: development cohort by year of the urine test; KNHANES: external validation cohort by survey year, "
                     "inputs coded X1. Positivity is any value above 0 (trace or more). Urobilinogen above normal: ≥1.0 on the model "
                     "scale (normal = 0.5 in both cohorts). SG: specific gravity."],
              source="dipstick_by_year.csv")


def level_share(frame, feat, lev_pred, yr):
    """Share (%) of persons of one year at the levels selected by lev_pred (supplement_derived.csv
    dipstick_levels_by_year); NaN when the year has no row for the field."""
    if not len(frame[(frame.feature == feat) & (frame.year == yr)]):
        return np.nan
    s = frame[(frame.feature == feat) & (frame.year == yr) & lev_pred(frame.level)]
    return float(s["pct"].sum())  # a level that is never coded contributes 0


TRACE = lambda x: np.isclose(x, 0.5)  # noqa: E731
ONE_PLUS = lambda x: np.isclose(x, 1.0)  # noqa: E731


def reporting_change(dev):
    """Data check of the statements about the reporting change: {field: True when the share at the named level in each
    year from 2022 is below a tenth of its 2021 value (a drop), False otherwise}."""
    out = {}
    for feat, pred in (("ua_ketone", TRACE), ("ua_bilirubin", ONE_PLUS), ("ua_glucose", TRACE), ("ua_protein", TRACE),
                       ("ua_blood", TRACE)):
        before = level_share(dev, feat, pred, 2021.0)
        after = [level_share(dev, feat, pred, float(y)) for y in sorted(dev.year.unique()) if y >= 2022]
        out[feat] = bool(after) and not isnan(before) and before > 0 and all(a < 0.1 * before for a in after)
    return out


def tab_levels(doc):
    der = rd("supplement_derived")
    lv = der[der.section == "dipstick_levels_by_year"]
    is_dev = lv.cohort.astype(str).str.startswith("development")
    dev, ext = lv[is_dev], lv[~is_dev]
    enc = rd("encoding")
    rows = []
    for name, frame in (("Development", dev), ("KNHANES adults", ext)):
        for yr in sorted(frame.year.unique()):
            g = lambda feat, pred: f(level_share(frame, feat, pred, yr), 2)  # noqa: E731
            rows.append([name, str(int(yr)), n_(frame[(frame.year == yr) & (frame.feature == "ua_ketone")]["n_year"].iloc[0]),
                         g("ua_ketone", TRACE), g("ua_ketone", lambda x: x >= 1),
                         g("ua_bilirubin", ONE_PLUS), g("ua_bilirubin", lambda x: x >= 2),
                         g("ua_glucose", TRACE), g("ua_glucose", lambda x: x >= 1),
                         g("ua_protein", TRACE), g("ua_blood", TRACE)])
    kt = {int(yr): level_share(ext, "ua_ketone", TRACE, yr) for yr in sorted(ext.year.unique())}
    yrs = sorted(kt)
    zero_from = next((y for y in yrs if all(kt[z] == 0 for z in yrs if z >= y)), None)
    kn_note = ("" if zero_from is None else
               f" KNHANES dipstick results also vary between survey years (for example, no KNHANES adult has a ketone trace "
               f"result from {zero_from} onward), but without a step at 2022; the external validation pools all survey years.")
    rc = reporting_change(dev)
    dropped = [FEAT[k_].lower() + (" trace" if k_ != "ua_bilirubin" else " 1+") for k_ in ("ua_ketone", "ua_bilirubin") if rc[k_]]
    stable = [FEAT[k_].lower() for k_ in ("ua_glucose", "ua_protein", "ua_blood") if not rc[k_]]
    change = (f"In the development data {' and '.join(dropped)} results almost disappear from 2022"
              + (f", while {', '.join(stable[:-1]) + ' and ' + stable[-1] if len(stable) > 1 else stable[0]} trace results show "
                 "no comparable drop" if stable else "") + ". This pattern indicates a change in the analyser, strip or result "
              "coding at the development laboratory, which took place at the end of 2021 (transition from July 2021 to January "
              "2022). " if dropped else "")
    bil2 = val(raw_map(enc, "ua_bilirubin", 2))
    doc.table("levels", "Dipstick results by level and year: the laboratory reporting change in the development data.",
              ["Cohort", "Year", "n", "Ketone trace, %", "Ketone ≥1+, %", "Bilirubin 1+ (model value 1), %",
               "Bilirubin ≥2 (model value), %", "Glucose trace, %", "Glucose ≥1+, %", "Protein trace, %", "Blood trace, %"], rows,
              notes=["Share of persons at each level of the model scale (0.5 = trace). " + change +
                     f"KNHANES bilirubin '+' is coded {bil2} under X1." + kn_note],
              source="supplement_derived.csv (dipstick_levels_by_year; computed by calc_supplement_derived.py from the analysis "
                     "cache), encoding.csv (raw_codes)")


def tab_ketone(doc):
    d = with_t(rd("fu_ketone"))
    rows = []
    for _, r in d.iterrows():
        if r["analysis"] == "O:E by ketone status":
            rows.append([TLAB[r["_t"]], "O:E by ketone status", r["cohort"], tidy(r["group"]), f"{n_(r['n'])} ({n_(r['events'])})",
                         f"{pc(r['prevalence'], 2)} / {pc(r['mean_pred'], 2)}", est(r["oe"], r["oe_lo"], r["oe_hi"], 2)])
        else:
            rows.append([TLAB[r["_t"]], tidy(neutral(r["analysis"])), r["cohort"], tidy(r["group"]), f"{n_(r['n'])} ({n_(r['events'])})",
                         f"{pc(r['mean_pred_before'], 2)} → {pc(r['mean_pred_after'], 2)} ({sg(r['rel_change_pct'], 1)}%)",
                         f"{f(r['oe_before'], 2)} → {f(r['oe_after'], 2)}; AUROC {f(r['auroc_before'])} → {f(r['auroc_after'])}"
                         if not isnan(r["auroc_before"]) else f"{f(r['oe_before'], 2)} → {f(r['oe_after'], 2)}"])
    sm = rd("encoding")
    sm = sm[sm.section == "summary"].set_index("feature")
    kpos_d, kpos_e = sm.loc["ua_ketone", "pos_dev"], sm.loc["ua_ketone", "pos_ext"]
    doc.table("ketone", "Ketone and calibration: O:E by ketone status and the effect of setting ketone results to negative.",
              ["Threshold", "Analysis", "Cohort", "Group", "n (events)", "Prevalence / mean predicted, %, or mean predicted before → after",
               "O:E (95% CI) or O:E before → after"], rows,
              notes=[f"Development ketone trace results almost disappeared from 2022 (laboratory reporting change at the end of "
                     f"2021, {T('levels')}), and ketone "
                     f"positivity was {pc(kpos_d)}% in the development cohort against {pc(kpos_e)}% in KNHANES adults. Counterfactual "
                     "rows re-score the same persons with the models after changing only the ketone input."],
              source="fu_ketone.csv (calc_followup.py), encoding.csv (summary)")


def tab_sgrestrict(doc):
    d = with_t(rd("fu_sg_restrict"))
    rows = [[TLAB[r["_t"]], tidy(r["group"]), f"{n_(r['n'])} ({n_(r['events'])})", f(r["pct_male"], 1),
             est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), est(r["oe"], r["oe_lo"], r["oe_hi"], 2)] for _, r in d.iterrows()]
    sgf = sg_facts()
    doc.table("sgrestrict", "KNHANES adults by specific gravity range.",
              ["Threshold", "Group", "n (events)", "Men, %", "AUROC (95% CI)", "O:E (95% CI)"], rows,
              notes=[f"KNHANES records specific gravity in steps of {sgf['ext_step']} up to {sgf['ext_max']}; the development "
                     f"strips report steps of {sgf['dev_step']} up to {sgf['dev_max']}. KNHANES measures specific gravity by "
                     "reflection refractometry and reads the other dipstick fields on Sysmex MEDITAPE UC-9A strips with a UC-3500 "
                     "analyzer (2022-2024; KDCA user guide for the 9th cycle); the development laboratory's instrument is not "
                     "documented, and its resolution is consistent with a reagent-pad reading."],
              source="fu_sg_restrict.csv (calc_followup.py), encoding.csv (summary, codebook)")


def tab_survey(doc, targets):
    k, ts = sfx(targets)
    sw = rd("survey_weighted")
    rows = []
    for t in targets:
        r = one(sw, target=t)
        rows.append([TLAB[t], n_(r["n"]), n_(r["positives"]),
                     pest(r["prevalence_unweighted"], r["prevalence_unweighted_lo"], r["prevalence_unweighted_hi"], 2),
                     pest(r["prevalence_weighted"], r["prevalence_weighted_lo"], r["prevalence_weighted_hi"], 2),
                     pc(r["mean_pred_unweighted"], 2), pc(r["mean_pred_weighted"], 2),
                     est(r["oe_unweighted"], r["oe_unweighted_lo"], r["oe_unweighted_hi"], 2),
                     est(r["oe_weighted"], r["oe_weighted_lo"], r["oe_weighted_hi"], 2),
                     est(r["auroc_unweighted"], r["auroc_unweighted_lo"], r["auroc_unweighted_hi"]),
                     est(r["auroc_weighted"], r["auroc_weighted_lo"], r["auroc_weighted_hi"])])
    r0 = sw.iloc[0]
    doc.table("survey" + k, "Unweighted and survey-weighted external results" + ts + ".",
              ["Threshold", "n", "Events", "Prevalence, unweighted, % (95% CI)", "Prevalence, weighted, % (95% CI)",
               "Mean predicted, unweighted, %", "Mean predicted, weighted, %", "O:E, unweighted (95% CI)", "O:E, weighted (95% CI)",
               "AUROC, unweighted (95% CI)", "AUROC, weighted (95% CI)"], rows,
              notes=[f"Weights: KNHANES examination weight: {clean_note(r0['weight_note'])}.",
                     f"{clean_note(r0['variance_note']).replace('NOT used', 'not used')}. The weighted estimates are a sensitivity "
                     "analysis; the unweighted complete-case sample is a national survey sample, not a nationally representative one."],
              source="survey_weighted.csv")


# ============================================================ S9 subgroups
SG_ORDER = ["All", "Age 19-39", "Age 40-59", "Age 60-79", "Age >=80", "Male", "Female"]


def sglab(s):
    return s.replace(">=", "≥").replace("Male", "Men").replace("Female", "Women")


def tab_subgroups(doc, targets):
    k, ts = sfx(targets)
    sgd = rd("subgroups")
    sgd["_side"] = sgd["cohort"].map(cohort_side)
    rows = []
    for t in targets:
        for co, lab in (("external", "External"), ("internal", "Internal")):
            g = sgd[(sgd.target == t) & (sgd._side == co)]
            assert set(g.subgroup) <= set(SG_ORDER), set(g.subgroup) - set(SG_ORDER)
            for s in SG_ORDER:
                x = g[g.subgroup == s]
                if not len(x):
                    continue
                r = x.iloc[0]
                few = r["positives"] < 5
                auc = f"{f(r['auroc'])} (CI not shown)" if few else est(r["auroc"], r["auroc_lo"], r["auroc_hi"])
                note = clean_note(r["note"])
                if few:
                    note = "fewer than 5 events: point estimate only, not interpretable"
                rows.append([TLAB[t], lab, sglab(s), n_(r["n"]), n_(r["positives"]), pc(r["prevalence"], 2), auc, note])
    # the AUROC pattern the note describes (men below overall at both thresholds, women below overall at >=7.0 and above it
    # at >=9.0) is described only where the data show it; female events at >=9.0 mg/dL read from subgroups.csv
    sgv = lambda t, co, s: one(sgd[sgd._side == co], target=t, subgroup=s)  # noqa: E731
    sides = ("external", "internal")
    men_low = all(sgv(t, co, "Male")["auroc"] < sgv(t, co, "All")["auroc"] for co in sides for t in MAIN)
    women7_low = all(sgv(">=7.0", co, "Female")["auroc"] < sgv(">=7.0", co, "All")["auroc"] for co in sides)
    women9_high = all(sgv(">=9.0", co, "Female")["auroc"] > sgv(">=9.0", co, "All")["auroc"] for co in sides)
    fem9 = {co: n_(sgv(">=9.0", co, "Female")["positives"]) for co in sides}
    expl = ""
    if men_low and women7_low:
        expl = (" In men at both thresholds, and in women at ≥7.0 mg/dL, the AUROC is lower than overall because sex, the "
                "strongest input, is constant inside the subgroup.")
    if women9_high:
        expl += (f" The higher female AUROC at ≥9.0 mg/dL rests on {fem9['external']} external and {fem9['internal']} internal "
                 "events and is not interpreted as better performance in women.")
    else:
        expl += f" Female events at ≥9.0 mg/dL: {fem9['external']} external and {fem9['internal']} internal."
    doc.table("subgroups" + k, "Subgroup discrimination of the models" + ts + ".",
              ["Threshold", "Cohort", "Subgroup", "n", "Events", "Prevalence, %", "AUROC (95% CI)", "Note"],
              rows, notes=["Bootstrap drawn within each subgroup, 1000 resamples, percentile CI. External: KNHANES adults; internal: "
                           "internal test persons." + expl],
              source="subgroups.csv")


def tab_sexdiff(doc, targets):
    k, ts = sfx(targets)
    sd = rd("sex_difference")
    rows = []
    for t in targets:
        for _, r in sd[sd.target == t].iterrows():
            coh = "External" if cohort_side(r["cohort"]) == "external" else "Internal"
            # both cohorts are adults, so the CSV label 'All ages' means all adults aged >=19 years
            rows.append([TLAB[t], coh, r["analysis"].replace("All ages", "All adults (≥19 years)"),
                         f"{n_(r['n_female'])} ({n_(r['pos_female'])})",
                         est(r["auroc_female"], r["auroc_female_lo"], r["auroc_female_hi"]),
                         f"{n_(r['n_male'])} ({n_(r['pos_male'])})", est(r["auroc_male"], r["auroc_male_lo"], r["auroc_male_hi"]),
                         sgci(r["diff_f_minus_m"], r["delong_lo"], r["delong_hi"]), pv(r["delong_p"]), ci(r["boot_lo"], r["boot_hi"]),
                         clean_note(r["note"])])
    doc.table("sexdiff" + k, "Female minus male AUROC" + ts + ".",
              ["Threshold", "Cohort", "Ages", "Women: n (events)", "Women AUROC (95% CI)", "Men: n (events)", "Men AUROC (95% CI)",
               "Difference (DeLong 95% CI)", "DeLong P", "Difference, bootstrap 95% CI", "Note"], rows,
              notes=["DeLong test for two independent ROC curves; bootstrap CI with each sex resampled independently (2000 resamples). "
                     "Age 40-69 is a sensitivity analysis with a more similar age distribution in the two sexes. Higher female AUROCs "
                     "at the severe thresholds rest on few events and are not interpreted as evidence of robustness."],
              source="sex_difference.csv")


# ============================================================ S10 nested and continuous
def nc_model(s):
    low = str(s).lower()
    if "soft-voting" in low or ("ensemble" in low and "lightgbm" not in low):
        return "Ensembles"
    if "continuous" in low:
        return "Continuous-urate LightGBM"
    if "binary lightgbm" in low:
        return "Separate binary LightGBM per threshold"
    raise KeyError(f"nested_coherence.csv model without a label: {s!r}")


def tab_nested(doc, comparisons, key, title):
    nc = rd("nested_coherence")
    rows = []
    for _, r in nc[nc.comparison.isin(comparisons)].iterrows():
        v = r["n_violations"]
        rows.append(["External (KNHANES adults)" if cohort_side(r["cohort"]) == "external" else "Internal test persons",
                     nc_model(r["model"]), r["comparison"].replace(">=", "≥"), n_(r["n"]),
                     f"{n_(v)} ({pc(r['prop_violations'], 2)})",
                     "" if v == 0 or isnan(r["median_violation"]) else f(r["median_violation"], 5),
                     "" if v == 0 or isnan(r["p95_violation"]) else f(r["p95_violation"], 4),
                     "" if v == 0 or isnan(r["max_violation"]) else f(r["max_violation"], 4),
                     "" if v == 0 or isnan(r["prop_violation_gt_0_01"]) else pc(r["prop_violation_gt_0_01"], 3),
                     "" if v == 0 or isnan(r["share_violators_p_lower_eq_0"]) else pc(r["share_violators_p_lower_eq_0"], 1)])
    doc.table(key, title,
              ["Cohort", "Model", "Violation", "n", "Persons violating, n (%)", "Median size", "95th percentile size", "Maximum size",
               "Persons with violation >0.01, %", "Violators whose lower-threshold risk is exactly 0, %"], rows,
              notes=["A violation is a person with a higher predicted probability for a stricter threshold than for a looser one "
                     "(for example P(urate ≥9.0) > P(urate ≥7.0)); size is the difference in probability. The continuous-urate model "
                     "is coherent by construction. The internal test persons are the same for the three thresholds; every model was "
                     "fitted on the training persons."],
              source="nested_coherence.csv")


def cont_model(s):
    s0 = str(s)
    low = s0.lower()
    if " minus " in low:
        part = {"formulation": "formulation", "ensemble/tuning": "ensemble and tuning", "total": "total"}
        tag = next((v for k_, v in part.items() if k_ in low), "")
        a, b = [x.strip() for x in re.sub(r"\s*\([^)]*\)\s*$", "", s0).split(" minus ", 1)]
        name = lambda x: ("continuous" if "continuous" in x.lower() else "binary" if "binary" in x.lower()  # noqa: E731
                          else "ensemble" if "ensemble" in x.lower() else neutral(x))
        return f"Δ {name(a)} minus {name(b)}" + (f" ({tag})" if tag else "")
    if "continuous" in low:
        return "Continuous-urate LightGBM"
    if "binary" in low:
        return "Binary LightGBM (fixed)"
    if "ensemble" in low:
        return "Ensemble"
    raise KeyError(f"continuous.csv model without a label: {s!r}")


def tab_continuous(doc, targets):
    k, ts = sfx(targets)
    cn = rd("continuous")
    cn["_side"] = cn["cohort"].map(cohort_side)
    coh_lab = {"external": "External (KNHANES adults)", "internal": "Internal test persons"}
    rows = []
    if targets != EXPL:
        for _, r in cn[cn.row_type == "regression"].iterrows():
            rows.append(["log urate (one regressor)", coh_lab[r["_side"]], "Continuous-urate LightGBM (regression)", n_(r["n"]), "",
                         f"MAE {f(r['mae_mgdl'], 3)} mg/dL; R² {f(r['r2_mgdl'], 3)}; r {f(r['pearson_r_mgdl'], 3)}", "", "", "", ""])
    for t in targets:
        for co in ("external", "internal"):
            g = cn[(cn.target == t) & (cn._side == co)]
            for _, r in g.iterrows():
                if r["row_type"] == "classification":
                    rows.append([TLAB[t], coh_lab[co], cont_model(r["model"]), n_(r["n"]), n_(r["positives"]),
                                 est(r["auroc"], r["auroc_lo"], r["auroc_hi"]), f(r["auprc"]), f(r["oe_ratio"], 2),
                                 f(r["calib_slope"], 2), f(r["scaled_brier"], 3)])
                else:
                    rows.append([TLAB[t], coh_lab[co], cont_model(r["model"]), "", "",
                                 f"{sgci(r['dauroc'], r['dauroc_lo'], r['dauroc_hi'])}; {pe(r['delong_p'])}", "", "", "", ""])
    doc.table("continuous" + k, "Continuous-urate model versus binary models (sensitivity analysis)" + ts + ".",
              ["Threshold", "Evaluation", "Model or contrast", "n", "Events", "AUROC (95% CI), accuracy, or Δ (95% CI); P", "AUPRC",
               "O:E", "Calibration slope", "Scaled Brier"], rows,
              notes=["Continuous model: one LightGBM regressor for log(urate) (fixed configuration); P(urate ≥ t) is the share of 5-fold "
                     "out-of-fold training residuals r with prediction + r ≥ log t (homoscedastic), so the three probabilities are "
                     "nested by construction. The binary LightGBM (fixed configuration, no tuning or calibration) is fitted per "
                     "threshold on the same persons to separate the formulation effect from the ensemble and tuning effect. All "
                     "models were fitted on the training persons and are compared on persons none of them was trained on. Δ: DeLong "
                     "95% CI and P for correlated curves. MAE: mean absolute error."],
              source="continuous.csv")


def tab_contfu(doc):
    d = rd("fu_continuous")
    rows = []
    for _, r in d.iterrows():
        a = str(r["analysis"])
        try:
            coh = {"external": "KNHANES adults", "internal": "Internal test persons"}[cohort_side(r["cohort"])]
        except KeyError:
            coh = neutral(r["cohort"])
        if a.startswith("observed vs predicted"):
            rows.append(["Decile of predicted urate", coh, f"decile {int(r['decile'])}", n_(r["n"]),
                         f"{f(r['mean_pred_mgdl'], 2)} vs {f(r['mean_obs_mgdl'], 2)}"])
        elif a.startswith("calibration-in-the-large"):
            rows.append(["Mean predicted vs observed urate", coh, "all", n_(r["n"]),
                         f"{f(r['mean_pred_mgdl'], 2)} vs {f(r['mean_obs_mgdl'], 2)}"])
        elif a == "classification":
            rows.append([f"P(urate ≥{int(r['threshold'])}.0 mg/dL)", coh, neutral(r["model"]), f"{n_(r['n'])} ({n_(r['events'])})",
                         f"AUROC {est(r['auroc'], r['auroc_lo'], r['auroc_hi'])}; O:E {f(r['oe'], 2)}"])
        else:
            rows.append([f"Comparison at ≥{int(r['threshold'])}.0 mg/dL", coh, neutral(r["model"]), "",
                         f"ΔAUROC {sgci(r['auroc'], r['auroc_lo'], r['auroc_hi'], 4 if abs(r['auroc']) < 0.005 else 3)}; "
                         f"DeLong {pe(r['delong_p'])}"])
    doc.table("contfu", "Continuous-urate models: observed versus predicted urate, and a heteroscedastic variant.",
              ["Analysis", "Cohort", "Group or model", "n (events)", "Result"], rows,
              notes=["Refitted comparison models, fitted on the training persons. Homoscedastic: one LightGBM regressor for log urate "
                     f"with a single pooled out-of-fold residual distribution (as in {P('continuous')}). Heteroscedastic: a second "
                     "LightGBM models the log squared residual, and exceedance probabilities use the empirical distribution of "
                     "standardised residuals. Predicted mean urate uses Duan smearing. Binary LightGBM: the equal-budget 12-input "
                     "classifier fitted on the training persons."],
              source="fu_continuous.csv (calc_followup.py)")


# ============================================================ S11 interpretation
def tab_perm(doc, targets):
    k, ts = sfx(targets)
    pi = with_t(rd("permutation_importance"))
    pi["_side"] = pi["cohort"].map(cohort_side)
    order = pi[(pi._t == targets[0]) & (pi._side == "external")].sort_values("auroc_drop_mean", ascending=False)["feature"]
    rows = []
    for ftr in order:
        row = [FEAT[ftr]]
        for t in targets:
            for co in ["external", "internal"]:
                r = one(pi, _t=t, _side=co, feature=ftr)
                row.append(f"{f(r['auroc_drop_mean'], 4)} ({f(r['auroc_drop_sd'], 4)}); {n_(r['rank'])}")
        rows.append(row)
    base = []
    for t in targets:
        for co in ["external", "internal"]:
            r = pi[(pi._t == t) & (pi._side == co)].iloc[0]
            base.append(f"{TLAB[t]} {co}: n = {n_(r['n_used'])} of {n_(r['n_cohort'])} ({n_(r['cases_used'])} events), base AUROC "
                        f"{f(r['auroc_base_used'])}")
    hdr = ["Input"]
    for t in targets:
        hdr += [f"{TLAB[t]} external: AUROC drop (SD); rank", f"{TLAB[t]} internal: AUROC drop (SD); rank"]
    doc.table("perm" + k, "Permutation importance of the whole calibrated ensemble" + ts + ".", hdr, rows,
              notes=["Mean drop in AUROC of the calibrated soft-voting ensemble when one input is randomly permuted (10 repeats, "
                     "seed 42). At most 20,000 persons per evaluation: all events kept and non-events sampled at random (AUROC does "
                     "not depend on the event fraction). " + "; ".join(base) + "."],
              source="permutation_importance.csv")


def tab_shap(doc, targets):
    k, ts = sfx(targets)
    sh = with_t(rd("shap_ensemble"))
    an = sh["analysis"].astype(str).str.lower()
    ext = sh[an.str.contains("external")]
    inte = sh[an.str.contains("internal")]
    assert len(ext) + len(inte) == len(sh), sorted(set(sh["analysis"]))
    order = ext[ext._t == targets[0]].sort_values("mean_abs_shap_logodds_fullcohort", ascending=False)["feature"]
    rows = []
    for ftr in order:
        row = [FEAT[ftr]]
        for t in targets:
            e = one(ext, _t=t, feature=ftr)
            i = one(inte, _t=t, feature=ftr)
            row += [f"{f(e['mean_abs_shap_logodds_fullcohort'], 3)}; {n_(e['rank_fullcohort'])}",
                    f"{n_(e['rank'])} ({n_(e['rank_min_across_learners'])}-{n_(e['rank_max_across_learners'])})",
                    n_(e["rank_prob"]), n_(i["rank_fullcohort"]),
                    "" if isnan(e["pdp_shape"]) else e["pdp_shape"]]
        rows.append(row)
    hdr = ["Input"]
    for t in targets:
        hdr += [f"{TLAB[t]}: external mean |SHAP|, full cohort; rank", "External rank, 1,000-person sample (range over 15 learners)",
                "External rank, probability-scale SHAP", "Internal rank, all test persons", "Partial dependence shape (external)"]
    doc.table("shap" + k, "Ensemble SHAP importance" + ts + ".", hdr, rows,
              notes=["TreeSHAP of all 15 base learners (3 learners x 5 calibration folds), mean absolute SHAP on the log-odds scale "
                     "averaged over learners, on all KNHANES adults (full cohort) and on a random sample of 1,000 (seed 42). Soft voting "
                     "averages probabilities, so averaged log-odds SHAP approximates the ensemble before isotonic calibration; the exact "
                     "probability-scale SHAP of the pre-calibration ensemble (interventional, 200 background persons) gives the "
                     "'probability-scale' rank. The shape column is from the calibrated partial dependence."],
              source="shap_ensemble.csv")


SG_LEVELS = ["<=1.005", "1.010", "1.015", "1.020", "1.025", ">=1.030"]
SG_TICKS = ["≤1.005", "1.010", "1.015", "1.020", "1.025", "≥1.030"]


def sg_pdp_shape(target=">=7.0"):
    """Shape label of the calibrated partial dependence on specific gravity in KNHANES adults (pdp_calibrated.csv)."""
    pdp = with_t(rd("pdp_calibrated"))
    g = pdp[(pdp._t == target) & (pdp.cohort.map(cohort_side) == "external") & (pdp.subgroup == "all") & (pdp.feature == "ua_sg")]
    return shape_text(g["pdp_shape"].iloc[0])


def tab_sglevel(doc, targets):
    k, ts = sfx(targets)
    sb = with_t(rd("shap_by_sg"))
    sb["_side"] = sb["cohort"].map(cohort_side)
    rows = []
    for t in targets:
        for co, lab in (("external", "External"), ("internal", "Internal")):
            g = sb[(sb._t == t) & (sb._side == co)]
            for lev in SG_LEVELS:
                row = [TLAB[t], lab, lev.replace("<=", "≤").replace(">=", "≥")]
                for sgp in ["all", "men", "women"]:
                    r = one(g, subgroup=sgp, sg_level=lev)
                    row += [n_(r["n_records"]), pc(r["observed_prevalence"], 2), pc(r["pdp_calibrated_sg_set_to_level"], 2),
                            f(r["pdp_calibrated_ratio_vs_1010"], 2)]
                rows.append(row)
    hdr = ["Threshold", "Cohort", "SG level"]
    for sgp in ["All", "Men", "Women"]:
        hdr += [f"{sgp}: n", f"{sgp}: observed, %", f"{sgp}: predicted with SG set to level, %", f"{sgp}: ratio vs 1.010"]
    shapes = "; ".join(f"{TLAB[t]} {sg_pdp_shape(t)}" for t in targets)
    doc.table("sglevel" + k, "Specific gravity: observed risk and calibrated partial dependence by level" + ts + ".", hdr, rows,
              notes=["Observed: prevalence among persons at that specific gravity (SG) level (KNHANES values, recorded in "
                     f"{sg_facts()['ext_step']} steps, are rounded to the nearest development level and clipped to "
                     f"{SG_LEVELS[0].lstrip('<=')}-{SG_LEVELS[-1].lstrip('>=')}). Predicted: calibrated partial "
                     "dependence, the mean predicted risk of the ensemble over the whole cohort (or sex subgroup) with SG set to the "
                     f"level. Shape of the model's SG effect in KNHANES adults (all persons): {shapes}."],
              source="shap_by_sg.csv, pdp_calibrated.csv (feature = ua_sg), encoding.csv (codebook)")


def not_converged(df):
    """Boolean mask of the sg_univariate.csv rows whose logistic model did not converge (converged == False; rows
    without a model have no value). Their estimates are never printed or used."""
    if "converged" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["converged"].astype(str).str.strip().str.lower().eq("false")


def sg_claims(su):
    """Which statements about specific gravity and urate the data support (sg_univariate.csv, development cohort)."""
    su = su[~not_converged(su)]
    dev = su[su.cohort.astype(str).str.startswith("development") & ~su.cohort.astype(str).str.contains(" vs ")]
    per = dev[dev.row_type == "logistic OR for urate >=7.0 per +0.010 SG"].set_index("model")["estimate"].astype(float)
    cat = dev[(dev.row_type == "logistic OR by SG level (reference 1.010)") & (dev.sg_level == ">=1.030")]
    c9 = cat[np.isclose(cat.threshold.astype(float), 9.0)].set_index("model")["estimate"].astype(float)
    out = dict(reverses=per.get("crude", np.nan) > 1 > per.get("adjusted for sex", np.nan),
               within_sex=(per.get("men: crude", np.inf) < 1.05 and per.get("women: crude", np.inf) < 1.05
                           and per.get("men: adjusted for age, BMI", np.inf) < 1 and per.get("women: adjusted for age, BMI", np.inf) < 1),
               inverse9=c9.get("crude", np.inf) < 1 and c9.get("adjusted for age, sex, BMI (primary)", np.inf) < c9.get("crude", np.nan))
    pv_ = su[(su.row_type == "prevalence by SG level and sex, both cohorts") & np.isclose(su.threshold.astype(float), 7.0)]
    s = pv_[pv_.sex == "all"].set_index("sg_level").loc[SG_LEVELS]
    out["men_rise"] = bool(np.all(np.diff(s["pct_men_dev"].astype(float)) > 0) and np.all(np.diff(s["pct_men_ext"].astype(float)) > 0))
    return out


def tab_sgconf(doc, targets):
    k, ts = sfx(targets)
    su = rd("sg_univariate")
    su["_nc"] = not_converged(su)
    coh_of = lambda s: "Development" if str(s).startswith("development") else "KNHANES adults"  # noqa: E731
    NE = "not estimable (model did not converge)"
    n_ne = 0
    rows = []
    if targets == MAIN:
        per = su[su.row_type == "logistic OR for urate >=7.0 per +0.010 SG"]
        for _, r in per.iterrows():
            n_ne += bool(r["_nc"])
            rows.append(["≥7.0 mg/dL", coh_of(r["cohort"]), f"Per +0.010 SG, {r['model']}", n_(r["n"]), "",
                         NE if r["_nc"] else est(r["estimate"], r["lo"], r["hi"], 2), "" if r["_nc"] else pv(r["p"]), ""])
        sp = su[su.row_type == "Spearman(SG, serum urate)"]
        for _, r in sp.iterrows():
            rows.append(["Serum urate", coh_of(r["cohort"]), f"Spearman correlation of SG with urate, {r['sg_level']}", n_(r["n"]), "",
                         f(r["estimate"], 3), "", ""])
    cat = su[(su.row_type == "logistic OR by SG level (reference 1.010)") & (su.threshold.isin([float(t[2:]) for t in targets]))]
    for _, r in cat.iterrows():
        if r["model"] not in ("crude", "adjusted for sex", "adjusted for age, sex, BMI (primary)",
                              "sensitivity: adjusted for sex x 10-year age band, BMI", "sensitivity: adjusted for the other inputs"):
            continue
        if r["sg_level"] != ">=1.030":
            continue
        n_ne += bool(r["_nc"])
        rows.append([TLAB[tnorm(r["threshold"])], coh_of(r["cohort"]), f"SG ≥1.030 vs 1.010, {r['model']}", n_(r["n"]), n_(r["cases"]),
                     NE if r["_nc"] else est(r["estimate"], r["lo"], r["hi"], 2), "",
                     "" if r["_nc"] else f"χ² {f(r['lr_chi2_sg'], 1)} (5 df), {pe(r['p_sg_joint'])}"])
    if n_ne:
        log(f"  sgconf{k}: {n_ne} row(s) of a model that did not converge printed as not estimable")
    cl = sg_claims(su)
    txt = ""
    if targets == MAIN:
        if cl["reverses"]:
            txt += ("In the development cohort the crude positive association with urate ≥7.0 mg/dL reverses after adjustment for "
                    "sex" + (", because men have both more concentrated urine and higher urate" if cl["men_rise"] else "") + ". ")
        if cl["within_sex"]:
            txt += "Within each sex the crude odds ratio per +0.010 is at most about 1 and the adjusted odds ratios are below 1. "
        if cl["inverse9"]:
            txt += ("At ≥9.0 mg/dL the association with the highest specific gravity level is inverse even before adjustment, and "
                    "adjustment strengthens it. ")
    doc.table("sgconf" + k, "Specific gravity and serum urate: confounding by sex, age and BMI" + ts + ".",
              ["Outcome", "Cohort", "Association", "n", "Events at the level", "Odds ratio (95% CI) or correlation", "P",
               "Likelihood-ratio test of SG (all levels)"], rows,
              notes=["Logistic regression of the outcome on SG (per +0.010, or categorical with 1.010 as reference), crude and adjusted. "
                     + txt + ("A model that did not converge is shown as not estimable. " if n_ne else "")
                     + "SG: specific gravity."],
              source="sg_univariate.csv")


def tab_agepdp(doc, targets):
    k, ts = sfx(targets)
    pdp = with_t(rd("pdp_calibrated"))
    pdp["_side"] = pdp["cohort"].map(cohort_side)
    g = pdp[(pdp.feature == "age") & (pdp._side == "external")]
    rows = []
    for a in sorted(g["value"].unique()):
        row = [n_(a)]
        for t in targets:
            for sgp in ["all", "men", "women"]:
                r = one(g, _t=t, subgroup=sgp, value=float(a))
                row.append(pc(r["mean_pred_calibrated"], 2))
        rows.append(row)
    gi = pdp[(pdp.feature == "age") & (pdp._side == "internal")]

    def change_40_70(frame, t):
        gg = frame[frame._t == t]
        v = {sgp: gg[gg.subgroup == sgp].set_index("value")["mean_pred_calibrated"] for sgp in ["men", "women"]}
        return {sgp: 100 * (v[sgp].loc[70.0] / v[sgp].loc[40.0] - 1) for sgp in v}
    shape, chg, chg_int, opposite = [], [], [], True
    for t in targets:
        for sgp in ["all", "men", "women"]:
            r = g[(g._t == t) & (g.subgroup == sgp)].iloc[0]
            shape.append(f"{TLAB[t]} {sgp}: {shape_text(r['pdp_shape'])}; {step_text(r['pdp_largest_step'])}")
        ce, cint = change_40_70(g, t), change_40_70(gi, t)
        chg.append(f"{TLAB[t]}: men {sg(ce['men'], 0)}%, women {sg(ce['women'], 0)}%")
        chg_int.append(f"{TLAB[t]}: men {sg(cint['men'], 0)}%, women {sg(cint['women'], 0)}%")
        opposite &= (ce["men"] < 0 < ce["women"]) or (ce["women"] < 0 < ce["men"])
    hdr = ["Age, years"]
    for t in targets:
        hdr += [f"{TLAB[t]} all, %", f"{TLAB[t]} men, %", f"{TLAB[t]} women, %"]
    doc.table("agepdp" + k, "Calibrated partial dependence of predicted risk on age, by sex, in KNHANES adults" + ts + ".", hdr, rows,
              notes=["Mean predicted risk of the ensemble with age set to the value for every KNHANES adult (sex kept at its "
                     "observed value within the men and women columns). Shapes: " + "; ".join(shape) + ". Change in predicted risk from "
                     "40 to 70 years: " + "; ".join(chg) + " (internal test persons, computed in the same way: " + "; ".join(chg_int)
                     + ")." + (" From 40 to 70 years the age effect runs in opposite directions in men and women." if opposite
                               else "")],
              source="pdp_calibrated.csv (feature = age)")


# ============================================================ S12 exploratory, condensed
def tab_expl_perf(doc):
    t = ">=10.0"
    dis, cal, rc = with_t(rd("discrimination")), with_t(rd("calibration")), with_t(rd("recalibration"))
    di, de = one(dis, _t=t, cohort="internal"), one(dis, _t=t, cohort="external")
    ci_, ce = one(cal, _t=t, cohort="internal"), one(cal, _t=t, cohort="external")
    r = one(rc, _t=t)
    rows = [["Persons, n (events)", f"{n_(di['n'])} ({n_(di['positives'])})", f"{n_(de['n'])} ({n_(de['positives'])})"],
            ["AUROC (95% CI)", est(di["auroc"], di["auroc_lo"], di["auroc_hi"]), est(de["auroc"], de["auroc_lo"], de["auroc_hi"])],
            ["AUPRC (95% CI)", est(di["auprc"], di["auprc_lo"], di["auprc_hi"]), est(de["auprc"], de["auprc_lo"], de["auprc_hi"])],
            ["O:E (95% CI)", est(ci_["oe"], ci_["oe_lo"], ci_["oe_hi"], 2), est(ce["oe"], ce["oe_lo"], ce["oe_hi"], 2)],
            ["Calibration intercept (95% CI)", est(ci_["intercept"], ci_["intercept_lo"], ci_["intercept_hi"], 2),
             est(ce["intercept"], ce["intercept_lo"], ce["intercept_hi"], 2)],
            ["Calibration slope (95% CI)", est(ci_["slope"], ci_["slope_lo"], ci_["slope_hi"], 2),
             est(ce["slope"], ce["slope_lo"], ce["slope_hi"], 2)],
            ["Scaled Brier score (95% CI)", est(ci_["scaled_brier"], ci_["scaled_brier_lo"], ci_["scaled_brier_hi"]),
             est(ce["scaled_brier"], ce["scaled_brier_lo"], ce["scaled_brier_hi"])],
            ["Predictions of exactly 0, n (events)", f"{n_(ci_['n_p0'])} ({n_(ci_['events_p0'])})", f"{n_(ce['n_p0'])} ({n_(ce['events_p0'])})"],
            ["After an intercept update: O:E; scaled Brier", "N/A",
             f"{est(r['oe_after_intercept'], r['oe_after_intercept_lo'], r['oe_after_intercept_hi'], 2)}; "
             f"{est(r['sbrier_intercept'], r['sbrier_intercept_lo'], r['sbrier_intercept_hi'])}"]]
    doc.table("expl_perf", "Exploratory ≥10.0 mg/dL model: discrimination and calibration.", ["Measure", "Internal test set",
              "KNHANES adults"], rows, notes=["The model as fitted; intercept update cross-validated over 100 half-splits. 95% CIs: "
                                              "percentile bootstrap, 1000 resamples."],
              source="discrimination.csv, calibration.csv, recalibration.csv")


def tab_expl_util(doc):
    t = ">=10.0"
    op, dcr = with_t(rd("operating_points")), with_t(rd("decision_curve_ranges"))
    rows = []
    for pol in ("youden", "sens80", "sens90", "top_1%", "top_2%", "top_5%", "top_10%", "top_20%"):
        for coh in ("internal", "external"):
            r = one(op, _t=t, cohort=coh, policy=pol)
            rows.append([tidy(pol).replace("_", " "), coh, f(r["threshold"], 4), pest(r["sens"], r["sens_lo"], r["sens_hi"]),
                         pest(r["ppv"], r["ppv_lo"], r["ppv_hi"]), pc(r["pos_rate"]), f(r["referrals_per_case"], 1)])
    g = dcr[dcr._t == t]
    dtxt = "; ".join(f"{r['cohort']}, {r['criterion']}: {r['ranges']}" for _, r in g.iterrows())
    grid = f"{val(g['grid_min'].min())}-{val(g['grid_max'].max())}"
    doc.table("expl_util", "Exploratory ≥10.0 mg/dL model: operating points and decision curves.",
              ["Rule", "Cohort", "Threshold", "Sensitivity, % (95% CI)", "PPV, % (95% CI)", "Flagged, %", "Referrals per case"], rows,
              notes=["Thresholds from the internal test set applied to KNHANES; top-k thresholds are cohort-specific. Decision-curve "
                     f"threshold ranges in which net benefit exceeded both defaults (grid {grid}): {dtxt}."],
              source="operating_points.csv, decision_curve_ranges.csv")


def tab_expl_value(doc):
    t = ">=10.0"
    ft, ev, co = with_t(rd("feature_tiers")), with_t(rd("ensemble_vs_simple")), with_t(rd("continuous"))
    ens = [m for m in dict.fromkeys(ev[ev.row_type == "model"]["model"]) if is_ensemble(m)]
    assert len(ens) == 1, ens
    rows = []
    for setting in ("internal", "external"):
        for mod in TIER_MODELS:
            r = one(ft[ft.row_type == "contrast_tier"], _t=t, setting=setting, model=mod, tier="12F-3F")
            rows.append(["Dipstick added to age, sex, BMI (refitted comparison models)", setting, model_label(mod),
                         sgci(r["delta_auroc"], r["delta_lo"], r["delta_hi"])])
        for comp in ("Logistic regression", "LR (age/BMI splines + age x male)", "LightGBM (GRID-tuned)"):
            r = one(ev[ev.row_type == "contrast"], _t=t, setting=setting, comparison=f"{ens[0]} vs {comp}")
            rows.append(["Ensemble minus simpler 12-input model", setting, model_label(comp),
                         sgci(r["delta_auroc"], r["delta_lo"], r["delta_hi"], 4 if abs(r["delta_auroc"]) < 0.005 else 3)])
    cc = co[(co.row_type == "comparison") & (co.cohort.map(cohort_side) == "external") & (co._t == t)]
    for _, r in cc.iterrows():
        rows.append(["Continuous-urate model", "external", cont_model(r["model"]), sgci(r["dauroc"], r["dauroc_lo"], r["dauroc_hi"])])
    n_ev = ci_items("cohort_info_v2")["KNHANES events >=10.0"]
    doc.table("expl_value", "Exploratory ≥10.0 mg/dL model: added value of the dipstick and comparison with simpler models "
              "(AUROC differences).", ["Analysis", "Setting", "Model or comparison", "ΔAUROC (95% CI)"], rows,
              notes=[f"Refitted comparison models as in Section {SEC('value')}; external: KNHANES adults (coding X1). External "
                     f"increments rest on {n_(n_ev)} events."],
              source="feature_tiers.csv, ensemble_vs_simple.csv, continuous.csv, cohort_info_v2.csv")


def tab_expl_sens(doc):
    t = ">=10.0"
    sgp, sx, ne = with_t(rd("subgroups")), with_t(rd("sex_difference")), rd("nested_coherence")
    rows = []
    ext_sg = sgp[(sgp._t == t) & (sgp.cohort.map(cohort_side) == "external")]
    for _, r in ext_sg.iterrows():
        rows.append(["Subgroup AUROC (KNHANES)", sglab(tidy(r["subgroup"])), f"{n_(r['n'])} ({n_(r['positives'])})",
                     est(r["auroc"], r["auroc_lo"], r["auroc_hi"])])
    # subgroups with fewer than 20 events, read from subgroups.csv
    few = ext_sg[ext_sg.positives < 20]
    few_txt = "; ".join(f"{sglab(tidy(r['subgroup'])).lower()}: {n_(r['positives'])}" for _, r in few.iterrows())
    for _, r in sx[(sx._t == t) & (sx.cohort.map(cohort_side) == "external")].iterrows():
        rows.append(["Female minus male AUROC", tidy(r["analysis"]).replace("All ages", "All adults (≥19 years)"), "",
                     f"{sgci(r['diff_f_minus_m'], r['delong_lo'], r['delong_hi'])}; {pe(r['delong_p'])}"])
    nc = ne[ne.model.map(nc_model).eq("Ensembles") & ne.comparison.isin(["P(>=10.0) > P(>=9.0)", "P(>=10.0) > P(>=7.0)"])]
    for _, r in nc.iterrows():
        coh = "KNHANES adults" if cohort_side(r["cohort"]) == "external" else "internal test persons"
        rows.append(["Nested-threshold violations, ensembles", f"{tidy(r['comparison'])}; {coh}", n_(r["n"]),
                     f"{n_(r['n_violations'])} ({pc(r['prop_violations'], 2)}%)"
                     + ("" if isnan(r["median_violation"]) else f"; median size {f(r['median_violation'], 5)}")])
    ms, mi = with_t(rd("mapping_sensitivity")), with_t(rd("mice_bmi"))
    md = map_desc(rd("encoding"))
    desc = {**{m: md[m][0].upper() + md[m][1:] for m in ("M1", "M2", "M3")},
            "X2": "As X1 except specific gravity snapped to the development grid"}
    for mp in ("M1", "M2", "M3", "X2"):
        r = one(ms, _t=t, mapping=mp)
        rows.append(["KNHANES coding (sensitivity)", f"{mp}: {desc[mp]}", "", f"AUROC {f(r['auroc'])}; O:E {f(r['oe_ratio'], 2)}"])
    lab = {r["imputation_model"]: tidy(clean_note(r["note"])) for _, r in mi[mi.section == "imputation_diagnostics"].iterrows()}
    for im in ("full", "age_sex"):
        r = one(mi[mi.section == "pooled"], _t=t, imputation_model=im)
        rows.append(["Excluded for missing BMI, BMI imputed", lab.get(im, im), f"{n_(r['n'])} ({n_(r['positives'])})",
                     f"AUROC {est(r['auroc'], r['auroc_lo'], r['auroc_hi'])}; O:E {f(r['oe'], 2)}; slope {f(r['cal_slope'], 2)}"])
    tp = with_t(rd("temporal"))
    p_tr, p_te = temporal_periods(tp)
    r = one(tp, _t=t, design="temporal", model="Ensemble-12F (tuned, refit)", evaluation="internal test")
    rows.append(["Temporal holdout (refit)", f"{p_tr} → {p_te}", f"{n_(r['n'])} ({n_(r['positives'])})",
                 f"AUROC {est(r['auroc'], r['auroc_lo'], r['auroc_hi'])}; O:E {f(r['oe'], 2)}"])
    sw = with_t(rd("survey_weighted"))
    r = one(sw, _t=t)
    rows.append(["KNHANES with survey weights", "", "", f"AUROC {f(r['auroc_weighted'])}; O:E {f(r['oe_weighted'], 2)}"])
    doc.table("expl_sens", "Exploratory ≥10.0 mg/dL model: subgroups, nested-threshold coherence and sensitivity analyses.",
              ["Analysis", "Group or variant", "n (events)", "Result"], rows,
              notes=[(f"Estimates with fewer than 20 events ({few_txt}) are not interpretable." if few_txt else
                      "Every subgroup has at least 20 events.") + f" Codings as in {T('mapping')}."],
              source="subgroups.csv, sex_difference.csv, nested_coherence.csv, mapping_sensitivity.csv, encoding.csv, "
                     "mice_bmi.csv, temporal.csv, survey_weighted.csv")


# ============================================================ figures
def load_style():
    """figstyle.py from this folder when present (shared with the main figures); otherwise a local equivalent."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    style = {}
    try:
        sys.path.insert(0, HERE)
        import figstyle as fsx  # noqa: F401
        style["module"] = "figstyle.py (working folder)"
        fsx.apply()
        style.update(NAVY=fsx.NAVY, RED=fsx.RED, GREEN=fsx.GREEN, GREY=fsx.GREY, AMBER=getattr(fsx, "AMBER", "#e08e0b"),
                     WIDTH_IN=getattr(fsx, "WIDTH_IN", 180 / 25.4), FS=getattr(fsx, "FS", None),
                     title=fsx.title, panel=fsx.panel, grid=fsx.grid)
    except ImportError:
        style["module"] = "local style (no figstyle.py in the working folder)"
        FS = dict(title=9.5, label=8.5, tick=8, legend=7.4, note=7.0, panel=13)
        NAVY = "#1f3b63"
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS["tick"], "axes.titlesize": FS["title"],
                             "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
                             "legend.fontsize": FS["legend"], "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
                             "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

        def title(ax, text):
            ax.set_title(text, fontsize=FS["title"], fontweight="bold", color=NAVY, pad=7, loc="left")

        def panel(ax, letter):
            ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-28, 6), textcoords="offset points",
                        fontsize=FS["panel"], fontweight="bold", ha="left", va="bottom", annotation_clip=False)

        def grid(ax, axis="y"):
            ax.grid(axis=axis, alpha=0.22, lw=0.6)
            ax.set_axisbelow(True)

        style.update(NAVY=NAVY, RED="#c0392b", GREEN="#2e7d5b", GREY="#7f8c8d", AMBER="#e08e0b", WIDTH_IN=180 / 25.4, FS=FS,
                     title=title, panel=panel, grid=grid)
    if style.get("FS") is None:
        style["FS"] = dict(title=9.5, label=8.5, tick=8, legend=7.4, note=7.0, panel=13)
    style["plt"] = plt
    return style


def new_fig(S, h, nrows, ncols, **kw):
    fig, axes = S["plt"].subplots(nrows, ncols, figsize=(S["WIDTH_IN"], h), constrained_layout=True, **kw)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.12)
    return fig, axes


def save_fig(S, fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300)
    S["plt"].close(fig)
    log(f"  figure {name}  {S['WIDTH_IN']:.2f} x {fig.get_figheight():.2f} in")
    return f"figures/{name}.png"


# panels of Figure S1: (title, dipstick_by_year column, field, level of the dashed line, level name)
YEAR_PANELS = [("Ketone", "pos_ketone", "ua_ketone", TRACE, "trace"),
               ("Bilirubin", "pos_bilirubin", "ua_bilirubin", ONE_PLUS, "1+"),
               ("Glucose", "pos_glucose", "ua_glucose", TRACE, "trace"),
               ("Protein", "pos_protein", "ua_protein", TRACE, "trace")]


def fig_years(S, name):
    dy = rd("dipstick_by_year")
    der = rd("supplement_derived")
    lv = der[der.section == "dipstick_levels_by_year"]
    dev = dy[(dy.cohort.astype(str).str.startswith("development")) & (dy.period_type == "year")].copy()
    ext = dy[(~dy.cohort.astype(str).str.startswith("development")) & (dy.period_type == "year")].copy()
    dev["yr"] = dev["period"].astype(int)
    ext["yr"] = ext["period"].astype(int)
    ldev = lv[lv.cohort.astype(str).str.startswith("development")]

    def lev(frame, feat, pred):
        s = frame[(frame.feature == feat) & pred(frame.level)].groupby("year")["pct"].sum()
        return s.index.astype(int), s.values

    fig, axes = new_fig(S, 5.6, 2, 2)
    for i, (ax, (lab, col, feat, pred, levname)) in enumerate(zip(axes.ravel(), YEAR_PANELS)):
        ax.axvline(2021.5, color=S["GREY"], lw=0.8, ls=":")
        ax.plot(dev["yr"], 100 * dev[col], color=S["NAVY"], marker="o", ms=3.5, lw=1.4, label="Development, any positive")
        x, y = lev(ldev, feat, pred)
        ax.plot(x, y, color=S["NAVY"], marker="^", ms=3.5, lw=1.0, ls="--",
                label="Development, level named in the panel title")
        ax.plot(ext["yr"], 100 * ext[col], color=S["RED"], marker="s", ms=3.5, lw=1.4, label="KNHANES adults, any positive")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.set_xlim(2009.5, 2025.5)
        ax.set_xticks(range(2010, 2026, 3))
        ax.set_ylabel("Persons, %" if i % 2 == 0 else "")
        S["grid"](ax, "y")
        S["title"](ax, f"{lab} (dashed: {levname})")
        S["panel"](ax, "ABCD"[i])
        ax.annotate("2022", xy=(2021.5, 1), xycoords=("data", "axes fraction"), xytext=(3, -2), textcoords="offset points",
                    fontsize=S["FS"]["note"], color="#555555", va="top")
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, handlelength=2.0)
    return save_fig(S, fig, name)


def fig_perm(S, targets, name):
    pi = with_t(rd("permutation_importance"))
    pi["_side"] = pi["cohort"].map(cohort_side)
    n = len(targets)
    fig, axes = new_fig(S, 3.6, 1, n)
    axes = np.atleast_1d(axes)
    for i, (ax, t) in enumerate(zip(axes, targets)):
        e = pi[(pi._t == t) & (pi._side == "external")].sort_values("auroc_drop_mean")
        it = pi[(pi._t == t) & (pi._side == "internal")].set_index("feature")
        y = np.arange(len(e))
        ax.barh(y, e["auroc_drop_mean"], xerr=e["auroc_drop_sd"], color=S["NAVY"], height=0.62, error_kw=dict(lw=0.7, capsize=1.5),
                label="KNHANES adults (bar, SD)")
        ax.scatter(it.loc[e["feature"], "auroc_drop_mean"], y, marker="D", s=16, facecolor="white", edgecolor=S["RED"],
                   linewidth=1.1, zorder=3, label="Internal test")
        ax.set_yticks(y)
        ax.set_yticklabels([FEAT[x] for x in e["feature"]])
        ax.axvline(0, color="#333333", lw=0.6)
        top = e["auroc_drop_mean"].max()
        ivals = it.loc[e["feature"], "auroc_drop_mean"].values
        for yy, v, sd, iv in zip(y, e["auroc_drop_mean"], e["auroc_drop_sd"], ivals):
            if v >= 0.01:
                xr = max(v + sd, iv) + top * 0.04
                ax.text(xr, yy, f(v, 3), va="center", fontsize=S["FS"]["note"], color="#333333")
        ax.set_xlim(min(0, e["auroc_drop_mean"].min()) - top * 0.02, top * 1.3)
        ax.set_xlabel("AUROC drop when permuted")
        S["grid"](ax, "x")
        S["title"](ax, TLAB[t])
        S["panel"](ax, "ABC"[i])
        ax.legend(loc="lower right")
    return save_fig(S, fig, name)


def fig_sg(S, targets, name, with_confounding=True):
    sb = with_t(rd("shap_by_sg"))
    sb["_side"] = sb["cohort"].map(cohort_side)
    su = rd("sg_univariate")
    x = np.arange(len(SG_LEVELS))
    sty = {"all": (S["NAVY"], "o", "-", "All"), "men": (S["RED"], "s", "--", "Men"), "women": (S["GREEN"], "^", "-.", "Women")}
    npan = len(targets) + (2 if with_confounding else 0)
    ncols = 2
    nrows = int(np.ceil(npan / ncols))
    fig, axes = new_fig(S, 2.75 * nrows, nrows, ncols)
    axes = np.atleast_1d(axes).ravel()
    k = 0
    for t in targets:
        ax = axes[k]
        g = sb[(sb._t == t) & (sb._side == "external")]
        ax.axhline(1, color=S["GREY"], lw=0.8, ls=":")
        for sgp, (col, mk, ls, lab) in sty.items():
            s = g[g.subgroup == sgp].set_index("sg_level").loc[SG_LEVELS]
            ax.plot(x, s["pdp_calibrated_ratio_vs_1010"], color=col, marker=mk, ls=ls, ms=4, lw=1.4, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(SG_TICKS)
        ax.set_xlabel("Specific gravity set to")
        ax.set_ylabel("Predicted risk relative to SG 1.010")
        S["grid"](ax, "y")
        S["title"](ax, f"Model partial dependence, {TLAB[t]}")
        S["panel"](ax, "ABCD"[k])
        ax.legend(loc="lower left")
        k += 1
    if with_confounding:
        pv_ = su[(su.row_type == "prevalence by SG level and sex, both cohorts") & (su.threshold == 7.0)]
        ax = axes[k]
        for coh_col, coh_lab, mkf in (("dev", "Development", "full"), ("ext", "KNHANES", "none")):
            for sgp, (col, mk, ls, lab) in sty.items():
                s = pv_[pv_.sex == sgp].set_index("sg_level").loc[SG_LEVELS]
                ax.plot(x, 100 * s[f"prev_{coh_col}"], color=col, marker=mk, ls="-" if coh_col == "dev" else ":", ms=4, lw=1.2,
                        fillstyle=mkf, label=f"{lab}, {coh_lab}")
        ax.set_xticks(x)
        ax.set_xticklabels(SG_TICKS)
        ax.set_xlabel("Specific gravity")
        ax.set_ylabel("Observed urate ≥7.0 mg/dL, %")
        S["grid"](ax, "y")
        S["title"](ax, "Observed prevalence by sex")
        S["panel"](ax, "ABCD"[k])
        ax.legend(loc="upper left", ncol=2, fontsize=S["FS"]["legend"] - 0.6, handlelength=1.8)
        ax.set_ylim(0, 100 * pv_["prev_dev"].max() * 1.45)
        k += 1
        ax = axes[k]
        s = pv_[pv_.sex == "all"].set_index("sg_level").loc[SG_LEVELS]
        ax.plot(x, 100 * s["pct_men_dev"], color=S["NAVY"], marker="o", ms=4, lw=1.4, label="Development")
        ax.plot(x, 100 * s["pct_men_ext"], color=S["RED"], marker="s", ms=4, lw=1.4, ls=":", label="KNHANES adults")
        ax.set_xticks(x)
        ax.set_xticklabels(SG_TICKS)
        ax.set_xlabel("Specific gravity")
        ax.set_ylabel("Men, %")
        S["grid"](ax, "y")
        S["title"](ax, "Share of men by specific gravity")
        S["panel"](ax, "ABCD"[k])
        ax.legend(loc="upper left")
        k += 1
    for j in range(k, len(axes)):
        axes[j].set_visible(False)
    return save_fig(S, fig, name)


def fig_expl(S, name):
    """>=10.0 exploratory: calibration, permutation importance, SG partial dependence."""
    t = ">=10.0"
    cc = rd("calibration_curve")
    cc = cc[(cc.model == ISO) & (cc.bins_requested == 10) & (cc.target == t)]
    cal = rd("calibration")
    pi = with_t(rd("permutation_importance"))
    pi["_side"] = pi["cohort"].map(cohort_side)
    sb = with_t(rd("shap_by_sg"))
    sb["_side"] = sb["cohort"].map(cohort_side)
    fig, axes = new_fig(S, 3.5, 1, 3, gridspec_kw=dict(width_ratios=[1, 1.25, 1]))
    ax = axes[0]
    hi = max(cc["mean_pred"].max(), cc["observed_hi"].max()) * 1.08
    ax.plot([0, hi], [0, hi], color=S["GREY"], lw=0.8, ls=":")
    for co, col, mk, lab in (("internal", S["NAVY"], "o", "Internal"), ("external", S["RED"], "s", "KNHANES")):
        s = cc[cc.cohort == co].sort_values("bin")
        r = one(cal, target=t, cohort=co)
        # Wilson lower bound of a 0-event bin is stored as ~5e-20 (floating point), so clip the bar lengths at 0
        ax.errorbar(s["mean_pred"], s["observed"], yerr=[(s["observed"] - s["observed_lo"]).clip(lower=0),
                                                         (s["observed_hi"] - s["observed"]).clip(lower=0)],
                    color=col, marker=mk, ms=4, lw=1.1, elinewidth=0.8, capsize=2, label=f"{lab}: O:E {f(r['oe'], 2)}")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed risk")
    S["grid"](ax, "both")
    S["title"](ax, "Calibration")
    S["panel"](ax, "A")
    ax.legend(loc="upper left")
    ax = axes[1]
    e = pi[(pi._t == t) & (pi._side == "external")].sort_values("auroc_drop_mean")
    y = np.arange(len(e))
    ax.barh(y, e["auroc_drop_mean"], xerr=e["auroc_drop_sd"], color=S["NAVY"], height=0.62, error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_yticks(y)
    ax.set_yticklabels([FEAT[v] for v in e["feature"]])
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_xlabel("Drop in AUROC (KNHANES)")
    S["grid"](ax, "x")
    S["title"](ax, "Importance")
    S["panel"](ax, "B")
    ax = axes[2]
    x = np.arange(len(SG_LEVELS))
    g = sb[(sb._t == t) & (sb._side == "external")]
    ax.axhline(1, color=S["GREY"], lw=0.8, ls=":")
    for sgp, col, mk, ls, lab in (("all", S["NAVY"], "o", "-", "All"), ("men", S["RED"], "s", "--", "Men"),
                                  ("women", S["GREEN"], "^", "-.", "Women")):
        s = g[g.subgroup == sgp].set_index("sg_level").loc[SG_LEVELS]
        ax.plot(x, s["pdp_calibrated_ratio_vs_1010"], color=col, marker=mk, ls=ls, ms=4, lw=1.3, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(SG_TICKS, rotation=35, ha="right")
    ax.set_xlabel("Specific gravity set to")
    ax.set_ylabel("Relative to SG 1.010")
    S["grid"](ax, "y")
    S["title"](ax, "Specific gravity")
    S["panel"](ax, "C")
    ax.legend(loc="lower left")
    return save_fig(S, fig, name)


# ============================================================ assembly
# Each table is written inside doc.group(<key>): its caption, then one panel per tab_* call, in STRUCTURE order (Doc
# checks the order); sections with doc.section(<key>) in SECTIONS order.
def section_main_tables(doc, S, figpaths):
    # ---------------- S1
    doc.section("flow")
    with doc.group("flowset"):
        tab_flow(doc)
        tab_setting_all(doc)
        tab_setting_dev(doc)
        tab_knflow(doc)
    # ---------------- S2
    doc.section("inputs")
    doc.p("The models receive the KNHANES dipstick fields under the label-matched coding X1 (" + T("mapping") + "), fixed "
          "before the models were fitted or evaluated. Three alternative codings of the semi-quantitative fields (M1, M2, M3) "
          "and one of specific gravity (X2) were scored as sensitivity analyses (" + T("mapsens") + ").")
    with doc.group("inputmap"):
        tab_inputs(doc)
        tab_mapping(doc)
    with doc.group("inputdist"):
        tab_codebook(doc)
        tab_shift(doc)
    with doc.group("codingsens"):
        tab_mapsens(doc, MAIN)
        tab_coding(doc, MAIN)
    # ---------------- S3
    doc.section("missing")
    with doc.group("missdata"):
        tab_missing(doc)
        tab_included(doc)
        tab_mcar(doc)
        tab_mice(doc, MAIN)
    with doc.group("bmilink"):
        tab_bmisource(doc)
        tab_bmiperf(doc, MAIN)
    with doc.group("verifbias"):
        tab_verif_model(doc)
        tab_ipw(doc, MAIN)
    # ---------------- S4
    doc.section("hyper")
    with doc.group("hyperparams"):
        tab_hyper(doc, MAIN)
    # ---------------- S5
    doc.section("value")
    with doc.group("addedvalue"):
        tab_tiers(doc, MAIN)
        tab_tier_contrasts(doc, MAIN)
        tab_tuned(doc, MAIN)
        tab_tiers2f(doc)
        tab_withinsex(doc)
    with doc.group("ensvssimple"):
        tab_ens(doc, MAIN)
    # ---------------- S6
    doc.section("calib")
    with doc.group("calibration"):
        tab_discrimination(doc, MAIN)
        tab_calib(doc, MAIN)
        tab_calbins(doc, MAIN)
        tab_platt(doc, MAIN)
        tab_isofloor(doc)
        tab_recal(doc, MAIN)
    # ---------------- S7
    doc.section("utility")
    with doc.group("operating"):
        tab_oppoints(doc, MAIN)
        tab_splithalf(doc, MAIN)
        tab_sexop(doc)
        tab_oof(doc)
        tab_recalop(doc)
    with doc.group("netbenefit"):
        tab_dca(doc, MAIN)
        tab_nbref(doc)
        tab_utility(doc, MAIN)
    # ---------------- S8
    doc.section("transport")
    with doc.group("overtime"):
        tab_temporal(doc, MAIN)
        tab_intperiod(doc)
        tab_periods(doc, MAIN)
        tab_knyear(doc)
    with doc.group("byyear"):
        tab_posyear(doc)
        tab_levels(doc)
    rc = reporting_change(rd("supplement_derived").query("section == 'dipstick_levels_by_year'")
                          .loc[lambda d: d.cohort.astype(str).str.startswith("development")])
    drop_txt = " and ".join(t_ for k_, t_ in (("ua_ketone", "ketone trace results"), ("ua_bilirubin", "bilirubin 1+")) if rc[k_])
    doc.figure("years", "Dipstick positivity by year: the laboratory reporting change in the development data.", figpaths["years"],
               notes=["Development: development cohort by year of the urine test; KNHANES: adults by survey year. Solid lines: any value "
                      "above negative; dashed lines: the single level named in the panel title (development only). The dotted vertical "
                      "line marks the start of 2022; the reporting change took place at the end of 2021 (transition from July 2021 to "
                      "January 2022)." + (f" {drop_txt[0].upper() + drop_txt[1:]} almost disappear from 2022 in the development data."
                                          if drop_txt else "")],
               source="dipstick_by_year.csv, supplement_derived.csv (dipstick_levels_by_year)")
    with doc.group("othersens"):
        tab_ketone(doc)
        tab_sgrestrict(doc)
        tab_survey(doc, MAIN)
    # ---------------- S9
    doc.section("subgroups")
    with doc.group("subgroupsex"):
        tab_subgroups(doc, MAIN)
        tab_sexdiff(doc, MAIN)
    # ---------------- S10
    doc.section("nested")
    with doc.group("nestedcont"):
        tab_nested(doc, ["P(>=9.0) > P(>=7.0)"], "nested",
                   "Nested-threshold coherence of the predicted probabilities: ≥9.0 versus ≥7.0 mg/dL.")
        tab_continuous(doc, MAIN)
        tab_contfu(doc)
    # ---------------- S11
    doc.section("interp")
    doc.p("The tables below explain the whole calibrated ensemble: permutation importance of the ensemble, SHAP values of all 15 "
          "base learners, and the calibrated partial dependence of predicted risk on specific gravity and age.")
    with doc.group("importance"):
        tab_perm(doc, MAIN)
        tab_shap(doc, MAIN)
    doc.figure("perm", "Permutation importance of the whole calibrated ensemble.", figpaths["perm"],
               notes=["Bars: mean drop in AUROC in KNHANES adults when the input is permuted (10 repeats; error bars, SD); open diamonds: "
                      "internal test persons."],
               source="permutation_importance.csv")
    with doc.group("sgage"):
        tab_sglevel(doc, MAIN)
        tab_sgconf(doc, MAIN)
        tab_agepdp(doc, MAIN)
    cl = sg_claims(rd("sg_univariate"))
    doc.figure("sg", "Specific gravity: what the models learnt and the crude association by sex.", figpaths["sg"],
               notes=["A-B: calibrated partial dependence of the ensembles in KNHANES adults, relative to specific gravity 1.010, "
                      "overall and within each sex. C: observed prevalence of serum urate ≥7.0 mg/dL by specific gravity and sex in both "
                      "cohorts (solid, development; dotted, KNHANES). D: share of men by specific gravity."
                      + (" Men have more concentrated urine and higher urate, so pooling the sexes produces a positive crude "
                         "association that disappears or reverses within sex." if cl["men_rise"] and cl["reverses"] else "")],
               source="shap_by_sg.csv, sg_univariate.csv")


NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def section_exploratory(doc, figpaths):
    n_tab = sum(1 for _, _, keys in STRUCTURE if all(k.startswith("expl_") for k in keys))
    doc.section("expl")
    doc.p("The ≥10.0 mg/dL model (the development institution's threshold for starting urate-lowering therapy) is exploratory. Its "
          f"results are summarised in {NUM_WORD[n_tab]} tables and one figure, and none is used for a conclusion in the main text. "
          "Event counts are small (" + EXPL_COUNTS + ").")
    with doc.group("expl10a"):
        tab_expl_perf(doc)
        tab_expl_util(doc)
    with doc.group("expl10b"):
        tab_expl_value(doc)
        tab_expl_sens(doc)
    doc.figure("expl", "Exploratory ≥10.0 mg/dL model: calibration, permutation importance and specific gravity.", figpaths["expl"],
               notes=["A: ten quantile bins of predicted risk with Wilson 95% CIs (internal test persons and KNHANES adults). B: drop in "
                      "AUROC in KNHANES adults when each input is permuted (10 repeats; SD). C: calibrated partial dependence relative to "
                      "specific gravity 1.010 in KNHANES adults."],
               source="calibration_curve.csv, calibration.csv, permutation_importance.csv, shap_by_sg.csv")


EXPL_COUNTS = ""


def conventions(doc):
    c = ci_items("cohort_info_v2")
    mp = rd("main_performance")
    n_dev, n_tr, n_te, n_ext = (int(c[k_]) for k_ in ("analytic cohort persons", "training persons", "test persons",
                                                      "KNHANES adults with 12 inputs and urate"))
    assert n_tr + n_te == n_dev, (n_tr, n_te, n_dev)
    # the evaluation cohorts of main_performance.csv are these persons
    assert {int(x) for x in mp[mp.cohort == "internal test"]["n"]} == {n_te}, "internal test persons differ"
    assert {int(x) for x in mp[mp.cohort == "KNHANES adults"]["n"]} == {n_ext}, "KNHANES adults differ"
    lt1 = int(c["serum urate '<1.0' persons (counted as 0.5)"])
    ck, nu = int(c["BMI source checkup (persons)"]), int(c["BMI source nursing (persons)"])
    assert ck + nu == n_dev, (ck, nu, n_dev)
    # formatting examples of the Statistics bullet: an interval of the main results (main_performance.csv, internal AUROC at
    # >=7.0 mg/dL) and one with a negative bound (calibration.csv, internal calibration intercept at >=7.0 mg/dL)
    ex_auc = one(mp, target=">=7.0", cohort="internal test")
    ex_neg = one(rd("calibration"), target=">=7.0", cohort="internal")
    assert float(ex_neg["intercept_lo"]) < 0 < float(ex_neg["intercept_hi"]), "formatting example needs a negative bound"
    doc.h(2, "How to read this appendix")
    doc.bullets([
        "Models. One model per threshold: a soft-voting ensemble of LightGBM, XGBoost and CatBoost (equal weights) with isotonic "
        "calibration (CalibratedClassifierCV, 5 folds), using 12 inputs: 9 urine dipstick fields (glucose, protein, ketone, "
        "bilirubin, occult blood, urobilinogen, nitrite, pH and specific gravity), age, sex and BMI. Leukocyte esterase is not an "
        "input because KNHANES does not measure it. The hyperparameters of the base learners were tuned on the training persons "
        f"only ({T('hyper')}). The data, inputs, split, KNHANES coding and modelling procedure were fixed before the models were "
        "fitted or evaluated. Results labelled 'ensemble' come from the saved model files. Models fitted for comparison or "
        "sensitivity analyses (feature tiers, tuned LightGBM, simpler comparators, temporal holdout, continuous-urate models, "
        "recalibration) are labelled as refitted comparison models.",
        f"Tables. Related results are grouped into {len(STRUCTURE)} tables, most of them with lettered panels; a reference such "
        f"as '{T('mcar')}' points to one panel, and every panel has its own notes and source line. The main internal and "
        f"external results are in main-text Table 2 and {T('discrim')}, {T('calib')}.",
        f"Development cohort. Adults aged ≥19 years with a first-visit urine dipstick test, a blood test within ±14 days, serum "
        f"urate and BMI recorded on the day of the urine test, and all 12 inputs (n = {n_(n_dev)} persons, one record each; "
        f"{T('flow')}). Serum urate reported as '<1.0' mg/dL is counted as 0.5 mg/dL ({n_(lt1)} persons). BMI comes from the "
        f"health-checkup record of the urine-test day ({n_(ck)} persons) or from height and weight in the nursing record of "
        f"that day ({n_(nu)} persons).",
        "Target population. The models are intended for adults aged ≥19 years at the urine test, the age range of the external "
        f"validation cohort; younger persons are excluded ({T('flow')}).",
        f"Internal test set. One person-level 80/20 split of the development cohort, stratified on serum urate bands and shared "
        f"by the three thresholds: {n_(n_tr)} training persons and {n_(n_te)} internal test persons. Every refitted comparison "
        "model is fitted on the same training persons and evaluated on the same internal test persons, unless a table states "
        f"otherwise (temporal holdout, {T('temporal')}).",
        f"External cohort. KNHANES {study_years('knhanes')} participants aged ≥19 years with the 12 inputs and serum urate "
        f"(n = {n_(n_ext)}), with the dipstick fields coded by the label-matched mapping X1 ({T('mapping')}); unweighted unless "
        "stated.",
        "Thresholds. ≥7.0 mg/dL is the primary model, ≥9.0 mg/dL secondary, ≥10.0 mg/dL exploratory; all ≥10.0 mg/dL results "
        f"are in Section {SEC('expl')}.",
        "Statistics. 95% CIs are percentile bootstrap intervals with 1000 resamples at full sample size (seed 42) unless stated. "
        "AUROC differences between models on the same persons: paired bootstrap CI (primary) and DeLong P. Ranges use a hyphen "
        f"({ci(ex_auc['auroc_lo'], ex_auc['auroc_hi'])}); intervals with a negative bound use 'to' "
        f"({ci(ex_neg['intercept_lo'], ex_neg['intercept_hi'], 2)}). P values follow JMIR style (P<.001).",
        "Provenance. Every table and figure is generated from the CSV outputs of the analysis scripts (and, for "
        f"{T('flow')} and {T('missing')}, one line of an analysis log) and names its source file. No result was typed by "
        "hand; design constants (thresholds, bootstrap resamples, seeds) are those of the analysis scripts.",
        "Abbreviations. AUPRC: area under the precision-recall curve; AUROC: area under the receiver operating characteristic "
        "curve; BMI: body mass index; CV: cross-validation; KNHANES: Korea National Health and Nutrition Examination Survey; "
        "NPV/PPV: negative/positive predictive value; O:E: observed-to-expected ratio; SD: standard deviation; SG: specific "
        "gravity; SHAP: Shapley additive explanations; 2F/3F/9F/11F/12F: models with age and sex / age, sex and BMI / the 9 "
        "dipstick fields / dipstick, age and sex / all 12 inputs.",
    ])


# ============================================================ renumbering (appendix of 2026-10-12 -> this appendix)
def write_renumbering_v5():
    """../revision_notes/supplement_renumbering_v5.csv: every row of the supplement_index.csv of 2026-10-12 (tables, the
    keys of their panels, the main-text alias 'adult', figures) -> its label in this appendix or 'removed', and the
    sections. The panel letters of 2026-10-12 come from the history file supplement_renumbering.csv (read only).
    Needs no analysis output; main() calls it after writing the appendix (check_structure() has already checked that every
    key of 2026-10-12 is kept or listed in REMOVED, so the assertion below cannot fail after the appendix is written)."""
    if not (os.path.exists(SNAP_INDEX) and os.path.exists(RENUMBER_HISTORY)):
        log(f"renumbering file not written: {SNAP_INDEX} or {RENUMBER_HISTORY} missing")
        return None
    tables, sections, figs = static_labels()
    old = pd.read_csv(SNAP_INDEX)
    hist = pd.read_csv(RENUMBER_HISTORY)
    old_panel = {r.old_key: ("" if isnan(r.panel) else str(r.panel)) for r in hist.itertuples()}
    old_table = {r.old_key: r.new_label for r in hist.itertuples()}
    rows = []
    for r in old.itertuples():
        key, lab = str(r.key), str(r.label)
        if r.kind == "figure":
            new = figs.get(key)
            rows.append(dict(kind="figure", old_key=key, old_label=lab, new_label=new or "removed",
                             note="" if new else "figure removed"))
            continue
        if key in old_table and key != "adult":
            assert old_table[key] == lab, (key, old_table[key], lab)
            olab = lab + (f" panel {old_panel[key]}" if old_panel.get(key) else "")
        else:
            olab = lab
        if key in tables:
            t_lab, letter = tables[key]
            rows.append(dict(kind="table", old_key=key, old_label=olab,
                             new_label=t_lab + (f" panel {letter}" if letter else ""), note=""))
        else:
            assert key in REMOVED, f"key {key} of the appendix of 2026-10-12 is neither kept nor listed in REMOVED"
            note = ("internal results of adults (former Table S56), resolved to main-text Table 2; no longer an appendix key"
                    if key == "adult" else "removed with the retraining of 2026-10-16")
            rows.append(dict(kind="table", old_key=key, old_label=olab, new_label="removed", note=note))
    for old_lab, title, new_key in OLD_SECTIONS:
        rows.append(dict(kind="section", old_key=old_lab, old_label=f"Section {old_lab}",
                         new_label=f"Section {sections[new_key]}" if new_key else "removed",
                         note=title.replace("≥", ">=")))
    df = pd.DataFrame(rows, columns=["kind", "old_key", "old_label", "new_label", "note"])
    assert all(str(v).isascii() for v in df.astype(str).to_numpy().ravel())
    os.makedirs(os.path.dirname(RENUMBER_V5), exist_ok=True)
    df.to_csv(RENUMBER_V5, index=False)
    log(f"-> {os.path.relpath(RENUMBER_V5, HERE)}: {len(df)} rows ({int((df.new_label == 'removed').sum())} removed)")
    return df


def main():
    global EXPL_COUNTS
    derived = os.path.join(HERE, "supplement_derived.csv")
    if not os.path.exists(derived):
        log("supplement_derived.csv missing: running calc_supplement_derived.py")
        import calc_supplement_derived
        calc_supplement_derived.main()
    os.makedirs(FIGDIR, exist_ok=True)
    check_structure()
    check_inputs()
    c = ci_items("cohort_info_v2")
    pvi = rd("platt_vs_isotonic")
    r10 = one(pvi, target=">=10.0", cohort="internal", method=ISO)
    assert int(r10["train_n"]) == int(c["training persons"]) and int(r10["train_events"]) == int(c["training events >=10.0"])
    EXPL_COUNTS = (f"{n_(c['training events >=10.0'])} training events, {n_(r10['calfold_events_min'])}-"
                   f"{n_(r10['calfold_events_max'])} events per isotonic calibration fold, {n_(c['test events >=10.0'])} events "
                   f"among the internal test persons, {n_(c['KNHANES events >=10.0'])} events in KNHANES adults, "
                   f"{n_(c['KNHANES female events >=10.0'])} of them in women")

    S = load_style()
    log(f"figure style: {S['module']}")
    figpaths = dict(years=fig_years(S, dict(FIGURES)["years"]),
                    perm=fig_perm(S, MAIN, dict(FIGURES)["perm"]),
                    sg=fig_sg(S, MAIN, dict(FIGURES)["sg"]),
                    expl=fig_expl(S, dict(FIGURES)["expl"]))

    doc = Doc()
    conventions(doc)
    section_main_tables(doc, S, figpaths)
    section_exploratory(doc, figpaths)
    assert doc.nt == len(STRUCTURE) and doc.np == sum(len(keys) for _, _, keys in STRUCTURE), (doc.nt, doc.np)
    assert len(doc.sections) == len(SECTIONS) and len(doc.figs) == len(FIGURES)
    assert set(_cache) == set(INPUTS) | {COHORT_FILE}, ("INPUTS differs from the files read",
                                                        sorted(set(_cache) ^ (set(INPUTS) | {COHORT_FILE})))
    tables, sections, figs = static_labels()
    assert {k_: (doc.tables[k_], doc.panel.get(k_, "")) for k_ in tables} == tables, "labels differ from STRUCTURE"
    assert doc.sections == sections and doc.figs == figs

    head = ("# Multimedia Appendix 1\n\n"
            "**Supplementary methods, tables and figures for: " + TITLE + ".**\n")
    log(f"appendix: {doc.nt} tables and {doc.nf} figures (no generation stamp in the text since 2026-10-17)")
    text = doc.render(head)
    # terminology: the appendix describes the models as a self-contained study (FORBIDDEN)
    m = forbidden_hit(text)
    if m:
        raise ValueError(f"terminology: {m.group(0)!r} near {text[max(0, m.start() - 160):m.end() + 60]!r}")
    # style checks: ASCII hyphens only (no en, em or minus dashes), no unresolved placeholders, no NaN leaking into the text
    bad = [ch for ch in ("–", "—", "−", "‐", "‑") if ch in text]
    if bad:
        raise ValueError(f"non-ASCII dash in output: {bad}")
    if "{{" in text or "}}" in text:
        raise ValueError("unresolved placeholder")
    nan_hits = [m.start() for m in re.finditer(r"\bnan\b", text)]
    if nan_hits:
        ctx = text[max(0, nan_hits[0] - 200):nan_hits[0] + 50]
        raise ValueError(f"'nan' in output near: {ctx}")
    caps = re.findall(r"(?m)^\*\*Table S(\d+)\. ", text)
    assert caps == [str(i) for i in range(1, len(STRUCTURE) + 1)], caps
    heads = re.findall(r"(?m)^## (S\d+)\. ", text)
    assert heads == [f"S{i}" for i in range(1, len(SECTIONS) + 1)], heads

    # supplement_index.csv (read by build_manuscript.py and make_checklist.py): every table (panel ''), the keys of its
    # panels as aliases of its label with their panel letter ('' in a table with one panel), the figures. Removed keys are
    # not listed.
    idx = []
    for new_key, _, keys in STRUCTURE:
        idx.append(dict(kind="table", key=new_key, label=doc.tables[new_key], panel=""))
        idx += [dict(kind="table", key=k_, label=doc.tables[k_], panel=doc.panel[k_]) for k_ in keys]
    idx += [dict(kind="figure", key=k_, label=lab, panel="") for k_, lab in doc.figs.items()]
    assert not {r["key"] for r in idx if r["kind"] == "table"} & REMOVED
    assert all((r["label"], r["panel"]) == tables[r["key"]] for r in idx if r["kind"] == "table")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    pd.DataFrame(idx, columns=["kind", "key", "label", "panel"]).to_csv(os.path.join(HERE, "supplement_index.csv"),
                                                                        index=False)
    write_renumbering_v5()
    na = text.count("| NA ") + text.count(" NA |")
    log(f"-> {OUT_MD}: {doc.nt} tables ({doc.np} panels), {len(doc.sections)} sections, {doc.nf} figures, {len(text):,} "
        f"characters, {na} NA cells")
    log(f"-> supplement_index.csv: {len(idx)} rows ({len(STRUCTURE)} tables, {doc.np} panel aliases, {len(doc.figs)} figures)")
    with open(LOG, "w") as fh:
        fh.write(__doc__.strip() + "\n\n" + "\n".join(LOGLINES) + "\n")


if __name__ == "__main__":
    main()
