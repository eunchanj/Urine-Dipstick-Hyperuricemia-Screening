"""Main-text Figures 1-5 of the R1 revision, drawn only from the analysis CSVs in this folder.

The models are the isotonic-calibrated ensembles with 12 inputs (9 dipstick fields, age, sex and BMI), trained on the
training persons of one person-level split that the thresholds share; the internal test persons are therefore the same
for every threshold, and the prediction files are checked for that (one record per person, same persons in the same
order for every target, in preds_internal_clean.csv and in preds_external_corrected.csv).

  Figure 1  Participant flow for the development and external cohorts. Development: the first urine test of every
            person (all ages) and its exclusion by age (the calc_followup.py log line of section a1), the adults and
            the adults with a blood test within +/-14 days (fu_setting.csv, a1), then the exclusion steps of the
            development cohort (cohort_info_v2.csv, items 'step: ...'; cross-checked against supplement_derived.csv
            dev_flow) to the development cohort (the final box) and its split into training and internal test persons
            (cohort_info_v2.csv; checked against preds_internal_clean.csv). The two sources meet at the adult window
            persons (asserted). The age box reads 'Younger than 19 years or implausible age', with 'age not
            determinable' added only when the log counts such persons. The study years in the column header are those
            of dipstick_by_year.csv. KNHANES: the sequential flow of supplement_derived.csv (knhanes_flow) to the
            external validation cohort (checked against cohort_info_v2.csv, preds_external_corrected.csv and, when
            present, fu_knhanes_flow.csv).
  Figure 2  ROC curves, internal test persons and external (KNHANES >= 19 y), >=7.0 and >=9.0 mg/dL.
            Curves from preds_internal_clean.csv / preds_external_corrected.csv; AUROC and bootstrap CI
            from discrimination.csv (the curve's AUROC is checked against it).
  Figure 3  Quantile-decile calibration, internal and external, isotonic ensembles, plus the
            external curve after an intercept-only update (logit(p') = logit(p) + a, slope fixed at 1).
            a = mean of the 100 half-split fits in recalibration.csv. A monotone shift keeps every record
            in its decile, so observed proportions and Wilson CIs are those of calibration_curve.csv (rows of the
            isotonic models: the one method key that starts with 'isotonic');
            only the mean predicted risk moves. O:E from calibration.csv and (after the update, fitted on
            one half and evaluated on the other) recalibration.csv.
  Figure 4  External decision curves: the ensemble as validated (decision_curve.csv, with its CI band)
            and after a cross-fitted intercept update, the 3-input spline logistic model after the same update,
            'test all men', treat all and treat none (fu_net_benefit.csv, fu_net_benefit_ranges.csv; the
            unrecalibrated ensemble and treat all must equal decision_curve.csv, so a stale file stops the run).
  Figure 5  External subgroup AUROC forest (subgroups.csv).

Run:  /opt/anaconda3/bin/python make_figures.py   (writes ../figures/Figure_{1..5}.pdf/.png)
"""
import re
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from sklearn.metrics import roc_curve, roc_auc_score
import figstyle as st
from figstyle import NAVY, GREY, AMBER, INK, GREEN, RED
from matplotlib.patches import FancyBboxPatch
from common import logit, expit, decile_curve

st.apply()
HERE = Path(__file__).resolve().parent
TGT = {"uric_hyper_70": ">=7.0", "uric_severe_9": ">=9.0"}
CODE = {v: k for k, v in TGT.items()}
NAME = {">=7.0": "Serum urate ≥7.0 mg/dL (primary)", ">=9.0": "Serum urate ≥9.0 mg/dL (secondary)"}
TITLE_EXT = {">=7.0": "Serum urate ≥7.0 mg/dL, external", ">=9.0": "Serum urate ≥9.0 mg/dL, external"}
SHORT = {"internal": "Internal test", "external": "KNHANES ≥19 y"}
COL = {"internal": AMBER, "external": NAVY}
COLLISIONS = []


def pct_range(s):
    """'0.02-0.44; 0.5' -> '2-44%, 50%' (thresholds as percentages, ASCII hyphen)."""
    def f(v):
        x = float(v) * 100
        return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"
    parts = []
    for seg in str(s).split(";"):
        seg = seg.strip()
        if not seg or seg == "none":
            continue
        a, *b = seg.split("-")
        parts.append(f(a) + (f"-{f(b[0])}" if b else ""))
    return ", ".join(p + "%" for p in parts) if parts else "none"


def _dense(xy, step=1.0):
    """Points along a polyline in display space, at most `step` px apart."""
    xy = np.asarray(xy, float)
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 2:
        return xy
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        k = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        out.append(a + (b - a) * np.linspace(0, 1, k + 1)[1:, None])
    return np.vstack(out)


def check_clear(fig, pad=2.5):
    """Report any legend/in-axes text that touches a plotted mark, another label or the axes edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); problems = []
    for ax in fig.axes:
        abox = ax.get_window_extent(r)
        labels = ([("legend", ax.get_legend().get_window_extent(r))] if ax.get_legend() else []) + \
                 [(t.get_text().split("\n")[0][:30], t.get_window_extent(r)) for t in ax.texts
                  if t.get_text().strip() and not isinstance(t, Annotation)]
        pts = []
        for ln in ax.lines:
            pts.append(_dense(ln.get_transform().transform(ln.get_xydata())))
        polys = []
        for c in ax.collections:
            if hasattr(c, "get_segments"):
                for sgm in c.get_segments():
                    pts.append(_dense(c.get_transform().transform(sgm)))
            else:
                polys += [c.get_transform().transform_path(p) for p in c.get_paths()]
        pts = np.vstack([p for p in pts if len(p)]) if pts else np.empty((0, 2))
        for i, (name, b) in enumerate(labels):
            x0, y0, x1, y1 = b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad
            hit = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)]
            if len(hit):
                problems.append(f"{ax.get_title(loc='left')[:24]} | '{name}' touches {len(hit)} mark points")
            gx, gy = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(y0, y1, 12))
            g = np.column_stack([gx.ravel(), gy.ravel()])
            if any(p.contains_points(g).any() for p in polys):
                problems.append(f"{ax.get_title(loc='left')[:24]} | '{name}' overlaps a shaded band")
            if b.x0 < abox.x0 or b.x1 > abox.x1 or b.y0 < abox.y0 or b.y1 > abox.y1:
                problems.append(f"{ax.get_title(loc='left')[:24]} | '{name}' extends outside the axes")
            for name2, b2 in labels[i + 1:]:
                if b.overlaps(b2):
                    problems.append(f"{ax.get_title(loc='left')[:24]} | '{name}' overlaps '{name2}'")
        tb = ax._left_title.get_window_extent(r)          # titles are set with loc="left"
        if tb.x1 > fig.bbox.x1 or tb.y1 > fig.bbox.y1:
            problems.append(f"title '{ax.get_title(loc='left')[:24]}' runs off the figure")
        for other in fig.axes:                             # panel letters vs every other panel's title
            for t in other.texts:
                if isinstance(t, Annotation) and other is not ax and t.get_window_extent(r).overlaps(tb):
                    problems.append(f"panel letter '{t.get_text()}' overlaps title '{ax.get_title(loc='left')[:24]}'")
    for p in problems:
        print("    COLLISION:", p)
    COLLISIONS.extend(problems)
    return problems


preds = {"internal": pd.read_csv(HERE / "preds_internal_clean.csv"),
         "external": pd.read_csv(HERE / "preds_external_corrected.csv")}
# one record per person, and the same persons in the same order for every threshold: the internal test persons of the
# one person-level split shared by the thresholds, and the KNHANES adults
for coh, pr in preds.items():
    assert set(CODE.values()) <= set(pr.target), (coh, "a target is missing")
    ids = [g["id"].astype(str).to_numpy() for _, g in pr.groupby("target", sort=False)]
    assert all(len(set(i)) == len(i) for i in ids), (coh, "a person appears twice for one target")
    assert all(np.array_equal(i, ids[0]) for i in ids[1:]), (coh, "the persons differ between thresholds")
disc = pd.read_csv(HERE / "discrimination.csv").set_index(["target", "cohort"])

# ────────────────────────────────────────────────────────────── Figure 1: participant flow
# Development (adults, the target population, as in KNHANES): the first urine test of every person (all ages) -> adults
# -> blood test within +/-14 days -> serum urate of the urine-test day -> BMI of the urine-test day -> all 9 dipstick
# fields and sex = development cohort (cohort_info_v2.csv: 'analytic cohort persons') -> one person-level split into
# training and internal test persons, shared by the
# thresholds. Up to the adults with a blood test within +/-14 days the counts are those of the first-visit urine tests
# (calc_followup.py section a1: fu_setting.csv and its log line); from there on they are the exclusion steps of
# cohort.dev_v2 (cohort_info_v2.csv). The two sources meet at the adult window persons (asserted). KNHANES: the
# sequential flow of supplement_derived.csv (knhanes_flow) to the external validation cohort.
ci2 = pd.read_csv(HERE / "cohort_info_v2.csv")
assert ci2["item"].is_unique, "cohort_info_v2.csv: an item is listed twice"
CI2 = dict(zip(ci2["item"].astype(str), ci2["value"]))


def ci2n(item):
    """Integer value of one item of cohort_info_v2.csv."""
    v = float(CI2[item])
    assert v == int(v), (item, v)
    return int(v)


def persons(n, unit="person"):
    return f"{n:,} {unit}" + ("" if n == 1 else "s")


# exclusion steps of cohort.dev_v2 in the order applied: window persons (all ages), adults, serum urate of the
# urine-test day, BMI of the urine-test day, all 9 dipstick fields and sex (analytic cohort)
STEPS = [(it[len("step: "):], ci2n(it)) for it in CI2 if it.startswith("step: ")]
STEP_KEYS = ("blood test within 14 days", "years or older", "serum urate", "BMI", "dipstick")
assert len(STEPS) == len(STEP_KEYS) and all(k in s for (s, _), k in zip(STEPS, STEP_KEYS)), STEPS
m_age = re.search(r"aged (\d+) years or older", STEPS[1][0])
assert m_age, STEPS[1][0]
ADULT = int(m_age.group(1))
n_win_all, n_win, n_urate, n_bmi, n_dev = (n for _, n in STEPS)
assert n_win_all >= n_win >= n_urate >= n_bmi >= n_dev == ci2n("analytic cohort persons")
n_tr, n_te = ci2n("training persons"), ci2n("test persons")
assert n_tr + n_te == n_dev
SPLIT = (round(100 * n_tr / n_dev), round(100 * n_te / n_dev))
assert sum(SPLIT) == 100, SPLIT
TT = (">=7.0", ">=9.0")
EV = {(g, t): ci2n(f"{g} events {t}") for g in ("cohort", "training", "test", "KNHANES") for t in TT}
for t in TT:
    assert EV[("training", t)] + EV[("test", t)] == EV[("cohort", t)], t
    d_ = preds["internal"][preds["internal"].target == CODE[t]]
    assert len(d_) == n_te and int(d_.y.sum()) == EV[("test", t)], (t, "preds_internal_clean.csv vs cohort_info_v2.csv")

fs = pd.read_csv(HERE / "fu_setting.csv")


def fsrow(section, group):
    r = fs[(fs.section == section) & (fs.group == group)]
    assert len(r) == 1, (section, group)
    return r.iloc[0]


def first_visit_age_exclusion():
    """First-visit persons of all ages and their exclusion by age, as logged by calc_followup.py section (a1). No CSV
    holds the excluded counts (fu_setting.csv is adults only), so they are parsed from the log line that
    make_supplement.py also reads for its flow table; the adult count is checked against fu_setting.csv below."""
    txt = (HERE / "calc_followup_log.txt").read_text(encoding="utf-8")
    m = re.search(r"first-visit persons ([\d,]+) \(all ages\); adult \((\d+)-\d+ y\) ([\d,]+) \([\d,]+ unique\); "
                  r"excluded: aged <(\d+) ([\d,]+), age <0 or >\d+ ([\d,]+), age not determinable[^;]*? ([\d,]+);", txt)
    assert m, "first-visit age exclusion not found in calc_followup_log.txt"
    num = lambda s: int(s.replace(",", ""))  # noqa: E731
    out = dict(all=num(m.group(1)), adult=num(m.group(3)), minor=num(m.group(5)), implausible=num(m.group(6)),
               unknown=num(m.group(7)))
    assert int(m.group(2)) == ADULT and int(m.group(4)) == ADULT
    assert out["all"] == out["adult"] + out["minor"] + out["implausible"] + out["unknown"], out
    return out


a_all = fsrow("a1 first-visit urine tests", "all")
a_uo = fsrow("a1 first-visit urine tests", "no blood within ±14 d (urine-only encounter)")
a_win = fsrow("a1 first-visit urine tests", "blood within ±14 d")
fva = first_visit_age_exclusion()
n_fv, n_adult, n_noblood = fva["all"], int(a_all.n), int(a_uo.n)
assert fva["adult"] == n_adult and float(a_all.pct_age_lt19) == 0.0
# adults minus the adults without a blood test within +/-14 days = the adult window persons of cohort.dev_v2
assert n_adult - n_noblood == int(a_win.n) == n_win, (n_adult, n_noblood, int(a_win.n), n_win)
n_age = n_fv - n_adult     # first-visit persons younger than 19 years or with an implausible or unknown age
assert n_fv >= n_win_all and n_age >= n_win_all - n_win >= 0
assert n_age == fva["minor"] + fva["implausible"] + fva["unknown"]
# label of the age box: the reasons that occur (an implausible age is <0 or above the adult range of cohort.py)
AGE_WHY = ([f"Younger than {ADULT} years"] + (["implausible age"] if fva["implausible"] else [])
           + (["age not determinable"] if fva["unknown"] else []))
# years of the development urine tests (dipstick_by_year.csv; the development cohort by calendar year)
dby = pd.read_csv(HERE / "dipstick_by_year.csv")
dev_yr = dby[(dby.period_type == "year") & dby.cohort.astype(str).str.lower().str.startswith("development")].period.astype(int)
assert len(dev_yr) and int(dby.loc[dev_yr.index, "n"].sum()) == n_dev, "dipstick_by_year.csv: development years"
DEV_YEARS = f"{dev_yr.min()}-{dev_yr.max()}"

der = pd.read_csv(HERE / "supplement_derived.csv")
dfl = der[(der.section == "dev_flow") & (der.order < 90)].sort_values("order")
if len(dfl):
    # calc_supplement_derived.py dev_flow: the same steps, with the age restriction split into <19 and >110 years
    rest = iter(int(v) for v in dfl.n_persons)
    assert all(any(v == n for v in rest) for _, n in STEPS), ("dev_flow differs from cohort_info_v2.csv",
                                                              list(dfl.n_persons))
else:
    print("  Figure 1: no dev_flow section in supplement_derived.csv; cross-check skipped")
kfl = der[(der.section == "knhanes_flow") & (der.order < 90)].sort_values("order").reset_index(drop=True)
assert len(kfl) >= 2 and int(kfl.order.iloc[0]) == 0, "supplement_derived.csv: no knhanes_flow steps"
n_kn, n_ext = int(kfl.n.iloc[0]), int(kfl.n.iloc[-1])
assert n_ext == ci2n("KNHANES adults with 12 inputs and urate")
for t in TT:
    d_ = preds["external"][preds["external"].target == CODE[t]]
    assert len(d_) == n_ext and int(d_.y.sum()) == EV[("KNHANES", t)], (t, "preds_external_corrected.csv vs "
                                                                         "cohort_info_v2.csv")
if (HERE / "fu_knhanes_flow.csv").exists():
    fk = pd.read_csv(HERE / "fu_knhanes_flow.csv")
    fk = fk[fk.section == "adult attrition"]
    assert fk.empty or int(fk.n.iloc[-1]) == n_ext, "fu_knhanes_flow.csv (adult attrition) does not end at the cohort"
# reason of each knhanes_flow step, recognised by keyword (the step labels of calc_supplement_derived.py; each step must
# match exactly one reason, so a relabelled or new step stops the run)
KN_WHY = [(r"\burate\b", "No serum urate"), (r"\bdipstick\b", "Dipstick field missing"),
          (r"\bage and sex\b", "Age or sex missing"), (r"\bBMI\b", "BMI missing"),
          (rf"\baged (?:>= ?)?{ADULT}\b|\b{ADULT} years or older\b|\badults\b", f"Aged <{ADULT} years")]
kn_x = []                  # (reason, participants excluded) in the order of the flow
for i in range(1, len(kfl)):
    r = kfl.iloc[i]
    why = [txt for pat, txt in KN_WHY if re.search(pat, str(r.step), flags=re.I)]
    assert len(why) == 1, ("knhanes_flow step not recognised", r.step)
    n_x = int(kfl.n.iloc[i - 1]) - int(r.n)
    assert n_x == int(r.n_excluded) and n_x >= 0, r.step
    kn_x.append((why[0], n_x))
assert sum(n for _, n in kn_x) == n_kn - n_ext

FIG1_H = 6.7                 # inches; line spacing and box padding (in points) are those of the 4.9-inch layout
SC = 4.9 / FIG1_H
fig = st.plt.figure(figsize=(st.WIDTH_IN, FIG1_H))
ax = fig.add_axes([0.005, 0.005, 0.99, 0.99]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
LH = 2.55 * SC   # line height in axis units
PAD = 2.2 * SC   # vertical padding inside a box
FLOW = []        # (first line, extent, text artists) of every box, for check_flow()


def hbox(n_lines):
    return LH * n_lines + PAD


def box(x0, yc, w, lines, kind="main", n_bold=1):
    h = hbox(len(lines))
    fc = {"main": "#eef2f7", "excl": "white", "note": "#f7f7f7"}[kind]
    ec = NAVY if kind == "main" else GREY
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc=fc, ec=ec, lw=0.9 if kind == "main" else 0.7))
    arts = [ax.text(x0 + 1.2, yc + h / 2 - 1.3 * SC - LH * (i + 0.5), ln, ha="left", va="center", fontsize=6.9,
                    color=INK, fontweight="bold" if i < n_bold else "normal") for i, ln in enumerate(lines)]
    FLOW.append((lines[0], (x0, yc - h / 2, x0 + w, yc + h / 2), arts))
    return yc - h / 2, yc + h / 2


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK,
                                                                  shrinkA=0, shrinkB=0, mutation_scale=7))


def check_flow(fig, notes):
    """Figure 1: every line of text inside its box, no two boxes overlapping, and each free-standing note clear of every
    box and inside the figure; problems are reported and collected like those of check_clear()."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); inv = ax.transData.inverted(); problems = []

    def ext(a):
        b = a.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform((b.x0, b.y0)), inv.transform((b.x1, b.y1))
        return x0, y0, x1, y1

    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    for name, bb, arts in FLOW:
        for a in arts:
            x0, y0, x1, y1 = ext(a)
            if x0 < bb[0] or x1 > bb[2] - 0.6 or y0 < bb[1] or y1 > bb[3]:
                problems.append(f"Figure 1 | '{a.get_text()[:30]}' runs outside its box")
    for i, (name, bb, _) in enumerate(FLOW):
        for name2, bb2, _ in FLOW[i + 1:]:
            if hit(bb, bb2):
                problems.append(f"Figure 1 | box '{name[:24]}' overlaps box '{name2[:24]}'")
    for t in notes:
        e = ext(t)
        if e[0] < 0 or e[2] > 100 or e[1] < 0 or e[3] > 100:
            problems.append(f"Figure 1 | note '{t.get_text()[:30]}' runs off the figure")
        for name, bb, _ in FLOW:
            if hit(e, bb):
                problems.append(f"Figure 1 | note '{t.get_text()[:30]}' overlaps box '{name[:24]}'")
    for p in problems:
        print("    COLLISION:", p)
    COLLISIONS.extend(problems)
    return problems


def events(g):
    return f"≥7.0 mg/dL: {EV[(g, '>=7.0')]:,}; ≥9.0 mg/dL: {EV[(g, '>=9.0')]:,}"


ax.text(1, 100 - 1.7 * SC, f"Development (tertiary hospital, {DEV_YEARS})", fontsize=8.2, fontweight="bold", color=NAVY,
        va="center")
ax.text(66, 100 - 1.7 * SC, "External validation (KNHANES)", fontsize=8.2, fontweight="bold", color=NAVY, va="center")
XM, WM, XE, WE, XK, WK = 1, 31, 35.5, 27.5, 66, 32
LM = [["First-visit urine test, one per person", f"{persons(n_fv)} (all ages)"],
      [f"Adults aged ≥{ADULT} years", persons(n_adult)],
      ["Blood test within ±14 days", persons(n_win)],
      ["Serum urate on the urine-test day", persons(n_urate)],
      ["BMI on the urine-test day", persons(n_bmi)],
      ["Development cohort", f"{persons(n_dev)} (one record each)", events("cohort")]]
age_why = (AGE_WHY[0] if len(AGE_WHY) == 1 else f"{AGE_WHY[0]} or {AGE_WHY[1]}" if len(AGE_WHY) == 2
           else ", ".join(AGE_WHY[:-1]) + f", or {AGE_WHY[-1]}")
LX = [[f"Excluded: {persons(n_age)}"] + textwrap.wrap(f"{age_why} (a)", width=30),
      [f"Excluded: {persons(n_noblood)}", "No blood test within ±14 days"],
      [f"Excluded: {persons(n_win - n_urate)}", "No serum urate on the urine-test day"],
      [f"Excluded: {persons(n_urate - n_bmi)}", "No BMI on the urine-test day"],
      [f"Excluded: {persons(n_bmi - n_dev)}", "Dipstick field missing"]]
LT = ["Training set", persons(n_tr), events("training")]
LV = ["Internal test set", persons(n_te), events("test")]
# vertical layout of the development column: each gap holds its side box (or, the last one, the split note and the
# branching arrows); the spare height is shared equally by the gaps
TOP, BOTTOM = 100 - 4.2 * SC, 1.0
MARGIN = 1.6 * SC            # least space between a side box and the main boxes above and below it
H_ROW = [hbox(len(m)) for m in LM] + [hbox(max(len(LT), len(LV)))]
NEED = [hbox(len(x)) + 2 * MARGIN for x in LX] + [2.6 * LH]
spare = TOP - BOTTOM - sum(H_ROW) - sum(NEED)
assert spare >= 0, f"Figure 1: {-spare:.1f} axis units short; increase FIG1_H"
yc, y = [], TOP
for h, g in zip(H_ROW, NEED + [0]):
    yc.append(y - h / 2)
    y -= h + g + spare / len(NEED)
bm = [box(XM, c, WM, m) for c, m in zip(yc, LM)]
bt, bv = box(XM, yc[-1], WM, LT), box(XE, yc[-1], WE, LV)
xc = XM + WM / 2
for i, lines in enumerate(LX):
    ym = (bm[i][0] + bm[i + 1][1]) / 2
    box(XE, ym, WE, lines, kind="excl")
    arrow(xc, bm[i][0], xc, bm[i + 1][1]); arrow(xc, ym, XE, ym)
# one person-level split, shared by the thresholds: training persons (below) and internal test persons (right)
y_br = bt[1] + 0.35 * (bm[-1][0] - bt[1])
xv = XE + WE / 2
arrow(xc, bm[-1][0], xc, bt[1])
ax.plot([xc, xv], [y_br, y_br], color=INK, lw=0.8)
arrow(xv, y_br, xv, bv[1])
split_note = ax.text(xc + 1.5, (bm[-1][0] + y_br) / 2,
                     f"Person-level {SPLIT[0]}/{SPLIT[1]} split, one for all thresholds",
                     ha="left", va="center", fontsize=6.6, color=INK)
# KNHANES column
# KNHANES holds one record per participant: the boxes count participants (Reviewer BZ comment 1; 2026-10-17)
c1 = box(XK, yc[0], WK, ["Survey participants, all ages", f"{n_kn:,} participants"])
c2 = box(XK, yc[len(LM) - 1], WK, ["External validation cohort", f"{n_ext:,} adults aged ≥{ADULT} years",
                                   events("KNHANES")])
c_ex = box(XK + 4, (c1[0] + c2[1]) / 2, WK - 4,
           [f"Excluded: {n_kn - n_ext:,} participants"]
           + [f"{why}: {n:,}" + (" (a)" if why.startswith("Aged") else "") for why, n in kn_x if n], kind="excl")
xk = XK + 2
ax.plot([xk, xk], [c1[0], c2[1]], color=INK, lw=0.8); arrow(xk, c2[1] + 0.01, xk, c2[1])
arrow(xk, (c_ex[0] + c_ex[1]) / 2, XK + 4, (c_ex[0] + c_ex[1]) / 2)
foot = ax.text(XK, c2[0] - 4.0 * SC, "(a) Outside the adult target population.", fontsize=6.6, color=INK, va="top",
               linespacing=1.4)
check_flow(fig, [split_note, foot])
st.save(fig, "Figure_1")

# ────────────────────────────────────────────────────────────── Figure 2: ROC
fig, axes = st.figure(3.75, 1, 2, wspace=0.08)
for ax, (L, t) in zip(axes, [("A", ">=7.0"), ("B", ">=9.0")]):
    ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.0)
    handles = []
    for coh, ls in [("internal", "--"), ("external", "-")]:
        d = preds[coh][preds[coh].target == CODE[t]]
        r = disc.loc[(t, coh)]
        assert len(d) == r.n and d.y.sum() == r.positives, (t, coh)
        assert abs(roc_auc_score(d.y, d.p) - r.auroc) < 1e-9, (t, coh)
        fpr, tpr, _ = roc_curve(d.y, d.p)
        h, = ax.plot(fpr, tpr, color=COL[coh], ls=ls, lw=1.6,
                     label=f"{SHORT[coh]}: {int(r.positives):,} cases / {int(r.n):,}\n"
                           f"AUROC {r.auroc:.3f} (95% CI {r.auroc_lo:.3f}-{r.auroc_hi:.3f})")
        handles.append(h)
    ax.legend(handles=handles, loc="lower right", handlelength=1.8, handletextpad=0.5, labelspacing=0.7,
              borderaxespad=0.3, fontsize=7.0)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    st.grid(ax, "both"); st.title(ax, NAME[t]); st.panel(ax, L)
check_clear(fig)
st.save(fig, "Figure_2")

# ────────────────────────────────────────────────────────────── Figure 3: calibration
cc = pd.read_csv(HERE / "calibration_curve.csv")
ISO = sorted(m for m in cc.model.unique() if str(m).startswith("isotonic"))
assert len(ISO) == 1, ("calibration_curve.csv: method key of the isotonic models", ISO)
cc = cc[(cc.model == ISO[0]) & (cc.bins_requested == 10)]
cal = pd.read_csv(HERE / "calibration.csv").set_index(["target", "cohort"])
rec = pd.read_csv(HERE / "recalibration.csv").set_index("target")
LIM = {">=7.0": 0.55, ">=9.0": 0.125}


def curve(t, coh, offset=0.0):
    """Decile curve with the binning of common.decile_curve; CIs joined from calibration_curve.csv."""
    d = preds[coh][preds[coh].target == CODE[t]]
    y, p = d.y.to_numpy(), d.p.to_numpy()
    ref = cc[(cc.target == t) & (cc.cohort == coh)].set_index("bin")
    raw = decile_curve(y, p, 10).set_index("bin")
    assert np.array_equal(raw.n.to_numpy(), ref.n.to_numpy()) and np.allclose(raw.mean_pred, ref.mean_pred) \
        and np.allclose(raw.observed, ref.observed), (t, coh, "curve does not reproduce calibration_curve.csv")
    if offset:
        c = decile_curve(y, expit(logit(p) + offset), 10).set_index("bin")
        assert np.array_equal(c.n.to_numpy(), ref.n.to_numpy()) and np.allclose(c.observed, ref.observed)
    else:
        c = raw
    return c.assign(lo=ref.observed_lo, hi=ref.observed_hi)


fig, axes = st.figure(3.75, 1, 2, wspace=0.08)
for ax, (L, t) in zip(axes, [("A", ">=7.0"), ("B", ">=9.0")]):
    lim = LIM[t]
    ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.1)
    a = rec.loc[t, "intercept_fit"]
    series = [("internal", 0.0, dict(color=AMBER, ls="-", marker="D", ms=3.8, mfc=AMBER)),
              ("external", 0.0, dict(color=NAVY, ls="--", marker="o", ms=4.0, mfc="white", mew=1.2)),
              ("external", a, dict(color=NAVY, ls="-", marker="o", ms=4.0, mfc=NAVY))]
    for coh, off, sty in series:
        c = curve(t, coh, off)
        assert c.mean_pred.max() < lim and c.hi.max() < lim, (t, coh, "outside axis limit")
        ax.errorbar(c.mean_pred, c.observed, yerr=[c.observed - c.lo, c.hi - c.observed], lw=1.4, elinewidth=0.8,
                    capsize=0, **sty)
    ci, ce = cal.loc[(t, "internal")], cal.loc[(t, "external")]
    r = rec.loc[t]
    leg = [Line2D([], [], color=AMBER, marker="D", ms=3.8, lw=1.4, label="Internal test, as validated"),
           Line2D([], [], color=NAVY, ls="--", marker="o", ms=4.0, mfc="white", mew=1.2, lw=1.4,
                  label="External, as validated"),
           Line2D([], [], color=NAVY, marker="o", ms=4.0, lw=1.4, label=f"External, intercept update ({a:+.2f})"),
           Line2D([], [], color=GREY, ls=":", lw=1.1, label="Ideal calibration")]
    lg = ax.legend(handles=leg, loc="upper left", borderaxespad=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted risk (decile of predicted risk)"); ax.set_ylabel("Observed proportion (95% CI)")
    st.grid(ax, "both"); st.title(ax, NAME[t]); st.panel(ax, L)
    fig.canvas.draw()
    lb = lg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(lb.x0 + 0.02, lb.y0 - 0.02,
            f"Observed/expected (95% CI)\n"
            f"Internal test: {ci.oe:.2f} ({ci.oe_lo:.2f}-{ci.oe_hi:.2f})\n"
            f"External: {ce.oe:.2f} ({ce.oe_lo:.2f}-{ce.oe_hi:.2f})\n"
            f"After update: {r.oe_after_intercept:.2f} ({r.oe_after_intercept_lo:.2f}-{r.oe_after_intercept_hi:.2f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=st.FS["note"], color=INK, linespacing=1.35)
check_clear(fig)
st.save(fig, "Figure_3")

# ────────────────────────────────────────────────────────────── Figure 4: decision curves (external)
dca = pd.read_csv(HERE / "decision_curve.csv")
fnb = pd.read_csv(HERE / "fu_net_benefit.csv")
frg = pd.read_csv(HERE / "fu_net_benefit_ranges.csv")
K = 1000   # net benefit per 1,000 persons
SPEC = {">=7.0": dict(xlim=(0, 50), ylim=(-12, 106), xt=np.arange(0, 51, 10)),
        ">=9.0": dict(xlim=(0, 10), ylim=(-5.6, 9.6), xt=np.arange(0, 10.1, 2))}
fig, axes = st.figure(3.8, 1, 2, wspace=0.08)
for ax, (L, t) in zip(axes, [("A", ">=7.0"), ("B", ">=9.0")]):
    d = dca[(dca.target == t) & (dca.cohort == "external")].sort_values("threshold")
    u = fnb[fnb.target == t].sort_values("threshold")
    assert np.allclose(u.threshold.to_numpy(), d.threshold.to_numpy())
    assert np.allclose(u.ensemble_unrecalibrated.to_numpy(), d.nb_model.to_numpy())
    assert np.allclose(u.treat_all.to_numpy(), d.nb_all.to_numpy())
    x = d.threshold * 100
    ax.axhline(0, color=INK, lw=1.0)
    ax.plot(x, d.nb_all * K, color=GREY, lw=1.3)
    ax.fill_between(x, d.nb_model_lo * K, d.nb_model_hi * K, color=NAVY, alpha=0.15, lw=0)
    ax.plot(x, d.nb_model * K, color=NAVY, lw=1.8)
    ax.plot(x, u.ensemble_recalibrated_crossfit * K, color=GREEN, lw=1.6)
    ax.plot(x, u["3F_splineLR_recalibrated_crossfit"] * K, color=AMBER, lw=1.4, ls="--")
    ax.plot(x, u.test_all_men * K, color=RED, lw=1.3, ls=(0, (1, 1.2)))
    s_ = SPEC[t]
    ax.set_xlim(*s_["xlim"]); ax.set_ylim(*s_["ylim"]); ax.set_xticks(s_["xt"])
    ax.set_xlabel("Threshold probability, %"); ax.set_ylabel("Net benefit per 1,000 persons")
    leg = [(Line2D([], [], color=NAVY, lw=1.8), Patch(color=NAVY, alpha=0.15, lw=0)),
           Line2D([], [], color=GREEN, lw=1.6), Line2D([], [], color=AMBER, lw=1.4, ls="--"),
           Line2D([], [], color=RED, lw=1.3, ls=(0, (1, 1.2))),
           Line2D([], [], color=GREY, lw=1.3), Line2D([], [], color=INK, lw=1.0)]
    lab = ["Ensemble, as validated (95% CI)", "Ensemble, intercept updated", "Age, sex, BMI model, intercept updated",
           "Test all men", "Treat all", "Treat none"]
    ax.legend(leg, lab, loc="upper right", borderaxespad=0.4, handlelength=2.4)
    rg = frg[frg.target == t].set_index("criterion").ranges
    st.grid(ax, "both"); st.title(ax, TITLE_EXT[t]); st.panel(ax, L)
    fig.canvas.draw()
    lb = ax.get_legend().get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(0.97, lb.y0 - 0.03,
            "Intercept-updated ensemble (95% CI > 0)\n"
            f"above both defaults: {pct_range(rg['ensemble recalibrated beats both defaults (CI lower > 0)'])}\n"
            f"above test all men: {pct_range(rg['ensemble recalibrated beats test all men (CI lower > 0)'])}",
            transform=ax.transAxes, ha="right", va="top", fontsize=st.FS["note"], color=INK, linespacing=1.35)
check_clear(fig)
st.save(fig, "Figure_4")

# ────────────────────────────────────────────────────────────── Figure 5: subgroup forest (external)
sg = pd.read_csv(HERE / "subgroups.csv")
sg = sg[sg.cohort.str.startswith("External")]
assert sg.groupby("target").cohort.nunique().eq(1).all(), "subgroups.csv: more than one external cohort per target"
ROWS = [("All adults", "All"), ("Age group", None), ("19-39 y", "Age 19-39"), ("40-59 y", "Age 40-59"),
        ("60-79 y", "Age 60-79"), ("≥80 y", "Age >=80"), ("Sex", None), ("Male", "Male"), ("Female", "Female")]
fig, axes = st.figure(3.3, 1, 2, wspace=0.05)
for j, (ax, (L, t)) in enumerate(zip(axes, [("A", ">=7.0"), ("B", ">=9.0")])):
    d = sg[sg.target == t].set_index("subgroup")
    assert d.n_boot_valid.eq(1000).all()
    overall = d.loc["All", "auroc"]
    ax.axvline(overall, color=GREY, ls="--", lw=0.9)
    for i, (lbl, key) in enumerate(ROWS):
        if key is None:
            continue
        r = d.loc[key]
        ax.plot([r.auroc_lo, r.auroc_hi], [i, i], color=NAVY, lw=1.6, solid_capstyle="butt")
        ax.plot(r.auroc, i, "D" if key == "All" else "s", color=NAVY, ms=5.2 if key == "All" else 4.4)
        ax.text(1.045, i, f"{int(r.positives):,} / {int(r.n):,}", ha="right", va="center", fontsize=st.FS["note"],
                color=INK)
    ax.text(1.045, -1.0, "Cases / n", ha="right", va="center", fontsize=st.FS["note"], color=INK, fontweight="bold")
    ax.set_ylim(len(ROWS) - 0.4, -1.6)
    ax.set_yticks(range(len(ROWS)))
    if j == 0:
        ax.set_yticklabels([lbl for lbl, _ in ROWS])
        for tl, (_, key) in zip(ax.get_yticklabels(), ROWS):
            if key is None:
                tl.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.set_xlim(0.55, 1.05); ax.set_xticks(np.arange(0.6, 1.001, 0.1))
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    st.grid(ax, "x"); st.title(ax, TITLE_EXT[t]); st.panel(ax, L)
check_clear(fig)
st.save(fig, "Figure_5")
print("collisions:", len(COLLISIONS))
