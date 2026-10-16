"""Sensitivity analysis for the nitrite parsing error in the development data.

Finding (confirmed upstream): raw nitrite results recorded as the text 'Positive' or 'POSITIVE' were mapped
to missing instead of 1 when the processed file was built; '+' was mapped to 1 and '-' / 'Negative' /
'NEGATIVE' to 0. Nitrite therefore looked (almost) constant in development, and nitrite-positive persons
were dropped from the complete-case cohort as "missing". This is a parsing error, not a laboratory
reporting convention. The other dipstick fields are unaffected.

(1) Recovery. The processed rows are each person's FIRST urine visit (the processed file carries the same
    '소변검사일시' as the raw first-visit file), so the raw text is merged on 일련번호 and the visit is
    confirmed by an identical urine date-time (full processed file) and date (window rows). Recovered
    coding (after stripping whitespace, case-insensitive): 'POSITIVE' / '+' -> 1, 'NEGATIVE' / '-' -> 0,
    anything else (raw '0', empty) stays missing. The recovered value must equal the processed value
    wherever the processed value is not missing. The corrected analytic cohort is cohort.dev_corrected()
    applied to the +/-14-day window with the processed nitrite filled from the recovered value (13 inputs
    and urate present, adults aged ADULT_AGE-AGE_MAX, one record per person); "recovered persons" are the
    persons it adds to the current corrected cohort (they were never in the submitted training or test rows,
    because all their rows had nitrite missing).
(2) Evaluation of the DEPLOYED calibrated soft-voting ensembles (loaded read-only, never refitted) on
    the recovered persons, with nitrite = 1 as input (as at deployment) and, as a sensitivity, nitrite = 0;
    and on the union of the current person-clean internal test rows of adults (as calc_original.py;
    predictions must match preds_internal_clean.csv) plus the recovered persons. Extra rows: the person-clean internal
    test alone (reference), women only (the recovered persons are mostly women), and a "proportional"
    union that adds only the 20% of recovered persons a stratified 80/20 split (seed 42) would have put in
    the test set. Metrics: AUROC and AUPRC with bootstrap CI, calibration intercept (offset MLE),
    calibration slope (exact ML, common.fit_logistic_1d), O:E, Brier, scaled Brier; 1000 resamples,
    seed 42, the common.boot_ci scheme; paired nitrite 1 vs 0 AUROC difference (common.paired_boot,
    DeLong).
(3) What the deployed models do with nitrite: the same records predicted with nitrite set to 0 and to 1
    (recovered persons, person-clean internal test, KNHANES adults), and the change decomposed by base
    learner (LightGBM / XGBoost / CatBoost, 5 calibration folds), with the number of tree splits on
    nitrite and the nitrite-positive rows among the ADULT rows of the submitted training data.

ADULT RESTRICTION (adult target population, decision of 2026-10-11): hospital data are adults only (age at the urine
test 19-110, cohort.ADULT_AGE / AGE_MAX). The raw first-visit file has no age column, so the age at the urine
test (age_at_urine_test) is joined by 일련번호 from the processed file, whose rows are the same first visit (the age
is constant within person, asserted); raw persons without a processed record have no determinable age and are
excluded. The processed file is restricted by its own age_at_urine_test and the +/-14-day window by
cohort.adult_window, so every recovery count, positivity and characteristic is adult-only; dev_corrected is adults
only by construction, so the recovered persons are adults. The full window is used only to rebuild the locked
models' submitted row frame and split (dev_submitted + split_submitted, seed 42): the internal test set is the
ADULT person-clean rows (= preds_internal_clean.csv) and nitrite-positive training rows are counted among adult
training rows. Exclusion counts (aged < 19, > 110, age not determinable) are logged and appended to
nitrite_recovery.csv as section "adult_restriction". KNHANES (ext_corrected, >= 19, and the two KNHANES record-level
positivity rows) is unchanged. Rows kept for structure: characteristics item "age < 19 n" is now 0; positivity
groups "window rows", "window persons (first record)" and "analytic cohort" are now computed on adult frames and
equal their ", adults >= 19" rows. The all-ages version of this script and its outputs is in
_snapshot_allages_261011/.

Outputs: nitrite_recovery.csv, nitrite_eval.csv, calc_nitrite_log.txt (all overwritten on re-run).
Run from the working folder:  /opt/anaconda3/bin/python calc_nitrite.py
"""
import os
N_THREADS = 4
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)
import json, tempfile, time, warnings
import numpy as np, pandas as pd

# cohort and common BEFORE lightgbm / xgboost (unpickling the deployed XGBoost learners can deadlock otherwise)
from cohort import (HERE, TARGETS, F13, ID, DATE, AGE_MAX, DATA_PATH, cached, dev_submitted, dev_corrected,
                    ext_corrected, split_submitted, labels, load_model, ADULT_AGE, is_adult, adult_window,
                    age_exclusion_counts)
from common import SEED, calib_stats, scaled_brier, auroc_ci, auprc_ci, paired_boot, delong_paired
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import catboost as cb
import lightgbm, xgboost  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)
threadpool_limits(limits=N_THREADS)

_cb_predict_proba = cb.CatBoostClassifier.predict_proba


def _cb_predict_proba_capped(self, X, ntree_start=0, ntree_end=0, thread_count=N_THREADS, verbose=None,
                             task_type="CPU"):
    return _cb_predict_proba(self, X, ntree_start, ntree_end, thread_count, verbose, task_type)


cb.CatBoostClassifier.predict_proba = _cb_predict_proba_capped

# Raw first-visit urine file (hospital data, not shareable): PROJ_U_DATA_ROOT overrides the folder.
RAW_PATH = os.path.join(os.environ.get("PROJ_U_DATA_ROOT", "/Users/chan/Desktop/Proj_U/00_data"),
                        "urine_wide_first_visit_raw.csv")
RAW_DT = "소변검사일시"
NIT = "ua_nitrite"
AGE_COL = "age_at_urine_test"     # age column of the processed file (cohort 'age' is this column)
N_BOOT = 1000
KN_REPORTED_PCT = 2.38            # KNHANES adult nitrite positivity quoted in the manuscript draft
LOG, REC, EVAL = [], [], []
AGEX = []                         # section "adult_restriction" of nitrite_recovery.csv, written after the other rows


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True); LOG.append(s)


def rec(section, group, coding, item, value, note=""):
    REC.append(dict(section=section, group=group, coding=coding, item=item, value=value, note=note))


def agex(group, ax, note=""):
    """Adult-restriction counts of one hospital frame (cohort.age_exclusion_counts) -> section 'adult_restriction'."""
    for item, key in (("rows", "rows"), (f"kept: adult (age {ADULT_AGE}-{AGE_MAX})", "adult"),
                      (f"excluded: age < {ADULT_AGE}", "minor"), (f"excluded: age > {AGE_MAX} (implausible)", "implausible"),
                      ("excluded: age not determinable", "missing")):
        AGEX.append(dict(section="adult_restriction", group=group, coding="check", item=item, value=int(ax[key]),
                         note=note))


def recover(text):
    """Raw nitrite text -> 1 / 0 / NaN."""
    s = text.astype("string").str.strip().str.upper()
    out = pd.Series(np.nan, index=text.index)
    out[s.isin(["POSITIVE", "+"]).fillna(False).to_numpy()] = 1.0
    out[s.isin(["NEGATIVE", "-"]).fillna(False).to_numpy()] = 0.0
    return out


def cap_fitted(model):
    for cc in model.calibrated_classifiers_:
        for est in cc.estimator.estimators_:
            if not isinstance(est, cb.CatBoostClassifier):
                est.set_params(n_jobs=N_THREADS)
    return model


def pct(a, b):
    return 100.0 * a / b if b else np.nan


t_start = time.time()
pd.set_option("display.width", 220)

# ============================================================================ (1) recovery
log("=" * 100); log("(1) NITRITE RECOVERY"); log("=" * 100)
# Adult restriction: the raw first-visit file has no age, so the age at the urine test is joined by 일련번호 from the
# processed file (its rows are the person's first visit, so the age is constant within person). Raw persons without
# a processed record have no determinable age and are excluded, as are ages < ADULT_AGE and > AGE_MAX.
proc = pd.read_csv(DATA_PATH, usecols=[ID, RAW_DT, NIT, "days_difference", AGE_COL], low_memory=False)
assert proc.groupby(ID)[AGE_COL].nunique(dropna=False).max() == 1, "age at the urine test differs within a person"
age_p = proc.groupby(ID)[AGE_COL].first()
raw = pd.read_csv(RAW_PATH, usecols=[ID, RAW_DT, NIT], encoding="utf-8-sig", dtype=str)
raw[ID] = raw[ID].astype(int)
assert raw[ID].is_unique, "raw first-visit file has repeated 일련번호"
raw["age"] = raw[ID].map(age_p)
ax = age_exclusion_counts(raw)
agex("raw first-visit file (one row per person)", ax,
     f"age = {AGE_COL} of the person's processed-file rows; not determinable = no processed-file record")
raw = adult_window(raw)
raw["nit_rec"] = recover(raw[NIT])
log(f"raw first-visit file: {ax['rows']:,} rows, 일련번호 unique; adult restriction (age at the urine test joined from "
    f"the processed file): kept {ax['adult']:,} adults; excluded {ax['minor']:,} aged < {ADULT_AGE}, {ax['implausible']} "
    f"aged > {AGE_MAX}, {ax['missing']:,} without a processed-file record (age not determinable)")
vc = raw[NIT].value_counts(dropna=False)
for k, v in vc.items():
    code = raw.loc[raw[NIT].isna() if pd.isna(k) else raw[NIT] == k, "nit_rec"].iloc[0]
    log(f"  raw {repr(k):>12}: {v:>8,}  -> recovered {code}")
    rec("raw_text", "raw first-visit file", "raw", f"count raw {repr(k)} -> {code}", int(v))
raw_i = raw.set_index(ID)

# --- full processed file (adult rows): same visit, agreement with processed coding, affected rows
axp = age_exclusion_counts(proc, col=AGE_COL)
agex("processed file (all rows)", axp, f"age = {AGE_COL}")
n_proc_pers = proc[ID].nunique()
proc = adult_window(proc, col=AGE_COL)
log(f"processed file adult restriction: {axp['rows']:,} rows ({n_proc_pers:,} persons) -> kept {axp['adult']:,} adult "
    f"rows ({proc[ID].nunique():,} persons); excluded {axp['minor']:,} rows aged < {ADULT_AGE}, {axp['implausible']} "
    f"aged > {AGE_MAX}, {axp['missing']} with age missing")
pm = proc.join(raw_i[[RAW_DT, NIT, "nit_rec"]].rename(columns={RAW_DT: "dt_raw", NIT: "nit_raw"}), on=ID)
n_unmatched = int((~pm[ID].isin(raw_i.index)).sum())
same_visit = (pm[RAW_DT].astype(str) == pm["dt_raw"].astype(str))
agree = pm[NIT].notna()
mism = int((pm.loc[agree, NIT] != pm.loc[agree, "nit_rec"]).sum())
aff = pm[NIT].isna() & (pm["nit_rec"] == 1)
aff0 = pm[NIT].isna() & (pm["nit_rec"] == 0)
still = pm[NIT].isna() & pm["nit_rec"].isna()
log(f"processed file, adult rows: {len(pm):,} rows, {pm[ID].nunique():,} distinct 일련번호 (rows repeat a person's first "
    f"visit when several lab dates were matched); rows without a raw record {n_unmatched}")
log(f"  same urine date-time as raw first visit: {int(same_visit.sum()):,}/{len(pm):,}")
log(f"  processed nitrite: 0 = {int((pm[NIT] == 0).sum()):,}, 1 = {int((pm[NIT] == 1).sum()):,}, "
    f"missing = {int(pm[NIT].isna().sum()):,}")
log(f"  recovered vs processed where processed not missing: {int(agree.sum()):,} compared, {mism} mismatches")
log(f"  processed missing -> recovered 1: {int(aff.sum()):,} rows ({pm.loc[aff, ID].nunique():,} persons); "
    f"-> recovered 0: {int(aff0.sum())}; still missing: {int(still.sum())} "
    f"(raw text {dict(pm.loc[still, 'nit_raw'].fillna('<empty>').value_counts())})")
log("  crosstab raw text x processed value:")
log(pd.crosstab(pm["nit_raw"].fillna("<empty>").map(repr), pm[NIT].fillna(-9).map(lambda v: "NaN" if v == -9 else int(v))).to_string())
assert n_unmatched == 0 and same_visit.all() and mism == 0
for item, val in (("rows", len(pm)), ("distinct persons (일련번호)", pm[ID].nunique()),
                  ("rows with the same urine date-time as the raw first visit", int(same_visit.sum())),
                  ("processed nitrite = 0", int((pm[NIT] == 0).sum())), ("processed nitrite = 1", int((pm[NIT] == 1).sum())),
                  ("processed nitrite missing", int(pm[NIT].isna().sum())),
                  ("recovered = processed where processed not missing (rows compared)", int(agree.sum())),
                  ("mismatches recovered vs processed", mism),
                  ("rows processed missing -> recovered 1 (parsing error)", int(aff.sum())),
                  ("persons processed missing -> recovered 1", int(pm.loc[aff, ID].nunique())),
                  ("rows processed missing -> recovered 0", int(aff0.sum())),
                  ("rows still missing after recovery", int(still.sum()))):
    rec("processed_file", "processed file (all adult rows)", "check", item, int(val))
n_meas_p = int(pm[NIT].notna().sum()); n_meas_r = int(pm["nit_rec"].notna().sum())
rec("positivity", "processed file rows", "processed", "nitrite positive % (n pos / n measured)",
    pct((pm[NIT] == 1).sum(), n_meas_p), f"{int((pm[NIT] == 1).sum())}/{n_meas_p}")
rec("positivity", "processed file rows", "recovered", "nitrite positive % (n pos / n measured)",
    pct((pm["nit_rec"] == 1).sum(), n_meas_r), f"{int((pm['nit_rec'] == 1).sum())}/{n_meas_r}")
del proc, pm

# --- +/-14-day development window, adult rows (the full window is kept only to rebuild the locked split, below)
win_all, kn = cached()
axw = age_exclusion_counts(win_all)
agex("development window rows (+/-14 d)", axw, "age = cohort 'age' (age_at_urine_test)")
win = adult_window(win_all)
log(f"\nwindow adult restriction: {axw['rows']:,} rows ({win_all[ID].nunique():,} persons) -> kept {axw['adult']:,} adult "
    f"rows ({win[ID].nunique():,} persons); excluded {axw['minor']:,} rows aged < {ADULT_AGE}, {axw['implausible']} "
    f"aged > {AGE_MAX}, {axw['missing']} with age missing")
w = win.join(raw_i[[RAW_DT, NIT, "nit_rec"]].rename(columns={RAW_DT: "dt_raw", NIT: "nit_raw"}), on=ID)
date_ok = pd.to_datetime(w["dt_raw"]).dt.strftime("%Y-%m-%d") == w[DATE].astype(str)
agree = w[NIT].notna()
mism_w = int((w.loc[agree, NIT] != w.loc[agree, "nit_rec"]).sum())
per_person = w.groupby(ID)["nit_rec"].nunique(dropna=False).max()
assert date_ok.all() and mism_w == 0 and per_person == 1
w_aff = w[NIT].isna() & (w["nit_rec"] == 1)
w_aff0 = w[NIT].isna() & (w["nit_rec"] == 0)
w_still = w[NIT].isna() & w["nit_rec"].isna()
win_r = win.copy()
win_r[NIT] = win[NIT].fillna(w["nit_rec"])
log(f"window (+/-14 d), adult rows: {len(w):,} rows, {w[ID].nunique():,} persons; urine date equal to raw first-visit "
    f"date in {int(date_ok.sum()):,}/{len(w):,}; recovered = processed in all {int(agree.sum()):,} non-missing rows; "
    f"nitrite constant within person")
log(f"  processed nitrite missing {int(w[NIT].isna().sum()):,} rows -> recovered 1: {int(w_aff.sum()):,} rows "
    f"({w.loc[w_aff, ID].nunique():,} persons); recovered 0: {int(w_aff0.sum())}; still missing {int(w_still.sum())}")
pos_ids = set(w.loc[win_r[NIT] == 1, ID])
log(f"  window persons nitrite-positive after recovery: {len(pos_ids):,} (processed: {w.loc[w[NIT] == 1, ID].nunique()})")
for item, val in (("rows", len(w)), ("persons", w[ID].nunique()),
                  ("rows with urine date equal to raw first-visit date", int(date_ok.sum())),
                  ("mismatches recovered vs processed", mism_w),
                  ("processed nitrite = 1 rows", int((w[NIT] == 1).sum())),
                  ("processed nitrite missing rows", int(w[NIT].isna().sum())),
                  ("rows processed missing -> recovered 1", int(w_aff.sum())),
                  ("persons processed missing -> recovered 1", int(w.loc[w_aff, ID].nunique())),
                  ("rows processed missing -> recovered 0", int(w_aff0.sum())),
                  ("rows still missing after recovery", int(w_still.sum())),
                  ("persons nitrite-positive, processed", int(w.loc[w[NIT] == 1, ID].nunique())),
                  ("persons nitrite-positive, recovered", len(pos_ids))):
    rec("window", "development window rows (+/-14 d)", "check", item, int(val))

# --- analytic cohorts
dc, _, _ = dev_corrected(win)
dr, _, _ = dev_corrected(win_r)
assert dc.index.isin(dr.index).all()
same = dr.loc[dc.index, F13 + ["urate"]].fillna(-999).eq(dc[F13 + ["urate"]].fillna(-999)).all().all()
assert same, "current cohort rows changed after recovery"
new = dr[~dr.index.isin(dc.index)].copy()
ds = dev_submitted(win_all)           # all ages: only to rebuild the locked models' row-level split in (2)
assert not new[ID].isin(ds[ID]).any(), "a recovered person has a row in the submitted development frame"
assert (new[NIT] == 1).all()
assert is_adult(dr["age"]).all() and dev_corrected(win_all)[0].index.equals(dc.index)   # same adult cohort either way
n_below = int((win.loc[new.index, "urate_sub"].isna()).sum())
log(f"\nanalytic cohort (13 inputs + urate, adults {ADULT_AGE}-{AGE_MAX}, one record per person): current {len(dc):,} -> "
    f"recovered {len(dr):,} (+{len(new):,} persons, all nitrite-positive; none in the submitted development "
    f"frame; urate below detection among them {n_below}); current cohort rows unchanged")

# why most nitrite-positive window persons still do not enter
wp = win_r[win_r[ID].isin(pos_ids)].copy()
wp.loc[wp["age"] > AGE_MAX, "age"] = np.nan
wp["_ok"] = wp[F13 + ["urate"]].notna().all(axis=1)
wp["_date"] = pd.to_datetime(wp[DATE], errors="coerce")
wp1 = wp.sort_values(["_ok", "_date"], ascending=[False, True]).drop_duplicates(ID, keep="first")
out_ = wp1[~wp1["_ok"]]
assert int(wp1["_ok"].sum()) == int((dr[NIT] == 1).sum()), "flow does not match the recovered adult cohort"
log(f"  nitrite-positive window persons {len(wp1):,}: enter the cohort {int(wp1['_ok'].sum()):,}, "
    f"stay out {len(out_):,}; missing on the selected record of those who stay out:")
rec("cohort", "nitrite-positive window persons (recovered)", "recovered", "persons", len(wp1))
rec("cohort", "nitrite-positive window persons (recovered)", "recovered", "enter the complete-case cohort", int(wp1["_ok"].sum()))
rec("cohort", "nitrite-positive window persons (recovered)", "recovered", "stay out (another input or urate missing)", len(out_))
for c in ["urate", "bmi_unified", "age"] + [f for f in F13 if f not in ("bmi_unified", "age")]:
    k = int(out_[c].isna().sum())
    if k:
        log(f"    {c:<18} missing {k:>6,} ({pct(k, len(out_)):.1f}%)")
        rec("cohort", "nitrite-positive persons staying out", "recovered", f"{c} missing", k, f"{pct(k, len(out_)):.1f}%")
hier = np.select([out_["urate"].isna(), out_["bmi_unified"].isna()], ["urate missing", "urate present, BMI missing"],
                 "urate and BMI present, other input missing")
for k, v in pd.Series(hier).value_counts().items():
    log(f"    exclusive: {k:<42} {v:>6,}")
    rec("cohort", "nitrite-positive persons staying out", "recovered", f"exclusive: {k}", int(v))


def describe(d, group, coding):
    fem = d["male"] == 0; n = len(d)
    items = [("persons", n), ("women n", int(fem.sum())), ("women %", pct(fem.sum(), n)),
             ("age mean", d["age"].mean()), ("age SD", d["age"].std()), ("age median", d["age"].median()),
             ("age Q1", d["age"].quantile(.25)), ("age Q3", d["age"].quantile(.75)),
             ("age < 19 n", int((d["age"] < 19).sum())), ("BMI mean", d["bmi_unified"].mean()),
             ("urate mean", d["urate"].mean()), ("urate SD", d["urate"].std()),
             ("nitrite positive n", int((d[NIT] == 1).sum())), ("nitrite positive %", pct((d[NIT] == 1).sum(), n))]
    for t, cfg in TARGETS.items():
        y = labels(d["urate"], cfg["threshold"])
        items += [(f"urate {cfg['short']} n", int(y.sum())), (f"urate {cfg['short']} %", 100 * y.mean())]
    for k, v in items:
        rec("characteristics", group, coding, k, v)
    return dict(items)


log("\ncharacteristics                         persons  women%  age mean(SD)  median  <19   urate  >=7.0%  >=9.0%  >=10.0%  nitrite+")
for g, d, coding in (("current corrected cohort", dc, "processed"), ("recovered persons (added)", new, "recovered"),
                     ("corrected cohort after recovery", dr, "recovered"),
                     ("current corrected cohort, women", dc[dc.male == 0], "processed"),
                     ("recovered persons, women", new[new.male == 0], "recovered"),
                     ("current corrected cohort, men", dc[dc.male == 1], "processed"),
                     ("recovered persons, men", new[new.male == 1], "recovered")):
    s = describe(d, g, coding)
    log(f"  {g:<36} {s['persons']:>8,}  {s['women %']:5.1f}  {s['age mean']:5.1f} ({s['age SD']:4.1f})  "
        f"{s['age median']:5.0f}  {s['age < 19 n']:>4}  {s['urate mean']:5.2f}  {s['urate >=7.0 %']:6.2f}  "
        f"{s['urate >=9.0 %']:6.2f}  {s['urate >=10.0 %']:7.2f}  {s['nitrite positive n']:>6,}")
# prevalence among recovered persons standardised to the current cohort's sex mix (context for the sex imbalance)
for t, cfg in TARGETS.items():
    thr = cfg["threshold"]; std = 0.0
    for sx in (0, 1):
        sub = new[new.male == sx]
        std += (dc.male == sx).mean() * (labels(sub["urate"], thr).mean() if len(sub) else np.nan)
    rec("characteristics", "recovered persons (added)", "recovered", f"urate {cfg['short']} % sex-standardised to current cohort", 100 * std)
    log(f"  recovered persons urate {cfg['short']}: crude {100 * labels(new['urate'], thr).mean():.2f}%, "
        f"sex-standardised to the current cohort {100 * std:.2f}% (current {100 * labels(dc['urate'], thr).mean():.2f}%)")

# --- nitrite positivity: development (processed vs recovered) vs KNHANES
ec = ext_corrected(kn)
wpers = win_r.assign(_nit_proc=win[NIT]).assign(_date=pd.to_datetime(win[DATE], errors="coerce")) \
    .sort_values("_date").drop_duplicates(ID, keep="first")
wpers.loc[wpers["age"] > AGE_MAX, "age"] = np.nan
AGE_BANDS = [19, 30, 40, 50, 60, 70, 200]


def positivity(group, s, coding, note=""):
    s = pd.Series(s).dropna(); n = len(s); k = int((s == 1).sum())
    rec("positivity", group, coding, "nitrite positive % (n pos / n measured)", pct(k, n), f"{k}/{n}" + (f"; {note}" if note else ""))
    return k, n


def std_to_knhanes(d, col):
    """Direct standardisation of adult positivity to the KNHANES adult (ext_corrected) age-sex distribution."""
    ref = ec.assign(b=pd.cut(ec["age"], AGE_BANDS, right=False))
    wt = ref.groupby(["male", "b"], observed=False).size() / len(ref)
    dd = d[d["age"] >= 19].assign(b=pd.cut(d["age"], AGE_BANDS, right=False))
    r = dd.groupby(["male", "b"], observed=False)[col].apply(lambda x: (x.dropna() == 1).mean())
    return 100 * float((wt * r).sum())


log("\nnitrite positivity (n positive / n measured)")
lines = []
wrows = win_r.assign(_nit_proc=win[NIT])
wrows.loc[wrows["age"] > AGE_MAX, "age"] = np.nan
for g, frame in (("window rows", wrows), ("window persons (first record)", wpers)):
    for sub_name, sub in (("", frame), (", adults >= 19", frame[frame["age"] >= 19]),
                          (", women >= 19", frame[(frame["age"] >= 19) & (frame.male == 0)]),
                          (", men >= 19", frame[(frame["age"] >= 19) & (frame.male == 1)])):
        kp, np_ = positivity(g + sub_name, sub["_nit_proc"], "processed")
        kr, nr = positivity(g + sub_name, sub[NIT], "recovered")
        lines.append((g + sub_name, kp, np_, kr, nr))
for g, dcur, drec in (("analytic cohort", dc, dr), ("analytic cohort, adults >= 19", dc[dc.age >= 19], dr[dr.age >= 19]),
                      ("analytic cohort, women >= 19", dc[(dc.age >= 19) & (dc.male == 0)], dr[(dr.age >= 19) & (dr.male == 0)]),
                      ("analytic cohort, men >= 19", dc[(dc.age >= 19) & (dc.male == 1)], dr[(dr.age >= 19) & (dr.male == 1)])):
    kp, np_ = positivity(g, dcur[NIT], "processed"); kr, nr = positivity(g, drec[NIT], "recovered")
    lines.append((g, kp, np_, kr, nr))
for g, kp, np_, kr, nr in lines:
    log(f"  {g:<42} processed {kp:>6,}/{np_:<8,} {pct(kp, np_):6.3f}%   recovered {kr:>6,}/{nr:<8,} {pct(kr, nr):6.3f}%")
sw = std_to_knhanes(wpers, NIT); sa = std_to_knhanes(dr, NIT)
rec("positivity", "window persons, adults >= 19, standardised to KNHANES adult age-sex mix", "recovered",
    "nitrite positive %", sw)
rec("positivity", "analytic cohort after recovery, adults >= 19, standardised to KNHANES adult age-sex mix", "recovered",
    "nitrite positive %", sa)
log(f"  window persons >= 19 standardised to KNHANES adult age-sex mix: recovered {sw:.3f}%")
log(f"  analytic cohort >= 19 standardised to KNHANES adult age-sex mix: recovered {sa:.3f}%")
for g, s in (("KNHANES adults, 13 inputs + urate (ext_corrected)", ec[NIT]),
             ("KNHANES adults, 13 inputs + urate, women", ec.loc[ec.male == 0, NIT]),
             ("KNHANES adults, 13 inputs + urate, men", ec.loc[ec.male == 1, NIT]),
             ("KNHANES all records with nitrite", kn[NIT]),
             ("KNHANES records aged >= 19 with nitrite", kn.loc[kn["age"] >= 19, NIT])):
    k, n = positivity(g, s, "KNHANES", "manuscript value 2.38%" if g.startswith("KNHANES adults, 13 inputs + urate (") else "")
    log(f"  {g:<52} {k:>6,}/{n:<8,} {pct(k, n):6.3f}%")
kn_pct = pct(int((ec[NIT] == 1).sum()), int(ec[NIT].notna().sum()))
assert abs(round(kn_pct, 2) - KN_REPORTED_PCT) < 1e-9, f"KNHANES positivity {kn_pct:.3f}% differs from 2.38%"

# ============================================================================ (2) evaluation
log("\n" + "=" * 100); log("(2) DEPLOYED MODELS ON THE RECOVERED PERSONS"); log("=" * 100)
pint_ref = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))


def _metrics(y, p):
    with np.errstate(all="ignore"):
        try:
            a, b, oe = calib_stats(y, p)
            if not (np.isfinite(a) and np.isfinite(b)): a = b = np.nan
        except np.linalg.LinAlgError:
            a = b = np.nan; oe = y.mean() / p.mean()
    return (roc_auc_score(y, p), average_precision_score(y, p), a, b, oe, brier_score_loss(y, p), scaled_brier(y, p))


NAMES = ("auroc", "auprc", "intercept", "slope", "oe", "brier", "scaled_brier")


def evaluate(y, p, check=False):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    out = dict(n=len(y), events=int(y.sum()), prevalence=y.mean(), mean_pred=p.mean())
    if y.min() == y.max():
        return out
    pt = _metrics(y, p)
    rng = np.random.RandomState(SEED); boot = []
    for _ in range(N_BOOT):                                   # identical resamples to common.boot_ci
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        boot.append(_metrics(y[i], p[i]))
    boot = np.array(boot, float)
    lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
    if check:
        ref = auroc_ci(y, p, N_BOOT, SEED); refp = auprc_ci(y, p, N_BOOT, SEED)
        assert np.allclose(ref, (pt[0], lo[0], hi[0])) and np.allclose(refp, (pt[1], lo[1], hi[1])), "bootstrap mismatch"
    for k, nm in enumerate(NAMES):
        out[nm] = pt[k]; out[nm + "_lo"] = lo[k]; out[nm + "_hi"] = hi[k]
    out["n_boot_used"] = len(boot); out["n_boot_slope_valid"] = int(np.isfinite(boot[:, 3]).sum())
    nz = p > 0
    if y[nz].min() != y[nz].max():
        with np.errstate(all="ignore"):
            a_nz, b_nz, _ = calib_stats(y[nz], p[nz])
        out["intercept_excl_p0"], out["slope_excl_p0"] = a_nz, b_nz
    out["n_p0"] = int((~nz).sum()); out["events_p0"] = int(y[~nz].sum())
    return out


def fold_learner_probs(model, X):
    """Per base learner (lgb, xgb, cat) mean over the 5 calibration folds of predict_proba, and the ensemble."""
    acc = {"lgb": 0.0, "xgb": 0.0, "cat": 0.0}
    for cc in model.calibrated_classifiers_:
        for (nm, _), est in zip(cc.estimator.estimators, cc.estimator.estimators_):
            acc[nm] = acc[nm] + est.predict_proba(X)[:, 1] / len(model.calibrated_classifiers_)
    return acc


def with_nit(X, v):
    X = X.copy(); X[NIT] = float(v); return X


LEARN = []
for tkey, cfg in TARGETS.items():
    thr = cfg["threshold"]; short = cfg["short"]; t0 = time.time()
    m = cap_fitted(load_model(tkey))
    tr, te, y = split_submitted(ds, thr)            # all-ages row frame: indices of the locked split only
    ids = ds[ID].values; ages = ds["age"].values
    clean = ~np.isin(ids[te], ids[tr])
    keep = clean & ~(ages[te] > AGE_MAX) & is_adult(ages[te])         # adult person-clean test rows
    tr_ad = tr[is_adult(ages[tr])]                                     # adult training rows (counts only)
    axt = age_exclusion_counts(pd.DataFrame({"age": ages[tr]}))
    axe = age_exclusion_counts(pd.DataFrame({"age": ages[te[clean]]}))
    agex(f"locked models' submitted training rows {short}", axt,
         "split_submitted (seed 42); used only to count nitrite-positive training rows; models not refitted")
    agex(f"person-clean internal test rows {short}", axe, "test rows of persons not in training; kept rows = "
         "preds_internal_clean.csv")
    X_in = ds[F13].iloc[te[keep]]; y_in = y[te[keep]]; id_in = ids[te[keep]]
    p_in = m.predict_proba(X_in)[:, 1]
    s = pint_ref[pint_ref.target == tkey]
    dmax = float(np.max(np.abs(s["p"].values - p_in))) if len(s) == len(p_in) else np.inf
    assert len(s) == len(p_in) and (s["y"].values == y_in).all() and dmax < 1e-12, "internal predictions differ"
    assert not np.isin(new[ID].values, id_in).any()
    n_tr_pos = int((ds[NIT].values[tr_ad] == 1).sum())
    y_new = labels(new["urate"], thr)
    X1 = with_nit(new[F13], 1); X0 = with_nit(new[F13], 0)
    p1 = m.predict_proba(X1)[:, 1]; p0 = m.predict_proba(X0)[:, 1]
    fem_new = new["male"].values == 0; fem_in = X_in["male"].values == 0
    _, prop_idx = train_test_split(np.arange(len(new)), test_size=0.2, random_state=SEED, stratify=y_new)
    log(f"\n{short}: submitted training rows {len(tr):,}, adults {len(tr_ad):,} (excluded {axt['minor']:,} aged < "
        f"{ADULT_AGE}, {axt['implausible']} aged > {AGE_MAX}; nitrite-positive adult training rows {n_tr_pos}); "
        f"person-clean internal test, adults {len(y_in):,} rows (events {int(y_in.sum())}; excluded {axe['minor']:,} "
        f"aged < {ADULT_AGE}, {axe['implausible']} aged > {AGE_MAX}; matches preds_internal_clean.csv, max |diff| "
        f"{dmax:.1e}); recovered persons {len(new):,} (events {int(y_new.sum())}); proportional 20% sample "
        f"{len(prop_idx)} (events {int(y_new[prop_idx].sum())})")
    sets = [("internal_clean", "as recorded", y_in, p_in, None),
            ("internal_clean_women", "as recorded", y_in[fem_in], p_in[fem_in], None),
            ("recovered", "1", y_new, p1, p0),
            ("recovered", "0", y_new, p0, None),
            ("recovered_women", "1", y_new[fem_new], p1[fem_new], p0[fem_new]),
            ("recovered_women", "0", y_new[fem_new], p0[fem_new], None),
            ("union_internal_clean+recovered", "1", np.r_[y_in, y_new], np.r_[p_in, p1], np.r_[p_in, p0]),
            ("union_internal_clean+recovered", "0", np.r_[y_in, y_new], np.r_[p_in, p0], None),
            ("union_internal_clean+recovered_20pct", "1", np.r_[y_in, y_new[prop_idx]], np.r_[p_in, p1[prop_idx]],
             np.r_[p_in, p0[prop_idx]]),
            ("union_internal_clean+recovered_20pct", "0", np.r_[y_in, y_new[prop_idx]], np.r_[p_in, p0[prop_idx]], None)]
    for name, inp, yy, pp, pp0 in sets:
        r = dict(part="performance", target=short, threshold=thr, set=name, nitrite_input_recovered=inp)
        r.update(evaluate(yy, pp, check=(name == "recovered" and inp == "1")))
        if pp0 is not None and yy.min() != yy.max():
            d, pv = delong_paired(yy, pp, pp0)
            lo, hi = paired_boot(yy, pp, pp0) if np.any(pp != pp0) else (0.0, 0.0)
            r.update(auroc_diff_nit1_minus_nit0=d, auroc_diff_lo=lo, auroc_diff_hi=hi, auroc_diff_delong_p=pv,
                     mean_pred_diff_nit1_minus_nit0=float(np.mean(pp - pp0)))
        EVAL.append(r)
        if "auroc" in r:
            extra = (f"  dAUROC(1-0) {r['auroc_diff_nit1_minus_nit0']:+.4f} ({r['auroc_diff_lo']:+.4f} to "
                     f"{r['auroc_diff_hi']:+.4f})" if "auroc_diff_nit1_minus_nit0" in r else "")
            log(f"  {name:<38} nit={inp:<11} n={r['n']:>6,} ev={r['events']:>5,}  AUROC {r['auroc']:.4f} "
                f"({r['auroc_lo']:.4f}-{r['auroc_hi']:.4f})  AUPRC {r['auprc']:.4f} ({r['auprc_lo']:.4f}-{r['auprc_hi']:.4f})  "
                f"int {r['intercept']:+.3f} ({r['intercept_lo']:+.3f} to {r['intercept_hi']:+.3f})  "
                f"slope {r['slope']:.3f} ({r['slope_lo']:.3f}-{r['slope_hi']:.3f})  O:E {r['oe']:.3f} "
                f"({r['oe_lo']:.3f}-{r['oe_hi']:.3f})  mean p {r['mean_pred']:.4f} vs obs {r['prevalence']:.4f}{extra}")
        else:
            log(f"  {name:<38} nit={inp:<11} n={r['n']:>6,} ev={r['events']:>5,}  single class, not evaluated")

    # ======================================================================== (3) nitrite 0 -> 1 switch
    ek = ec[F13]; ye = labels(ec["urate"], thr); nit_e = ec[NIT].values == 1
    for name, X, yy in (("recovered (recorded nitrite 1)", new[F13], y_new),
                        ("internal_clean (recorded nitrite 0 except rare 1)", X_in, y_in),
                        ("KNHANES adults (recorded nitrite 2.4% positive)", ek, ye)):
        q0 = m.predict_proba(with_nit(X, 0))[:, 1]; q1 = m.predict_proba(with_nit(X, 1))[:, 1]
        dlt = q1 - q0; ch = np.abs(dlt) > 1e-12
        r = dict(part="switch_0_to_1", target=short, threshold=thr, set=name, nitrite_input_recovered="0 vs 1",
                 n=len(yy), events=int(yy.sum()), prevalence=float(np.mean(yy)),
                 mean_pred_nit0=q0.mean(), mean_pred_nit1=q1.mean(), mean_diff=dlt.mean(),
                 relative_mean_change=q1.mean() / q0.mean() - 1, median_diff=float(np.median(dlt)),
                 mean_abs_diff=np.abs(dlt).mean(), p95_abs_diff=float(np.quantile(np.abs(dlt), .95)),
                 max_abs_diff=np.abs(dlt).max(), frac_changed=ch.mean(), frac_up=(dlt > 1e-12).mean(),
                 frac_down=(dlt < -1e-12).mean(), spearman_nit0_nit1=spearmanr(q0, q1)[0],
                 auroc_nit0=roc_auc_score(yy, q0), auroc_nit1=roc_auc_score(yy, q1),
                 oe_nit0=np.mean(yy) / q0.mean(), oe_nit1=np.mean(yy) / q1.mean())
        bl0 = fold_learner_probs(m, with_nit(X, 0)); bl1 = fold_learner_probs(m, with_nit(X, 1))
        for nm in ("lgb", "xgb", "cat"):
            r[f"{nm}_mean_diff"] = float(np.mean(bl1[nm] - bl0[nm]))
            r[f"{nm}_mean_abs_diff"] = float(np.mean(np.abs(bl1[nm] - bl0[nm])))
        sv = (bl1["lgb"] + bl1["xgb"] + bl1["cat"]) / 3 - (bl0["lgb"] + bl0["xgb"] + bl0["cat"]) / 3
        r["softvote_mean_diff_before_isotonic"] = float(sv.mean())
        EVAL.append(r)
        log(f"  switch 0->1 {name:<50} mean p {q0.mean():.4f} -> {q1.mean():.4f} (diff {dlt.mean():+.5f}, "
            f"{100 * r['relative_mean_change']:+.1f}%); mean|diff| {r['mean_abs_diff']:.5f}, max|diff| {r['max_abs_diff']:.4f}; "
            f"changed {100 * ch.mean():.1f}% (up {100 * r['frac_up']:.1f}%, down {100 * r['frac_down']:.1f}%); "
            f"AUROC {r['auroc_nit0']:.4f} -> {r['auroc_nit1']:.4f}; base learners mean diff lgb {r['lgb_mean_diff']:+.2e} "
            f"xgb {r['xgb_mean_diff']:+.2e} cat {r['cat_mean_diff']:+.2e}")
    # KNHANES nitrite-positive adults: as recorded vs set to 0
    qa = m.predict_proba(ek)[:, 1]; qz = m.predict_proba(with_nit(ek, 0))[:, 1]
    r = dict(part="switch_0_to_1", target=short, threshold=thr, set="KNHANES nitrite-positive adults (recorded 1 vs set 0)",
             nitrite_input_recovered="0 vs 1", n=int(nit_e.sum()), events=int(ye[nit_e].sum()),
             prevalence=float(ye[nit_e].mean()), mean_pred_nit0=qz[nit_e].mean(), mean_pred_nit1=qa[nit_e].mean(),
             mean_diff=float(np.mean(qa[nit_e] - qz[nit_e])), relative_mean_change=qa[nit_e].mean() / qz[nit_e].mean() - 1,
             mean_abs_diff=float(np.mean(np.abs(qa[nit_e] - qz[nit_e]))), max_abs_diff=float(np.max(np.abs(qa[nit_e] - qz[nit_e]))),
             oe_nit0=ye[nit_e].mean() / qz[nit_e].mean(), oe_nit1=ye[nit_e].mean() / qa[nit_e].mean())
    EVAL.append(r)
    log(f"  KNHANES nitrite-positive adults (n={r['n']:,}, observed {r['prevalence']:.4f}): mean p recorded(1) "
        f"{r['mean_pred_nit1']:.4f} vs set 0 {r['mean_pred_nit0']:.4f}")

    # tree splits on nitrite in each base learner
    k_nit = F13.index(NIT)
    for f, cc in enumerate(m.calibrated_classifiers_):
        lg, xg, ct = cc.estimator.estimators_
        imp = dict(zip(lg.booster_.feature_name(), lg.booster_.feature_importance("split")))
        xs = xg.get_booster().get_score(importance_type="weight")
        with tempfile.TemporaryDirectory() as tmp:
            pth = os.path.join(tmp, "cat.json"); ct.save_model(pth, format="json")
            with open(pth) as fh:
                trees = json.load(fh)["oblivious_trees"]
        kc = ct.feature_names_.index(NIT)
        n_cat = sum(1 for trd in trees for sp in trd["splits"] if sp.get("float_feature_index") == kc)
        for nm, n_split, n_trees in (("lgb", int(imp.get(NIT, 0)), lg.booster_.num_trees()),
                                     ("xgb", int(xs.get(NIT, xs.get(f"f{k_nit}", 0))), len(xg.get_booster().get_dump())),
                                     ("cat", n_cat, len(trees))):
            LEARN.append(dict(part="learner_splits", target=short, threshold=thr, set=f"fold {f + 1}",
                              learner=nm, nitrite_splits=n_split, trees=n_trees, train_nitrite_pos_rows=n_tr_pos))
    ls = pd.DataFrame([x for x in LEARN if x["target"] == short]).groupby("learner")[["nitrite_splits", "trees"]].sum()
    log(f"  splits on nitrite summed over 5 folds: " + ", ".join(f"{k} {int(v.nitrite_splits)}/{int(v.trees)} trees"
                                                           for k, v in ls.iterrows()) + f"   [{time.time() - t0:.0f}s]")

EVAL.extend(LEARN)
ev = pd.DataFrame(EVAL)
ev.to_csv(os.path.join(HERE, "nitrite_eval.csv"), index=False)
REC.extend(AGEX)                                  # adult-restriction counts after the existing sections
pd.DataFrame(REC).to_csv(os.path.join(HERE, "nitrite_recovery.csv"), index=False)

log("\n" + "=" * 100); log("SUMMARY"); log("=" * 100)
perf = ev[ev.part == "performance"]
cols = ["target", "set", "nitrite_input_recovered", "n", "events", "auroc", "auroc_lo", "auroc_hi", "auprc",
        "intercept", "slope", "oe"]
log(perf[cols].round(4).to_string(index=False))
log("\nadult restriction (adult target population; nitrite_recovery.csv section 'adult_restriction'):")
log(pd.DataFrame(AGEX)[["group", "item", "value"]].to_string(index=False))
log(f"\nOutputs: nitrite_recovery.csv ({len(REC)} rows, of which {len(AGEX)} adult_restriction), nitrite_eval.csv "
    f"({len(ev)} rows); "
    f"runtime {time.time() - t_start:.0f}s")
with open(os.path.join(HERE, "calc_nitrite_log.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
