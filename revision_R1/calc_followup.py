"""Follow-up analyses requested by the internal pre-resubmission review of manuscript v3.0 (JMIR Medical Informatics
#107515, revision R1), for the retrained models.

MODELS AND COHORTS (decision of 2026-10-16)
  Models: the retrained calibrated soft-voting ensembles (bestmodel_v5, cohort.load_model): Optuna-tuned LightGBM +
  XGBoost + CatBoost, soft voting, isotonic CalibratedClassifierCV(cv=5), 12 inputs (cohort.FMODEL: the 9 dipstick
  fields without leukocyte esterase, age, male, BMI). They are loaded read-only and never refitted; their predictions
  are read from preds_internal_clean.csv / preds_external_corrected.csv (calc_main.py), aligned by person and
  recomputed from the models here (asserted equal). Refits are labelled as such and live only in memory.
  Cohorts: cohort.main_cohort() -> d = the corrected development cohort dev_v2 (adults aged 19-110 years at the
  first-visit urine test, one record per person; serum urate and BMI of the urine-test day; nitrite parsed correctly)
  with ONE person-level 80/20 split shared by the thresholds: training persons d.iloc[tr] (103,628) and internal test
  persons d.iloc[te] (25,907, the same persons for every threshold); ext = KNHANES adults aged >= 19 (49,987) with the
  dipstick inputs coded by the label-matched mapping X1. "internal" = d.iloc[te], "external" = ext. Every refit is
  fitted on the training persons and evaluated on d.iloc[te] and on ext. Counts are asserted against
  cohort_info_v2.csv, and the AUROC and O:E of the retrained models on a full set against main_performance.csv.
  Thresholds: >=7.0 and >=9.0 mg/dL (MAIN_T), as in the previous version of this script.
Every output is a file prefixed fu_.

(a) fu_setting.csv  Care setting and urine-only encounters.
      a1 first-visit urine tests of adults 2010-2025 (raw first-visit file, one per person) by blood-test status
         (blood test within +/-14 days = the adult window persons of dev_v2, asserted; only 15-365 days away; none
         within 365 days), with age, sex, same-day health-checkup record, BMI of the urine-test day as dev_v2 defines
         it (a health-checkup BMI dated on the urine-test day, otherwise height and weight in the nursing record of
         that day; the plausibility rules of add_bmi_unified) and calendar period. Age and sex come from the nursing
         (encounter) file and the health-checkup file (lookups, not analysis populations). Persons aged < 19, with
         age < 0 or > 110, or with no determinable age are excluded and counted in the log line that make_figures.py
         and make_supplement.py parse; adult status from the birth date is asserted equal to adult status from the
         processed file's age_at_urine_test for every person with a processed-file record.
      a2 the analytic cohort (dev_v2) by whether a health-checkup record is dated on the urine-test date (the only
         setting marker in the data; outpatient, inpatient and emergency encounters cannot be distinguished).
      a3 the adult window persons (cohort.persons_v2: one record per person, before the urate, BMI and completeness
         exclusions) by the same marker: BMI and serum urate of the urine-test day available.
      a4 the retrained models on the internal test persons by the same marker: AUROC and O:E (bootstrap CIs).
(b) fu_bmi_source.csv  How the BMI of the analytic cohort was recorded. Persons whose BMI came from a health checkup on
      the urine-test day: checkup BMI recomputed from the checkup height and weight, and terminal-digit (digit
      preference) analysis of the checkup height and weight; persons whose BMI came from the nursing record of that day:
      terminal digits of the nursing height and weight. Model-free. Each person's processed-file row is the cohort
      record itself (row number = index of the cached window; person id and checkup BMI asserted equal).
(c) fu_sex_operating.csv  Operating points overall and by sex for every rule of Table 3: the retrained ensemble at the
      thresholds of operating_points.csv (chosen on the internal test persons) and at the top 5/10/20%; 'test all men';
      the 3-input (age, sex, BMI) spline LR and LightGBM (fixed) refits at thresholds chosen on their own internal test
      predictions; sex-specific thresholds of the retrained model (80% and 90% sensitivity within each sex on the
      internal test persons). 95% CIs: bootstrap within each group (1000; thresholds fixed).
(d) fu_net_benefit.csv  External net benefit of: the retrained ensemble unrecalibrated (equal to decision_curve.csv,
      asserted) and after a cross-fitted intercept update (2-fold: fitted on one random half of KNHANES, applied to the
      other, both ways; 100 random splits, point = mean); the 3F spline LR refit (unrecalibrated and cross-fitted);
      'test all men'; treat all. Paired bootstrap CI (1000) for the ensemble minus 'test all men' and the defaults.
    fu_net_benefit_ranges.csv  threshold ranges where each strategy beats the defaults and 'test all men'.
    fu_recal_operating.csv  operating points at fixed thresholds on the recalibrated risk scale (cross-fitted).
(e) fu_within_sex.csv  Dipstick increment within each sex in KNHANES (refits): 12F minus 3F and 11F minus 2F
      (LightGBM fixed, spline LR), and the retrained ensemble minus the 3F refits; paired bootstrap (1000) and DeLong.
(f) fu_tiers_2F.csv  Age-and-sex-only models (2F) and the increments over them: 11F minus 2F (dipstick added to age and
      sex), 3F minus 2F (BMI added to age and sex), 12F minus 3F (dipstick added to age, sex and BMI) and 12F minus 11F
      (BMI added to dipstick, age and sex); LightGBM (fixed), main-effects LR, spline LR; internal and external. These
      are the fits, bootstrap draws and DeLong statistics of feature_tiers.csv (calc_feature_tiers.py), so every value
      is asserted equal to it; the file is kept for the table generators.
(g) fu_oof_thresholds.csv  Thresholds chosen on out-of-fold training predictions of the retrained models.
      cohort.cal_folds returns each model's 5 calibration folds (StratifiedKFold(5) without shuffling on
      cohort.train_xy, as CalibratedClassifierCV uses; each fold's isotonic map is verified by refitting it). Each
      training person is scored by the fold model whose base learners did not see it (its isotonic map was fitted on
      that fold). Thresholds (Youden, sensitivity 80% and 90%) chosen on these predictions are applied to the training
      predictions, the internal test persons and KNHANES.
(h) fu_internal_period.csv  Retrained models on the internal test persons by period of the urine test (2010-2021 vs
      2022-2025) and by year: O:E and AUROC with bootstrap CIs.
(i) fu_ketone.csv  Ketone and calibration: O:E by ketone status (internal, KNHANES); change in predicted risk when
      internal ketone trace results (0.5) are set to 0, and when all ketone results are set to 0; KNHANES O:E with
      ketone set to 0.
(j) fu_knhanes_year.csv  KNHANES calibration and discrimination by survey year.
(k) fu_sg_restrict.csv  KNHANES adults with specific gravity <= 1.030 (the range of the training persons, asserted) vs
      > 1.030.
(l) fu_knhanes_flow.csv  KNHANES file composition (cache knhanes_v2, X1 coding): records per survey year, serum urate
      availability by age band, and the complete-case attrition among adults (ends at ext, asserted).
(m) fu_isotonic_floor.csv  Isotonic outputs of exactly 0: effect of flooring at 0.001, of replacing zeros by the
      logistic-calibrator value, and of the logistic calibrator throughout (same fold learners and calibration folds;
      preds_calibration_methods.csv of calc_calibration.py, whose method key isotonic_deployed holds the retrained
      isotonic models).
(n) fu_continuous.csv  Continuous-urate models (refits on the training persons): the homoscedastic model of
      calc_sex_nested.py reproduced (AUROCs asserted equal to continuous.csv), a heteroscedastic location-scale variant
      (LightGBM for the mean and for the log squared residual; empirical standardised residuals), observed versus
      predicted urate by decile, and AUROC comparisons with the equal-budget binary LightGBM (the 12F LightGBM (fixed)
      of feature_tiers.csv) on the same persons (paired bootstrap), internally and externally at both thresholds.

Removed with the retraining (decision of 2026-10-16; the previous version and its outputs are in
_snapshot_original_models_261016/working/):
  * the reconstruction of the original models' row-level split (dev_submitted, split_submitted, dev_corrected,
    ext_corrected, cached()), the person-clean and adult masks on their test rows, and the bookkeeping of their training
    rows of persons aged < 19 (in (g) their folds were rebuilt from all rows and only adult rows were scored). The v2
    cohort is adult and person-level, and the internal test persons are the same for every threshold;
  * the harmonised KNHANES coding of the refits (urobilinogen 0.5, nitrite 0; calc_feature_tiers.HARM) and the
    "external as coded" 13F LightGBM predictions used in (n): X1 codes every KNHANES field with the development label,
    so every model sees the same external inputs;
  * the adult restriction of the processed data file in (b), which is now linked record by record to the analytic
    cohort;
  * the restriction of the internal comparisons in (n) to >=7.0 (the binary comparator at >=9.0 came from a split
    stratified at 9.0). The split is shared now, so the internal >=9.0 comparisons are reported too.
Changed definitions: BMI availability in (a1) is the BMI of the urine-test day (dev_v2), no longer a checkup BMI from any
date (column pct_checkup_bmi_any -> pct_checkup_bmi_same_day); (a3) counts the adult window persons with the serum urate
and BMI of the urine-test day; (b) takes the checkup and nursing BMI sources from dev_v2 (bmi_v2_source).

Bootstrap: resampling with replacement at full size, percentile 95% CI, seed 42. The bootstraps of (e), (f) and (n) run
in 4 threads with the same draws as a serial run. Threads are capped at 4.
Run from the working folder:  /opt/anaconda3/bin/python calc_followup.py
(the xlsx sources are read with the calamine engine when python-calamine is importable, otherwise openpyxl;
the engine changes only speed.)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import time, warnings
import numpy as np, pandas as pd
from scipy import stats

# Import order (workflow rule of 2026-10-11): cohort and common first, then the boosting libraries.
from cohort import (HERE, TARGETS, F9, FMODEL, ID, DATE, AGE_MAX, ADULT_AGE, DATA_PATH, RAW_FIRST, RAW_DT, labels,
                    load_model, main_cohort, cohort_steps, train_xy, cal_folds, cached_v2, persons_v2, is_adult,
                    adult_window, age_exclusion_counts)
from common import (SEED, LGBM_FIXED, NB_GRID, logit, expit, fit_offset_intercept, calib_stats, scaled_brier,
                    youden_threshold, threshold_for_sensitivity, delong_paired, paired_boot, auroc_ci)
import xgboost  # noqa: F401  (before lightgbm; see calc_feature_tiers.py)
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import KFold, train_test_split
from calc_feature_tiers import TIERS, lgbm, logreg, logreg_spline, cap_threads

warnings.filterwarnings("ignore", category=UserWarning)
threadpool_limits(limits=4)
N_THREADS, N_BOOT, N_SPLIT = 4, 1000, 100
OUT = HERE                                         # output folder (fu_*.csv and the log)
# Hospital data folder (not shareable): PROJ_U_DATA_ROOT overrides; the fallback is the original value.
ROOT = os.environ.get("PROJ_U_DATA_ROOT", "/Users/chan/Desktop/Proj_U/00_data")
NURSING = [os.path.join(ROOT, "medical_record", f"medical_record{i}.xlsx") for i in range(1, 8)]
CHECKUP = os.path.join(ROOT, "demographic", "demographic.xlsx")
MAIN_T = {"uric_hyper_70": 7.0, "uric_severe_9": 9.0}
SH = {t: cfg["short"] for t, cfg in TARGETS.items()}
ENS = "Retrained ensemble"
LGB, LRM, SPL = "LightGBM (fixed)", "Logistic regression", "Spline LR"
CLASSES = ((LGB, lambda f: lgbm()), (LRM, logreg), (SPL, logreg_spline))      # the model classes of calc_feature_tiers
FT_MODEL = {LGB: "LightGBM (fixed)", LRM: "Logistic regression", SPL: "LR (age/BMI splines + age x male)"}
FU_TIERS = ("2F", "3F", "11F", "12F")              # input sets refitted here (TIERS of calc_feature_tiers.py)
ISO_KEY = "isotonic_deployed"                      # calc_calibration.py's method key for the retrained isotonic models
LOG = []
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)
    with open(os.path.join(OUT, "calc_followup_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def out(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    log(f"  -> {name} ({len(df)} rows) [{time.time() - T0:.0f}s]")


def xlsx(path, **kw):
    try:
        import python_calamine  # noqa: F401
        return pd.read_excel(path, engine="calamine", **kw)
    except ImportError:
        return pd.read_excel(path, engine="openpyxl", **kw)


def boot_idx(n, B=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    for _ in range(B):
        yield rng.randint(0, n, n)


def boot_weights(n, B=N_BOOT, seed=SEED):
    for i in boot_idx(n, B, seed):
        yield np.bincount(i, minlength=n).astype(float)


def pci(a):
    a = np.asarray(a, float)
    return np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)


def oe_ci(y, p, B=N_BOOT):
    y, p = np.asarray(y, float), np.asarray(p, float)
    vals = [y[i].sum() / p[i].sum() for i in boot_idx(len(y), B)]
    lo, hi = pci(vals)
    return y.sum() / p.sum(), float(lo), float(hi)


def auc_ci(y, p):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan, np.nan, np.nan
    return auroc_ci(y, p, N_BOOT)


def job_auc(y, p):
    """AUROC with its percentile bootstrap CI (common.auroc_ci)."""
    return auroc_ci(y, p, N_BOOT)


def job_contrast(y, pa, pb):
    """AUROC(a) - AUROC(b) on the same persons: paired DeLong and paired bootstrap CI."""
    dlt, pval = delong_paired(y, pa, pb)
    lo, hi = paired_boot(y, pa, pb, N_BOOT)
    return dlt, pval, lo, hi


def par(jobs):
    """Run [(function, args)] in N_THREADS threads. Every job draws its resamples from its own generator with a fixed
    seed, so the results equal a serial run (threads, not processes: no worker process is started)."""
    return Parallel(n_jobs=N_THREADS, prefer="threads")(delayed(f)(*a) for f, a in jobs)


def assert_same(a, b, what, tol=1e-12):
    """a == b to tol: the same computation in another script (same persons, fits and predictions)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and np.allclose(a, b, rtol=0, atol=tol, equal_nan=True), (what, a, b)


def assert_same_ci(a, b, what):
    """Bootstrap CIs equal to another file's: the same function, seed and number of resamples."""
    assert_same(a, b, what + " (95% CI)")


# ======================================================================== data
d, tr, te, ext = main_cohort()
dte = d.iloc[te]
win2, kn2 = cached_v2()
_ci2 = pd.read_csv(os.path.join(HERE, "cohort_info_v2.csv"))
CI2 = dict(zip(_ci2["item"].astype(str), _ci2["value"]))
N_TR, N_TE, N_EXT = int(CI2["training persons"]), int(CI2["test persons"]), int(CI2["KNHANES adults with 12 inputs and urate"])
assert (len(tr), len(te), len(ext)) == (N_TR, N_TE, N_EXT) == (103_628, 25_907, 49_987), (len(tr), len(te), len(ext))
assert d[ID].is_unique and not np.isin(d[ID].to_numpy()[te], d[ID].to_numpy()[tr]).any(), "a person in both sets"
assert is_adult(d["age"]).all() and (ext["age"] >= ADULT_AGE).all() and ext["id"].is_unique
pint = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))
pext = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"))
pcal = pd.read_csv(os.path.join(HERE, "preds_calibration_methods.csv"))
MP = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
MP_COH = {"internal": "internal test", "external": "KNHANES adults"}
models = {t: cap_threads(load_model(t)) for t in MAIN_T}
male_e = ext["male"].to_numpy(int)
log(f"calc_followup.py: retrained models (bestmodel_v5, {len(FMODEL)} inputs); dev_v2 {len(d):,} persons (adults "
    f"{ADULT_AGE}-{AGE_MAX} y, one record per person): training {len(tr):,}, internal test {len(te):,} (one person-level "
    f"split shared by the thresholds); KNHANES adults (X1 coding) {len(ext):,}")


def check_mp(t, cohort, y, p, what, auroc_ci_=None):
    """n, events, AUROC and O:E of the retrained model on a full set equal main_performance.csv (and its AUROC CI when
    given). Returns the AUROC."""
    r = MP[(MP.target == SH[t]) & (MP.cohort == MP_COH[cohort])]
    assert len(r) == 1, (t, cohort)
    r = r.iloc[0]
    a = roc_auc_score(y, p)
    assert len(y) == int(r.n) and int(np.sum(y)) == int(r.events), (what, t, cohort, len(y), int(np.sum(y)))
    assert abs(a - float(r.auroc)) < 1e-9 and round(a, 4) == round(float(r.auroc), 4), (what, t, cohort, a, r.auroc)
    assert abs(np.sum(y) / np.sum(p) - float(r.oe)) < 1e-9, (what, t, cohort)
    if auroc_ci_ is not None:
        assert_same_ci(auroc_ci_, (r.auroc_lo, r.auroc_hi), f"{what} {t} {cohort} AUROC vs main_performance.csv")
    return a


INT, EXT = {}, {}
year_i = pd.to_datetime(dte[DATE]).dt.year.to_numpy()
for t, thr in MAIN_T.items():
    assert TARGETS[t]["threshold"] == thr and list(models[t].feature_names_in_) == FMODEL
    ri, re_ = pint[pint.target == t], pext[pext.target == t]
    yi, ye = labels(dte["urate"], thr), labels(ext["urate"], thr)
    assert len(ri) == N_TE and np.array_equal(ri["id"].to_numpy(), dte[ID].to_numpy()) and np.array_equal(ri.y.to_numpy(), yi)
    assert np.array_equal(ri.male.to_numpy(), dte["male"].to_numpy()) and np.allclose(ri.age.to_numpy(), dte["age"].to_numpy())
    assert len(re_) == N_EXT and np.array_equal(re_["id"].astype(str).to_numpy(), ext["id"].astype(str).to_numpy())
    assert np.array_equal(re_.y.to_numpy(), ye) and np.array_equal(re_.male.to_numpy(), male_e)
    assert int(yi.sum()) == int(CI2[f"test events {SH[t]}"]) and int(ye.sum()) == int(CI2[f"KNHANES events {SH[t]}"])
    # the stored predictions are the retrained models' own (recomputed here on every person)
    err = max(np.abs(models[t].predict_proba(dte[FMODEL])[:, 1] - ri.p.to_numpy()).max(),
              np.abs(models[t].predict_proba(ext[FMODEL])[:, 1] - re_.p.to_numpy()).max())
    assert err < 1e-9, (t, err)
    INT[t] = dict(y=yi, p=ri.p.to_numpy(float), male=dte["male"].to_numpy(int), age=dte["age"].to_numpy(float),
                  year=year_i, X=dte[FMODEL])
    EXT[t] = dict(y=ye, p=re_.p.to_numpy(float))
    a_i = check_mp(t, "internal", yi, INT[t]["p"], "data")
    a_e = check_mp(t, "external", ye, EXT[t]["p"], "data")
    log(f"  {SH[t]}: internal test {len(yi):,} persons ({int(yi.sum()):,} events), KNHANES {len(ye):,} ({int(ye.sum()):,} "
        f"events); predictions recomputed from the model (max |diff| {err:.1e}); AUROC {a_i:.4f} / {a_e:.4f} "
        f"(= main_performance.csv)")

# ======================================================================== (a) setting
log("\n(a) Care setting and urine-only encounters (adults)")
fv = pd.read_csv(RAW_FIRST, usecols=[ID, RAW_DT], low_memory=False)
fv["udate"] = pd.to_datetime(fv[RAW_DT]).dt.normalize()
# processed data file; its row number is the index of the cached window (and of dev_v2), checked in (b)
proc = pd.read_csv(DATA_PATH, usecols=[ID, "days_difference", "age_at_urine_test", "체고", "체중", "BMI_demo", "키",
                                       "몸무게"], low_memory=False)
mg = proc[[ID, "days_difference", "age_at_urine_test"]]
assert mg.groupby(ID)["age_at_urine_test"].nunique(dropna=False).max() == 1, "age at the urine test differs within a person"
age_mg = mg.groupby(ID)["age_at_urine_test"].first()     # used only for the adult-status agreement check below
axm = age_exclusion_counts(mg, col="age_at_urine_test")
mg = adult_window(mg, col="age_at_urine_test")           # processed file: adult rows only (adult target population)
log(f"  processed file rows {axm['rows']:,}: adult {axm['adult']:,}; excluded: minor {axm['minor']:,}, age >{AGE_MAX} "
    f"{axm['implausible']}, age missing {axm['missing']}")
mind = mg.assign(ad=mg["days_difference"].abs()).groupby(ID)["ad"].min()
fv["blood"] = np.where(fv[ID].map(mind) <= 14, "blood within ±14 d",
                       np.where(fv[ID].isin(mind.index), "blood 15-365 d only", "no blood within 365 d"))
nur = pd.concat([xlsx(p, usecols=["순번", " 검사일자", "성별", "생년월일", "키", "몸무게"]) for p in NURSING],
                ignore_index=True)
nur.columns = [c.strip() for c in nur.columns]
nur["date"] = pd.to_datetime(nur["검사일자"], errors="coerce").dt.normalize()
h = pd.to_numeric(nur["키"], errors="coerce"); w = pd.to_numeric(nur["몸무게"], errors="coerce")
h = h.where((h >= 50) & (h <= 220)); w = w.where((w >= 10) & (w <= 300))
nur["bmi_ok"] = (w / (h / 100) ** 2).between(10, 70)    # the plausibility rules of add_bmi_unified (and dev_v2)
log(f"  nursing (encounter) file (lookup of sex, birth date and same-day height/weight; all ages): {len(nur):,} rows, "
    f"{nur['순번'].nunique():,} persons")
chk = xlsx(CHECKUP, usecols=["일련번호", "수진일자", "성별구분코드", "환자생일일자", "BMI"])
chk["date"] = pd.to_datetime(chk["수진일자"], errors="coerce").dt.normalize()
cbmi = pd.to_numeric(chk["BMI"], errors="coerce")
chk["bmi_any"], chk["bmi_ok"] = cbmi.notna(), cbmi.between(10, 70)
log(f"  health-checkup file (lookup of sex, birth date, checkup date and BMI; all ages): {len(chk):,} rows, "
    f"{chk['일련번호'].nunique():,} persons")
demo = (nur.dropna(subset=["성별"]).drop_duplicates("순번").set_index("순번")[["성별", "생년월일"]])
demo2 = chk.drop_duplicates("일련번호").set_index("일련번호")[["성별구분코드", "환자생일일자"]]
fv["sex"] = fv[ID].map(demo["성별"]).fillna(fv[ID].map(demo2["성별구분코드"]))
birth = pd.to_datetime(fv[ID].map(demo["생년월일"]), errors="coerce").fillna(
    pd.to_datetime(fv[ID].map(demo2["환자생일일자"]), errors="coerce"))
fv["age"] = (fv["udate"] - birth).dt.days / 365.25
n_fv, n_undet = len(fv), int(fv["age"].isna().sum())
n_impl = int(((fv["age"] < 0) | (fv["age"] > AGE_MAX)).sum())
fv.loc[(fv["age"] < 0) | (fv["age"] > AGE_MAX), "age"] = np.nan
# adult first-visit tests only (adult target population): age at the urine test 19-110 from the birth date. For every
# person with a processed-file record, adult status must equal adult status by the processed file's age_at_urine_test.
inp = fv[ID].isin(age_mg.index).values
assert (is_adult(fv["age"]).values[inp] == is_adult(fv[ID].map(age_mg)).values[inp]).all(), \
    "adult status from the birth date differs from age_at_urine_test"
n_minor = int((fv["age"] < ADULT_AGE).sum())
fv = fv[is_adult(fv["age"])].copy()
assert n_fv - len(fv) == n_minor + n_impl + n_undet
# this line is parsed by make_figures.py and make_supplement.py (participant flow): keep its wording
log(f"  first-visit persons {n_fv:,} (all ages); adult ({ADULT_AGE}-{AGE_MAX} y) {len(fv):,} ({fv[ID].nunique():,} unique); "
    f"excluded: aged <{ADULT_AGE} {n_minor:,}, age <0 or >{AGE_MAX} {n_impl}, age not determinable (no birth or urine-test "
    f"date) {n_undet}; "
    f"adult status agrees with age_at_urine_test for all {int(inp.sum()):,} persons with a processed-file record")
log(f"  adult first-visit persons by blood status: " + "; ".join(f"{k} {v:,}" for k, v in fv["blood"].value_counts().items()))
key_n = set(zip(nur.loc[nur["bmi_ok"], "순번"], nur.loc[nur["bmi_ok"], "date"]))
key_c = set(zip(chk["일련번호"], chk["date"]))
key_cb = set(zip(chk.loc[chk["bmi_ok"], "일련번호"], chk.loc[chk["bmi_ok"], "date"]))
key_cb_any = set(zip(chk.loc[chk["bmi_any"], "일련번호"], chk.loc[chk["bmi_any"], "date"]))
pairs = list(zip(fv[ID], fv["udate"]))
fv["nurse_bmi_same_day"] = [k in key_n for k in pairs]
fv["checkup_same_day"] = [k in key_c for k in pairs]
fv["checkup_bmi_same_day"] = [k in key_cb for k in pairs]
ck_bmi_any = np.array([k in key_cb_any for k in pairs])
# BMI of the urine-test day as dev_v2 builds it: the same-day checkup BMI when one is recorded (missing when outside
# 10-70 kg/m2), otherwise the nursing height and weight of that day
fv["bmi_available"] = fv["checkup_bmi_same_day"] | (~ck_bmi_any & fv["nurse_bmi_same_day"])
fv["period_2022"] = fv["udate"].dt.year >= 2022
# the adult first-visit persons with a blood test within +/-14 days are the adult window persons of dev_v2
pv = persons_v2()
pva = pv[pv["adult"]].copy()
STEPS = cohort_steps()
fv_win = fv[fv["blood"] == "blood within ±14 d"]
assert len(pva) == STEPS[1][1] == len(fv_win) and set(fv_win[ID]) == set(pva[ID]), "a1 window persons differ from dev_v2"
fb = fv_win.set_index(ID).loc[pva[ID].to_numpy(), "bmi_available"].to_numpy(bool)
hb = pva["has_bmi"].to_numpy(bool)
log(f"  adult window persons ({len(pva):,} = dev_v2 step '{STEPS[1][0]}'): BMI of the urine-test day available "
    f"{100 * fb.mean():.2f}% from the nursing and checkup files vs {100 * hb.mean():.2f}% in dev_v2 (processed file); "
    f"person-level agreement {100 * (fb == hb).mean():.2f}% ({int((fb & ~hb).sum()):,} available in the source files "
    f"only, {int((~fb & hb).sum()):,} in dev_v2 only)")
rows = []


def setting_row(section, group, dd, extra=None):
    a = dd["age"]
    r = dict(section=section, group=group, n=len(dd), age_known=int(a.notna().sum()), age_mean=a.mean(), age_sd=a.std(),
             pct_age_lt19=100 * (a < 19).mean(), pct_age_ge65=100 * (a >= 65).mean(),
             pct_male=100 * (dd["sex"] == "M").sum() / dd["sex"].notna().sum() if "sex" in dd else np.nan,
             pct_checkup_same_day=100 * dd["checkup_same_day"].mean(),
             pct_bmi_available=100 * dd["bmi_available"].mean(),
             pct_checkup_bmi_same_day=100 * dd["checkup_bmi_same_day"].mean() if "checkup_bmi_same_day" in dd else np.nan,
             pct_nurse_bmi_same_day=100 * dd["nurse_bmi_same_day"].mean() if "nurse_bmi_same_day" in dd else np.nan,
             pct_2022_2025=100 * dd["period_2022"].mean() if "period_2022" in dd else np.nan)
    r.update(extra or {})
    rows.append(r)
    log(f"  [{section}] {group}: n {len(dd):,}; age {r['age_mean']:.1f} ({r['age_sd']:.1f}), <19 {r['pct_age_lt19']:.1f}%, "
        f">=65 {r['pct_age_ge65']:.1f}%; male {r['pct_male']:.1f}%; same-day checkup {r['pct_checkup_same_day']:.1f}%; "
        f"BMI of the urine-test day {r['pct_bmi_available']:.1f}% (same-day checkup BMI {r['pct_checkup_bmi_same_day']:.1f}%, "
        f"same-day nursing height/weight {r['pct_nurse_bmi_same_day']:.1f}%); 2022-2025 {r['pct_2022_2025']:.1f}%")


setting_row("a1 first-visit urine tests", "all", fv)
for g in ("blood within ±14 d", "blood 15-365 d only", "no blood within 365 d"):
    setting_row("a1 first-visit urine tests", g, fv[fv["blood"] == g])
setting_row("a1 first-visit urine tests", "no blood within ±14 d (urine-only encounter)", fv[fv["blood"] != "blood within ±14 d"])


def same_day_checkup(frame):
    """A health-checkup record dated on the urine-test day (parsed as cohort.write_cache_v2 does for the BMI rule)."""
    ck = pd.to_datetime(frame["수진일자"].astype(str), errors="coerce", format="mixed").dt.normalize()
    return (ck == pd.to_datetime(frame[DATE]).dt.normalize()).to_numpy()


# a2 analytic cohort by same-day checkup record
ck_d = same_day_checkup(win2.loc[d.index])
src = d["bmi_v2_source"].to_numpy()
assert not (src == "checkup")[~ck_d].any(), "checkup BMI without a same-day checkup record"
dcs = d.copy()
dcs["checkup_same_day"] = ck_d
dcs["sex"] = np.where(dcs["male"] == 1, "M", "F")
dcs["bmi_available"] = dcs["bmi_unified"].notna()
assert dcs["bmi_available"].all()
dcs["period_2022"] = pd.to_datetime(dcs[DATE]).dt.year >= 2022
for g, dd in (("all", dcs), ("same-day health-checkup record", dcs[dcs.checkup_same_day]),
              ("no same-day health-checkup record", dcs[~dcs.checkup_same_day])):
    setting_row("a2 development cohort", g, dd, dict(prev_ge7=100 * (dd["urate"] >= 7).mean(),
                                                      prev_ge9=100 * (dd["urate"] >= 9).mean(),
                                                      pct_bmi_from_checkup=100 * (dd["bmi_v2_source"] == "checkup").mean()))
log(f"  [a2] BMI from the same-day checkup {int((src == 'checkup').sum()):,}, from the nursing record "
    f"{int((src == 'nursing').sum()):,} (of whom {int(((src == 'nursing') & ck_d).sum()):,} had a same-day checkup "
    f"without a BMI)")
# a3 adult window persons (one record per person, before the urate, BMI and completeness exclusions)
pva["checkup_same_day"] = same_day_checkup(pva)
for g, dd in (("window records, same-day checkup record", pva[pva.checkup_same_day]),
              ("window records, no same-day checkup record", pva[~pva.checkup_same_day])):
    rows.append(dict(section="a3 window records", group=g, n=len(dd), pct_bmi_available=100 * dd["has_bmi"].mean(),
                     pct_urate_measured=100 * dd["has_urate"].mean(), age_mean=dd["age"].mean(),
                     pct_male=100 * dd["male"].mean()))
    log(f"  [a3] {g} (adult window persons, one record per person): n {len(dd):,}; BMI of the urine-test day "
        f"{100 * dd['has_bmi'].mean():.1f}%; serum urate of the urine-test day {100 * dd['has_urate'].mean():.1f}%; "
        f"age {dd['age'].mean():.1f}; male {100 * dd['male'].mean():.1f}%")
# a4 retrained models on the internal test persons by same-day checkup record
ck_te = ck_d[te]
for t in MAIN_T:
    I = INT[t]
    for g, msk in (("same-day health-checkup record", ck_te), ("no same-day health-checkup record", ~ck_te)):
        a, lo, hi = auc_ci(I["y"][msk], I["p"][msk]); oe, olo, ohi = oe_ci(I["y"][msk], I["p"][msk])
        rows.append(dict(section="a4 internal test persons (retrained)", group=g, target=SH[t], n=int(msk.sum()),
                         positives=int(I["y"][msk].sum()), auroc=a, auroc_lo=lo, auroc_hi=hi, oe=oe, oe_lo=olo, oe_hi=ohi,
                         age_mean=I["age"][msk].mean(), pct_male=100 * I["male"][msk].mean()))
        log(f"  [a4] {SH[t]} {g}: n {int(msk.sum()):,} ({int(I['y'][msk].sum())} events); AUROC {a:.3f} ({lo:.3f}-{hi:.3f}); "
            f"O:E {oe:.2f} ({olo:.2f}-{ohi:.2f})")
out(pd.DataFrame(rows), "fu_setting.csv")
del nur, chk, key_n, key_c, key_cb, key_cb_any, pairs

# ======================================================================== (b) BMI source
log("\n(b) BMI of the analytic cohort: checkup-record BMI recomputation and digit preference")
link = proc.loc[d.index]                                   # the cohort records in the processed file
assert np.array_equal(link[ID].to_numpy(), d[ID].to_numpy()), "processed-file rows are not the cohort records"
assert_same(pd.to_numeric(link["BMI_demo"], errors="coerce"), d["BMI_demo"], "processed-file checkup BMI", tol=0)
is_ck, is_nu = src == "checkup", src == "nursing"
assert (is_ck | is_nu).all()
assert_same(d["bmi_unified"].to_numpy()[is_ck], d["BMI_demo"].to_numpy()[is_ck], "checkup-source BMI", tol=0)
assert_same(d["bmi_unified"].to_numpy()[is_nu], d["bmi_calc"].to_numpy()[is_nu], "nursing-source BMI", tol=0)
rows = []
ch = pd.to_numeric(link.loc[is_ck, "체고"], errors="coerce").to_numpy()
cw = pd.to_numeric(link.loc[is_ck, "체중"], errors="coerce").to_numpy()
dif = np.abs(cw / (ch / 100) ** 2 - d["BMI_demo"].to_numpy()[is_ck])
rows.append(dict(item="checkup BMI: persons (BMI from a health checkup on the urine-test day)", value=int(is_ck.sum())))
rows.append(dict(item="nursing BMI: persons (height and weight in the nursing record of the urine-test day)",
                 value=int(is_nu.sum())))
rows.append(dict(item="checkup BMI: height and weight recorded", value=int((~np.isnan(ch) & ~np.isnan(cw)).sum())))
for tol in (0.1, 0.2, 0.5):
    rows.append(dict(item=f"checkup BMI within {tol} kg/m2 of weight/height^2 from the checkup record, %",
                     value=100 * (dif <= tol).mean()))
for lab, x, scale in (("checkup weight (0.1 kg)", pd.Series(cw), 10), ("checkup height (1 cm)", pd.Series(ch), 1),
                      ("nursing weight (0.1 kg), same-day source", pd.to_numeric(link.loc[is_nu, "몸무게"], errors="coerce"), 10),
                      ("nursing height (0.1 cm), same-day source", pd.to_numeric(link.loc[is_nu, "키"], errors="coerce"), 10)):
    x = x.dropna()
    dig = (np.round(x * scale) % 10).astype(int)
    cnt = np.bincount(dig, minlength=10)
    chi2, p = stats.chisquare(cnt)
    rows.append(dict(item=f"{lab}: n", value=len(x)))
    rows.append(dict(item=f"{lab}: terminal digit shares 0-9, %", value=" / ".join(f"{100 * c / cnt.sum():.1f}" for c in cnt)))
    rows.append(dict(item=f"{lab}: terminal digit 0 or 5, %", value=100 * (cnt[0] + cnt[5]) / cnt.sum()))
    rows.append(dict(item=f"{lab}: chi-square test of uniform terminal digits (df 9)", value=f"{chi2:.1f}; P={p:.3g}"))
    if scale == 10:
        rows.append(dict(item=f"{lab}: recorded with a nonzero decimal, %", value=100 * (dig != 0).mean()))
    log(f"  {lab}: n {len(x):,}; digit 0/5 {100 * (cnt[0] + cnt[5]) / cnt.sum():.1f}%; chi2 {chi2:.1f} (P={p:.3g}); "
        f"digits {' '.join(f'{100 * c / cnt.sum():.1f}' for c in cnt)}")
log(f"  BMI source: checkup on the urine-test day {int(is_ck.sum()):,}, nursing record of that day {int(is_ck.size - is_ck.sum()):,}; "
    f"checkup BMI vs weight/height^2: within 0.2 kg/m2 {100 * (dif <= 0.2).mean():.1f}%; within 0.5 {100 * (dif <= 0.5).mean():.1f}%")
out(pd.DataFrame(rows), "fu_bmi_source.csv")
del proc, link

# ======================================================================== refits used by (c)-(f) and (n)
log("\nRefits on the training persons (in memory): LightGBM (fixed), main-effects LR and spline LR for the input sets "
    f"{', '.join(FU_TIERS)}; internal = test persons, external = KNHANES adults (X1)")
ft = pd.read_csv(os.path.join(HERE, "feature_tiers.csv"))
REF = {}
for t, thr in MAIN_T.items():
    y = labels(d["urate"], thr)
    R = dict(yi=y[te], ye=EXT[t]["y"])
    for tier in FU_TIERS:
        feats = TIERS[tier]
        X, Xe = d[feats].to_numpy(float), ext[feats].to_numpy(float)
        for mname, make in CLASSES:
            m = make(feats).fit(X[tr], y[tr])
            R[(tier, mname, "internal")] = m.predict_proba(X[te])[:, 1]
            R[(tier, mname, "external")] = m.predict_proba(Xe)[:, 1]
    REF[t] = R
    worst = 0.0
    for tier in FU_TIERS:
        for mname, _ in CLASSES:
            for coh in ("internal", "external"):
                r = ft[(ft.row_type == "tier") & (ft.target == SH[t]) & (ft.setting == coh) & (ft.model == FT_MODEL[mname])
                       & (ft.tier == tier)]
                assert len(r) == 1, (t, tier, mname, coh)
                yy = R["yi"] if coh == "internal" else R["ye"]
                worst = max(worst, abs(roc_auc_score(yy, R[(tier, mname, coh)]) - float(r.auroc.iloc[0])))
    log(f"  {SH[t]}: refits reproduce feature_tiers.csv AUROCs to {worst:.1e} ({'/'.join(FU_TIERS)}, 3 model classes, "
        f"internal and external)")
    assert worst < 1e-12, worst

# ======================================================================== (c) operating points by sex
log("\n(c) Operating points overall and by sex")
opp = pd.read_csv(os.path.join(HERE, "operating_points.csv"))


def opp_thr(t, rule):
    r = opp[(opp.target == SH[t]) & (opp.cohort == "internal") & (opp.policy == rule)]
    assert len(r) == 1, (t, rule)
    return float(r.threshold.iloc[0])


def flags_topk(p, frac):
    """Fractional flags for the top fraction (records tied at the cut-off flagged with equal fractions)."""
    n = len(p); k = frac * n
    order = np.sort(p)[::-1]
    cut = order[min(int(np.ceil(k)) - 1, n - 1)]
    above = p > cut; tie = p == cut
    f = above.astype(float)
    f[tie] = (k - above.sum()) / tie.sum()
    return f, float(cut)


def op_by_sex(y, f, male, B=N_BOOT):
    """sens, PPV, flagged share, share of flagged who are men; overall and within sex, bootstrap within group."""
    y = np.asarray(y, float); f = np.asarray(f, float); male = np.asarray(male)
    res = {}
    for g, msk in (("all", np.ones(len(y), bool)), ("men", male == 1), ("women", male == 0)):
        yy, ff, mm = y[msk], f[msk], male[msk]
        tp, fp = (ff * yy).sum(), (ff * (1 - yy)).sum()
        pt = dict(n=int(msk.sum()), events=int(yy.sum()), sens=tp / yy.sum(), ppv=tp / (tp + fp) if tp + fp > 0 else np.nan,
                  flagged=(tp + fp) / len(yy), spec=((1 - ff) * (1 - yy)).sum() / (1 - yy).sum(),
                  n_flagged=tp + fp, tp=tp, share_flagged_men=(ff * mm).sum() / ff.sum() if ff.sum() > 0 else np.nan)
        bs = {k: [] for k in ("sens", "ppv", "flagged")}
        for wv in boot_weights(len(yy), B, SEED + (0 if g == "all" else 1 if g == "men" else 2)):
            tpb, fpb = (wv * ff * yy).sum(), (wv * ff * (1 - yy)).sum()
            bs["sens"].append(tpb / (wv * yy).sum()); bs["flagged"].append((tpb + fpb) / wv.sum())
            bs["ppv"].append(tpb / (tpb + fpb) if tpb + fpb > 0 else np.nan)
        for k, v in bs.items():
            lo, hi = pci(v); pt[k + "_lo"], pt[k + "_hi"] = float(lo), float(hi)
        res[g] = pt
    return res


rows = []


def add_rows(t, strategy, rule, cohort, threshold, y, f, male, threshold_source):
    res = op_by_sex(y, f, male)
    for g, r in res.items():
        rows.append(dict(target=SH[t], strategy=strategy, rule=rule, cohort=cohort, group=g, threshold=threshold,
                         threshold_source=threshold_source, **r))
    a, m_, w_ = res["all"], res["men"], res["women"]
    log(f"  {SH[t]} {strategy:<34} {rule:<9} {cohort:<8} thr {threshold if isinstance(threshold, str) else f'{threshold:.4f}'}: "
        f"all sens {100 * a['sens']:.1f} PPV {100 * a['ppv']:.1f} flagged {100 * a['flagged']:.1f} (men {100 * a['share_flagged_men']:.1f}% of flagged) | "
        f"men sens {100 * m_['sens']:.1f} PPV {100 * m_['ppv']:.1f} flagged {100 * m_['flagged']:.1f} | "
        f"women sens {100 * w_['sens']:.1f} ({w_['sens'] * w_['events']:.0f}/{w_['events']}) PPV {100 * w_['ppv']:.1f} flagged {100 * w_['flagged']:.1f}")


for t in MAIN_T:
    I, E, R = INT[t], EXT[t], REF[t]
    for rule in ("youden", "sens80", "sens90"):
        thr_i = opp_thr(t, rule)
        add_rows(t, ENS, rule, "internal", thr_i, I["y"], (I["p"] >= thr_i).astype(float), I["male"],
                 "internal test (apparent)")
        add_rows(t, ENS, rule, "external", thr_i, E["y"], (E["p"] >= thr_i).astype(float), male_e, "internal test")
    for frac in (0.05, 0.10, 0.20):
        for coh, yy, pp, mm in (("internal", I["y"], I["p"], I["male"]), ("external", E["y"], E["p"], male_e)):
            f, cut = flags_topk(pp, frac)
            add_rows(t, ENS, f"top{int(frac * 100)}%", coh, cut, yy, f, mm, "cohort-specific")
    for coh, yy, mm in (("internal", I["y"], I["male"]), ("external", E["y"], male_e)):
        add_rows(t, "Test all men", "all men", coh, "n/a", yy, (mm == 1).astype(float), mm, "none")
    # 3F refits (age, sex, BMI): thresholds from their own predictions for the internal test persons
    for mname in (SPL, LGB):
        pi3, pe3 = R[("3F", mname, "internal")], R[("3F", mname, "external")]
        for rule, fn in (("youden", youden_threshold), ("sens80", lambda a, b: threshold_for_sensitivity(a, b, 0.8)),
                         ("sens90", lambda a, b: threshold_for_sensitivity(a, b, 0.9))):
            th = float(fn(R["yi"], pi3))
            add_rows(t, f"3F {mname} (refit)", rule, "external", th, E["y"], (pe3 >= th).astype(float), male_e,
                     "internal test (refit)")
        for frac in (0.05, 0.10, 0.20):
            f, cut = flags_topk(pe3, frac)
            add_rows(t, f"3F {mname} (refit)", f"top{int(frac * 100)}%", "external", cut, E["y"], f, male_e, "cohort-specific")
    # sex-specific thresholds for the retrained model (80% / 90% sensitivity within each sex, chosen on the internal test)
    for rule, s in (("sens80_sexspecific", 0.8), ("sens90_sexspecific", 0.9)):
        thm = float(threshold_for_sensitivity(I["y"][I["male"] == 1], I["p"][I["male"] == 1], s))
        thf = float(threshold_for_sensitivity(I["y"][I["male"] == 0], I["p"][I["male"] == 0], s))
        for coh, yy, pp, mm in (("internal", I["y"], I["p"], I["male"]), ("external", E["y"], E["p"], male_e)):
            f = np.where(mm == 1, pp >= thm, pp >= thf).astype(float)
            add_rows(t, f"{ENS}, sex-specific thresholds", rule, coh, f"men {thm:.4f}; women {thf:.4f}",
                     yy, f, mm, "internal test within sex")
sexop = pd.DataFrame(rows)
for coh, n_ in (("internal", N_TE), ("external", N_EXT)):
    assert (sexop[(sexop.cohort == coh) & (sexop.group == "all")].n == n_).all(), coh
out(sexop, "fu_sex_operating.csv")

# ======================================================================== (d) net benefit incl. recalibration
log("\n(d) External net benefit: recalibrated (cross-fitted) ensemble, 3F spline LR, 'test all men'")
dca = pd.read_csv(os.path.join(HERE, "decision_curve.csv"))


def nb_curve(y, p, grid, w=None, F=None):
    w = np.ones(len(y)) if w is None else w
    F = (p[:, None] >= grid[None, :]).astype(float) if F is None else F
    n = w.sum(); odds = grid / (1 - grid)
    tp = (w * y) @ F; fp = (w * (1 - y)) @ F
    return tp / n - fp / n * odds


def nb_rule(y, f, grid, w=None):
    w = np.ones(len(y)) if w is None else w
    n = w.sum(); odds = grid / (1 - grid)
    return (w * y * f).sum() / n - (w * (1 - y) * f).sum() / n * odds


def nb_all(y, grid, w=None):
    w = np.ones(len(y)) if w is None else w
    prev = (w * y).sum() / w.sum()
    return prev - (1 - prev) * grid / (1 - grid)


def crossfit_intercept(y, p, n_rep=N_SPLIT, seed=SEED):
    """2-fold cross-fitted intercept update, repeated over random splits: list of full-length prediction vectors."""
    lp = logit(p); idx = np.arange(len(y)); outv = []
    for r in range(n_rep):
        a, b = train_test_split(idx, test_size=0.5, random_state=seed + r, stratify=y)
        q = np.empty(len(y))
        q[b] = expit(fit_offset_intercept(y[a], lp[a]) + lp[b])
        q[a] = expit(fit_offset_intercept(y[b], lp[b]) + lp[a])
        outv.append(q)
    return outv


def ranges(grid, flag):
    runs, start, prev = [], None, None
    for g, f in zip(grid, flag):
        if f and start is None: start = g
        if not f and start is not None: runs.append((start, prev)); start = None
        prev = g
    if start is not None: runs.append((start, prev))
    return "; ".join(f"{a:g}-{b:g}" if a != b else f"{a:g}" for a, b in runs) or "none"


rows, rrows, orows = [], [], []
for t, thr in MAIN_T.items():
    grid = NB_GRID[thr]; y = EXT[t]["y"].astype(float); pdp = EXT[t]["p"]
    p3 = REF[t][("3F", SPL, "external")]
    men = (male_e == 1).astype(float)
    cf_dep = crossfit_intercept(y, pdp); cf_3 = crossfit_intercept(y, p3)
    S = {"ensemble_unrecalibrated": nb_curve(y, pdp, grid),
         "ensemble_recalibrated_crossfit": np.mean([nb_curve(y, q, grid) for q in cf_dep], axis=0),
         "3F_splineLR_unrecalibrated": nb_curve(y, p3, grid),
         "3F_splineLR_recalibrated_crossfit": np.mean([nb_curve(y, q, grid) for q in cf_3], axis=0),
         "test_all_men": nb_rule(y, men, grid), "treat_all": nb_all(y, grid)}
    dc_ = dca[(dca.target == SH[t]) & (dca.cohort == "external")].sort_values("threshold")
    assert_same(dc_.threshold.to_numpy(), grid, f"{t} decision-curve grid")
    assert_same(dc_.nb_model.to_numpy(), S["ensemble_unrecalibrated"], f"{t} net benefit vs decision_curve.csv")
    assert_same(dc_.nb_all.to_numpy(), S["treat_all"], f"{t} treat all vs decision_curve.csv")
    split_sd = np.std([nb_curve(y, q, grid) for q in cf_dep], axis=0)
    # paired bootstrap (records): ensemble variants minus 'test all men' and minus best default
    q0, q3 = cf_dep[0], cf_3[0]
    B = {k: [] for k in ("unrec_minus_men", "recal_minus_men", "recal_minus_default", "unrec_minus_default",
                         "recal_minus_3Frecal", "men_minus_default", "recal_nb")}
    Fu = (pdp[:, None] >= grid[None, :]).astype(float)
    Fr = (q0[:, None] >= grid[None, :]).astype(float)
    F3m = (q3[:, None] >= grid[None, :]).astype(float)
    for wv in boot_weights(len(y)):
        nu, nr = nb_curve(y, pdp, grid, wv, Fu), nb_curve(y, q0, grid, wv, Fr)
        nm, na = nb_rule(y, men, grid, wv), nb_all(y, grid, wv)
        n3 = nb_curve(y, q3, grid, wv, F3m)
        best = np.maximum(na, 0)
        B["unrec_minus_men"].append(nu - nm); B["recal_minus_men"].append(nr - nm)
        B["recal_minus_default"].append(nr - best); B["unrec_minus_default"].append(nu - best)
        B["recal_minus_3Frecal"].append(nr - n3); B["men_minus_default"].append(nm - best); B["recal_nb"].append(nr)
    CI = {k: pci(v) for k, v in B.items()}
    best = np.maximum(S["treat_all"], 0)
    for j, pt in enumerate(grid):
        r = dict(target=SH[t], threshold=pt, **{k: v[j] for k, v in S.items()}, recal_split_sd=split_sd[j],
                 recal_nb_lo=CI["recal_nb"][0][j], recal_nb_hi=CI["recal_nb"][1][j])
        for k in ("unrec_minus_men", "recal_minus_men", "recal_minus_default", "unrec_minus_default",
                  "recal_minus_3Frecal", "men_minus_default"):
            r[k + "_lo"], r[k + "_hi"] = CI[k][0][j], CI[k][1][j]
        rows.append(r)
    crit = [("ensemble unrecalibrated beats both defaults (point)", S["ensemble_unrecalibrated"] > best),
            ("ensemble recalibrated beats both defaults (point)", S["ensemble_recalibrated_crossfit"] > best),
            ("ensemble recalibrated beats both defaults (CI lower > 0)", CI["recal_minus_default"][0] > 0),
            ("test all men beats both defaults (point)", S["test_all_men"] > best),
            ("ensemble unrecalibrated beats test all men (point)", S["ensemble_unrecalibrated"] > S["test_all_men"]),
            ("ensemble unrecalibrated beats test all men (CI lower > 0)", CI["unrec_minus_men"][0] > 0),
            ("ensemble recalibrated beats test all men (point)", S["ensemble_recalibrated_crossfit"] > S["test_all_men"]),
            ("ensemble recalibrated beats test all men (CI lower > 0)", CI["recal_minus_men"][0] > 0),
            ("ensemble recalibrated beats both defaults and test all men (CI lower > 0 for both)",
             (CI["recal_minus_default"][0] > 0) & (CI["recal_minus_men"][0] > 0)),
            ("3F spline LR recalibrated beats both defaults (point)", S["3F_splineLR_recalibrated_crossfit"] > best),
            ("ensemble recalibrated beats 3F spline LR recalibrated (CI lower > 0)", CI["recal_minus_3Frecal"][0] > 0)]
    for c, fl in crit:
        rrows.append(dict(target=SH[t], criterion=c, ranges=ranges(grid, fl), grid_min=grid[0], grid_max=grid[-1]))
        log(f"  {SH[t]} {c}: {ranges(grid, fl)}")
    for pt in ((0.05, 0.08, 0.10, 0.20) if thr == 7.0 else (0.005, 0.01, 0.02, 0.03)):
        j = int(np.argmin(np.abs(grid - pt)))
        log(f"  {SH[t]} pt={grid[j]:g}: NB ensemble {S['ensemble_unrecalibrated'][j]:.4f}, recalibrated "
            f"{S['ensemble_recalibrated_crossfit'][j]:.4f}, test all men {S['test_all_men'][j]:.4f}, 3F spline LR "
            f"{S['3F_splineLR_unrecalibrated'][j]:.4f} (recal {S['3F_splineLR_recalibrated_crossfit'][j]:.4f}), treat all "
            f"{S['treat_all'][j]:.4f}; recal - men {S['ensemble_recalibrated_crossfit'][j] - S['test_all_men'][j]:+.4f} "
            f"({CI['recal_minus_men'][0][j]:+.4f} to {CI['recal_minus_men'][1][j]:+.4f})")
    # operating points on the recalibrated scale (cross-fitted; mean over splits)
    a_full = fit_offset_intercept(y, logit(pdp))
    for pt in ((0.05, 0.10, 0.15, 0.20, 0.30) if thr == 7.0 else (0.005, 0.01, 0.02, 0.03, 0.05)):
        acc = []
        for q in cf_dep:
            fl = (q >= pt).astype(float)
            res = {}
            for g, msk in (("all", np.ones(len(y), bool)), ("men", male_e == 1), ("women", male_e == 0)):
                tp = (fl[msk] * y[msk]).sum(); fp = (fl[msk] * (1 - y[msk])).sum()
                res[g] = (tp / y[msk].sum(), tp / (tp + fp) if tp + fp else np.nan, (tp + fp) / msk.sum())
            acc.append(res)
        for g in ("all", "men", "women"):
            arr = np.array([a[g] for a in acc], float)
            orows.append(dict(target=SH[t], recalibrated_threshold=pt, group=g, sens=np.nanmean(arr[:, 0]),
                              ppv=np.nanmean(arr[:, 1]), flagged=np.nanmean(arr[:, 2]),
                              model_scale_threshold=float(expit(logit(pt) - a_full))))
        a_ = [o for o in orows if o["target"] == SH[t] and o["recalibrated_threshold"] == pt]
        log(f"  {SH[t]} recalibrated risk >= {pt:g} (model-scale {a_[0]['model_scale_threshold']:.4f}): "
            + "; ".join(f"{o['group']} sens {100 * o['sens']:.1f} PPV {100 * o['ppv']:.1f} flagged {100 * o['flagged']:.1f}" for o in a_))
    log(f"  {SH[t]} full-sample intercept update {a_full:+.3f} (log-odds)")
out(pd.DataFrame(rows), "fu_net_benefit.csv")
out(pd.DataFrame(rrows), "fu_net_benefit_ranges.csv")
out(pd.DataFrame(orows), "fu_recal_operating.csv")

# ======================================================================== (e) within-sex increments
log("\n(e) Dipstick increment within each sex (KNHANES, refits on the training persons, X1 coding)")
specs, jobs = [], []
for t in MAIN_T:
    R, ye, pens = REF[t], EXT[t]["y"], EXT[t]["p"]
    for g, msk in (("men", male_e == 1), ("women", male_e == 0), ("all", np.ones(len(ye), bool))):
        for lab, a, b_ in ((f"{LGB}: 12F minus 3F", R[("12F", LGB, "external")], R[("3F", LGB, "external")]),
                           (f"{SPL}: 12F minus 3F", R[("12F", SPL, "external")], R[("3F", SPL, "external")]),
                           (f"{LGB}: 11F minus 2F", R[("11F", LGB, "external")], R[("2F", LGB, "external")]),
                           (f"{SPL}: 11F minus 2F", R[("11F", SPL, "external")], R[("2F", SPL, "external")]),
                           (f"{ENS} minus 3F spline LR", pens, R[("3F", SPL, "external")]),
                           (f"{ENS} minus 3F {LGB}", pens, R[("3F", LGB, "external")])):
            specs.append((t, g, lab, ye[msk], a[msk], b_[msk]))
            jobs.append((job_contrast, (ye[msk], a[msk], b_[msk])))
rows = []
for (t, g, lab, yy, a, b_), (dlt, pval, lo, hi) in zip(specs, par(jobs)):
    aa, ab = roc_auc_score(yy, a), roc_auc_score(yy, b_)
    if g == "all" and lab.startswith(ENS):
        check_mp(t, "external", yy, a, "(e)")
    rows.append(dict(target=SH[t], group=g, comparison=lab, n=len(yy), events=int(yy.sum()), auroc_a=aa, auroc_b=ab,
                     diff=dlt, diff_lo=lo, diff_hi=hi, delong_p=pval))
    log(f"  {SH[t]} {g:<5} {lab:<44} {aa:.3f} vs {ab:.3f}: {dlt:+.4f} ({lo:+.4f} to {hi:+.4f}), P={pval:.3g}")
out(pd.DataFrame(rows), "fu_within_sex.csv")

# ======================================================================== (f) 2F tiers
log("\n(f) Age-and-sex-only models and increments (refits; the fits and bootstrap draws of feature_tiers.csv)")
CONTRASTS_2F = [("11F", "2F", "dipstick added to age and sex"), ("3F", "2F", "BMI added to age and sex"),
                ("12F", "3F", "dipstick added to age, sex and BMI"), ("12F", "11F", "BMI added to dipstick, age and sex")]
specs, jobs = [], []
for t in MAIN_T:
    R = REF[t]
    for coh, yy in (("internal", R["yi"]), ("external", R["ye"])):
        for mname, _ in CLASSES:
            for tier in FU_TIERS:
                specs.append((t, coh, yy, mname, "tier", tier, None))
                jobs.append((job_auc, (yy, R[(tier, mname, coh)])))
            for A, Bt, meaning in CONTRASTS_2F:
                specs.append((t, coh, yy, mname, "contrast", (A, Bt), meaning))
                jobs.append((job_contrast, (yy, R[(A, mname, coh)], R[(Bt, mname, coh)])))
rows, worst_p = [], 0.0
for (t, coh, yy, mname, kind, tier, meaning), res in zip(specs, par(jobs)):
    fr = ft[(ft.target == SH[t]) & (ft.setting == coh) & (ft.model == FT_MODEL[mname])]
    if kind == "tier":
        a, lo, hi = res
        r = fr[(fr.row_type == "tier") & (fr.tier == tier)]
        assert len(r) == 1 and int(r.n.iloc[0]) == len(yy) and int(r.positives.iloc[0]) == int(yy.sum())
        assert_same(a, r.auroc.iloc[0], f"(f) {t} {coh} {mname} {tier} AUROC vs feature_tiers.csv")
        assert_same_ci((lo, hi), (r.auroc_lo.iloc[0], r.auroc_hi.iloc[0]), f"(f) {t} {coh} {mname} {tier}")
        rows.append(dict(target=SH[t], setting=coh, model=mname, row_type="tier", tier=tier, auroc=a,
                         auroc_lo=lo, auroc_hi=hi, n=len(yy), events=int(yy.sum())))
    else:
        dlt, pval, lo, hi = res
        A, Bt = tier
        r = fr[(fr.row_type == "contrast_tier") & (fr.tier == f"{A}-{Bt}")]
        assert len(r) == 1, (t, coh, mname, tier)
        assert_same(dlt, r.delta_auroc.iloc[0], f"(f) {t} {coh} {mname} {A}-{Bt} vs feature_tiers.csv", tol=1e-9)
        assert_same_ci((lo, hi), (r.delta_lo.iloc[0], r.delta_hi.iloc[0]), f"(f) {t} {coh} {mname} {A}-{Bt}")
        assert pval == 0 or abs(pval - float(r.delong_p.iloc[0])) < 1e-9, (t, coh, mname, tier, pval, r.delong_p.iloc[0])
        rows.append(dict(target=SH[t], setting=coh, model=mname, row_type="contrast", tier=f"{A} minus {Bt}",
                         meaning=meaning, auroc=dlt, auroc_lo=lo, auroc_hi=hi, delong_p=pval, n=len(yy), events=int(yy.sum())))
        log(f"  {SH[t]} {coh:<8} {mname:<20} {A}-{Bt}: {dlt:+.4f} ({lo:+.4f} to {hi:+.4f})  [{meaning}]")
for t in MAIN_T:
    R = REF[t]
    for coh, yy in (("internal", R["yi"]), ("external", R["ye"])):
        for mname, _ in CLASSES:
            log(f"  {SH[t]} {coh:<8} {mname:<20} AUROC " + ", ".join(
                f"{tier} {roc_auc_score(yy, R[(tier, mname, coh)]):.3f}" for tier in FU_TIERS))
f2 = pd.DataFrame(rows)
for coh, n_ in (("internal", N_TE), ("external", N_EXT)):
    assert (f2[f2.setting == coh].n == n_).all(), coh
log(f"  every AUROC, increment, 95% CI and DeLong P equals feature_tiers.csv (calc_feature_tiers.py)")
out(f2, "fu_tiers_2F.csv")

# ======================================================================== (g) OOF thresholds
log("\n(g) Thresholds from out-of-fold training predictions of the retrained models (cohort.cal_folds)")
rows = []
for t, thr in MAIN_T.items():
    m, I, E = models[t], INT[t], EXT[t]
    X_tr, ytr = train_xy(thr)
    assert len(ytr) == N_TR and np.array_equal(ytr, labels(d["urate"], thr)[tr])
    assert int(ytr.sum()) == int(CI2[f"training events {SH[t]}"])
    folds = cal_folds(m, X_tr, ytr)            # the model's 5 calibration folds, verified against its isotonic maps
    oof = np.full(len(ytr), np.nan)
    for k, (_, b_) in enumerate(folds):
        oof[b_] = m.calibrated_classifiers_[k].predict_proba(X_tr.iloc[b_])[:, 1]
    assert not np.isnan(oof).any()
    mtr = X_tr["male"].to_numpy(int)
    log(f"  {SH[t]}: 5 calibration folds of the {len(ytr):,} training persons ({int(ytr.sum()):,} events; fold sizes "
        f"{[len(b) for _, b in folds]}, events {[int(ytr[b].sum()) for _, b in folds]}); every fold's isotonic map "
        f"reproduced; OOF AUROC {roc_auc_score(ytr, oof):.3f}; OOF O:E {ytr.sum() / oof.sum():.3f}; OOF predictions of "
        f"exactly 0: {int((oof == 0).sum()):,} ({int(ytr[oof == 0].sum())} events)")
    for rule, fn in (("youden", youden_threshold), ("sens80", lambda a, b: threshold_for_sensitivity(a, b, 0.8)),
                     ("sens90", lambda a, b: threshold_for_sensitivity(a, b, 0.9))):
        th = float(fn(ytr, oof))
        th_test = opp_thr(t, rule)
        for coh, yy, pp, mm in (("training OOF", ytr, oof, mtr), ("internal", I["y"], I["p"], I["male"]),
                                ("external", E["y"], E["p"], male_e)):
            res = op_by_sex(yy, (pp >= th).astype(float), mm, B=200 if coh == "training OOF" else N_BOOT)
            for g, r in res.items():
                rows.append(dict(target=SH[t], rule=rule, threshold_oof=th, threshold_testset=th_test, cohort=coh, group=g, **r))
            a = res["all"]
            log(f"  {SH[t]} {rule}: OOF threshold {th:.4f} (test-set threshold {th_test:.4f}) -> {coh}: sens {100 * a['sens']:.1f} "
                f"({100 * a['sens_lo']:.1f}-{100 * a['sens_hi']:.1f}) spec {100 * a['spec']:.1f} PPV {100 * a['ppv']:.1f} "
                f"flagged {100 * a['flagged']:.1f}")
out(pd.DataFrame(rows), "fu_oof_thresholds.csv")

# ======================================================================== (h) internal by period
log("\n(h) Retrained models on the internal test persons by period of the urine test")
rows = []
for t in MAIN_T:
    I = INT[t]; yr = I["year"]
    groups = [("2010-2021", yr <= 2021), ("2022-2025", yr >= 2022), ("all", np.ones(len(yr), bool))] + \
             [(str(v), yr == v) for v in sorted(np.unique(yr))]
    for g, msk in groups:
        yy, pp = I["y"][msk], I["p"][msk]
        oe, lo, hi = oe_ci(yy, pp) if g in ("2010-2021", "2022-2025", "all") else (yy.sum() / pp.sum(), np.nan, np.nan)
        a, alo, ahi = auc_ci(yy, pp) if g in ("2010-2021", "2022-2025", "all") else (
            roc_auc_score(yy, pp) if yy.min() != yy.max() else np.nan, np.nan, np.nan)
        if g == "all":
            check_mp(t, "internal", yy, pp, "(h)", (alo, ahi))
        rows.append(dict(target=SH[t], period=g, n=int(msk.sum()), events=int(yy.sum()), prevalence=yy.mean(),
                         mean_pred=pp.mean(), oe=oe, oe_lo=lo, oe_hi=hi, auroc=a, auroc_lo=alo, auroc_hi=ahi))
        if g in ("2010-2021", "2022-2025", "all"):
            log(f"  {SH[t]} {g}: n {int(msk.sum()):,} events {int(yy.sum())}; prevalence {100 * yy.mean():.2f}% mean pred "
                f"{100 * pp.mean():.2f}%; O:E {oe:.2f} ({lo:.2f}-{hi:.2f}); AUROC {a:.3f} ({alo:.3f}-{ahi:.3f})")
out(pd.DataFrame(rows), "fu_internal_period.csv")

# ======================================================================== (i) ketone
log("\n(i) Ketone and calibration")
rows = []
for t in MAIN_T:
    I, E, m = INT[t], EXT[t], models[t]
    ki = I["X"]["ua_ketone"].to_numpy(); ke = ext["ua_ketone"].to_numpy()
    yr = I["year"]
    for coh, yy, pp, kk in (("internal", I["y"], I["p"], ki), ("external", E["y"], E["p"], ke)):
        for g, msk in (("ketone negative", kk == 0), ("ketone trace", kk == 0.5), ("ketone 1+ or more", kk >= 1),
                       ("ketone trace or more", kk > 0)):
            if msk.sum() == 0:
                continue
            oe, lo, hi = oe_ci(yy[msk], pp[msk])
            rows.append(dict(target=SH[t], analysis="O:E by ketone status", cohort=coh, group=g, n=int(msk.sum()),
                             events=int(yy[msk].sum()), mean_pred=pp[msk].mean(), prevalence=yy[msk].mean(), oe=oe, oe_lo=lo, oe_hi=hi))
            log(f"  {SH[t]} {coh} {g}: n {int(msk.sum()):,} ({100 * msk.mean():.1f}%); prevalence {100 * yy[msk].mean():.2f}% mean pred "
                f"{100 * pp[msk].mean():.2f}%; O:E {oe:.2f} ({lo:.2f}-{hi:.2f})")
    # counterfactual re-prediction with the retrained model (only the ketone input changed)
    for lab, Xc in (("internal, ketone trace set to 0", I["X"].assign(ua_ketone=np.where(ki == 0.5, 0.0, ki))),
                    ("internal, all ketone set to 0", I["X"].assign(ua_ketone=0.0))):
        pc = m.predict_proba(Xc)[:, 1]
        for g, msk in (("all", np.ones(len(ki), bool)), ("records changed", (ki == 0.5) if "trace" in lab else (ki > 0)),
                       ("2010-2021", yr <= 2021), ("2022-2025", yr >= 2022)):
            rows.append(dict(target=SH[t], analysis=lab, cohort="internal", group=g, n=int(msk.sum()),
                             events=int(I["y"][msk].sum()), mean_pred_before=I["p"][msk].mean(), mean_pred_after=pc[msk].mean(),
                             rel_change_pct=100 * (pc[msk].mean() / I["p"][msk].mean() - 1),
                             oe_before=I["y"][msk].sum() / I["p"][msk].sum(), oe_after=I["y"][msk].sum() / pc[msk].sum(),
                             auroc_before=roc_auc_score(I["y"][msk], I["p"][msk]) if I["y"][msk].min() != I["y"][msk].max() else np.nan,
                             auroc_after=roc_auc_score(I["y"][msk], pc[msk]) if I["y"][msk].min() != I["y"][msk].max() else np.nan))
            r = rows[-1]
            if g == "all":
                check_mp(t, "internal", I["y"], I["p"], "(i)")
            log(f"  {SH[t]} {lab} [{g}]: n {r['n']:,}; mean pred {100 * r['mean_pred_before']:.2f}% -> {100 * r['mean_pred_after']:.2f}% "
                f"({r['rel_change_pct']:+.1f}%); O:E {r['oe_before']:.2f} -> {r['oe_after']:.2f}")
    pc = m.predict_proba(ext[FMODEL].assign(ua_ketone=0.0))[:, 1]
    for g, msk in (("all", np.ones(len(ke), bool)), ("records changed", ke > 0)):
        rows.append(dict(target=SH[t], analysis="external, all ketone set to 0", cohort="external", group=g, n=int(msk.sum()),
                         events=int(E["y"][msk].sum()), mean_pred_before=E["p"][msk].mean(), mean_pred_after=pc[msk].mean(),
                         rel_change_pct=100 * (pc[msk].mean() / E["p"][msk].mean() - 1),
                         oe_before=E["y"][msk].sum() / E["p"][msk].sum(), oe_after=E["y"][msk].sum() / pc[msk].sum(),
                         auroc_before=roc_auc_score(E["y"][msk], E["p"][msk]), auroc_after=roc_auc_score(E["y"][msk], pc[msk])))
        r = rows[-1]
        if g == "all":
            check_mp(t, "external", E["y"], E["p"], "(i)")
        log(f"  {SH[t]} external, all ketone set to 0 [{g}]: n {r['n']:,}; mean pred {100 * r['mean_pred_before']:.2f}% -> "
            f"{100 * r['mean_pred_after']:.2f}%; O:E {r['oe_before']:.2f} -> {r['oe_after']:.2f}; AUROC {r['auroc_before']:.3f} -> {r['auroc_after']:.3f}")
out(pd.DataFrame(rows), "fu_ketone.csv")

# ======================================================================== (j) KNHANES by year
log("\n(j) KNHANES calibration by survey year")
rows = []
yrs = ext["year"].to_numpy()
for t in MAIN_T:
    E = EXT[t]
    for v in sorted(np.unique(yrs)):
        msk = yrs == v
        oe, lo, hi = oe_ci(E["y"][msk], E["p"][msk])
        a = roc_auc_score(E["y"][msk], E["p"][msk])
        rows.append(dict(target=SH[t], year=int(v), n=int(msk.sum()), events=int(E["y"][msk].sum()),
                         prevalence=E["y"][msk].mean(), mean_pred=E["p"][msk].mean(), oe=oe, oe_lo=lo, oe_hi=hi, auroc=a))
    yd = pd.DataFrame([r for r in rows if r["target"] == SH[t]])
    assert yd.n.sum() == N_EXT
    log(f"  {SH[t]}: O:E by year " + ", ".join(f"{int(r.year)} {r.oe:.2f}" for _, r in yd.iterrows())
        + f"; range {yd.oe.min():.2f}-{yd.oe.max():.2f}; AUROC range {yd.auroc.min():.3f}-{yd.auroc.max():.3f}")
out(pd.DataFrame(rows), "fu_knhanes_year.csv")

# ======================================================================== (k) SG restriction
log("\n(k) KNHANES adults by specific gravity range")
sg_dev = float(d["ua_sg"].to_numpy()[tr].max())
assert abs(sg_dev - 1.030) < 1e-9, sg_dev                   # the 'development range' of the labels below
rows = []
sg = ext["ua_sg"].to_numpy()
for t in MAIN_T:
    E = EXT[t]
    for g, msk in (("SG <=1.030 (development range)", sg <= 1.030 + 1e-9), ("SG >1.030", sg > 1.030 + 1e-9),
                   ("all", np.ones(len(sg), bool))):
        a, lo, hi = auc_ci(E["y"][msk], E["p"][msk]); oe, olo, ohi = oe_ci(E["y"][msk], E["p"][msk])
        if g == "all":
            check_mp(t, "external", E["y"][msk], E["p"][msk], "(k)", (lo, hi))
        rows.append(dict(target=SH[t], group=g, n=int(msk.sum()), events=int(E["y"][msk].sum()), auroc=a, auroc_lo=lo,
                         auroc_hi=hi, oe=oe, oe_lo=olo, oe_hi=ohi, pct_male=100 * male_e[msk].mean()))
        log(f"  {SH[t]} {g}: n {int(msk.sum()):,} events {int(E['y'][msk].sum())}; AUROC {a:.3f} ({lo:.3f}-{hi:.3f}); "
            f"O:E {oe:.2f} ({olo:.2f}-{ohi:.2f}); male {100 * male_e[msk].mean():.1f}%")
out(pd.DataFrame(rows), "fu_sg_restrict.csv")

# ======================================================================== (l) KNHANES flow
log("\n(l) KNHANES file composition and serum urate availability (knhanes_v2, X1 coding)")
rows = []
k = kn2.copy()
for v, n in k["year"].value_counts().sort_index().items():
    rows.append(dict(section="records per survey year", group=str(int(v)), n=int(n)))
ab = pd.cut(k["age"], [-1, 9, 18, 39, 59, 79, 200], labels=["<10", "10-18", "19-39", "40-59", "60-79", ">=80"])
for g in ab.cat.categories:
    msk = ab == g
    rows.append(dict(section="serum urate available by age", group=g, n=int(msk.sum()),
                     n_with_urate=int(k.loc[msk, "urate"].notna().sum()), pct_with_urate=100 * k.loc[msk, "urate"].notna().mean()))
    log(f"  age {g}: n {int(msk.sum()):,}; with serum urate {int(k.loc[msk, 'urate'].notna().sum()):,} "
        f"({100 * k.loc[msk, 'urate'].notna().mean():.1f}%)")
ad = k[k["age"] >= ADULT_AGE]
steps = [("adults aged >=19", ad), ("with serum urate", ad[ad["urate"].notna()])]
steps.append(("with the 9 measured dipstick fields", steps[-1][1].dropna(subset=list(F9))))
steps.append(("with age, sex and BMI", steps[-1][1].dropna(subset=["age", "male", "bmi_unified"])))
for lab, dd in steps:
    rows.append(dict(section="adult attrition", group=lab, n=len(dd)))
    log(f"  adults: {lab}: {len(dd):,}")
assert len(steps[-1][1]) == N_EXT and set(steps[-1][1]["id"]) == set(ext["id"]), "adult attrition does not end at ext"
noua = ad[ad["urate"].isna()]
rows.append(dict(section="adults without serum urate", group="n; mean age; % aged >=65; % male", n=len(noua),
                 age_mean=noua["age"].mean(), pct_age_ge65=100 * (noua["age"] >= 65).mean(), pct_male=100 * noua["male"].mean()))
wua = ad[ad["urate"].notna()]
rows.append(dict(section="adults with serum urate", group="n; mean age; % aged >=65; % male", n=len(wua),
                 age_mean=wua["age"].mean(), pct_age_ge65=100 * (wua["age"] >= 65).mean(), pct_male=100 * wua["male"].mean()))
log(f"  adults without serum urate: {len(noua):,}; mean age {noua['age'].mean():.1f}; >=65 {100 * (noua['age'] >= 65).mean():.1f}%; "
    f"male {100 * noua['male'].mean():.1f}% | with urate: mean age {wua['age'].mean():.1f}; >=65 {100 * (wua['age'] >= 65).mean():.1f}%; "
    f"male {100 * wua['male'].mean():.1f}%")
out(pd.DataFrame(rows), "fu_knhanes_flow.csv")

# ======================================================================== (m) isotonic zeros
log("\n(m) Isotonic outputs of exactly 0: flooring and replacement (retrained models)")
rows = []
for t in MAIN_T:
    for coh in ("internal", "external"):
        g = pcal[(pcal.target == t) & (pcal.cohort == coh)]
        iso = g[g.method == ISO_KEY]; lg = g[g.method == "logistic_logodds"]
        y = iso.y.to_numpy(); pi_ = iso.p.to_numpy(float); pl = lg.p.to_numpy(float)
        ref = INT[t] if coh == "internal" else EXT[t]
        assert np.array_equal(y, lg.y.to_numpy()) and np.array_equal(y, ref["y"])
        assert_same(pi_, ref["p"], f"{t} {coh}: preds_calibration_methods.csv isotonic rows vs the stored predictions")
        variants = {"isotonic as fitted": pi_, "isotonic floored at 0.001": np.maximum(pi_, 0.001),
                    "isotonic, zeros replaced by logistic calibrator": np.where(pi_ == 0, pl, pi_),
                    "logistic calibrator (same fold learners)": pl}
        for v, pp in variants.items():
            a_, s_, oe = calib_stats(y, pp)
            if v == "isotonic as fitted":
                check_mp(t, coh, y, pp, "(m)")
            rows.append(dict(target=SH[t], cohort=coh, variant=v, n=len(y), n_zero=int((pp == 0).sum()),
                             events_at_zero=int(y[pp == 0].sum()), min_pred=pp.min(), auroc=roc_auc_score(y, pp),
                             brier=brier_score_loss(y, pp), scaled_brier=scaled_brier(y, pp),
                             log_loss=log_loss(y, np.clip(pp, 1e-15, 1 - 1e-15)), oe=oe, cal_intercept=a_, cal_slope=s_))
            r = rows[-1]
            log(f"  {SH[t]} {coh:<8} {v:<48}: zeros {r['n_zero']:,} (events {r['events_at_zero']}); AUROC {r['auroc']:.4f}; "
                f"Brier {r['brier']:.5f}; log loss {r['log_loss']:.4f}; O:E {oe:.3f}; slope {s_:.2f}")
out(pd.DataFrame(rows), "fu_isotonic_floor.csv")

# ======================================================================== (n) continuous models
log("\n(n) Continuous-urate models (refits on the training persons)")
REG = {**LGBM_FIXED, "objective": "regression", "n_jobs": N_THREADS}     # as calc_sex_nested.py (REG_PARAMS)
X = d[FMODEL]; z = np.log(d["urate"].to_numpy(float))
oof = np.full(len(tr), np.nan)
for a, b_ in KFold(n_splits=5, shuffle=True, random_state=SEED).split(tr):
    oof[b_] = LGBMRegressor(**REG).fit(X.iloc[tr[a]], z[tr[a]]).predict(X.iloc[tr[b_]])
resid = z[tr] - oof
reg = LGBMRegressor(**REG).fit(X.iloc[tr], z[tr])
smear = float(np.mean(np.exp(resid)))
# heteroscedastic: model log(r^2) on the OOF residuals; standardised residuals give the error distribution
oof_s = np.full(len(tr), np.nan)
lr2 = np.log(resid ** 2 + 1e-8)
for a, b_ in KFold(n_splits=5, shuffle=True, random_state=SEED + 1).split(tr):
    oof_s[b_] = LGBMRegressor(**REG).fit(X.iloc[tr[a]], lr2[a]).predict(X.iloc[tr[b_]])
sreg = LGBMRegressor(**REG).fit(X.iloc[tr], lr2)
sd_oof = np.exp(oof_s / 2)
zres = np.sort(resid / sd_oof)
rs = np.sort(resid)
log(f"  training persons {len(tr):,}: OOF log-residual SD {resid.std():.4f}; smearing factor {smear:.4f}")


def p_homo(mu, t):
    return (len(rs) - np.searchsorted(rs, np.log(t) - mu, side="left")) / len(rs)


def p_hetero(mu, sdv, t):
    return (len(zres) - np.searchsorted(zres, (np.log(t) - mu) / sdv, side="left")) / len(zres)


cn = pd.read_csv(os.path.join(HERE, "continuous.csv"))
T_OF = {7.0: "uric_hyper_70", 9.0: "uric_severe_9"}
blocks, specs, jobs = {"external": [], "internal": []}, [], []     # rows per cohort, in the order of the former file
for coh, Xc, u in (("external", ext[FMODEL], ext["urate"].to_numpy(float)), ("internal", dte[FMODEL], dte["urate"].to_numpy(float))):
    mu = reg.predict(Xc); sdv = np.exp(sreg.predict(Xc) / 2)
    pred_mg = np.exp(mu) * smear
    dec = pd.qcut(pred_mg, 10, labels=False, duplicates="drop")
    for q in range(int(dec.max()) + 1):
        msk = dec == q
        blocks[coh].append(dict(analysis="observed vs predicted urate by decile (homoscedastic mean, smearing)",
                                cohort=coh, decile=q + 1, n=int(msk.sum()), mean_pred_mgdl=pred_mg[msk].mean(),
                                mean_obs_mgdl=u[msk].mean()))
    log(f"  {coh}: deciles predicted vs observed mean urate (mg/dL): "
        + "; ".join(f"{r['mean_pred_mgdl']:.2f}/{r['mean_obs_mgdl']:.2f}" for r in blocks[coh]))
    blocks[coh].append(dict(analysis="calibration-in-the-large of predicted urate", cohort=coh, n=len(u),
                            mean_pred_mgdl=pred_mg.mean(), mean_obs_mgdl=u.mean()))
    log(f"  {coh}: mean predicted {pred_mg.mean():.3f} vs observed {u.mean():.3f} mg/dL (n {len(u):,})")
    for thr in (7.0, 9.0):
        t = T_OF[thr]
        y = (u >= thr).astype(int)
        assert np.array_equal(y, INT[t]["y"] if coh == "internal" else EXT[t]["y"])
        pb = REF[t][("12F", LGB, coh)]                  # equal-budget binary LightGBM (12F of feature_tiers.csv)
        ph, pz = p_homo(mu, thr), p_hetero(mu, sdv, thr)
        for lab, pp in (("homoscedastic", ph), ("heteroscedastic", pz)):
            specs.append(("classification", coh, thr, lab, y, pp))
            jobs.append((job_auc, (y, pp)))
        for lab, pa_, pb_ in (("heteroscedastic minus homoscedastic", pz, ph), ("heteroscedastic minus binary LightGBM", pz, pb),
                              ("homoscedastic minus binary LightGBM", ph, pb)):
            specs.append(("comparison", coh, thr, lab, y, None))
            jobs.append((job_contrast, (y, pa_, pb_)))
        # calc_sex_nested.py fitted the same homoscedastic model and binary LightGBM configuration on the same persons
        cc = cn[(cn.row_type == "classification") & (cn.target == f">={thr}")
                & cn.cohort.str.startswith("External" if coh == "external" else "Internal")]
        r_h = cc[cc.model.str.startswith("Simple homoscedastic")]
        r_b = cc[cc.model.str.startswith("Equal-budget binary")]
        assert len(r_h) == 1 and len(r_b) == 1, (coh, thr)
        assert_same(roc_auc_score(y, ph), r_h.auroc.iloc[0], f"(n) {coh} >={thr} homoscedastic AUROC vs continuous.csv")
        log(f"  {coh} >={thr}: homoscedastic AUROC equals continuous.csv ({float(r_h.auroc.iloc[0]):.4f}); binary LightGBM "
            f"(12F of feature_tiers.csv) {roc_auc_score(y, pb):.6f} vs continuous.csv {float(r_b.auroc.iloc[0]):.6f}")
for (kind, coh, thr, lab, y, pp), res in zip(specs, par(jobs)):
    if kind == "classification":
        a, lo, hi = res
        rr = cn[(cn.row_type == "classification") & (cn.target == f">={thr}") & cn.model.str.startswith("Simple homoscedastic")
                & cn.cohort.str.startswith("External" if coh == "external" else "Internal")]
        if lab == "homoscedastic":
            assert_same_ci((lo, hi), (rr.auroc_lo.iloc[0], rr.auroc_hi.iloc[0]), f"(n) {coh} >={thr} homoscedastic")
        blocks[coh].append(dict(analysis="classification", cohort=coh, threshold=thr, model=lab, auroc=a, auroc_lo=lo,
                                auroc_hi=hi, oe=y.sum() / pp.sum(), n=len(y), events=int(y.sum())))
        log(f"  {coh} >={thr}: {lab} AUROC {a:.3f} ({lo:.3f}-{hi:.3f}); O:E {y.sum() / pp.sum():.2f}")
    else:
        dlt, pval, lo, hi = res
        blocks[coh].append(dict(analysis="comparison", cohort=coh, threshold=thr, model=lab, auroc=dlt, auroc_lo=lo,
                                auroc_hi=hi, delong_p=pval))
        log(f"  {coh} >={thr}: {lab} {dlt:+.4f} ({lo:+.4f} to {hi:+.4f}), P={pval:.3g}")
fc = pd.DataFrame(blocks["external"] + blocks["internal"])
for coh, n_ in (("internal", N_TE), ("external", N_EXT)):
    assert (fc[(fc.cohort == coh) & (fc.analysis == "classification")].n == n_).all(), coh
out(fc, "fu_continuous.csv")
log(f"\nDone in {(time.time() - T0) / 60:.1f} min")
