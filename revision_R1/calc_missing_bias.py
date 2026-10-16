"""Missing data, MCAR, multiple imputation of BMI, BMI source and verification bias for the retrained models
(bestmodel_v5, 12 inputs; uric-acid revision R1, JMIR Medical Informatics #107515).

Responds to
  Editor            report missing-data proportions; test MCAR; consider multiple imputation
  Reviewer AW #6    verification bias: serum urate exists only where a clinician ordered it
  Reviewer AW #9    BMI linked by nearest date with no window; questionnaire vs measured BMI

Analysis frame
  cohort.persons_v2() restricted to adults (age 19-110 y at the urine test; persons younger than 19 and implausible
  ages are logged): one row per person of the +/-14-day window of first-visit urine tests (every window row of a
  person repeats the same urine test, so records = persons), with the v2 fields of the retrained models: nitrite
  parsed from the raw first-visit text, serum urate of the urine-test day ('<1.0' = 0.5 mg/dL, below every threshold)
  and BMI of the urine-test day (checkup BMI only when the health checkup was on that day, otherwise weight/height^2
  from the nursing record of that day; values outside 10-70 set to missing). The analytic cohort (dev_v2) is the
  complete-case subset of this frame. Model inputs = FMODEL (9 dipstick fields, age, sex, BMI); leukocyte esterase is
  not a model input.

Parts and outputs
  (a) missing_table.csv      missingness of the 12 inputs and same-day urate among adult window persons, the
                             attrition to the analytic cohort (= the dev_v2 exclusion steps, checked against
                             cohort.cohort_steps()), persons excluded only because same-day BMI was missing vs
                             included persons
  (b) mcar.csv               Little's MCAR test (EM under multivariate normality) on all adult window persons (age,
                             sex, 9 dipstick fields, BMI, urate), on 10 random subsamples of 5,000 (plus a sample-size
                             series), missingness-pattern counts
  (c) mice_bmi.csv           retrained models on adults with a same-day urate but no BMI of that day (never used);
                             BMI multiply imputed (IterativeImputer, sample_posterior, m=20); AUROC pooled by Rubin's
                             rules (logit scale); complete-case reference = the internal test persons
  (d) bmi_window.csv         agreement of the same-day checkup BMI with the nursing-record BMI of the same day;
                             retrained-model performance on the internal test persons by BMI source (checkup vs
                             nursing record), overall and within age bands; case mix by source
  (e) verification_bias.csv  P(same-day urate measured | dipstick, age, sex, calendar year) among adults; IPW
                             internal AUROC of the retrained models
  calc_missing_bias_log.txt  key printed numbers

The retrained ensembles are evaluated as saved; nothing is refitted or overwritten. Internal evaluations use the
internal test persons of split_v2 (25,907 persons, the same for every threshold), checked person by person against
preds_internal_clean.csv; their AUROC equals main_performance.csv.

Method notes (kept from the independent review of the first version)
  (c) sklearn 1.6.1 BayesianRidge.predict(return_std=True) computes x' Sigma x on UNCENTRED X although Sigma (sigma_)
      was estimated on centred X; with unscaled predictors (ua_sg mean 1.02, SD 0.007) every posterior draw carried
      excess noise. Imputation predictors are therefore standardised (the BMI column and its 10-70 bounds are
      untouched), and the script asserts that the imputed BMI SD equals sqrt(var(fitted) + residual variance) and that
      the imputer's predictive SD equals the residual SD.
  (d) age-stratified BMI-source comparison; CIs and tests suppressed for groups with < 10 positives.
  (b) multivariate-normality caveat, per-pattern chi-square contributions and runs without ua_nitrite / without rare
      patterns.
  (e) case-mix profile of the P(measured) fifths, fifths within age bands, alternative weight cap and notes on the
      fixed propensity model.

Removed with the retraining (decision of 2026-10-16; the previous version, which evaluated the original models on the
person-clean test rows of their row-level split, is in _snapshot_original_models_261016/)
  * bmi_window.csv: the gap_distribution section, the checkup-vs-nursing agreement rows for checkups 1-365 d and
    > 365 d away, and every performance subgroup or contrast by BMI gap (contemporaneous, > 365 d, > 5 y, same-day
    checkup subgroup) and the case-mix column pct_gap_gt365: BMI is now taken only from the urine-test day.
  * missing_table.csv: the reference row of the submitted row-level frame (it concerned only the original models), the
    'one record per person' attrition step (the frame is one row per person) and the 'records, n' comparison row
    (records = persons); leukocyte esterase (not a model input) is no longer listed, compared or used as a predictor
    in (a), (b), (c) or (e).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import functools, time  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import BayesianRidge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from cohort import (HERE, TARGETS, F9, FMODEL, ID, DATE, AGE_MAX, ADULT_AGE, DATA_PATH, labels,  # noqa: E402
                    load_model, main_cohort, cohort_steps, persons_v2, age_exclusion_counts)
from common import SEED, auroc_ci, delong_var, delong_unpaired, calib_stats  # noqa: E402

T0 = time.time()
N_BOOT = 1000
M_IMP = 20
N_TEST, N_EXT = 25907, 49987                     # sanity checks of the task (internal test persons, KNHANES adults)
LOG = []
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def path(name):
    return os.path.join(HERE, name)


def cap_threads(model):
    """Limit every booster inside the calibrated soft-voting ensemble to 4 threads."""
    for cc in model.calibrated_classifiers_:
        for est in cc.estimator.estimators_:
            name = type(est).__name__
            if name.startswith("LGBM"):
                est.set_params(n_jobs=4)
            elif name.startswith("XGB"):
                est.set_params(n_jobs=4)
                est.get_booster().set_param({"nthread": 4})
            elif name.startswith("CatBoost"):
                est.predict_proba = functools.partial(type(est).predict_proba, est, thread_count=4)
    return model


def smd_mean(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def smd_prop(p1, p2):
    d = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / d if d > 0 else np.nan


def oe_ratio(y, p):
    return float(np.mean(y) / np.mean(p))


# =============================================================================== data
P_ALL = persons_v2()                                    # one row per window person (all ages), v2 fields
# age_at_urine_test keeps the implausible ages (> AGE_MAX) that persons_v2 sets to missing in 'age'
AX = age_exclusion_counts(P_ALL, col="age_at_urine_test")
P = P_ALL[P_ALL["adult"]].copy()                        # adult window persons; keeps the window index labels
assert len(P) == AX["adult"] and P[ID].is_unique and P.index.is_unique
log(f"adult restriction (age {ADULT_AGE}-{AGE_MAX} y at the urine test, adult target population): window persons "
    f"{AX['rows']:,} -> {AX['adult']:,}; excluded: aged < {ADULT_AGE} {AX['minor']:,}, age > {AGE_MAX} "
    f"{AX['implausible']:,}, age not determinable {AX['missing']:,}")
P["age_c"] = P["age"]
P["date"] = pd.to_datetime(P[DATE])
P["year"] = P["date"].dt.year
has_bmi = P["has_bmi"]
urm = P["has_urate"]                                    # urate of the urine-test day, incl. '<1.0' (kept as 0.5)
dip_ok = P[F9].notna().all(axis=1)
# same-day checkup BMI exactly as cohort.write_cache_v2 defines it (health checkup on the urine-test day)
ck_day = pd.to_datetime(P["수진일자"].astype(str), errors="coerce", format="mixed").dt.normalize()
same_ck = ck_day == P["date"].dt.normalize()
demo_any = pd.to_numeric(P["BMI_demo"], errors="coerce")
demo_same = demo_any.where(same_ck)
calc_ok = P["bmi_calc"].notna()
assert ((P["bmi_v2_source"] == "checkup") == demo_same.notna()).all()
assert ((P["bmi_v2_source"] == "nursing") == (demo_same.isna() & calc_ok)).all()
P["bmi_src"] = np.select([has_bmi & (P["bmi_v2_source"] == "checkup"), has_bmi & (P["bmi_v2_source"] == "nursing")],
                         ["checkup", "nursing"], "missing")
other_ck = demo_any.notna() & ~same_ck                  # checkup BMI of another day (not used)
ck_gap = (ck_day - P["date"].dt.normalize()).dt.days
N_OTHER_LATER = int((other_ck & (ck_gap > 0)).sum())
N = len(P)
log(f"adult window persons {N:,} (one first-visit urine test each); checkup BMI of another day (not used) "
    f"{int(other_ck.sum()):,}, of which later than the urine test {N_OTHER_LATER:,}")

# sex is coded male = (성별 == 'M'), which would hide a missing sex; check the source column directly
# (processed data file read directly: .loc[P.index] keeps only the rows of the adult window persons)
raw = pd.read_csv(DATA_PATH, usecols=[ID, "성별"], low_memory=False).loc[P.index]
assert (raw[ID].values == P[ID].values).all()
sex_missing = int((~raw["성별"].isin(["M", "F"])).sum())
log(f"source sex values of the adult window persons: {raw['성별'].value_counts(dropna=False).to_dict()} -> "
    f"missing/other {sex_missing}")
del raw

# retrained models and the internal test persons (must equal preds_internal_clean.csv and main_performance.csv)
d, tr, te, ext = main_cohort()
dte = d.iloc[te]
assert len(dte) == N_TEST and len(ext) == N_EXT
assert (P.loc[dte.index, ID].to_numpy() == dte[ID].to_numpy()).all(), "internal test persons not in the adult frame"
assert P.loc[d.index, "in_cohort"].all() and int(P["in_cohort"].sum()) == len(d)
ref = pd.read_csv(path("preds_internal_clean.csv"))
MP = pd.read_csv(path("main_performance.csv"))
MODELS, INTERNAL, AUC_MAIN = {}, {}, {}
for t, cfg in TARGETS.items():
    m = cap_threads(load_model(t))
    assert list(m.feature_names_in_) == FMODEL, "retrained model inputs differ from FMODEL"
    MODELS[t] = m
    y = labels(dte["urate"], cfg["threshold"])
    p = m.predict_proba(dte[FMODEL])[:, 1]
    r = ref[ref["target"] == t]
    assert len(r) == len(dte) and (r["y"].values == y).all() and (r["id"].values == dte[ID].values).all()
    assert np.allclose(r["p"].values, p, rtol=0, atol=1e-10), "internal predictions differ from preds_internal_clean"
    AUC_MAIN[t] = float(MP.loc[(MP["target"] == cfg["short"]) & (MP["cohort"] == "internal test"), "auroc"].iloc[0])
    a_chk = roc_auc_score(y, p)
    assert f"{a_chk:.4f}" == f"{AUC_MAIN[t]:.4f}", (t, a_chk, AUC_MAIN[t])
    INTERNAL[t] = pd.DataFrame({"row": dte.index.values, "y": y, "p": p})
    log(f"{cfg['short']}: internal test persons n={len(y):,}, positives={int(y.sum()):,}, AUROC {a_chk:.4f} "
        f"(matches preds_internal_clean.csv and main_performance.csv)")


def check_auc(t, a):
    """AUROC of the retrained model on the full internal test set must equal main_performance.csv to 4 decimals."""
    assert f"{a:.4f}" == f"{AUC_MAIN[t]:.4f}", (t, a, AUC_MAIN[t])


# =============================================================================== (a) missing table
log("\n(a) MISSING DATA")
A = []
n_below = int((P["urate_day_below"] & urm).sum())
nit_na = P["ua_nitrite"].isna()
NIT_NOTE = ("parsed from the raw first-visit text ('Positive'/'POSITIVE' = positive); "
            + ("every person still missing has none of the 9 dipstick fields recorded"
               if (nit_na <= P[F9].isna().all(axis=1)).all() else
               f"{int((nit_na & ~P[F9].isna().all(axis=1)).sum())} persons still missing have other dipstick fields"))
notes = {"age": f"age at the urine test; adults aged {ADULT_AGE}-{AGE_MAX} y (adult target population): "
                f"{AX['minor']:,} persons aged < {ADULT_AGE} y and {AX['implausible']} persons with age > {AGE_MAX} y "
                f"(implausible) were excluded before this table",
         "male": f"source sex column is F/M for every adult window person ({sex_missing} missing)",
         "bmi_unified": "BMI of the urine-test day: checkup BMI when the health checkup was on that day, otherwise "
                        "weight/height^2 from the nursing record of that day; values outside 10-70 set to missing",
         "ua_nitrite": NIT_NOTE,
         "urate": f"serum urate not measured on the urine-test day; the {n_below} '<1.0' results count as measured"}
for v in FMODEL + ["urate"]:
    na = P[v].isna()
    A.append(dict(section="missing", item=v, n_total=N, n_missing=int(na.sum()), pct_missing=100 * na.mean(),
                  n_missing_urate_measured=int((na & urm).sum()),
                  pct_missing_urate_measured=100 * (na & urm).sum() / urm.sum(), note=notes.get(v, "dipstick field")))
extra = [
    ("age implausible (> 110 y)", P["age_at_urine_test"] > AGE_MAX,
     f"excluded with the persons aged < {ADULT_AGE} y by the adult restriction before this table "
     f"({AX['implausible']} persons among all window persons)"),
    ("urate reported as '<1.0'", P["urate_day_below"] & urm,
     "counted as measured, as 0.5 mg/dL (below every threshold)"),
    ("BMI: no checkup BMI of the urine-test day and no nursing-record height/weight", demo_same.isna() & ~calc_ok, ""),
    ("BMI: checkup BMI only from another day (not used) and no nursing-record height/weight", other_ck & ~calc_ok,
     f"included in the row above; every checkup BMI of another day ({int(other_ck.sum()):,} persons) is from a checkup "
     f"later than the urine test ({N_OTHER_LATER:,}); the submitted pipeline linked it by nearest date"),
    ("BMI: same-day checkup BMI outside 10-70", demo_same.notna() & ~has_bmi,
     "no fallback to the nursing-record height/weight (plausibility rule unchanged)"),
    ("BMI: nursing-record BMI outside 10-70 (no same-day checkup BMI)", demo_same.isna() & calc_ok & ~has_bmi, ""),
    ("BMI available from the same-day checkup record", P["bmi_src"] == "checkup",
     "count of persons WITH BMI from this source"),
    ("BMI available from the nursing record (urine-test day)", P["bmi_src"] == "nursing",
     "count of persons WITH BMI from this source"),
    ("any of the 12 inputs missing", P[FMODEL].isna().any(axis=1), ""),
    ("any of the 12 inputs or urate missing", P[FMODEL + ["urate"]].isna().any(axis=1),
     "complement = complete-case persons = analytic cohort"),
]
assert N_OTHER_LATER == int(other_ck.sum()), "a checkup BMI of another day is not later than the urine test"
for item, mask, note in extra:
    A.append(dict(section="missing", item=item, n_total=N, n_missing=int(mask.sum()), pct_missing=100 * mask.mean(),
                  n_missing_urate_measured=int((mask & urm).sum()),
                  pct_missing_urate_measured=100 * (mask & urm).sum() / urm.sum(), note=note))
assert N - int(P[FMODEL + ["urate"]].isna().any(axis=1).sum()) == len(d)
for r in A:
    log(f"  {r['item']:<85s} {r['n_missing']:>8,} ({r['pct_missing']:5.2f}%)   among urate measured "
        f"{r['n_missing_urate_measured']:>8,} ({r['pct_missing_urate_measured']:5.2f}%)")

# attrition: the dev_v2 exclusion steps among the adult window persons (records = persons)
steps = [(f"adults aged {ADULT_AGE}-{AGE_MAX} y with a first-visit urine test and a blood test within 14 days "
          "(adult window persons)", pd.Series(True, index=P.index))]
steps.append(("serum urate measured on the urine-test day (incl. '<1.0')", steps[-1][1] & urm))
steps.append(("BMI recorded on the urine-test day", steps[-1][1] & has_bmi))
steps.append(("all 9 dipstick fields and sex recorded = analytic cohort (one record per person)",
              steps[-1][1] & dip_ok & P["male"].notna()))
CS = cohort_steps()
assert [int(m_.sum()) for _, m_ in steps] == [n for _, n in CS[1:]], ([int(m_.sum()) for _, m_ in steps], CS)
assert (steps[-1][1] == P["in_cohort"]).all()
prev = None
for k, (name, mask) in enumerate(steps):
    n_rows = int(mask.sum())
    A.append(dict(section="attrition", item=f"{k}. {name}", n_rows=n_rows, n_persons=int(P.loc[mask, ID].nunique()),
                  n_excluded=(prev - n_rows) if prev is not None else np.nan))
    prev = n_rows
for r in A:
    if r["section"] == "attrition":
        log(f"  {r['item']:<130s} persons {r['n_persons']:>8,}  excluded "
            f"{'' if pd.isna(r['n_excluded']) else format(int(r['n_excluded']), ',')}")
log(f"  (steps equal cohort.cohort_steps(); before the adult restriction: {CS[0][1]:,} window persons)")

# included vs excluded because same-day BMI missing: persons complete on everything except BMI
base = urm & dip_ok & P["age_c"].notna()
inc, exc = P[base & has_bmi], P[base & ~has_bmi]
assert len(inc) == len(d)
SEMI = ["ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen"]


def any_abnormal(f):
    return ((f[SEMI] >= 1).any(axis=1) | (f["ua_nitrite"] >= 1)).astype(float)


cmp_rows = [("persons, n", inc[ID].nunique(), exc[ID].nunique(), np.nan, ""),
            ("age, mean", inc["age_c"].mean(), exc["age_c"].mean(), smd_mean(inc["age_c"], exc["age_c"]), "SMD"),
            ("age, SD", inc["age_c"].std(), exc["age_c"].std(), np.nan, ""),
            ("age, median", inc["age_c"].median(), exc["age_c"].median(), np.nan, "")]
for lab, fn in (("age >= 65 y, %", lambda f: (f["age_c"] >= 65).astype(float)),
                ("male, %", lambda f: f["male"].astype(float))):
    a, b = fn(inc).mean(), fn(exc).mean()
    cmp_rows.append((lab, 100 * a, 100 * b, smd_prop(a, b), "SMD"))
for c in SEMI:
    a, b = (inc[c] >= 1).mean(), (exc[c] >= 1).mean()
    cmp_rows.append((f"{c} >= 1+, %", 100 * a, 100 * b, smd_prop(a, b), "trace (0.5) counted negative; "
                     "urobilinogen 0.5 is the normal grade" if c == "ua_urobilinogen" else "trace (0.5) counted negative"))
a, b = (inc["ua_nitrite"] >= 1).mean(), (exc["ua_nitrite"] >= 1).mean()
cmp_rows.append(("ua_nitrite positive, %", 100 * a, 100 * b, smd_prop(a, b), ""))
a, b = any_abnormal(inc).mean(), any_abnormal(exc).mean()
cmp_rows.append(("any dipstick field >= 1+ or nitrite positive, %", 100 * a, 100 * b, smd_prop(a, b),
                 "the 6 semi-quantitative fields and nitrite of the model inputs"))
for c in ("ua_pH", "ua_sg"):
    cmp_rows.append((f"{c}, mean", inc[c].mean(), exc[c].mean(), smd_mean(inc[c], exc[c]), "SMD"))
cmp_rows.append(("serum urate, mean mg/dL", inc["urate"].mean(), exc["urate"].mean(),
                 smd_mean(inc["urate"], exc["urate"]), "SMD"))
for t, cfg in TARGETS.items():
    a, b = labels(inc["urate"], cfg["threshold"]).mean(), labels(exc["urate"], cfg["threshold"]).mean()
    cmp_rows.append((f"urate {cfg['short']} mg/dL, %", 100 * a, 100 * b, smd_prop(a, b), "SMD"))
cmp_rows.append(("calendar year, median", inc["year"].median(), exc["year"].median(),
                 smd_mean(inc["year"], exc["year"]), "SMD of mean year"))
# broader population ignoring urate: how often same-day urate was measured, by same-day BMI availability
pop2 = dip_ok & P["age_c"].notna()
a, b = urm[pop2 & has_bmi].mean(), urm[pop2 & ~has_bmi].mean()
cmp_rows.append(("serum urate measured, % (persons with complete dipstick/age/sex, urate ignored)",
                 100 * a, 100 * b, smd_prop(a, b), f"denominators {int((pop2 & has_bmi).sum()):,} vs "
                                                   f"{int((pop2 & ~has_bmi).sum()):,}"))
for lab, a, b, s, note in cmp_rows:
    A.append(dict(section="bmi_missing_comparison", item=lab, included_bmi_available=a, excluded_bmi_missing=b,
                  smd=s, note=note))
log("  included (same-day BMI available) vs excluded (same-day BMI missing), persons complete on everything else:")
for lab, a, b, s, _ in cmp_rows:
    log(f"    {lab:<85s} {a:>12.3f} {b:>12.3f}   SMD {s: .3f}")
pd.DataFrame(A).to_csv(path("missing_table.csv"), index=False, encoding="utf-8")


# =============================================================================== (b) Little's MCAR test
def em_mvn(X, tol=1e-8, max_iter=5000):
    """ML mean/covariance of a multivariate normal with missing values (EM), grouped by missingness pattern."""
    n, p = X.shape
    miss = np.isnan(X)
    pats, inv = np.unique(miss, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    groups = [(pats[k], X[inv == k]) for k in range(len(pats))]
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0))
    for it in range(1, max_iter + 1):
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))
        for pat, R in groups:
            o, mi, nk = ~pat, pat, len(R)
            if not mi.any():
                T1 += R.sum(0)
                T2 += R.T @ R
                continue
            F = R.copy()
            if o.any():
                B = S[np.ix_(mi, o)] @ np.linalg.pinv(S[np.ix_(o, o)])
                F[:, mi] = mu[mi] + (R[:, o] - mu[o]) @ B.T
                C = S[np.ix_(mi, mi)] - B @ S[np.ix_(o, mi)]
            else:
                F[:] = mu
                C = S.copy()
            T1 += F.sum(0)
            T2 += F.T @ F
            T2[np.ix_(mi, mi)] += nk * C
        mu_new = T1 / n
        S_new = T2 / n - np.outer(mu_new, mu_new)
        delta = max(np.abs(mu_new - mu).max(), np.abs(S_new - S).max())
        mu, S = mu_new, S_new
        if delta < tol:
            break
    return mu, S, it, groups


def log10_chi2_sf(x, df):
    """log10 of the chi-square upper tail, exact in arbitrary precision (scipy underflows below ~1e-308)."""
    import mpmath
    return float(mpmath.log10(mpmath.gammainc(mpmath.mpf(df) / 2, mpmath.mpf(x) / 2, mpmath.inf, regularized=True)))


def little_mcar(X, names, return_patterns=False):
    """Little (1988) chi-square test of MCAR. Variables are standardised (the test is affine invariant);
    variables with no variation among observed values are dropped and reported. With return_patterns=True the
    per-pattern contributions to the chi-square statistic are returned as well."""
    X = np.asarray(X, float)
    sd = np.nanstd(X, axis=0)
    ok = (np.sum(~np.isnan(X), axis=0) >= 2) & (sd > 0)
    dropped = [nm for nm, k in zip(names, ok) if not k]
    kept = np.array([nm for nm, k in zip(names, ok) if k])
    X = X[:, ok]
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    X = X[~np.isnan(X).all(axis=1)]
    mu, S, n_iter, groups = em_mvn(X)
    d2, df, contrib = 0.0, 0, []
    for pat, R in groups:
        o = ~pat
        if not o.any():
            continue
        diff = R[:, o].mean(0) - mu[o]
        c = len(R) * float(diff @ np.linalg.pinv(S[np.ix_(o, o)]) @ diff)
        d2 += c
        df += int(o.sum())
        contrib.append(dict(pattern="complete" if not pat.any() else "missing: " + "+".join(kept[pat]),
                            count=len(R), n_observed_vars=int(o.sum()), chi2_contribution=c))
    df -= X.shape[1]
    out = dict(n=len(X), n_vars=int(X.shape[1]), n_patterns=len(groups), chi2=d2, df=df,
               p=float(stats.chi2.sf(d2, df)), log10_p=log10_chi2_sf(d2, df),
               chi2_over_df=d2 / df if df > 0 else np.nan, em_iterations=n_iter, dropped_vars=";".join(dropped))
    return (out, contrib) if return_patterns else out


log("\n(b) LITTLE'S MCAR TEST")
MCAR_VARS = ["age_c", "male"] + F9 + ["bmi_unified", "urate"]
MCAR_NAMES = ["age", "male"] + F9 + ["bmi_unified", "urate"]
Xm = P[MCAR_VARS].to_numpy(float)
NO_NIT = [k for k, nm in enumerate(MCAR_NAMES) if nm != "ua_nitrite"]
_nit = Xm[:, MCAR_NAMES.index("ua_nitrite")]                        # counts for the note
N_NIT_POS = int(np.nansum(_nit >= 1))
Z_NIT_POS = float((1 - np.nanmean(_nit)) / np.nanstd(_nit))         # the standardisation used inside little_mcar
B_rows = []
t1 = time.time()
res, contrib = little_mcar(Xm, MCAR_NAMES, return_patterns=True)
B_rows.append(dict(section="little_full", sample="all adult window persons", note="", **res))   # note filled below
log(f"  all adult window persons: n={res['n']:,} patterns={res['n_patterns']} chi2={res['chi2']:.1f} df={res['df']} "
    f"p={res['p']:.3g} (log10 p={res['log10_p']:.1f}); EM iterations {res['em_iterations']} ({time.time() - t1:.0f}s)")
contrib = sorted(contrib, key=lambda c: -c["chi2_contribution"])
for c in contrib:
    B_rows.append(dict(section="little_full_pattern_contribution", sample="all adult window persons",
                       pattern=c["pattern"], count=c["count"], chi2_contribution=c["chi2_contribution"],
                       pct_of_chi2=100 * c["chi2_contribution"] / res["chi2"], df_contribution=c["n_observed_vars"]))
nit_share = sum(c["chi2_contribution"] for c in contrib if "ua_nitrite" in c["pattern"]) / res["chi2"]
log(f"  chi2 contributions: top pattern [{contrib[0]['pattern']}] n={contrib[0]['count']:,} "
    f"{contrib[0]['chi2_contribution']:.0f} ({100 * contrib[0]['chi2_contribution'] / res['chi2']:.1f}%); patterns "
    f"involving missing ua_nitrite {100 * nit_share:.1f}%")
# full-frame sensitivity runs: without ua_nitrite, without rare patterns (n < 30), both
pat_key = pd.Series([tuple(r) for r in np.isnan(Xm)])
pat_n = pat_key.map(pat_key.value_counts()).to_numpy()
pat_key_nn = pd.Series([tuple(r) for r in np.isnan(Xm[:, NO_NIT])])
pat_n_nn = pat_key_nn.map(pat_key_nn.value_counts()).to_numpy()
SENS = []
for lab, Xs, nms in (("all adult window persons, ua_nitrite dropped", Xm[:, NO_NIT], [MCAR_NAMES[k] for k in NO_NIT]),
                     ("all adult window persons, patterns with n >= 30 only", Xm[pat_n >= 30], MCAR_NAMES),
                     ("all adult window persons, ua_nitrite dropped and patterns with n >= 30 only",
                      Xm[pat_n_nn >= 30][:, NO_NIT], [MCAR_NAMES[k] for k in NO_NIT])):
    r = little_mcar(Xs, nms)
    SENS.append(r)
    B_rows.append(dict(section="little_full_sensitivity", sample=lab, **r))
    log(f"  sensitivity [{lab}]: n={r['n']:,} patterns={r['n_patterns']} chi2={r['chi2']:.1f} "
        f"df={r['df']} chi2/df={r['chi2_over_df']:.1f} log10 p={r['log10_p']:.1f}")
rng = np.random.default_rng(SEED)
series, SUB_IDX = {}, {}
for size in (5000, 1000, 20000, 100000):          # 5,000 = the pre-specified subsample size; others show the trend
    ps = []
    SUB_IDX[size] = []
    for rep in range(10):
        idx = rng.choice(N, size, replace=False)
        SUB_IDX[size].append(idx)
        r = little_mcar(Xm[idx], MCAR_NAMES)
        ps.append(r["p"])
        B_rows.append(dict(section="little_subsample" if size == 5000 else "little_size_series",
                           sample=f"random subsample {rep + 1} of {size:,}", **r))
    series[size] = ps
    sub = [b for b in B_rows if b["sample"].endswith(f"of {size:,}")]
    B_rows.append(dict(section="little_subsample_summary" if size == 5000 else "little_size_series_summary",
                       sample=f"10 subsamples of {size:,}", n=size, p=float(np.median(ps)),
                       chi2=float(np.median([b["chi2"] for b in sub])), df=float(np.median([b["df"] for b in sub])),
                       chi2_over_df=float(np.median([b["chi2_over_df"] for b in sub])),
                       n_rejected_at_005=int(np.sum(np.array(ps) < 0.05)),
                       note="p = median p; chi2, df, chi2_over_df = medians"))
    log(f"  10 subsamples of {size:>7,}: median p={np.median(ps):.3g}; rejected at 0.05 in "
        f"{int(np.sum(np.array(ps) < 0.05))}/10; median chi2/df={np.median([b['chi2_over_df'] for b in sub]):.2f}")
# the same subsamples of 5,000 and 1,000 with ua_nitrite dropped (rejection not driven by nitrite)
REJ_NN = {}
for size in (5000, 1000):
    rs = [little_mcar(Xm[idx][:, NO_NIT], [MCAR_NAMES[k] for k in NO_NIT]) for idx in SUB_IDX[size]]
    for rep, r in enumerate(rs):
        B_rows.append(dict(section="little_subsample_no_nitrite",
                           sample=f"random subsample {rep + 1} of {size:,}, ua_nitrite dropped", **r))
    ps = np.array([r["p"] for r in rs])
    REJ_NN[size] = int(np.sum(ps < 0.05))
    n_auto = sum("ua_nitrite" in str(b.get("dropped_vars", "")) for b in B_rows
                 if b["section"] in ("little_subsample", "little_size_series") and b["sample"].endswith(f"of {size:,}"))
    B_rows.append(dict(section="little_subsample_no_nitrite_summary",
                       sample=f"10 subsamples of {size:,}, ua_nitrite dropped (same subsamples as above)", n=size,
                       p=float(np.median(ps)), chi2=float(np.median([r["chi2"] for r in rs])),
                       df=float(np.median([r["df"] for r in rs])),
                       chi2_over_df=float(np.median([r["chi2_over_df"] for r in rs])),
                       n_rejected_at_005=int(np.sum(ps < 0.05)),
                       note="p = median p; chi2, df, chi2_over_df = medians. "
                            + (f"In {n_auto}/10 of these subsamples ua_nitrite had no positive result and was already "
                               "dropped automatically in the main run (zero variance), so those subsamples give "
                               "identical results with and without it" if n_auto else
                               "ua_nitrite had positive results in every subsample, so every main run included it")))
    log(f"  10 subsamples of {size:>7,}, ua_nitrite dropped: median p={np.median(ps):.3g}; rejected at 0.05 in "
        f"{int(np.sum(ps < 0.05))}/10; median chi2/df={np.median([r['chi2_over_df'] for r in rs]):.2f} "
        f"(ua_nitrite auto-dropped, no positives, in {n_auto}/10 in the main run)")
# caveat note of the full-frame test, written from the results above
REJ = {s: int(np.sum(np.array(series[s]) < 0.05)) for s in (5000, 1000)}
top_share = 100 * contrib[0]["chi2_contribution"] / res["chi2"]
sens_all = all(r["p"] < 0.05 for r in SENS)
MVN_NOTE = ("Little's test assumes multivariate normality, which is violated here (male is binary, the dipstick "
            f"fields are ordinal, ua_nitrite is binary with {N_NIT_POS:,} positives among the adult window persons, "
            f"standardised value about {Z_NIT_POS:.0f}), and with n = {res['n']:,} even a small departure is "
            "detected, so the chi-square magnitude is not a measure of the size of the departure from MCAR; the "
            f"largest share of the chi-square, {top_share:.0f}%, comes from the pattern [{contrib[0]['pattern']}] "
            "(see little_full_pattern_contribution); only the rejection is interpreted, and it "
            + ("survives" if sens_all else "does not survive every run of")
            + f" dropping ua_nitrite and rare patterns and holds in {REJ[5000]}/10 random subsamples of 5,000 and "
              f"{REJ[1000]}/10 of 1,000 ({REJ_NN[5000]}/10 and {REJ_NN[1000]}/10 with ua_nitrite dropped)")
B_rows[0]["note"] = MVN_NOTE
log("  CAVEAT: " + MVN_NOTE)
# missingness patterns over the 13 variables
upats, ucnts = np.unique(P[MCAR_VARS].isna().to_numpy(), axis=0, return_counts=True)
pc = pd.Series(ucnts, index=["complete" if not u.any() else "missing: " + "+".join(np.array(MCAR_NAMES)[u])
                             for u in upats]).sort_values(ascending=False, kind="mergesort")
for lab, cnt in pc.items():
    B_rows.append(dict(section="patterns", sample="all adult window persons", pattern=lab, count=int(cnt),
                       pct=100 * cnt / N))
log(f"  {len(pc)} missingness patterns; top:")
for lab, cnt in pc.head(8).items():
    log(f"    {lab:<60s} {cnt:>8,} ({100 * cnt / N:5.2f}%)")
pd.DataFrame(B_rows).to_csv(path("mcar.csv"), index=False, encoding="utf-8")


# =============================================================================== (c) multiple imputation of BMI
def rubin(q, u, n_obs):
    """Rubin's rules with the Barnard-Rubin small-sample degrees of freedom."""
    q, u = np.asarray(q, float), np.asarray(u, float)
    m = len(q)
    qbar, W, Bv = q.mean(), u.mean(), q.var(ddof=1)
    T = W + (1 + 1 / m) * Bv
    lam = (1 + 1 / m) * Bv / T
    riv = (1 + 1 / m) * Bv / W
    nu_old = (m - 1) / lam ** 2 if lam > 0 else np.inf
    nu_com = n_obs - 1
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    nu = 1 / (1 / nu_old + 1 / nu_obs)
    fmi = (riv + 2 / (nu + 3)) / (riv + 1)
    tq = stats.t.ppf(0.975, nu)
    return dict(qbar=qbar, W=W, B=Bv, T=T, df=nu, riv=riv, fmi=fmi,
                lo=qbar - tq * np.sqrt(T), hi=qbar + tq * np.sqrt(T))


def lg(x):
    return np.log(x / (1 - x))


def ex(z):
    return 1 / (1 + np.exp(-z))


log("\n(c) MULTIPLE IMPUTATION OF BMI (adults with a same-day urate but no BMI of that day; never used)")
elig = dip_ok & P["age_c"].notna() & urm                             # complete except (possibly) BMI
cand = elig & ~has_bmi
in_dev = P[ID].isin(set(d[ID]))
assert not (cand & in_dev).any(), "a person without same-day BMI is in the analytic cohort"
ev = P[cand].copy()                                                  # one row per person
assert ev[ID].is_unique
log(f"  adults complete on dipstick/age/sex with a same-day urate: {int(elig.sum()):,}; of these same-day BMI missing "
    f"(evaluation set, one record per person, none in the analytic cohort): {len(ev):,}")
IMP_MODELS = {"full": ["age_c", "male"] + F9 + ["urate"], "age_sex": ["age_c", "male"]}
IMP_LABEL = {"full": "BMI imputed from age, sex, 9 dipstick fields and serum urate",
             "age_sex": "BMI imputed from age and sex only (sensitivity)"}
C_rows, POOL = [], {}
Y_EV = {t: labels(ev["urate"], cfg["threshold"]) for t, cfg in TARGETS.items()}
bmi_obs = P.loc[elig & has_bmi, "bmi_unified"]
for key, preds in IMP_MODELS.items():
    cols = preds + ["bmi_unified"]
    Xi_all = P.loc[elig, cols].to_numpy(float)
    # Standardise the predictor columns. sklearn's BayesianRidge.predict(return_std=True) evaluates x' Sigma x on
    # uncentred X while Sigma was estimated on centred X; with unscaled predictors (ua_sg mean 1.02, SD 0.007) this
    # inflates every posterior draw. Standardising is an affine change that leaves the fitted regression unchanged.
    # The BMI column (the one imputed) and its 10-70 bounds are left on the original scale.
    x_mu, x_sd = Xi_all[:, :-1].mean(0), Xi_all[:, :-1].std(0)
    x_sd[x_sd == 0] = 1.0
    Xi_all[:, :-1] = (Xi_all[:, :-1] - x_mu) / x_sd
    lo_v = np.full(len(cols), -np.inf)
    hi_v = np.full(len(cols), np.inf)
    lo_v[-1], hi_v[-1] = 10.0, 70.0                                   # same plausibility range as bmi_unified
    obs = ~np.isnan(Xi_all[:, -1])
    Z = np.column_stack([np.ones(obs.sum()), Xi_all[obs, :-1]])
    beta = np.linalg.lstsq(Z, Xi_all[obs, -1], rcond=None)[0]
    resid_var = np.var(Xi_all[obs, -1] - Z @ beta)
    r2 = 1 - resid_var / np.var(Xi_all[obs, -1])
    pos = P.index[elig].get_indexer(ev.index)
    assert (pos >= 0).all()
    # the SD the imputed BMI should have among the evaluation persons: sqrt(var(fitted) + residual variance)
    fit_ev = np.column_stack([np.ones(len(pos)), Xi_all[pos, :-1]]) @ beta
    exp_sd = float(np.sqrt(np.var(fit_ev, ddof=1) + resid_var))
    per = {t: [] for t in TARGETS}
    imp_means, imp_sds, pred_sds = [], [], []
    t1 = time.time()
    for k in range(M_IMP):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, random_state=k, max_iter=10,
                               skip_complete=True, min_value=lo_v, max_value=hi_v)
        bmi_k = imp.fit_transform(Xi_all)[pos, -1]
        # predictive SD used for the draws (must equal the residual SD up to negligible parameter uncertainty)
        trip = [tr_ for tr_ in imp.imputation_sequence_ if tr_.feat_idx == len(cols) - 1][-1]
        _, sd_k = trip.estimator.predict(Xi_all[pos][:, trip.neighbor_feat_idx], return_std=True)
        pred_sds.append(float(sd_k.mean()))
        imp_means.append(bmi_k.mean())
        imp_sds.append(bmi_k.std(ddof=1))
        Xev = ev[FMODEL].copy()
        Xev["bmi_unified"] = bmi_k
        for t, cfg in TARGETS.items():
            y = Y_EV[t]
            p = MODELS[t].predict_proba(Xev[FMODEL])[:, 1]
            auc, var = delong_var(y, p)
            ci_, cs_, oe_ = calib_stats(y, p)
            per[t].append(dict(auc=auc, var=var, oe=oe_, cal_int=ci_, cal_slope=cs_, mean_p=p.mean()))
            C_rows.append(dict(section="per_imputation", imputation_model=key, target=cfg["short"], imputation=k + 1,
                               n=len(y), positives=int(y.sum()), auroc=auc, auroc_se_delong=np.sqrt(var),
                               oe=oe_, cal_intercept=ci_, cal_slope=cs_, mean_pred=p.mean(),
                               imputed_bmi_mean=bmi_k.mean(), imputed_bmi_sd=bmi_k.std(ddof=1),
                               imputer_predictive_sd=pred_sds[-1]))
    log(f"  [{key}] imputation R^2 (linear, observed BMI) = {r2:.3f}; imputed BMI mean {np.mean(imp_means):.2f} "
        f"(SD {np.mean(imp_sds):.3f}; expected sqrt(var(fitted)+residual var) = {exp_sd:.3f}) vs observed "
        f"{bmi_obs.mean():.2f} (SD {bmi_obs.std():.2f}); imputer predictive SD {np.mean(pred_sds):.3f} vs residual "
        f"SD {np.sqrt(resid_var):.3f}; {M_IMP} imputations in {time.time() - t1:.0f}s")
    # the draws must carry exactly the residual noise (the uncentred-X quirk made them noisier)
    assert abs(np.mean(pred_sds) / np.sqrt(resid_var) - 1) < 0.01, \
        f"[{key}] imputer predictive SD {np.mean(pred_sds):.3f} != residual SD {np.sqrt(resid_var):.3f}"
    assert abs(np.mean(imp_sds) / exp_sd - 1) < 0.02, \
        f"[{key}] imputed BMI SD {np.mean(imp_sds):.3f} != expected {exp_sd:.3f}"
    C_rows.append(dict(section="imputation_diagnostics", imputation_model=key, note=IMP_LABEL[key],
                       n=len(ev), imputation_r2=r2, imputed_bmi_mean=float(np.mean(imp_means)),
                       imputed_bmi_sd=float(np.mean(imp_sds)), expected_imputed_bmi_sd=exp_sd,
                       imputer_predictive_sd=float(np.mean(pred_sds)), residual_sd=float(np.sqrt(resid_var)),
                       observed_bmi_mean=bmi_obs.mean(),
                       observed_bmi_sd=bmi_obs.std(), n_imputation_fit=int(elig.sum()), n_bmi_observed=int(obs.sum()),
                       method_note="predictors standardised before IterativeImputer (sklearn BayesianRidge "
                                   "return_std uses uncentred X); expected SD = sqrt(var(fitted among evaluation "
                                   "persons) + residual variance). The BayesianRidge fit is identical across the "
                                   "m imputations (predictors complete), so draws add residual noise plus per-row "
                                   "parameter noise rather than one coherent parameter draw per imputation; with "
                                   f"{int(obs.sum()):,} observed BMI values parameter uncertainty is negligible"))
    for t, cfg in TARGETS.items():
        dd = pd.DataFrame(per[t])
        y = Y_EV[t]
        # AUROC: Rubin on the logit scale, within-imputation variance from DeLong (delta method)
        q = lg(dd["auc"].values)
        u = dd["var"].values / (dd["auc"].values * (1 - dd["auc"].values)) ** 2
        ra = rubin(q, u, len(y))
        # O:E: Rubin on the log scale, within variance of log(observed rate) = (1 - ybar) / (n ybar)
        yb = y.mean()
        ro = rubin(np.log(dd["oe"].values), np.full(M_IMP, (1 - yb) / (len(y) * yb)), len(y))
        POOL[(key, t)] = ex(ra["qbar"])
        C_rows.append(dict(section="pooled", imputation_model=key, target=cfg["short"], note=IMP_LABEL[key],
                           n=len(y), positives=int(y.sum()), m=M_IMP,
                           auroc=ex(ra["qbar"]), auroc_lo=ex(ra["lo"]), auroc_hi=ex(ra["hi"]),
                           auroc_min=dd["auc"].min(), auroc_max=dd["auc"].max(), rubin_df=ra["df"], fmi=ra["fmi"],
                           riv=ra["riv"], oe=np.exp(ro["qbar"]), oe_lo=np.exp(ro["lo"]), oe_hi=np.exp(ro["hi"]),
                           oe_min=dd["oe"].min(), oe_max=dd["oe"].max(), cal_intercept=dd["cal_int"].mean(),
                           cal_slope=dd["cal_slope"].mean(), mean_pred=dd["mean_p"].mean(), prevalence=yb))
# complete-case internal reference (internal test persons, retrained models)
for t, cfg in TARGETS.items():
    di = INTERNAL[t]
    a, lo_, hi_ = auroc_ci(di["y"].values, di["p"].values, n=N_BOOT)
    check_auc(t, a)
    ci_, cs_, oe_ = calib_stats(di["y"].values, di["p"].values)
    # DeLong CI on the logit scale, the same construction as the pooled MI interval (like-for-like comparison)
    a_dl, v_dl = delong_var(di["y"].values, di["p"].values)
    se_l = np.sqrt(v_dl) / (a_dl * (1 - a_dl))
    C_rows.append(dict(section="complete_case_internal", imputation_model="none (complete case)", target=cfg["short"],
                       note="internal test persons; auroc_lo/hi = bootstrap 95% CI; auroc_lo_delong/"
                            "auroc_hi_delong = DeLong CI on the logit scale (same construction as the MI interval)",
                       n=len(di), positives=int(di["y"].sum()), auroc=a, auroc_lo=lo_, auroc_hi=hi_,
                       auroc_lo_delong=ex(lg(a_dl) - 1.96 * se_l), auroc_hi_delong=ex(lg(a_dl) + 1.96 * se_l), oe=oe_,
                       cal_intercept=ci_, cal_slope=cs_, mean_pred=di["p"].mean(), prevalence=di["y"].mean()))
cdf = pd.DataFrame(C_rows)
cdf.to_csv(path("mice_bmi.csv"), index=False, encoding="utf-8")
log("  pooled (Rubin) vs complete-case internal:")
show = cdf[cdf["section"].isin(["pooled", "complete_case_internal"])]
for _, r in show.iterrows():
    log(f"    {r['target']:>6s} {r['imputation_model']:<22s} n={int(r['n']):>7,} pos={int(r['positives']):>6,} "
        f"AUROC {r['auroc']:.4f} ({r['auroc_lo']:.4f}-{r['auroc_hi']:.4f})  O:E {r['oe']:.3f}"
        + (f" ({r['oe_lo']:.3f}-{r['oe_hi']:.3f}) FMI {r['fmi']:.3f}" if r["section"] == "pooled" else "")
        + (f"  [DeLong {r['auroc_lo_delong']:.4f}-{r['auroc_hi_delong']:.4f}]"
           if r["section"] == "complete_case_internal" else "")
        + f"  slope {r['cal_slope']:.3f}")


# =============================================================================== (d) BMI source (urine-test day)
log("\n(d) BMI OF THE URINE-TEST DAY: CHECKUP VS NURSING RECORD")
D_rows = []

# agreement of the same-day checkup BMI with the nursing-record BMI of the same day (persons with both)
both = demo_same.between(10, 70) & P["bmi_calc"].between(10, 70)
a_, b_ = demo_same[both], P.loc[both, "bmi_calc"]
d_ = a_ - b_
D_rows.append(dict(section="bmi_demo_vs_height_weight",
                   population="adult window persons with both a checkup BMI and a nursing-record BMI of the urine-test "
                              "day",
                   n=int(both.sum()), diff_median=d_.median(), diff_mean=d_.mean(), abs_diff_median=d_.abs().median(),
                   pct_abs_diff_le05=100 * (d_.abs() <= 0.5).mean(), pct_abs_diff_le1=100 * (d_.abs() <= 1).mean(),
                   pct_abs_diff_gt2=100 * (d_.abs() > 2).mean(),
                   pearson_r=np.corrcoef(a_, b_)[0, 1] if both.sum() > 2 else np.nan))
r = D_rows[-1]
log(f"  same-day checkup BMI vs nursing-record BMI: n={r['n']:>7,} median diff {r['diff_median']:+.3f}; "
    f"|diff|<=0.5 {r['pct_abs_diff_le05']:.1f}%; |diff|<=1 {r['pct_abs_diff_le1']:.1f}%; r={r['pearson_r']:.3f}")

# retrained-model performance by BMI source on the internal test persons
assert set(P.loc[dte.index, "bmi_src"]) <= {"checkup", "nursing"}
assert (P.loc[d.index, "bmi_src"].to_numpy() == d["bmi_v2_source"].to_numpy()).all()
SRC = {"checkup": "checkup record", "nursing": "nursing record"}     # both of the urine-test day by definition
GROUPS = [("all", lambda f: np.ones(len(f), bool)),
          (f"source {SRC['checkup']}", lambda f: f["bmi_src"] == "checkup"),
          (f"source {SRC['nursing']}", lambda f: f["bmi_src"] == "nursing")]
MIN_POS = 10                    # groups with fewer positives: point estimate only, CI and tests suppressed
FEW_NOTE = f"fewer than {MIN_POS} positives: CI and tests suppressed (uninformative)"


def perf_row(yy, pp, **kw):
    row = dict(kw, n=len(yy), positives=int(yy.sum()), prevalence=yy.mean() if len(yy) else np.nan)
    if 0 < yy.sum() < len(yy):
        row.update(auroc=roc_auc_score(yy, pp), oe=oe_ratio(yy, pp), mean_pred=pp.mean())
        if yy.sum() >= MIN_POS:
            _, lo_, hi_ = auroc_ci(yy, pp, n=N_BOOT)
            ci_, cs_, _ = calib_stats(yy, pp)
            row.update(auroc_lo=lo_, auroc_hi=hi_, cal_intercept=ci_, cal_slope=cs_)
    row["note"] = FEW_NOTE if yy.sum() < MIN_POS else ""
    return row


def diff_row(ya, pa, yb, pb, **kw):
    row = dict(kw, n_a=len(ya), positives_a=int(ya.sum()), n_b=len(yb), positives_b=int(yb.sum()))
    if min(ya.sum(), yb.sum()) >= MIN_POS:
        diff, lo_, hi_, pv = delong_unpaired(ya, pa, yb, pb)
        row.update(auroc_diff=diff, auroc_diff_lo=lo_, auroc_diff_hi=hi_, p_delong=pv, note="")
    else:
        row.update(auroc_diff=(roc_auc_score(ya, pa) - roc_auc_score(yb, pb)
                               if 0 < ya.sum() < len(ya) and 0 < yb.sum() < len(yb) else np.nan),
                   note=f"a group has fewer than {MIN_POS} positives: CI and test suppressed (uninformative)")
    return row


def fmt_ci(r, lo="auroc_lo", hi="auroc_hi", sign=""):
    return (f"({r[lo]:{sign}.4f} to {r[hi]:{sign}.4f})" if pd.notna(r.get(lo, np.nan)) else "(CI suppressed)")


for t, cfg in TARGETS.items():
    di = INTERNAL[t]
    f = P.loc[di["row"].values, ["bmi_src"]].reset_index(drop=True)
    y, p = di["y"].values, di["p"].values
    masks = {}
    for lab, fn in GROUPS:
        mk = np.asarray(fn(f), bool)
        masks[lab] = mk
        row = perf_row(y[mk], p[mk], section="performance", population=f"internal test: {lab}", target=cfg["short"])
        if lab == "all":
            check_auc(t, row["auroc"])
        D_rows.append(row)
        log(f"  {cfg['short']:>6s} {lab:<40s} n={row['n']:>6,} pos={row['positives']:>5,} AUROC "
            f"{row.get('auroc', np.nan):.4f} {fmt_ci(row)} O:E {row.get('oe', np.nan):.3f}")
    a_lab, b_lab = f"source {SRC['checkup']}", f"source {SRC['nursing']}"
    row = diff_row(y[masks[a_lab]], p[masks[a_lab]], y[masks[b_lab]], p[masks[b_lab]], section="performance_difference",
                   population=f"{a_lab}  minus  {b_lab}", target=cfg["short"])
    D_rows.append(row)
    log(f"  {cfg['short']:>6s} AUROC difference [{a_lab}] - [{b_lab}] = {row['auroc_diff']:+.4f} "
        f"{fmt_ci(row, 'auroc_diff_lo', 'auroc_diff_hi', '+')}, DeLong p="
        f"{row['p_delong'] if 'p_delong' in row else np.nan:.3g}  {row['note']}")

# BMI source difference within age bands: is the source gap explained by age mix?
AGE_STRATA = [("age 19-64 y", lambda a: (a >= 19) & (a < 65)), ("age >= 65 y", lambda a: a >= 65)]
for t, cfg in TARGETS.items():
    di = INTERNAL[t]
    f = P.loc[di["row"].values, ["bmi_src", "age_c"]].reset_index(drop=True)
    y, p = di["y"].values, di["p"].values
    for a_lab, a_fn in AGE_STRATA:
        am = np.asarray(a_fn(f["age_c"]), bool)
        msk, prow = {}, {}
        for src in ("checkup", "nursing"):
            mk = am & np.asarray(f["bmi_src"] == src, bool)
            msk[src] = mk
            row = perf_row(y[mk], p[mk], section="performance_by_source_age",
                           population=f"internal test, {a_lab}: source {SRC[src]}", target=cfg["short"])
            row["age_mean"] = f.loc[mk, "age_c"].mean()
            prow[src] = row
            D_rows.append(row)
        row = diff_row(y[msk["checkup"]], p[msk["checkup"]], y[msk["nursing"]], p[msk["nursing"]],
                       section="performance_difference_by_source_age",
                       population=f"{a_lab}: source {SRC['checkup']}  minus  source {SRC['nursing']}",
                       target=cfg["short"])
        D_rows.append(row)
        ra_, rb_ = prow["checkup"], prow["nursing"]
        log(f"  {cfg['short']:>6s} {a_lab:<12s} checkup {ra_.get('auroc', np.nan):.4f} (n={ra_['n']:,}, "
            f"pos={ra_['positives']}) vs nursing {rb_.get('auroc', np.nan):.4f} (n={rb_['n']:,}, "
            f"pos={rb_['positives']}): diff {row['auroc_diff']:+.4f} "
            f"{fmt_ci(row, 'auroc_diff_lo', 'auroc_diff_hi', '+')}, DeLong p={row.get('p_delong', np.nan):.3g}"
            f"  {row['note']}")
# case mix by BMI source (context for AUROC differences between sources)
for lab, fr in (("analytic cohort", P.loc[d.index]), ("internal test", P.loc[dte.index])):
    for src in ("checkup", "nursing"):
        g = fr[fr["bmi_src"] == src]
        row = dict(section="case_mix_by_source", population=f"{lab}: source {SRC[src]}", n=len(g),
                   age_mean=g["age_c"].mean(), pct_age_lt19=100 * (g["age_c"] < 19).mean(),
                   pct_age_ge65=100 * (g["age_c"] >= 65).mean(), pct_male=100 * g["male"].mean(),
                   bmi_mean=g["bmi_unified"].mean(), year_median=g["year"].median(),
                   note="case mix is context only for the BMI-source AUROC contrast (adults only); the contrasts within "
                        "age bands are in performance_by_source_age / performance_difference_by_source_age")
        for t, cfg in TARGETS.items():
            row[f"prevalence_{cfg['short']}"] = labels(g["urate"], cfg["threshold"]).mean()
        D_rows.append(row)
        log(f"  case mix {row['population']:<45s} n={row['n']:>7,} age {row['age_mean']:.1f} (>=65 y "
            f"{row['pct_age_ge65']:.1f}%) male {row['pct_male']:.1f}% BMI {row['bmi_mean']:.1f} "
            f"urate>=7.0 {100 * row['prevalence_>=7.0']:.1f}%")
pd.DataFrame(D_rows).to_csv(path("bmi_window.csv"), index=False, encoding="utf-8")


# =============================================================================== (e) verification bias
log("\n(e) VERIFICATION BIAS (same-day serum urate)")
vmask = dip_ok & P["age_c"].notna() & P["male"].notna()
V = P.loc[vmask]
measured = V["has_urate"].astype(int).values
# 10-year age bands; age 19 forms its own band (adults only), reference band 40-49
AGE_BINS = [18, 19, 29, 39, 49, 59, 69, 79, 200]
AGE_LABS = ["19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
assert V["age_c"].min() >= ADULT_AGE and V["age_c"].max() <= AGE_MAX
DIP_TERMS = {c: (V[c].astype(float), "per grade (0, trace=0.5, 1+, 2+, 3+, 4+)") for c in SEMI}
DIP_TERMS["ua_nitrite"] = (V["ua_nitrite"].astype(float), "positive vs negative")
DIP_TERMS["ua_pH"] = (V["ua_pH"].astype(float), "per 1.0 pH unit")
DIP_TERMS["ua_sg"] = ((V["ua_sg"].astype(float) - 1.0) * 100, "per 0.010 specific gravity")
assert sorted(DIP_TERMS) == sorted(F9)
Xdip = pd.DataFrame({k: v for k, (v, _) in DIP_TERMS.items()}, index=V.index)
age_band = pd.Categorical(pd.cut(V["age_c"], AGE_BINS, labels=AGE_LABS), categories=AGE_LABS)
Xage = pd.get_dummies(age_band, prefix="age", dtype=float).drop(columns="age_40-49")
Xage.index = V.index
Xsex = V[["male"]].astype(float)
Xyear = pd.get_dummies(V["year"].astype(int), prefix="year", dtype=float)
Xyear = Xyear.drop(columns=f"year_{int(V['year'].min())}")


def fit_logit(parts):
    X = sm.add_constant(pd.concat(parts, axis=1), has_constant="add")
    res_ = sm.Logit(measured, X).fit(disp=0, method="newton", maxiter=100)
    return res_, X


fit_full, X_full = fit_logit([Xdip, Xage, Xsex, Xyear])
fit_nodip, _ = fit_logit([Xage, Xsex, Xyear])
fit_dip, _ = fit_logit([Xdip])
for _nm, _f in (("full", fit_full), ("without dipstick", fit_nodip), ("dipstick only", fit_dip)):
    assert _f.mle_retvals["converged"], f"measurement model ({_nm}) did not converge"
auc_full = roc_auc_score(measured, fit_full.predict())
auc_nodip = roc_auc_score(measured, fit_nodip.predict())
auc_dip = roc_auc_score(measured, fit_dip.predict())
lr = 2 * (fit_full.llf - fit_nodip.llf)
lr_df = Xdip.shape[1]
E_rows = [dict(section="measurement_model", item="population",
               value=f"adult window persons (aged {ADULT_AGE}-{AGE_MAX} y, one first-visit urine test each) with "
                     f"complete dipstick (9 fields), age and sex (n={len(V):,}); serum urate measured on the "
                     f"urine-test day in {measured.sum():,} ({100 * measured.mean():.1f}%)"),
          dict(section="measurement_model", item="AUROC full model (dipstick + age band + sex + calendar year)",
               estimate=auc_full, n=len(V)),
          dict(section="measurement_model", item="AUROC without dipstick (age band + sex + calendar year)",
               estimate=auc_nodip, n=len(V)),
          dict(section="measurement_model", item="AUROC dipstick only", estimate=auc_dip, n=len(V)),
          dict(section="measurement_model", item="likelihood-ratio test, dipstick terms (full vs without dipstick)",
               estimate=lr, df=lr_df, p=float(stats.chi2.sf(lr, lr_df)),
               log10_p=log10_chi2_sf(lr, lr_df)),
          dict(section="measurement_model", item="McFadden pseudo-R2 full / without dipstick / dipstick only",
               value=f"{fit_full.prsquared:.4f} / {fit_nodip.prsquared:.4f} / {fit_dip.prsquared:.4f}")]
log(f"  P(same-day urate measured): n={len(V):,}, measured {100 * measured.mean():.1f}%; AUROC full {auc_full:.4f}, "
    f"without dipstick {auc_nodip:.4f}, dipstick only {auc_dip:.4f}; LR chi2(dipstick, df={lr_df}) = {lr:.0f}")
ci_tab = fit_full.conf_int()
units = {k: u for k, (_, u) in DIP_TERMS.items()}
units.update({c: "vs age 40-49" for c in Xage.columns})
units.update({"male": "male vs female"})
units.update({c: f"vs {int(V['year'].min())}" for c in Xyear.columns})
for term in fit_full.params.index:
    if term == "const":
        continue
    E_rows.append(dict(section="odds_ratio", item=term, unit=units.get(term, ""),
                       estimate=float(np.exp(fit_full.params[term])), lo=float(np.exp(ci_tab.loc[term, 0])),
                       hi=float(np.exp(ci_tab.loc[term, 1])), p=float(fit_full.pvalues[term])))
    if term in DIP_TERMS or term == "male":
        r = E_rows[-1]
        log(f"    OR {term:<18s} {r['estimate']:.3f} ({r['lo']:.3f}-{r['hi']:.3f}) {r['unit']}")
ps_all = pd.Series(np.asarray(fit_full.predict()), index=V.index)

# sensitivity: the same measurement model fitted only where same-day BMI is available, i.e. in the population the
# 12-input model can actually score (IPW then targets all persons with the 12 inputs, measured or not)
vb = has_bmi.loc[V.index].values
Xb = X_full[vb]
# terms with < 20 non-zero values are dropped (guard against quasi-separation; the dropped terms are logged)
keep_cols = (Xb.columns == "const") | (((Xb != 0).sum() >= 20) & (Xb.std() > 0))
dropped_bmi = [c for c, k in zip(Xb.columns, keep_cols) if not k]
Xb = Xb.loc[:, keep_cols]
fit_bmi = sm.Logit(measured[vb], Xb).fit(disp=0, method="newton", maxiter=100)
assert fit_bmi.mle_retvals["converged"], "sensitivity measurement model did not converge"
auc_bmi = roc_auc_score(measured[vb], fit_bmi.predict())
ps_bmi = pd.Series(np.asarray(fit_bmi.predict()), index=V.index[vb])
E_rows.append(dict(section="measurement_model",
                   item="AUROC full model refitted among persons with same-day BMI available (sensitivity)",
                   estimate=auc_bmi, n=int(vb.sum()),
                   value=f"urate measured in {int(measured[vb].sum()):,} of {int(vb.sum()):,} "
                         f"({100 * measured[vb].mean():.1f}%); terms dropped (< 20 non-zero): "
                         f"{';'.join(dropped_bmi) if dropped_bmi else 'none'}"))
log(f"  sensitivity measurement model among persons with same-day BMI: n={int(vb.sum()):,}, measured "
    f"{100 * measured[vb].mean():.1f}%, AUROC {auc_bmi:.4f}; dropped terms {dropped_bmi}")
PS_MODELS = [("spec", "P(measured) model on records with complete dipstick, age and sex (BMI ignored)", ps_all),
             ("bmi_available", "P(measured) model on records with all 12 inputs (sensitivity)", ps_bmi)]


def weighted_auc_ci(y, p, w, n=N_BOOT, seed=SEED):
    """Weighted AUROC and paired bootstrap CIs for it and for (weighted - unweighted)."""
    rng = np.random.RandomState(seed)
    aw, du = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a_w = roc_auc_score(y[i], p[i], sample_weight=w[i])
        aw.append(a_w)
        du.append(a_w - roc_auc_score(y[i], p[i]))
    return (roc_auc_score(y, p, sample_weight=w), np.percentile(aw, 2.5), np.percentile(aw, 97.5),
            np.percentile(du, 2.5), np.percentile(du, 97.5))


IPW_NOTE = ("bootstrap CI treats the propensity model as fixed (not refitted per resample) and the weight cap is the "
            "99th percentile of the internal-test weights; both are negligible here (propensity fitted on "
            f"{len(V):,} persons; alternative cap from all measured persons reported separately)")
# alternative cap: 99th percentile of 1/P(measured) over all measured adult window persons (pre-specified model)
CAP_ALL = float(np.percentile(1 / ps_all[measured == 1], 99))
AGE_BANDS_E = [("age < 40 y", lambda a: a < 40), ("age 40-64 y", lambda a: (a >= 40) & (a < 65)),
               ("age >= 65 y", lambda a: a >= 65)]
for t, cfg in TARGETS.items():
    di = INTERNAL[t]
    y, p = di["y"].values, di["p"].values
    a_u, lo_u, hi_u = auroc_ci(y, p, n=N_BOOT)
    check_auc(t, a_u)
    E_rows.append(dict(section="auroc", item="unweighted (internal test)", target=cfg["short"], n=len(y),
                       positives=int(y.sum()), estimate=a_u, lo=lo_u, hi=hi_u, oe=oe_ratio(y, p),
                       prevalence=y.mean()))
    msg = f"  {cfg['short']:>6s} internal AUROC unweighted {a_u:.4f} ({lo_u:.4f}-{hi_u:.4f})"
    for key, desc, ps_series in PS_MODELS:
        ps = ps_series.reindex(di["row"].values).values
        assert not np.isnan(ps).any()
        w_raw = 1 / ps
        cap = np.percentile(w_raw, 99)
        w = np.minimum(w_raw, cap)
        a_w, lo_w, hi_w, dlo, dhi = weighted_auc_ci(y, p, w)
        a_raw = roc_auc_score(y, p, sample_weight=w_raw)
        ess = w.sum() ** 2 / (w ** 2).sum()
        E_rows.append(dict(section="weights", item=f"internal test IPW weights [{desc}]", target=cfg["short"],
                           propensity_model=key, n=len(y), ps_min=ps.min(), ps_median=np.median(ps), ps_max=ps.max(),
                           w_median=np.median(w_raw), w_p99_cap=cap, w_max_untruncated=w_raw.max(),
                           n_truncated=int((w_raw > cap).sum()), effective_n=ess))
        E_rows.append(dict(section="auroc", item=f"IPW, 1/P(measured) truncated at 99th percentile [{desc}]",
                           target=cfg["short"], propensity_model=key, n=len(y), positives=int(y.sum()),
                           estimate=a_w, lo=lo_w, hi=hi_w, diff_vs_unweighted=a_w - a_u, diff_lo=dlo, diff_hi=dhi,
                           oe=float(np.sum(w * y) / np.sum(w * p)), prevalence=float(np.sum(w * y) / np.sum(w)),
                           note=IPW_NOTE))
        E_rows.append(dict(section="auroc", item=f"IPW, untruncated weights [{desc}]", target=cfg["short"],
                           propensity_model=key, n=len(y), positives=int(y.sum()), estimate=a_raw,
                           oe=float(np.sum(w_raw * y) / np.sum(w_raw * p)),
                           prevalence=float(np.sum(w_raw * y) / np.sum(w_raw))))
        msg += (f" | IPW[{key}] {a_w:.4f} ({lo_w:.4f}-{hi_w:.4f}), diff {a_w - a_u:+.4f} ({dlo:+.4f} to {dhi:+.4f}),"
                f" untruncated {a_raw:.4f}, P(measured) median {np.median(ps):.3f} [{ps.min():.3f}-{ps.max():.3f}],"
                f" cap w={cap:.2f}, ESS {ess:,.0f}")
        if key == "spec":
            w_alt = np.minimum(w_raw, CAP_ALL)
            a_alt = roc_auc_score(y, p, sample_weight=w_alt)
            E_rows.append(dict(section="auroc", item=f"IPW, truncated at the 99th percentile of weights among all "
                                                     f"measured adult window persons (cap {CAP_ALL:.3f}) [{desc}]",
                               target=cfg["short"], propensity_model=key, n=len(y), positives=int(y.sum()),
                               estimate=a_alt, diff_vs_unweighted=a_alt - a_u,
                               oe=float(np.sum(w_alt * y) / np.sum(w_alt * p)),
                               prevalence=float(np.sum(w_alt * y) / np.sum(w_alt))))
            msg += f", alt cap {CAP_ALL:.2f} -> {a_alt:.4f}"
    log(msg)
    # AUROC within fifths of P(measured) (pre-specified model): does discrimination depend on testing likelihood?
    ps = ps_all.reindex(di["row"].values).values
    ft = P.loc[di["row"].values, ["age_c", "male"]].reset_index(drop=True)
    qb = pd.qcut(ps, 5, labels=False, duplicates="drop")
    for qq in np.unique(qb):
        mk = qb == qq
        row = dict(section="auroc_by_measurement_propensity", item=f"P(measured) fifth {qq + 1}", target=cfg["short"],
                   propensity_model="spec", n=int(mk.sum()), positives=int(y[mk].sum()),
                   ps_range=f"{ps[mk].min():.3f}-{ps[mk].max():.3f}", prevalence=y[mk].mean(),
                   age_mean=ft.loc[mk, "age_c"].mean(), pct_age_lt19=100 * (ft.loc[mk, "age_c"] < 19).mean(),
                   pct_age_ge65=100 * (ft.loc[mk, "age_c"] >= 65).mean(), pct_male=100 * ft.loc[mk, "male"].mean(),
                   note="case-mix profile of the fifth (the P(measured) model contains age band, sex and year)")
        if 0 < y[mk].sum() < mk.sum():
            a, lo_, hi_ = auroc_ci(y[mk], p[mk], n=N_BOOT)
            row.update(estimate=a, lo=lo_, hi=hi_, oe=oe_ratio(y[mk], p[mk]))
        E_rows.append(row)
    fifths = [r for r in E_rows if r["section"] == "auroc_by_measurement_propensity" and r["target"] == cfg["short"]]
    log("         AUROC by P(measured) fifth (lowest to highest): "
        + ", ".join(f"{r.get('estimate', np.nan):.3f}" for r in fifths)
        + " | mean age " + ", ".join(f"{r['age_mean']:.0f}" for r in fifths)
        + " | % >= 65 y " + ", ".join(f"{r['pct_age_ge65']:.0f}" for r in fifths)
        + " | % male " + ", ".join(f"{r['pct_male']:.0f}" for r in fifths))
    # the same gradient within age bands and within sex x adult age band (fifths re-cut within each stratum);
    # primary threshold only. If the gradient vanishes within sex x age, it reflects case mix (the fifths differ
    # in sex and age composition), not a dependence of discrimination on testing likelihood itself.
    if cfg["threshold"] != 7.0:
        continue
    age_v, male_v = ft["age_c"].values, ft["male"].values
    strata = [(lab, np.asarray(fn(age_v), bool)) for lab, fn in AGE_BANDS_E]
    for s_val, s_lab in ((0, "women"), (1, "men")):
        for a_lab, lo_a, hi_a in (("19-39 y", 19, 40), ("40-64 y", 40, 65), (">= 65 y", 65, np.inf)):
            strata.append((f"{s_lab} {a_lab}", (male_v == s_val) & (age_v >= lo_a) & (age_v < hi_a)))
    for b_lab, bm in strata:
        qb_b = pd.qcut(ps[bm], 5, labels=False, duplicates="drop")
        yb_, pb_, psb = y[bm], p[bm], ps[bm]
        ests = []
        for qq in np.unique(qb_b):
            mk = qb_b == qq
            row = dict(section="auroc_by_measurement_propensity_within_stratum",
                       item=f"{b_lab}: P(measured) fifth {qq + 1}",
                       target=cfg["short"], propensity_model="spec", n=int(mk.sum()), positives=int(yb_[mk].sum()),
                       ps_range=f"{psb[mk].min():.3f}-{psb[mk].max():.3f}", prevalence=yb_[mk].mean(),
                       age_mean=age_v[bm][mk].mean(), pct_male=100 * male_v[bm][mk].mean(),
                       note="fifths of P(measured) re-cut within the stratum")
            if 0 < yb_[mk].sum() < mk.sum():
                row["estimate"] = roc_auc_score(yb_[mk], pb_[mk])
                row["oe"] = oe_ratio(yb_[mk], pb_[mk])
                if yb_[mk].sum() >= MIN_POS:
                    _, row["lo"], row["hi"] = auroc_ci(yb_[mk], pb_[mk], n=N_BOOT)
                else:
                    row["note"] += "; " + FEW_NOTE
            E_rows.append(row)
            ests.append(row)
        log(f"         {b_lab:<16s} AUROC by P(measured) fifth within stratum: "
            + ", ".join(f"{r.get('estimate', np.nan):.3f}" for r in ests)
            + " | prevalence " + ", ".join(f"{100 * r['prevalence']:.0f}%" for r in ests)
            + " | % male " + ", ".join(f"{r['pct_male']:.0f}" for r in ests))
pd.DataFrame(E_rows).to_csv(path("verification_bias.csv"), index=False, encoding="utf-8")

log(f"\nruntime {(time.time() - T0) / 60:.1f} min")
with open(path("calc_missing_bias_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
