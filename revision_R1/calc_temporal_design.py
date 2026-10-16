"""Temporal validation, KNHANES survey period, survey weights and development age of the retrained models (R1).

Reviewer AW #11 (random split, not temporal), our own finding that "nationally representative" was claimed without
survey weights, and the age restriction of the development cohort.

Retrained models (bestmodel_v5, decision of 2026-10-16): 12 inputs (cohort.FMODEL = 9 dipstick fields + age, sex, BMI;
leukocyte esterase is not an input), trained on the corrected development cohort dev_v2 (cohort.main_cohort: adults aged
>= 19 at the urine test, first-visit urine test, serum urate and BMI of the urine-test day, nitrite parsed correctly, one
record per person) with ONE person-level 80/20 split shared by the three thresholds (training persons d.iloc[tr],
internal test persons d.iloc[te]). External: KNHANES adults aged >= 19 with the 12 inputs and serum urate (cohort ext,
dipstick inputs coded by the label-matched mapping X1).

(a) temporal.csv
    dev_v2 persons; year = year of the urine test. Train on persons tested 2010-2021, test on those tested 2022-2025.
    Models:
      LGBM-12F, LGBM-3F   common.LGBM_FIXED (equal modelling budget), n_jobs=4
      LR-12F              StandardScaler + LogisticRegression(C=1, max_iter=5000) (as in the liver audit)
      Ensemble-12F        the retrained soft-voting template, clone(load_model(t).estimator) (same tuned
                          hyperparameters and seeds), wrapped in isotonic CalibratedClassifierCV(cv=5) and refit on the
                          early period only (NOT re-tuned; the saved model files are untouched)
    Each model is also fitted, the same way, on the training persons of the random person-level split of the retrained
    models (cohort.main_cohort tr, shared by the three thresholds) and tested on its internal test persons (te): the
    reference with a similar training size. For the ensemble the reference refit uses the retrained model's own
    calibration folds (cohort.cal_folds on cohort.train_xy, i.e. what cv=5 forms on those rows), so it re-creates the
    retrained model; its predictions are checked against preds_internal_clean.csv / preds_external_corrected.csv and its
    AUROC against main_performance.csv. Reported: internal test AUROC (95% CI), calibration intercept / slope / O:E,
    Brier; the same on KNHANES adults; and the LGBM 12F-minus-3F AUROC contrast (paired bootstrap CI, DeLong p).
    Calibration slopes come from common.calib_stats (exact maximum likelihood, Newton-Raphson). prevalence_train and
    mean_pred_train (in-sample prediction on the training persons) are added so the O:E can be split into the change in
    prevalence and the change in mean predicted risk.
(b) knhanes_by_year.csv   retrained models, KNHANES adults, by survey period 2016-18 / 2019-21 / 2022-24.
(c) survey_weighted.csv   retrained models, KNHANES adults, weighted (wt_itvex) vs unweighted prevalence, O:E and
                          AUROC. Strata / PSU are NOT used for variance.
(d) adult_dev.csv         retrained models, internal test persons (all adults by construction of dev_v2; group
                          "age >= 19"). The former "all ages" and "age < 19" rows and the adults-minus-<19 contrast were
                          removed on 2026-10-11 (adult target population); the columns are kept (contrast columns empty,
                          the note column says why) so that make_tables.py and make_supplement.py still read the file.
(e) dipstick_by_year.csv  inputs and outcome by calendar year (development, dev_v2, 2010-2025) and by survey year
                          (KNHANES adults, X1 coding as fed to the models, 2016-2024), plus pooled periods: positivity
                          (value > 0) of each dipstick field; urobilinogen above normal (>= 1.0 on the model scale; normal
                          = 0.5 in both cohorts under X1); leukocyte esterase positivity as observed in development (not a
                          model input; kept as a marker of the 2022 laboratory reporting change; not measured in KNHANES);
                          mean specific gravity and pH; mean age, male fraction, mean BMI; prevalence of serum urate
                          >= 7.0 / 9.0 / 10.0 mg/dL. The log then interprets the temporal test (time plus possible
                          measurement change).
Removed on 2026-10-16 (retrained models): the LR-11F sensitivity model (13F minus nitrite and urobilinogen), fitted only
because development nitrite was near-constant through the parsing error and the submission's KNHANES coding put the
urobilinogen normal at 0 against 0.5 in development (both corrected: nitrite is parsed correctly in dev_v2 and X1 codes
the KNHANES normal 0.5); the window-level adult-restriction counts and the nitrite-defect footprint by year (they
described the submission's processed data; dev_v2 is adult and one first-visit record per person by construction, its
exclusion steps are cohort.cohort_steps()).

Bootstrap: 1000 resamples with replacement at full size, percentile 95% CI, seed 42.
Run from the working folder:  /opt/anaconda3/bin/python calc_temporal_design.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import time, warnings
import numpy as np, pandas as pd

# cohort and common MUST be imported before lightgbm / xgboost: otherwise unpickling the saved
# XGBoost learners (load_model) can deadlock.
from cohort import (HERE, TARGETS, F3, FMODEL, LE, ID, DATE, main_cohort, cohort_steps, persons_v2, train_xy, cal_folds,
                    labels, load_model, ADULT_AGE, is_adult)
from common import (SEED, LGBM_FIXED, auroc_ci, boot_ci, paired_boot, delong_paired, delong_unpaired, calib_stats)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
NJ = 4                      # thread cap per model
N_BOOT = 1000
EARLY_MAX, LATE_MIN = 2021, 2022
PERIODS = [("2016-2018", 2016, 2018), ("2019-2021", 2019, 2021), ("2022-2024", 2022, 2024)]
VAR_NOTE = ("95% CIs from 1000 simple record-level bootstrap resamples (each record keeps its weight); "
            "KNHANES strata (kstrata) and PSU (psu) are NOT used, so the CIs ignore the complex design "
            "and are likely too narrow")
WT_NOTE = ("wt_itvex as supplied for each survey year; dividing by the 9 pooled years is a constant "
           "rescaling and leaves weighted prevalence, O:E and AUROC unchanged; weights were not "
           "re-adjusted for complete-case exclusion (12 inputs + serum urate)")
MIN_POS = 10                # below this many positives a subgroup CI / calibration fit / contrast is not reported
MAIN_M, DEMO_M = "Ensemble-12F (tuned, refit)", "LGBM-3F"
CONTRAST = "LGBM 12F minus 3F (AUROC difference)"
ENS_NOTE = {"temporal": ("hyperparameters of the retrained model (Optuna-tuned on the training persons of the random "
                         "split, which include persons tested in 2022 or later); refit on the persons tested 2010-2021 "
                         "only, not re-tuned"),
            "random (reference)": ("retrained template refit on the retrained model's training persons with its own "
                                   "calibration folds (re-creates the retrained model; checked against its saved "
                                   "predictions)")}
DEV_NOTE = ("corrected development cohort dev_v2 (nitrite parsed correctly); urobilinogen normal = 0.5, above normal = "
            ">= 1.0; leukocyte esterase as observed, not a model input")
KN_NOTE = ("inputs as fed to the retrained models (label-matched mapping X1); urobilinogen normal and trace coded 0.5, "
           "above normal = >= 1.0 (+ or more); leukocyte esterase not measured in KNHANES (not a model input; positivity "
           "not reported)")
# Dipstick fields reported as positivity (> 0); urobilinogen as above normal (>= UROB_ABOVE). Leukocyte esterase is not
# a model input; its development positivity is kept as a marker of the 2022 laboratory reporting change.
POS_FIELDS = [("ua_glucose", "glucose"), ("ua_protein", "protein"), ("ua_ketone", "ketone"),
              ("ua_bilirubin", "bilirubin"), ("ua_blood", "blood"), ("ua_urobilinogen", "urobilinogen_above_normal"),
              ("ua_nitrite", "nitrite"), (LE, "leukocyte_esterase")]
UROB_ABOVE = 1.0
STEP_REL = 0.30             # a "step" at 2022 needs a >= 30% relative change 2021 -> 2022 AND no overlap of the
                            # annual values before and after (every 2022+ year beyond every earlier year)
P0_NOTE = "DeLong p below floating-point resolution (reported as 0; p < 1e-15)"

T0 = time.time()
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def oe(y, p):
    return float(np.mean(y) / np.mean(p))


def summary(y, p):
    """AUROC with bootstrap CI, calibration intercept / slope / O:E, Brier."""
    y = np.asarray(y); p = np.asarray(p)
    auc, lo, hi = auroc_ci(y, p, N_BOOT, SEED)
    ci, cs, r = calib_stats(y, p)
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()), mean_pred=float(p.mean()),
                auroc=auc, auroc_lo=lo, auroc_hi=hi, cal_intercept=ci, cal_slope=cs, oe=r,
                brier=brier_score_loss(y, p))


def build_models(target, ens_cv):
    """Comparators and the ensemble refit for one threshold; ens_cv = the ensemble's isotonic calibration folds
    (5 = StratifiedKFold(5) without shuffling on the training rows passed, or an explicit list of folds)."""
    lgb_cfg = {**LGBM_FIXED, "n_jobs": NJ}
    v = clone(load_model(target).estimator)    # unfitted retrained template: same tuned hyperparameters and seeds
    v.set_params(n_jobs=1, lgb__n_jobs=NJ, xgb__n_jobs=NJ, cat__thread_count=NJ)
    ens = CalibratedClassifierCV(v, method="isotonic", cv=ens_cv, n_jobs=1)
    return {"LGBM-12F": (FMODEL, LGBMClassifier(**lgb_cfg)),
            "LGBM-3F": (F3, LGBMClassifier(**lgb_cfg)),
            "LR-12F": (FMODEL, make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))),
            MAIN_M: (FMODEL, ens)}


def wboot(y, p, w, fn, n=N_BOOT, seed=SEED):
    """Record-level bootstrap of a weighted statistic; same draws as common.boot_ci."""
    rng = np.random.RandomState(seed); out = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        out.append(fn(y[i], p[i], w[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ============================================================================ data
d0, tr, te, ext = main_cohort()
d = d0.copy()                                   # positions tr / te refer to this frame (same row order)
d["year"] = pd.to_datetime(d[DATE], errors="coerce").dt.year
assert d["year"].notna().all(), "undated development records"
assert bool(is_adult(d["age"]).all()) and d[ID].is_unique, "dev_v2 must hold one adult record per person"
# observed leukocyte esterase of the cohort persons (not a model input; for dipstick_by_year.csv only)
_pv = persons_v2()
d[LE] = _pv.loc[_pv["in_cohort"]].set_index(ID)[LE].reindex(d[ID]).to_numpy()
ext = ext.copy()
early = np.where(d["year"].values <= EARLY_MAX)[0]
late = np.where(d["year"].values >= LATE_MIN)[0]
assert len(early) + len(late) == len(d)
assert len(te) == 25907 and len(ext) == 49987

for desc, n in cohort_steps():
    log(f"dev_v2 step: {desc}: {n:,}")
log(f"dev_v2 persons {len(d):,} (adults aged >= {ADULT_AGE}; age {int(d.age.min())}-{int(d.age.max())}); years "
    f"{int(d.year.min())}-{int(d.year.max())}; random split of the retrained models: training {len(tr):,}, internal "
    f"test {len(te):,}")
log(f"  early (<= {EARLY_MAX}) {len(early):,}   late (>= {LATE_MIN}) {len(late):,} "
    f"(years {int(d.year.values[late].min())}-{int(d.year.values[late].max())})")
log(f"  leukocyte esterase (observed, not an input) missing for {int(d[LE].isna().sum())} cohort persons")
log(f"KNHANES adults (X1 coding) {len(ext):,}")
for f in ("ua_nitrite", "ua_urobilinogen"):
    log(f"  coding check {f}: development {d[f].value_counts().sort_index().to_dict()} | "
        f"KNHANES {ext[f].value_counts().sort_index().to_dict()}")

# ============================================================================ retrained-model predictions
# round_trip: the predictions exactly as calc_main.py wrote them (pandas' default float parser can be off by ~1e-16)
pe = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"), float_precision="round_trip")
pi = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"), float_precision="round_trip")
mperf = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
dte = d.iloc[te]
for t, cfg in TARGETS.items():                      # row alignment with the internal test persons and with ext
    s = pe[pe.target == t]
    assert len(s) == len(ext) and np.array_equal(s.year.values, ext.year.values)
    assert np.array_equal(s.id.values, ext.id.values) and np.allclose(s.wt.values, ext.wt_itvex.values)
    assert np.array_equal(s.y.values, labels(ext.urate, cfg["threshold"]))
    s = pi[pi.target == t]
    assert len(s) == len(te) and np.array_equal(s.id.values, dte[ID].values)
    assert np.array_equal(s.y.values, labels(dte.urate, cfg["threshold"]))


def main_auroc(t, cohort):
    r = mperf[(mperf.target == TARGETS[t]["short"]) & (mperf.cohort == cohort)].iloc[0]
    return int(r["n"]), float(r["auroc"])


# ============================================================================ (a) temporal design
log("\n(a) Temporal validation")
rows_a = []
LATE_BY_YEAR = {}           # (threshold short, model) -> DataFrame of observed / mean predicted by 2022+ year
for t, cfg in TARGETS.items():
    thr = cfg["threshold"]
    y = labels(d["urate"], thr); y_ext = labels(ext["urate"], thr)
    # calibration folds of the retrained model (checked against its saved isotonic calibrators), positions in d.iloc[tr]
    X_tr, y_tr = train_xy(thr)
    assert np.array_equal(y_tr, y[tr]) and np.array_equal(X_tr.to_numpy(float), d[FMODEL].iloc[tr].to_numpy(float))
    folds = cal_folds(load_model(t), X_tr, y_tr)
    designs = {"temporal": (early, late, f"2010-{EARLY_MAX}", f"{LATE_MIN}-{int(d.year.max())}", 5),
               "random (reference)": (tr, te, "2010-2025 random 80%", "2010-2025 random 20%", folds)}
    log(f"\n  {cfg['short']}: prevalence early {y[early].mean():.4f} ({int(y[early].sum())}), "
        f"late {y[late].mean():.4f} ({int(y[late].sum())}); KNHANES {y_ext.mean():.4f} ({int(y_ext.sum())})")
    for dname, (trn, tst, trp, tep, ens_cv) in designs.items():
        preds = {}
        for mname, (feats, model) in build_models(t, ens_cv).items():
            t1 = time.time()
            m = clone(model).fit(d[feats].iloc[trn], y[trn])
            p_tr = m.predict_proba(d[feats].iloc[trn])[:, 1]          # in-sample, for the O:E decomposition
            p_int = m.predict_proba(d[feats].iloc[tst])[:, 1]
            p_ext = m.predict_proba(ext[feats])[:, 1]
            preds[mname] = (p_int, p_ext)
            if dname == "temporal":
                yr = d["year"].values[tst]
                LATE_BY_YEAR[(cfg["short"], mname)] = (pd.DataFrame({"year": yr, "y": y[tst], "p": p_int})
                                                      .groupby("year").agg(n=("y", "size"), observed=("y", "mean"),
                                                                           mean_pred=("p", "mean")))
            for ev, yy, pp in (("internal test", y[tst], p_int), ("external KNHANES >=19", y_ext, p_ext)):
                rows_a.append(dict(target=cfg["short"], role=cfg["role"], design=dname, train_period=trp,
                                   test_period=tep if ev == "internal test" else "KNHANES 2016-2024",
                                   model=mname, n_features=len(feats), evaluation=ev, n_train=len(trn),
                                   pos_train=int(y[trn].sum()), prevalence_train=float(y[trn].mean()),
                                   mean_pred_train=float(p_tr.mean()), **summary(yy, pp),
                                   note=ENS_NOTE[dname] if mname == MAIN_M else ""))
            r_i = rows_a[-2]; r_e = rows_a[-1]
            log(f"    [{dname:18s}] {mname[:28]:28s} int AUROC {r_i['auroc']:.4f} ({r_i['auroc_lo']:.4f}-{r_i['auroc_hi']:.4f}) "
                f"int/slope {r_i['cal_intercept']:+.3f}/{r_i['cal_slope']:.3f} O:E {r_i['oe']:.3f} | "
                f"ext AUROC {r_e['auroc']:.4f} ({r_e['auroc_lo']:.4f}-{r_e['auroc_hi']:.4f}) "
                f"O:E {r_e['oe']:.3f}  [{time.time() - t1:.0f}s]")
            if dname == "random (reference)" and mname == MAIN_M:
                # the reference refit re-creates the retrained model: same template, training persons, row order and folds
                dp_i = float(np.max(np.abs(p_int - pi[pi.target == t].p.values)))
                dp_e = float(np.max(np.abs(p_ext - pe[pe.target == t].p.values)))
                (n_i, a_i), (n_e, a_e) = main_auroc(t, "internal test"), main_auroc(t, "KNHANES adults")
                log(f"    [{dname:18s}] {'':28s} vs the saved retrained model: max |dp| internal {dp_i:.3g}, "
                    f"KNHANES {dp_e:.3g}; AUROC {r_i['auroc']:.4f} / {r_e['auroc']:.4f} vs main_performance.csv "
                    f"{a_i:.4f} / {a_e:.4f}")
                assert r_i["n"] == n_i == 25907 and r_e["n"] == n_e == 49987
                assert round(r_i["auroc"], 4) == round(a_i, 4) and round(r_e["auroc"], 4) == round(a_e, 4), \
                    "reference refit does not reproduce the retrained model's AUROC"
        # 12F minus 3F (LGBM_FIXED): paired bootstrap CI + DeLong
        for ev, yy, k in (("internal test", y[tst], 0), ("external KNHANES >=19", y_ext, 1)):
            p12, p3 = preds["LGBM-12F"][k], preds["LGBM-3F"][k]
            dd, pval = delong_paired(yy, p12, p3)
            lo, hi = paired_boot(yy, p12, p3, N_BOOT, SEED)
            rows_a.append(dict(target=cfg["short"], role=cfg["role"], design=dname, train_period=trp,
                               test_period=tep if ev == "internal test" else "KNHANES 2016-2024",
                               model=CONTRAST, n_features=np.nan, evaluation=ev,
                               n_train=len(trn), pos_train=int(y[trn].sum()), n=len(yy), positives=int(yy.sum()),
                               prevalence=float(yy.mean()), auroc=dd, auroc_lo=lo, auroc_hi=hi, delong_p=pval,
                               note=P0_NOTE if pval == 0 else ""))
            log(f"    [{dname:18s}] 12F-3F {ev:22s} dAUROC {dd:+.4f} ({lo:+.4f} to {hi:+.4f}) DeLong p={pval:.2e}")
temporal = pd.DataFrame(rows_a)
temporal = temporal[[c for c in temporal.columns if c != "note"] + ["note"]]
temporal.to_csv(os.path.join(HERE, "temporal.csv"), index=False)
log(f"  -> temporal.csv ({len(temporal)} rows)  [{(time.time() - T0) / 60:.1f} min]")

# ============================================================================ (b) KNHANES by survey period
log("\n(b) Retrained models on KNHANES adults by survey period")
rows_b = []
for t, cfg in TARGETS.items():
    s = pe[pe.target == t]
    ref = None
    for per, a, b in PERIODS + [("2016-2024 (all)", 2016, 2024)]:
        g = s[(s.year >= a) & (s.year <= b)]
        yy, pp = g.y.values, g.p.values
        r = summary(yy, pp)
        r["oe_lo"], r["oe_hi"] = boot_ci(yy, pp, oe, N_BOOT, SEED)
        if per == "2016-2018":
            ref = (yy, pp); dd = dict(auroc_diff_vs_2016_2018=np.nan, diff_lo=np.nan, diff_hi=np.nan, diff_p=np.nan)
        elif per.startswith("2016-2024"):
            dd = dict(auroc_diff_vs_2016_2018=np.nan, diff_lo=np.nan, diff_hi=np.nan, diff_p=np.nan)
            n_m, a_m = main_auroc(t, "KNHANES adults")
            assert r["n"] == n_m == 49987 and round(r["auroc"], 4) == round(a_m, 4)
        else:
            dl, lo, hi, pv = delong_unpaired(yy, pp, *ref)
            dd = dict(auroc_diff_vs_2016_2018=dl, diff_lo=lo, diff_hi=hi, diff_p=pv)
        rows_b.append(dict(target=cfg["short"], role=cfg["role"], period=per, **r, **dd))
        log(f"  {cfg['short']:6s} {per:16s} n={r['n']:6d} pos={r['positives']:5d} AUROC {r['auroc']:.4f} "
            f"({r['auroc_lo']:.4f}-{r['auroc_hi']:.4f})  O:E {r['oe']:.3f} ({r['oe_lo']:.3f}-{r['oe_hi']:.3f})"
            + (f"  vs 2016-18 {dd['auroc_diff_vs_2016_2018']:+.4f} p={dd['diff_p']:.3f}"
               if not np.isnan(dd["auroc_diff_vs_2016_2018"]) else ""))
by_year = pd.DataFrame(rows_b)
by_year.to_csv(os.path.join(HERE, "knhanes_by_year.csv"), index=False)
log("  -> knhanes_by_year.csv")

# ============================================================================ (c) survey weights
log("\n(c) Retrained models on KNHANES adults: survey-weighted vs unweighted")
rows_c = []
w_auc = lambda y, p, w: roc_auc_score(y, p, sample_weight=w)
w_prev = lambda y, p, w: float(np.sum(w * y) / np.sum(w))
w_oe = lambda y, p, w: float(np.sum(w * y) / np.sum(w * p))
for t, cfg in TARGETS.items():
    s = pe[pe.target == t]
    yy, pp, ww = s.y.values, s.p.values, s.wt.values
    auc_u, lo_u, hi_u = auroc_ci(yy, pp, N_BOOT, SEED)
    n_m, a_m = main_auroc(t, "KNHANES adults")
    assert len(yy) == n_m == 49987 and round(auc_u, 4) == round(a_m, 4)
    prev_u_lo, prev_u_hi = boot_ci(yy, pp, lambda y, p: float(np.mean(y)), N_BOOT, SEED)
    oe_u_lo, oe_u_hi = boot_ci(yy, pp, oe, N_BOOT, SEED)
    r = dict(target=cfg["short"], role=cfg["role"], n=len(yy), positives=int(yy.sum()),
             prevalence_unweighted=float(yy.mean()), prevalence_unweighted_lo=prev_u_lo, prevalence_unweighted_hi=prev_u_hi,
             prevalence_weighted=w_prev(yy, pp, ww), **dict(zip(("prevalence_weighted_lo", "prevalence_weighted_hi"),
                                                               wboot(yy, pp, ww, w_prev))),
             mean_pred_unweighted=float(pp.mean()), mean_pred_weighted=float(np.sum(ww * pp) / np.sum(ww)),
             oe_unweighted=oe(yy, pp), oe_unweighted_lo=oe_u_lo, oe_unweighted_hi=oe_u_hi,
             oe_weighted=w_oe(yy, pp, ww), **dict(zip(("oe_weighted_lo", "oe_weighted_hi"), wboot(yy, pp, ww, w_oe))),
             auroc_unweighted=auc_u, auroc_unweighted_lo=lo_u, auroc_unweighted_hi=hi_u,
             auroc_weighted=w_auc(yy, pp, ww), **dict(zip(("auroc_weighted_lo", "auroc_weighted_hi"),
                                                          wboot(yy, pp, ww, w_auc))),
             weight_variable="wt_itvex", weight_note=WT_NOTE, variance_note=VAR_NOTE)
    rows_c.append(r)
    log(f"  {cfg['short']:6s} prevalence unweighted {r['prevalence_unweighted']:.4f} vs weighted {r['prevalence_weighted']:.4f} | "
        f"O:E {r['oe_unweighted']:.3f} vs {r['oe_weighted']:.3f} | AUROC {auc_u:.4f} ({lo_u:.4f}-{hi_u:.4f}) vs "
        f"weighted {r['auroc_weighted']:.4f} ({r['auroc_weighted_lo']:.4f}-{r['auroc_weighted_hi']:.4f})")
log(f"  note: {VAR_NOTE}")
log(f"  note: {WT_NOTE}")
pd.DataFrame(rows_c).to_csv(os.path.join(HERE, "survey_weighted.csv"), index=False)
log("  -> survey_weighted.csv")

# ============================================================================ (d) development age
# The corrected cohort dev_v2 holds adults only, so preds_internal_clean.csv (all internal test persons) has only the
# "age >= 19" group. The former "all ages" / "age < 19" rows and the adults-minus-<19 DeLong contrast were removed on
# 2026-10-11; the contrast columns stay (empty) so that make_tables.py and make_supplement.py still read the file.
AD_NOTE = ("adults only: the corrected development cohort (dev_v2) includes persons aged >= 19 only; the all-ages and "
           "<19 rows and the adults-minus-<19 contrast were removed (adult target population, 2026-10-11)")
log("\n(d) Retrained models, internal test persons, adults (age >= 19; the whole internal test set)")
assert bool(is_adult(pi["age"]).all()), "preds_internal_clean.csv must hold adults only"
rows_d = []
for t, cfg in TARGETS.items():
    s = pi[pi.target == t]
    for grp, mask in (("age >= 19", is_adult(s.age.values)),):
        g = s[mask]
        r = summary(g.y.values, g.p.values)
        n_m, a_m = main_auroc(t, "internal test")
        assert r["n"] == n_m == 25907 and round(r["auroc"], 4) == round(a_m, 4)
        note = AD_NOTE
        if r["positives"] < MIN_POS:            # CI / calibration fit meaningless with a handful of positives
            r.update(auroc_lo=np.nan, auroc_hi=np.nan, cal_intercept=np.nan, cal_slope=np.nan)
            note = f"only {r['positives']} positive(s): point estimates only, not interpretable; {AD_NOTE}"
        dd = dict(auroc_diff_vs_under19=np.nan, diff_lo=np.nan, diff_hi=np.nan, diff_p=np.nan)
        rows_d.append(dict(target=cfg["short"], role=cfg["role"], group=grp, **r, **dd, note=note))
        log(f"  {cfg['short']:6s} {grp:10s} n={r['n']:6d} pos={r['positives']:5d} AUROC {r['auroc']:.4f} "
            f"({r['auroc_lo']:.4f}-{r['auroc_hi']:.4f}) int/slope {r['cal_intercept']:+.3f}/{r['cal_slope']:.3f} "
            f"O:E {r['oe']:.3f} Brier {r['brier']:.4f}")
log(f"  note: {AD_NOTE}")
pd.DataFrame(rows_d).to_csv(os.path.join(HERE, "adult_dev.csv"), index=False)
log("  -> adult_dev.csv")

# ============================================================================ (e) dipstick inputs by year
log("\n(e) Dipstick inputs, demographics and urate prevalence by year")
THR_COLS = [(cfg["threshold"], f"{cfg['threshold']:g}") for cfg in TARGETS.values()]


def positivity(v, f):
    v = np.asarray(v, float)
    return float(np.mean(v >= UROB_ABOVE)) if f == "ua_urobilinogen" else float(np.mean(v > 0))


def input_row(g, cohort_name, period, kind, knhanes):
    r = dict(cohort=cohort_name, period=period, period_type=kind, n=len(g))
    for f, nm in POS_FIELDS:
        r[f"pos_{nm}"] = np.nan if (knhanes and f == LE) else positivity(g[f], f)
    r.update(mean_sg=float(g["ua_sg"].mean()), mean_pH=float(g["ua_pH"].mean()), mean_age=float(g["age"].mean()),
             male_frac=float(g["male"].mean()), mean_bmi=float(g["bmi_unified"].mean()))
    for thr, s in THR_COLS:
        yy = labels(g["urate"], thr)
        r[f"n_urate_ge{s}"] = int(yy.sum()); r[f"prev_urate_ge{s}"] = float(yy.mean())
    r["note"] = KN_NOTE if knhanes else DEV_NOTE
    return r


DEV_C, KN_C = "development (dev_v2)", "KNHANES adults (X1 coding)"
assert d[LE].notna().all(), "leukocyte esterase positivity needs the observed value of every cohort person"
rows_e = []
ext_year = ext["year"].astype(int)
for cname, frame, yrs, pooled, is_kn in (
        (DEV_C, d, d["year"].astype(int), [("2010-2021", 2010, 2021), ("2019-2021", 2019, 2021), ("2022-2025", 2022, 2025)],
         False),
        (KN_C, ext, ext_year, [("2016-2021", 2016, 2021), ("2019-2021", 2019, 2021), ("2022-2024", 2022, 2024)], True)):
    for yv in sorted(yrs.unique()):
        rows_e.append(input_row(frame[yrs.values == yv], cname, str(yv), "year", is_kn))
    for lab, a, b in pooled:
        rows_e.append(input_row(frame[(yrs.values >= a) & (yrs.values <= b)], cname, lab, "pooled", is_kn))
dip = pd.DataFrame(rows_e)
dip.to_csv(os.path.join(HERE, "dipstick_by_year.csv"), index=False)
POS_COLS = [f"pos_{nm}" for _, nm in POS_FIELDS]
show = dip[["cohort", "period", "n"] + POS_COLS + ["mean_sg", "mean_age", "male_frac", "mean_bmi"]
           + [f"prev_urate_ge{s}" for _, s in THR_COLS]].copy()
show.columns = [c.replace("pos_", "").replace("urobilinogen_above_normal", "uro>N").replace("leukocyte_esterase", "LE")
                 .replace("prev_urate_ge", "prev>=").replace("mean_", "").replace("male_frac", "male") for c in show.columns]
for c in show.columns[3:]:
    show[c] = show[c].map(lambda v: "" if pd.isna(v) else (f"{v:.4f}" if c in ("sg",) else
                                                           f"{v:.1f}" if c in ("age", "bmi") else f"{100 * v:.2f}"))
show["cohort"] = show["cohort"].str.split(" ").str[0]
log("  positivity, male and prevalence in %; sg, age and BMI as means; uro>N = urobilinogen >= 1.0; "
    "LE = leukocyte esterase (observed, not a model input)")
for line in show.to_string(index=False).splitlines():
    log("  " + line)
log(f"  note (development): {DEV_NOTE}")
log(f"  note (KNHANES): {KN_NOTE}")
log(f"  -> dipstick_by_year.csv ({len(dip)} rows)")


def step_at(cohort_name, first, pivot, col):
    """Annual values of `col` before `pivot` (from `first`) vs `pivot` onward: step / trend / overlap."""
    dd = dip[(dip.cohort == cohort_name) & (dip.period_type == "year")].assign(yr=lambda x: x.period.astype(int))
    e = dd.loc[(dd.yr >= first) & (dd.yr < pivot), col].values; l = dd.loc[dd.yr >= pivot, col].values
    v_prev = float(dd.loc[dd.yr == pivot - 1, col].iloc[0]); v_piv = float(dd.loc[dd.yr == pivot, col].iloc[0])
    rel = (v_piv - v_prev) / v_prev if v_prev > 0 else np.nan
    if np.isnan(e).all() or np.isnan(l).all():
        kind = "not available"
    elif l.max() < e.min():
        kind = "step down" if (not np.isnan(rel) and rel <= -STEP_REL) else "lower in every later year, no abrupt step"
    elif l.min() > e.max():
        kind = "step up" if (not np.isnan(rel) and rel >= STEP_REL) else "higher in every later year, no abrupt step"
    else:
        kind = "overlaps earlier years"
    return dict(kind=kind, early_min=float(np.nanmin(e)), early_max=float(np.nanmax(e)), late_min=float(np.nanmin(l)),
                late_max=float(np.nanmax(l)), v_prev=v_prev, v_piv=v_piv, rel=rel)


def field_name(nm):
    return nm.replace("_", " ") + (" (not a model input)" if nm == "leukocyte_esterase" else "")


STEP_DEV = {nm: step_at(DEV_C, 2010, LATE_MIN, f"pos_{nm}") for _, nm in POS_FIELDS}
STEP_DEV["mean_sg"] = step_at(DEV_C, 2010, LATE_MIN, "mean_sg")
STEP_KN = {nm: step_at(KN_C, 2016, LATE_MIN, f"pos_{nm}") for _, nm in POS_FIELDS if nm != "leukocyte_esterase"}
STEP_KN["mean_sg"] = step_at(KN_C, 2016, LATE_MIN, "mean_sg")
log(f"\n  Change at {LATE_MIN} (step = every {LATE_MIN}+ year beyond every earlier year AND >= {STEP_REL:.0%} relative "
    f"change {LATE_MIN - 1} -> {LATE_MIN}):")
for lab, steps in (("development 2010-2021 vs 2022-2025", STEP_DEV), ("KNHANES 2016-2021 vs 2022-2024", STEP_KN)):
    log(f"    {lab}")
    for nm, s in steps.items():
        fmt = (lambda v: f"{v:.4f}") if nm == "mean_sg" else (lambda v: f"{100 * v:.2f}%")
        log(f"      {nm:26s} {s['kind']:42s} earlier years {fmt(s['early_min'])}-{fmt(s['early_max'])}, "
            f"{LATE_MIN}+ years {fmt(s['late_min'])}-{fmt(s['late_max'])}; {LATE_MIN - 1} {fmt(s['v_prev'])} -> "
            f"{LATE_MIN} {fmt(s['v_piv'])}" + (f" ({s['rel']:+.0%})" if not np.isnan(s["rel"]) else ""))

# ============================================================================ (f) interpretation of the temporal test
log("\n(f) Interpretation of the temporal test (2010-2021 -> 2022-2025): time plus possible measurement change")
log("  O:E in 2022+ = [observed prevalence 2022+ / training prevalence] x [training prevalence / mean predicted 2022+]")
log("  (first factor: change in outcome frequency; second: change in predicted risk, i.e. covariate shift through "
    "the model's inputs; in-sample mean predicted risk on the training persons shown for reference)")
ta = temporal[(temporal.design == "temporal") & (temporal.evaluation == "internal test")
              & ~temporal.model.str.contains("minus")]
interp = {}
for t, cfg in TARGETS.items():
    s = ta[ta.target == cfg["short"]]
    prev_tr = float(s.prevalence_train.iloc[0]); prev_te = float(s.prevalence.iloc[0]); ratio = prev_te / prev_tr
    log(f"\n  {cfg['short']} ({cfg['role']}): training prevalence {prev_tr:.4f}; 2022+ observed {prev_te:.4f} "
        f"(ratio {ratio:.3f})")
    for _, r in s.iterrows():
        shift = r.mean_pred / prev_tr
        log(f"    {r.model[:32]:32s} mean predicted train {r.mean_pred_train:.4f} -> 2022+ {r.mean_pred:.4f} "
            f"(x{shift:.3f} of training prevalence) | O:E {r.oe:.3f} = {ratio:.3f} x {prev_tr / r.mean_pred:.3f} | "
            f"slope {r.cal_slope:.3f}, AUROC {r.auroc:.4f}")
    m = s[s.model == MAIN_M].iloc[0]; d3 = s[s.model == DEMO_M].iloc[0]
    interp[cfg["short"]] = dict(ratio=ratio, prev_tr=prev_tr, prev_te=prev_te, oe=m.oe, shift=m.mean_pred / prev_tr,
                                shift3=d3.mean_pred / prev_tr, oe3=d3.oe, slope=m.cal_slope)
    for mn in (MAIN_M, DEMO_M):
        yb = LATE_BY_YEAR[(cfg["short"], mn)]
        log(f"    {mn[:32]:32s} by year observed / mean predicted: " + "; ".join(
            f"{int(k)} {v.observed:.4f}/{v.mean_pred:.4f}" for k, v in yb.iterrows()))

log("\n  Reading")
for t, cfg in TARGETS.items():
    k = interp[cfg["short"]]
    close = abs(k["shift"] - 1) < 0.05
    if close:
        log(f"  * {cfg['short']}: the temporal O:E drop ({k['oe']:.3f}, {MAIN_M}) matches the fall in prevalence "
            f"({k['prev_tr']:.4f} -> {k['prev_te']:.4f}, ratio {k['ratio']:.3f}); mean predicted risk in 2022+ stays at "
            f"the training prevalence (x{k['shift']:.3f}), so the miscalibration is a change in outcome frequency, "
            f"not in the inputs.")
    elif k["shift"] > 1 and k["ratio"] < 1:
        part = np.log(k["shift"]) / (np.log(k["shift"]) - np.log(k["ratio"]))
        demo_txt = (f"The age/sex/BMI-only model (LGBM-3F) also predicts above the training prevalence in 2022+ "
                    f"(x{k['shift3']:.3f}, roughly {np.log(k['shift3']) / np.log(k['shift']):.0%} of the 12F rise on "
                    f"the log scale; different models, so indicative only), so part of this shift is in age, sex and "
                    f"BMI and not only in the dipstick inputs." if k["shift3"] > 1.05 else
                    f"The age/sex/BMI-only model (LGBM-3F) does not (x{k['shift3']:.3f}), so the shift runs through the "
                    f"dipstick inputs.")
        log(f"  * {cfg['short']}: observed prevalence falls ({k['prev_tr']:.4f} -> {k['prev_te']:.4f}, ratio "
            f"{k['ratio']:.3f}) while mean predicted risk in 2022+ rises above the training prevalence "
            f"(x{k['shift']:.3f}, {MAIN_M}); O:E {k['oe']:.3f} = {k['ratio']:.3f} x {1 / k['shift']:.3f}, so about "
            f"{part:.0%} of the log O:E reflects covariate shift rather than the fall in prevalence. {demo_txt}")
    else:
        log(f"  * {cfg['short']}: O:E {k['oe']:.3f}, prevalence ratio {k['ratio']:.3f}, mean predicted x{k['shift']:.3f} "
            f"of training prevalence (no simple pattern).")

_pd = dip[(dip.cohort == DEV_C) & (dip.period_type == "pooled")].set_index("period")
_e, _l = _pd.loc[f"2010-{EARLY_MAX}"], _pd.loc[f"{LATE_MIN}-2025"]
log(f"  * Demographics, development 2010-{EARLY_MAX} -> {LATE_MIN}-2025: mean age {_e.mean_age:.1f} -> {_l.mean_age:.1f}, "
    f"male {100 * _e.male_frac:.1f}% -> {100 * _l.male_frac:.1f}%, mean BMI {_e.mean_bmi:.2f} -> {_l.mean_bmi:.2f} "
    f"(the inputs of the LGBM-3F model).")
steps_down = [nm for nm in ("bilirubin", "ketone") if STEP_DEV[nm]["kind"] == "step down"]
other_steps = [f"{field_name(nm)} {STEP_DEV[nm]['kind']} ({100 * STEP_DEV[nm]['v_prev']:.1f}% -> "
               f"{100 * STEP_DEV[nm]['v_piv']:.1f}%)" for nm in STEP_DEV
               if nm not in ("bilirubin", "ketone") and STEP_DEV[nm]["kind"] in ("step down", "step up")]
if steps_down:
    txt = " and ".join(f"{nm} positivity steps down in {LATE_MIN} ({100 * STEP_DEV[nm]['v_prev']:.1f}% in "
                       f"{LATE_MIN - 1} -> {100 * STEP_DEV[nm]['v_piv']:.1f}% in {LATE_MIN}; every {LATE_MIN}+ year "
                       f"{100 * STEP_DEV[nm]['late_max']:.1f}% or less vs at least {100 * STEP_DEV[nm]['early_min']:.1f}% "
                       f"in every year 2010-{LATE_MIN - 1})" for nm in steps_down)
    shifted = [s for s, k in interp.items() if k["shift"] > 1.05 and k["ratio"] < 1]
    log(f"  * Measurement: in the development cohort {txt}" + (f"; also {', '.join(other_steps)}" if other_steps else "")
        + "." + (f" The covariate shift at {' / '.join(shifted)} coincides with this step change in dipstick "
                 f"positivity (a coincidence in time; the step is not shown to cause the shift)." if shifted else ""))
else:
    log(f"  * Measurement: the table does not show a step down in bilirubin or ketone positivity at {LATE_MIN} "
        f"(bilirubin: {STEP_DEV['bilirubin']['kind']}; ketone: {STEP_DEV['ketone']['kind']})"
        + (f"; steps elsewhere: {', '.join(other_steps)}" if other_steps else "") + ".")
trend = [f"{field_name(nm)} ({STEP_DEV[nm]['kind']}; {100 * STEP_DEV[nm]['early_min']:.1f}-"
         f"{100 * STEP_DEV[nm]['early_max']:.1f}% before, {100 * STEP_DEV[nm]['late_min']:.1f}-"
         f"{100 * STEP_DEV[nm]['late_max']:.1f}% from {LATE_MIN})"
         for nm in STEP_DEV if "no abrupt step" in STEP_DEV[nm]["kind"]]
if trend:
    log(f"    Gradual changes without an abrupt {LATE_MIN} step: {', '.join(trend)}.")
kn_steps = [f"{nm} ({STEP_KN[nm]['kind']})" for nm in STEP_KN if STEP_KN[nm]["kind"] in ("step down", "step up")]
log(f"  * KNHANES: " + (f"step changes at {LATE_MIN}: {', '.join(kn_steps)}" if kn_steps else
                        f"no step change at {LATE_MIN} by the same rule")
    + "; bilirubin and ketone in KNHANES: " + "; ".join(
        f"{nm} {STEP_KN[nm]['kind']} (2016-2021 {100 * STEP_KN[nm]['early_min']:.2f}-{100 * STEP_KN[nm]['early_max']:.2f}%, "
        f"{LATE_MIN}+ {100 * STEP_KN[nm]['late_min']:.2f}-{100 * STEP_KN[nm]['late_max']:.2f}%)" for nm in ("bilirubin", "ketone"))
    + ".")
_ky = dip[(dip.cohort == KN_C) & (dip.period_type == "year")].set_index("period")
log("    KNHANES survey-year range (min-max, year of max): " + "; ".join(
    f"{nm} {100 * _ky[f'pos_{nm}'].min():.2f}-{100 * _ky[f'pos_{nm}'].max():.2f}% ({_ky[f'pos_{nm}'].idxmax()})"
    for nm in ("bilirubin", "ketone", "blood", "protein"))
    + ", i.e. KNHANES dipstick positivity also moves between survey years"
    + (f", but not as a step at {LATE_MIN}" if not kn_steps else "") + "; the external validation pools 2016-2024.")
_ny = dip[(dip.cohort == DEV_C) & (dip.period_type == "year")]
log(f"  * Nitrite (parsed correctly in dev_v2): development positivity {100 * _ny.pos_nitrite.min():.2f}-"
    f"{100 * _ny.pos_nitrite.max():.2f}% by year ({STEP_DEV['nitrite']['kind']} at {LATE_MIN}); KNHANES "
    f"{100 * _ky.pos_nitrite.min():.2f}-{100 * _ky.pos_nitrite.max():.2f}% by survey year.")
_ai = temporal[(temporal.model == MAIN_M) & (temporal.evaluation == "internal test")]
auc_txt = "; ".join(
    f"{cfg['short']} {_ai[(_ai.target == cfg['short']) & (_ai.design == 'temporal')].auroc.iloc[0]:.4f} vs "
    f"{_ai[(_ai.target == cfg['short']) & (_ai.design == 'random (reference)')].auroc.iloc[0]:.4f}"
    for cfg in TARGETS.values())
log(f"  * Framing: the {LATE_MIN}+ test set differs from training in calendar time AND in the distribution of several "
    f"dipstick results (a possible change in strip, analyser or result coding that the data do not document). The "
    f"temporal test therefore measures time plus possible measurement change, not temporal drift alone. "
    f"{MAIN_M} internal AUROC, temporal vs random-split reference: {auc_txt}; calibration-in-the-large moved with "
    f"the fall in prevalence and, at the higher thresholds, with the input shift.")

log(f"\nDone in {(time.time() - T0) / 60:.1f} min")
with open(os.path.join(HERE, "calc_temporal_design_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
