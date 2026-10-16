"""Calibration of the retrained uric-acid ensembles (bestmodel_v5) and two parametric alternatives.

Reviewer AW #16: isotonic calibration on few positives may overfit; give a parametric alternative
and a calibration slope. Our own check: the submitted calibration figure showed external data only
and understated over-prediction.

MODELS AND COHORTS (decision of 2026-10-16): the retrained ensembles (cohort.load_model, bestmodel_v5): 12 inputs
(cohort.FMODEL = 9 dipstick fields, age, sex, BMI; leukocyte esterase is not an input), Optuna-tuned LightGBM +
XGBoost + CatBoost, soft voting, isotonic CalibratedClassifierCV(cv=5), trained on the training persons of the
corrected development cohort dev_v2 (cohort.main_cohort: adults, one record per person, ONE person-level 80/20 split
shared by the three thresholds). "internal" = the internal test persons d.iloc[te] (preds_internal_clean.csv, written
by calc_main.py); "external" = KNHANES adults aged >= 19 with the dipstick inputs coded by the label-matched mapping
X1 (preds_external_corrected.csv). The models are loaded read-only; they are not refitted or re-calibrated here.

(a) calibration.csv        retrained isotonic models: calibration intercept (offset MLE), slope, O:E,
                           Brier, scaled Brier (all with bootstrap 95% CI), and the exact-0 prediction counts;
                           internal test persons and KNHANES adults.
(b) calibration_curve.csv  quantile-bin curves (common.decile_curve), 10 bins for every target and
                           also 5 bins for >=9.0 and >=10.0; isotonic model (and, for comparison, the two
                           parametric alternatives from (d)); Wilson 95% CI for the observed risk. Ties in the
                           isotonic output can merge quantile bins (bins_realized < bins_requested).
(c) recalibration.csv      common.recalibrate on external (100 stratified half-splits): intercept-only
                           and logistic recalibration, scaled Brier before/after, O:E after
                           (plus O:E before and after logistic recalibration on the same half-splits).
(d) platt_vs_isotonic.csv  method 'sigmoid_refit': sklearn's sigmoid (Platt) calibrator, fitted as
                           CalibratedClassifierCV(method='sigmoid') fits it (sklearn.calibration.
                           _fit_calibrator on the fold learner's soft-vote probability; checked against
                           CalibratedClassifierCV(FrozenEstimator(fold learner), method='sigmoid')), on each of the
                           retrained isotonic model's own 5 calibration folds (cohort.cal_folds on cohort.train_xy:
                           StratifiedKFold(5) without shuffling on the training persons in the order they were
                           passed, verified against the saved isotonic calibrators), and applied to that fold's
                           learner; outputs averaged over the 5 folds as CalibratedClassifierCV(ensemble=True) does.
                           Every training person is an adult, so every record of each calibration fold is used.
                           sklearn applies the sigmoid to the soft-vote PROBABILITY (VotingClassifier
                           has no decision_function), so its output cannot fall below expit(-b).
                           Supplementary method 'logistic_logodds': the same fold learners and calibration folds,
                           with an unpenalised logistic calibrator on the LOG-ODDS of the soft-vote
                           probability (logistic/Cox calibration), averaged over the 5 folds as
                           CalibratedClassifierCV(ensemble=True) does. Both are evaluated on the internal test
                           persons and on KNHANES adults against the isotonic model (paired AUROC and Brier
                           differences).
Method key of the isotonic model: the rows of the retrained isotonic models carry the key 'isotonic_deployed' in
calibration_curve.csv (model), platt_vs_isotonic.csv and preds_calibration_methods.csv (method). The key name is
kept because make_figures.py, make_supplement.py and calc_followup.py select rows by it.

The sigmoid models (the retrained fold learners with the sigmoid calibrators) are written to new files
(refit_sigmoid_<target>.joblib) in this folder, and all predictions to preds_calibration_methods.csv.
Bootstrap: 1000 resamples with replacement at full size, percentile CI, seed 42, single-class
resamples skipped (identical to common.boot_ci). All evaluations are unweighted (KNHANES survey
weights not applied).

Calibration slopes are exact maximum-likelihood estimates (common.fit_logistic_1d, Newton-Raphson);
the point estimates are checked against statsmodels Logit and the log reports the largest difference.
The script's own logistic calibrators use sklearn newton-cg (tol 1e-10) and are checked against
common.fit_logistic_1d. p = 0 is clipped to 1e-9 (logit -20.7) by common.logit.
Sanity checks: the isotonic predictions equal the stored prediction files (rows, labels, person ids), and the
isotonic AUROC (with its bootstrap CI), Brier, O:E, calibration intercept and slope equal main_performance.csv.

The log ends with an interpretation block written from this run's numbers: (1) the external
over-prediction as a calibration-in-the-large difference between settings (O:E, intercept, slope), present with
every calibrator, and what an intercept update fitted on local data does; mean predicted vs observed risk
(case mix); possible contributors that these data cannot separate (different outcome risk at the same predictors;
measurement differences in the inputs); KNHANES O:E by nitrite status. (2) isotonic vs the log-odds logistic
calibrator on held-out data, and the exact-zero isotonic outputs as a limitation.
Run from the working folder:  /opt/anaconda3/bin/python calc_calibration.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import time, warnings
import numpy as np, pandas as pd, joblib

# cohort and common BEFORE lightgbm/xgboost: otherwise unpickling the saved XGBoost learners can deadlock
from cohort import (HERE, TARGETS, FMODEL, ID, main_cohort, train_xy, cal_folds, labels, load_model)
from common import (SEED, calib_stats, scaled_brier, recalibrate, decile_curve, boot_ci, auroc_ci,
                    paired_boot, delong_paired, logit, expit, fit_logistic_1d)

from threadpoolctl import threadpool_limits
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
import catboost as cb
import lightgbm, xgboost  # noqa: F401  (loaded before threadpool_limits so their OpenMP pools are capped)

warnings.filterwarnings("ignore", category=UserWarning)      # xgboost: 'use_label_encoder' not used
N_THREADS = 4
N_BOOT = 1000
ISO = "isotonic_deployed"      # method key of the retrained isotonic models (name kept for the downstream readers)
METHODS = (ISO, "sigmoid_refit", "logistic_logodds")
threadpool_limits(limits=N_THREADS)                          # OpenMP/BLAS pools (LightGBM, XGBoost)

# VotingClassifier calls CatBoost predict_proba(X) whose thread_count defaults to -1 (all cores);
# default it to the cap instead.
_cb_predict_proba = cb.CatBoostClassifier.predict_proba


def _cb_predict_proba_capped(self, X, ntree_start=0, ntree_end=0, thread_count=N_THREADS, verbose=None,
                             task_type="CPU"):
    return _cb_predict_proba(self, X, ntree_start, ntree_end, thread_count, verbose, task_type)


cb.CatBoostClassifier.predict_proba = _cb_predict_proba_capped

LOG = []


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True); LOG.append(s)


def cap_fitted(model):
    """Thread cap on the fitted LightGBM/XGBoost learners of a loaded model (in memory only).
    Fitted CatBoost models refuse set_params; their predict threads are capped by the wrapper above."""
    for cc in model.calibrated_classifiers_:
        for est in cc.estimator.estimators_:
            if not isinstance(est, cb.CatBoostClassifier):
                est.set_params(n_jobs=N_THREADS)
    return model


MLE_CHECK = []                 # (key, |slope - statsmodels slope|, max |score|) for every point estimate


def slope_check(key, y, p):
    """Exact-MLE slope (as in common.calib_stats) against statsmodels Logit, plus the score at the solution."""
    lp = logit(p); b0, b1 = fit_logistic_1d(lp, y)
    score = np.abs(np.column_stack([np.ones_like(lp), lp]).T @ (y - expit(b0 + b1 * lp))).max()
    ref = sm.Logit(y, sm.add_constant(lp)).fit(disp=0, method="newton", maxiter=200, tol=1e-12).params[1]
    MLE_CHECK.append((key, abs(b1 - ref), score))


def evaluate(y, p, key=None):
    """Point estimates and bootstrap CIs of the calibration metrics (one shared set of resamples)."""
    y = np.asarray(y); p = np.asarray(p, float)
    a, b, oe = calib_stats(y, p)
    nz = p > 0
    a_nz, b_nz, _ = calib_stats(y[nz], p[nz])
    slope_check(key, y, p)
    c10 = decile_curve(y, p, 10)                    # ECE over quantile bins (ties can merge bins)
    ece = float((c10['n'] * (c10['mean_pred'] - c10['observed']).abs()).sum() / c10['n'].sum())
    rng = np.random.RandomState(SEED); boot = []
    for _ in range(N_BOOT):                         # same resampling scheme as common.boot_ci
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        bi = calib_stats(y[i], p[i])
        boot.append((brier_score_loss(y[i], p[i]), scaled_brier(y[i], p[i]), bi[0], bi[1], bi[2]))
    boot = np.array(boot)
    assert np.isfinite(boot).all(), f"non-finite bootstrap calibration statistics for {key}"
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    ref = boot_ci(y, p, brier_score_loss, N_BOOT, SEED)
    assert np.allclose(ref, (lo[0], hi[0])), "bootstrap resamples differ from common.boot_ci"
    n_p0, ev_p0 = int((~nz).sum()), int(y[~nz].sum())
    p0_upper = float(proportion_confint(ev_p0, n_p0, alpha=0.05, method="beta")[1]) if n_p0 else np.nan
    out = dict(n=len(y), events=int(y.sum()), prevalence=y.mean(), mean_pred=p.mean(),
               oe=oe, oe_lo=lo[4], oe_hi=hi[4],
               intercept=a, intercept_lo=lo[2], intercept_hi=hi[2],
               slope=b, slope_lo=lo[3], slope_hi=hi[3],
               brier=brier_score_loss(y, p), brier_lo=lo[0], brier_hi=hi[0],
               scaled_brier=scaled_brier(y, p), scaled_brier_lo=lo[1], scaled_brier_hi=hi[1],
               n_boot_used=len(boot), ece_quantile10=ece,
               intercept_excl_p0=a_nz, slope_excl_p0=b_nz,
               n_distinct_p=int(np.unique(p).size), n_p0=n_p0, events_p0=ev_p0,
               p_min=p.min(), p_max=p.max(),
               share_p0=n_p0 / len(y),                         # share of records with predicted risk exactly 0
               share_events_p0=ev_p0 / max(int(y.sum()), 1),   # share of true cases given p = 0
               obs_risk_p0_upper95=p0_upper)                   # Clopper-Pearson upper bound, observed risk at p = 0
    return out


def curve_rows(tkey, cohort, model_name, y, p, bins):
    c = decile_curve(y, p, bins)
    c["events"] = np.rint(c["n"] * c["observed"]).astype(int)
    lo, hi = proportion_confint(c["events"], c["n"], alpha=0.05, method="wilson")
    c["observed_lo"], c["observed_hi"] = lo, hi
    c["pred_to_obs"] = c["mean_pred"] / c["observed"].replace(0, np.nan)
    c.insert(0, "bins_realized", len(c)); c.insert(0, "bins_requested", bins)
    c.insert(0, "model", model_name); c.insert(0, "cohort", cohort)
    c.insert(0, "threshold", TARGETS[tkey]["threshold"]); c.insert(0, "target", TARGETS[tkey]["short"])
    return c


def recal_supplement(y, p, n_rep=100, seed=SEED):
    """O:E before and after logistic recalibration on the SAME half-splits as common.recalibrate."""
    y = np.asarray(y); p = np.asarray(p, float); lp = logit(p); idx = np.arange(len(y)); rows = []
    for r in range(n_rep):
        tr, te = train_test_split(idx, test_size=0.5, random_state=seed + r, stratify=y)
        m = LogisticRegression(penalty=None, solver='newton-cg', tol=1e-10, max_iter=5000).fit(lp[tr].reshape(-1, 1), y[tr])
        p_s = m.predict_proba(lp[te].reshape(-1, 1))[:, 1]
        rows.append(dict(oe_raw=y[te].mean() / p[te].mean(), oe_after_logistic=y[te].mean() / p_s.mean(),
                         sbrier_raw_check=scaled_brier(y[te], p[te]), slope_newtoncg_check=float(m.coef_[0][0])))
    d = pd.DataFrame(rows); out = {}
    for c in d.columns:
        out[c] = d[c].mean(); out[c + "_lo"] = d[c].quantile(0.025); out[c + "_hi"] = d[c].quantile(0.975)
    return out


# ---------------------------------------------------------------------------------------------- data
t_start = time.time()
pint = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))
pext = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"))
mperf = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
MP_COHORT = {"internal": "internal test", "external": "KNHANES adults"}     # main_performance.csv cohort labels
d, tr, te, ext = main_cohort()
dte = d.iloc[te]
log(f"dev_v2 {len(d):,} persons (adults, one record per person); shared person-level split: training {len(tr):,}, "
    f"internal test {len(te):,} | KNHANES adults (X1 coding) {len(ext):,} | retrained models (bestmodel_v5), inputs "
    f"({len(FMODEL)}): {', '.join(FMODEL)}")

# ---------------------------------------------------------------- (d) sigmoid refits (run first: the
# sigmoid predictions also feed the comparison curves in (b))
P = {}                         # (target, cohort, method) -> (y, p)
fold_info = {}
for tkey, cfg in TARGETS.items():
    thr = cfg["threshold"]; t0 = time.time()
    iso = cap_fitted(load_model(tkey))
    assert list(iso.feature_names_in_) == FMODEL, "retrained model inputs differ from FMODEL"
    X_tr, y_tr = train_xy(thr)                                        # training persons, as passed to the recipe
    folds = cal_folds(iso, X_tr, y_tr)                                # the model's own 5 calibration folds (verified)
    X_in, y_in = dte[FMODEL], labels(dte["urate"], thr)
    X_ex, y_ex = ext[FMODEL], labels(ext["urate"], thr)

    # isotonic predictions must match the stored prediction files exactly (rows, labels, person ids)
    p_iso_in = iso.predict_proba(X_in)[:, 1]; p_iso_ex = iso.predict_proba(X_ex)[:, 1]
    s_in = pint[pint.target == tkey]; s_ex = pext[pext.target == tkey]
    assert len(s_in) == len(te) and np.array_equal(s_in.id.values, dte[ID].values), "internal rows differ"
    assert len(s_ex) == len(ext) and np.array_equal(s_ex.id.astype(str).values, ext["id"].astype(str).values)
    assert np.array_equal(s_in.y.values, y_in) and np.allclose(s_in.p.values, p_iso_in, rtol=0, atol=1e-12)
    assert np.array_equal(s_ex.y.values, y_ex) and np.allclose(s_ex.p.values, p_iso_ex, rtol=0, atol=1e-12)

    iso_steps = [int(np.unique(cc.calibrators[0].y_thresholds_).size) for cc in iso.calibrated_classifiers_]
    # sigmoid (Platt): an independent in-memory copy of the retrained model whose 5 fold calibrators are replaced
    # by sklearn's sigmoid calibrator fitted on the same calibration fold; the fold learners stay those of the model
    sig = cap_fitted(load_model(tkey)); sig.set_params(method="sigmoid")
    cal_n, cal_events, sig_vs_api = [], [], []
    # supplementary: logistic calibration on the log-odds, same fold learners and calibration folds
    p_lo_in = np.zeros(len(y_in)); p_lo_ex = np.zeros(len(y_ex)); lo_ab = []; lr_vs_nr = []
    for k, (_, cal_idx) in enumerate(folds):
        est = iso.calibrated_classifiers_[k].estimator
        f_cal, y_cal = est.predict_proba(X_tr.iloc[cal_idx])[:, 1], y_tr[cal_idx]
        cal_n.append(len(cal_idx)); cal_events.append(int(y_cal.sum()))
        sk = sig.calibrated_classifiers_[k].estimator                 # = est (same saved learner)
        sig.calibrated_classifiers_[k] = _fit_calibrator(sk, f_cal.reshape(-1, 1), y_cal, sig.classes_, "sigmoid")
        api = CalibratedClassifierCV(FrozenEstimator(sk), method="sigmoid").fit(X_tr.iloc[cal_idx], y_cal)
        c_new, c_api = sig.calibrated_classifiers_[k].calibrators[0], api.calibrated_classifiers_[0].calibrators[0]
        sig_vs_api.append(max(abs(c_new.a_ - c_api.a_), abs(c_new.b_ - c_api.b_)))
        lr = LogisticRegression(penalty=None, solver='newton-cg', tol=1e-10, max_iter=5000).fit(logit(f_cal).reshape(-1, 1), y_cal)
        lo_ab.append((float(lr.intercept_[0]), float(lr.coef_[0][0])))
        nr = fit_logistic_1d(logit(f_cal), y_cal)                     # exact MLE, same data
        lr_vs_nr.append(max(abs(lo_ab[-1][0] - nr[0]), abs(lo_ab[-1][1] - nr[1])))
        p_lo_in += lr.predict_proba(logit(est.predict_proba(X_in)[:, 1]).reshape(-1, 1))[:, 1] / len(folds)
        p_lo_ex += lr.predict_proba(logit(est.predict_proba(X_ex)[:, 1]).reshape(-1, 1))[:, 1] / len(folds)
    assert sum(cal_n) == len(y_tr) and sum(cal_events) == int(y_tr.sum())
    joblib.dump(sig, os.path.join(HERE, f"refit_sigmoid_{tkey}.joblib"))
    p_sig_in = sig.predict_proba(X_in)[:, 1]; p_sig_ex = sig.predict_proba(X_ex)[:, 1]
    # fold learners of the sigmoid model vs those of the isotonic model (the same saved learners: 0 expected)
    diffs = [np.abs(sig.calibrated_classifiers_[k].estimator.predict_proba(X_ex)[:, 1]
                    - iso.calibrated_classifiers_[k].estimator.predict_proba(X_ex)[:, 1]).max() for k in range(5)]
    sig_ab = [(float(cc.calibrators[0].a_), float(cc.calibrators[0].b_)) for cc in sig.calibrated_classifiers_]

    fold_info[tkey] = dict(train_n=len(y_tr), train_events=int(y_tr.sum()),
                           calfold_events_min=min(cal_events), calfold_events_max=max(cal_events),
                           iso_levels_per_fold_min=min(iso_steps), iso_levels_per_fold_mean=np.mean(iso_steps),
                           iso_levels_per_fold_max=max(iso_steps),
                           sigmoid_a_mean=np.mean([a for a, _ in sig_ab]), sigmoid_b_mean=np.mean([b for _, b in sig_ab]),
                           sigmoid_floor=float(np.mean([expit(-b) for _, b in sig_ab])),
                           logodds_intercept_mean=np.mean([a for a, _ in lo_ab]),
                           logodds_slope_mean=np.mean([b for _, b in lo_ab]),
                           base_maxabs_diff_vs_isotonic=max(diffs))
    P[(tkey, "internal", ISO)] = (y_in, p_iso_in)
    P[(tkey, "external", ISO)] = (y_ex, p_iso_ex)
    P[(tkey, "internal", "sigmoid_refit")] = (y_in, p_sig_in)
    P[(tkey, "external", "sigmoid_refit")] = (y_ex, p_sig_ex)
    P[(tkey, "internal", "logistic_logodds")] = (y_in, p_lo_in)
    P[(tkey, "external", "logistic_logodds")] = (y_ex, p_lo_ex)
    log(f"[d] {cfg['short']}: training persons {len(y_tr):,} (events {int(y_tr.sum()):,}); internal test persons "
        f"{len(y_in):,} (events {int(y_in.sum()):,}); KNHANES adults {len(y_ex):,} (events {int(y_ex.sum()):,}); "
        f"persons per calibration fold {cal_n}, calibration-fold events {cal_events}; isotonic output levels per fold "
        f"{iso_steps}; sklearn sigmoid (a,b) per fold {[(round(a, 2), round(b, 2)) for a, b in sig_ab]} -> output floor "
        f"expit(-b) {fold_info[tkey]['sigmoid_floor']:.4f} (vs CalibratedClassifierCV(FrozenEstimator) max|diff| "
        f"{max(sig_vs_api):.1e}); log-odds calibrator (intercept, slope) per fold "
        f"{[(round(a, 2), round(b, 2)) for a, b in lo_ab]} (newton-cg vs exact MLE max|diff| {max(lr_vs_nr):.1e}); "
        f"fold-learner max|diff| sigmoid model vs isotonic model {max(diffs):.2e}; {time.time() - t0:.0f}s")
    assert max(sig_vs_api) < 1e-8 and max(diffs) == 0, "sigmoid calibrators or fold learners differ from the reference"

pd.concat([pd.DataFrame(dict(target=k[0], cohort=k[1], method=k[2], y=v[0], p=v[1])) for k, v in P.items()]
          ).to_csv(os.path.join(HERE, "preds_calibration_methods.csv"), index=False)

# ----------------------------------------------------------------------------- evaluate every (y, p)
E = {}
for k, (y, p) in P.items():
    t0 = time.time(); E[k] = evaluate(y, p, key=k)
    log(f"  evaluated {k} in {time.time() - t0:.0f}s")

# --------------------------------------------------------------------------- (a) calibration.csv
rows = []
for tkey, cfg in TARGETS.items():
    for cohort in ("internal", "external"):
        rows.append(dict(target=cfg["short"], threshold=cfg["threshold"], cohort=cohort, **E[(tkey, cohort, ISO)]))
cal = pd.DataFrame(rows)
cal.to_csv(os.path.join(HERE, "calibration.csv"), index=False)

# --------------------------------------------------------------------- (b) calibration_curve.csv
curves = []
for tkey, cfg in TARGETS.items():
    for cohort in ("internal", "external"):
        for method in METHODS:
            y, p = P[(tkey, cohort, method)]
            for bins in ((10,) if cfg["threshold"] == 7.0 else (10, 5)):
                curves.append(curve_rows(tkey, cohort, method, y, p, bins))
curve = pd.concat(curves, ignore_index=True)
curve.to_csv(os.path.join(HERE, "calibration_curve.csv"), index=False)

# ------------------------------------------------------------------------ (c) recalibration.csv
rows = []; RECAL_SLOPE_DIFF = {}
for tkey, cfg in TARGETS.items():
    y, p = P[(tkey, "external", ISO)]
    t0 = time.time()
    rc = recalibrate(y, p, n_rep=100, seed=SEED)
    sp = recal_supplement(y, p, n_rep=100, seed=SEED)
    assert abs(sp.pop("sbrier_raw_check") - rc["sbrier_raw"]) < 1e-12, "half-splits differ from common.recalibrate"
    RECAL_SLOPE_DIFF[cfg["short"]] = abs(sp.pop("slope_newtoncg_check") - rc["slope_fit"])   # newton-cg vs exact MLE
    for c in ("sbrier_raw_check_lo", "sbrier_raw_check_hi", "slope_newtoncg_check_lo", "slope_newtoncg_check_hi"):
        sp.pop(c)
    rows.append(dict(target=cfg["short"], threshold=cfg["threshold"], cohort="external", n=len(y),
                     events=int(y.sum()), **rc, **sp))
    log(f"  recalibration {cfg['short']} {time.time() - t0:.0f}s")
rec = pd.DataFrame(rows)
rec.to_csv(os.path.join(HERE, "recalibration.csv"), index=False)

# ---------------------------------------------------------------------- (d) platt_vs_isotonic.csv
rows = []
for tkey, cfg in TARGETS.items():
    for cohort in ("internal", "external"):
        y, p_iso = P[(tkey, cohort, ISO)]
        for method in METHODS:
            _, p = P[(tkey, cohort, method)]
            auc, auc_lo, auc_hi = auroc_ci(y, p, N_BOOT, SEED)
            r = dict(target=cfg["short"], threshold=cfg["threshold"], cohort=cohort, method=method,
                     auroc=auc, auroc_lo=auc_lo, auroc_hi=auc_hi, **E[(tkey, cohort, method)])
            if method != ISO:
                d_auc, p_dl = delong_paired(y, p, p_iso)
                r["d_auroc_vs_isotonic"] = d_auc
                r["d_auroc_lo"], r["d_auroc_hi"] = paired_boot(y, p, p_iso, N_BOOT, SEED)
                r["d_auroc_delong_p"] = p_dl
                rng = np.random.RandomState(SEED); db = []
                for _ in range(N_BOOT):
                    i = rng.randint(0, len(y), len(y))
                    if y[i].min() == y[i].max(): continue
                    db.append(brier_score_loss(y[i], p[i]) - brier_score_loss(y[i], p_iso[i]))
                r["d_brier_vs_isotonic"] = brier_score_loss(y, p) - brier_score_loss(y, p_iso)
                r["d_brier_lo"], r["d_brier_hi"] = np.percentile(db, 2.5), np.percentile(db, 97.5)
            r.update(fold_info[tkey])
            rows.append(r)
pvi = pd.DataFrame(rows)
pvi.to_csv(os.path.join(HERE, "platt_vs_isotonic.csv"), index=False)

# ------------------------------------------------- sanity: isotonic results = main_performance.csv (calc_main.py)
MP_DIFF = 0.0
for tkey, cfg in TARGETS.items():
    for cohort in ("internal", "external"):
        m = mperf[(mperf.target == cfg["short"]) & (mperf.cohort == MP_COHORT[cohort])]
        assert len(m) == 1, (cfg["short"], cohort)
        m = m.iloc[0]
        r = pvi[(pvi.target == cfg["short"]) & (pvi.cohort == cohort) & (pvi.method == ISO)].iloc[0]
        assert int(r.n) == int(m.n) and int(r.events) == int(m.events), (cfg["short"], cohort, "n/events")
        for a_, b_ in (("auroc", "auroc"), ("auroc_lo", "auroc_lo"), ("auroc_hi", "auroc_hi")):
            assert round(float(r[a_]), 4) == round(float(m[b_]), 4), (cfg["short"], cohort, a_, r[a_], m[b_])
        pairs = (("auroc", "auroc"), ("auroc_lo", "auroc_lo"), ("auroc_hi", "auroc_hi"), ("brier", "brier"),
                 ("oe", "oe"), ("intercept", "cal_intercept"), ("slope", "cal_slope"), ("mean_pred", "mean_pred"))
        for a_, b_ in pairs:
            MP_DIFF = max(MP_DIFF, abs(float(r[a_]) - float(m[b_])))
assert MP_DIFF < 1e-9, f"isotonic results differ from main_performance.csv (max |diff| {MP_DIFF:.1e})"

# ------------------------------------------------------------------------------------------ log
f = lambda v, lo, hi, d=3: f"{v:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"
log("\n(a) Retrained isotonic models: calibration")
for _, r in cal.iterrows():
    log(f"  {r.target:>6} {r.cohort:8} n={r.n:,} events={r.events:,} prev={r.prevalence:.4f} mean_pred={r.mean_pred:.4f} "
        f"O:E {f(r.oe, r.oe_lo, r.oe_hi, 2)} | intercept {f(r.intercept, r.intercept_lo, r.intercept_hi, 2)} | "
        f"slope {f(r.slope, r.slope_lo, r.slope_hi, 2)} [excl p=0: {r.slope_excl_p0:.2f}] | "
        f"Brier {f(r.brier, r.brier_lo, r.brier_hi, 4)} | scaled Brier {f(r.scaled_brier, r.scaled_brier_lo, r.scaled_brier_hi)} | "
        f"ECE10 {r.ece_quantile10:.4f} | distinct p {r.n_distinct_p:,}, p=0 {r.n_p0:,} ({r.share_p0:.1%}; "
        f"{r.events_p0} of {r.events:,} events)")
log("\n(b) Highest-risk bin, isotonic (mean predicted vs observed)")
top = curve[curve.model == ISO].groupby(["target", "cohort", "bins_requested"], sort=False).tail(1)
for _, r in top.iterrows():
    log(f"  {r.target:>6} {r.cohort:8} {r.bins_requested:>2} bins (realized {r.bins_realized}): n={r.n:,} "
        f"pred {r.mean_pred:.3f} vs obs {r.observed:.3f} ({r.observed_lo:.3f}-{r.observed_hi:.3f}); pred/obs {r.pred_to_obs:.2f}")
log("\n(c) External recalibration (100 half-splits; mean and 2.5-97.5th percentiles)")
for _, r in rec.iterrows():
    log(f"  {r.target:>6} intercept fit {f(r.intercept_fit, r.intercept_fit_lo, r.intercept_fit_hi, 2)}; "
        f"slope fit {f(r.slope_fit, r.slope_fit_lo, r.slope_fit_hi, 2)}; scaled Brier raw "
        f"{f(r.sbrier_raw, r.sbrier_raw_lo, r.sbrier_raw_hi)} -> intercept {f(r.sbrier_intercept, r.sbrier_intercept_lo, r.sbrier_intercept_hi)}"
        f" -> logistic {f(r.sbrier_slope, r.sbrier_slope_lo, r.sbrier_slope_hi)}; O:E raw {r.oe_raw:.2f} -> after intercept "
        f"{f(r.oe_after_intercept, r.oe_after_intercept_lo, r.oe_after_intercept_hi, 2)} -> after logistic {r.oe_after_logistic:.2f}")
log("\n(d) Sigmoid (Platt, sklearn) refit and logistic log-odds calibration vs isotonic (same fold learners and folds)")
for _, r in pvi.iterrows():
    extra = (f" | dAUROC {r.d_auroc_vs_isotonic:+.4f} ({r.d_auroc_lo:+.4f} to {r.d_auroc_hi:+.4f}), DeLong p={r.d_auroc_delong_p:.3f};"
             f" dBrier {r.d_brier_vs_isotonic:+.5f} ({r.d_brier_lo:+.5f} to {r.d_brier_hi:+.5f})") if r.method != ISO else ""
    log(f"  {r.target:>6} {r.cohort:8} {r.method:17} AUROC {f(r.auroc, r.auroc_lo, r.auroc_hi, 4)} | int {r.intercept:+.2f} "
        f"slope {r.slope:.2f} O:E {r.oe:.2f} Brier {r.brier:.5f} sBrier {r.scaled_brier:.3f} ECE10 {r.ece_quantile10:.4f} | "
        f"distinct p {r.n_distinct_p:,} p=0 {r.n_p0:,} p_min {r.p_min:.4f}{extra}")

# ------------------------------------------------------------ (e) nitrite in the two cohorts (log only)
nit_dev = d["ua_nitrite"].to_numpy(float); nit_ext = ext["ua_nitrite"].to_numpy(float)
log("\n(e) Nitrite (dev_v2: parsed from the raw first-visit text; KNHANES: X1 coding) and KNHANES calibration by "
    "nitrite status (isotonic)")
log(f"  nitrite = 1: development persons {int((nit_dev == 1).sum()):,} of {len(d):,} ({(nit_dev == 1).mean():.2%}; "
    f"female {(d.loc[d['ua_nitrite'] == 1, 'male'] == 0).mean():.1%}); KNHANES adults {int((nit_ext == 1).sum()):,} of "
    f"{len(ext):,} ({(nit_ext == 1).mean():.2%}; female {(ext.loc[ext['ua_nitrite'] == 1, 'male'] == 0).mean():.1%})")
NIT = {}
for tkey, cfg in TARGETS.items():
    y, p = P[(tkey, "external", ISO)]
    s = {lab: dict(n=int(m.sum()), events=int(y[m].sum()), prev=y[m].mean(), mean_pred=p[m].mean(),
                   oe=y[m].mean() / p[m].mean()) for lab, m in (("pos", nit_ext == 1), ("neg", nit_ext == 0))}
    NIT[cfg["short"]] = s
    log(f"  {cfg['short']:>6} KNHANES isotonic: nitrite+ n={s['pos']['n']:,} events={s['pos']['events']} prev "
        f"{s['pos']['prev']:.4f} mean_pred {s['pos']['mean_pred']:.4f} O:E {s['pos']['oe']:.2f}; nitrite- "
        f"n={s['neg']['n']:,} events={s['neg']['events']:,} prev {s['neg']['prev']:.4f} mean_pred "
        f"{s['neg']['mean_pred']:.4f} O:E {s['neg']['oe']:.2f}")

# ---------------------------------------------------------------------------- numerical checks
log("\nNumerical checks")
worst = max(MLE_CHECK, key=lambda t: t[1])
log(f"  calibration slope (exact MLE, common.fit_logistic_1d) vs statsmodels Logit, {len(MLE_CHECK)} point estimates: "
    f"max |diff| {worst[1]:.1e} ({worst[0]}); max |score| at the solution {max(t[2] for t in MLE_CHECK):.1e}")
log(f"  O:E after logistic recalibration uses sklearn newton-cg: mean slope vs common.recalibrate exact MLE max |diff| "
    f"{max(RECAL_SLOPE_DIFF.values()):.1e}; per-fold log-odds calibrators: see the [d] lines above")
log(f"  isotonic predictions = preds_internal_clean.csv / preds_external_corrected.csv (rows, labels, ids); isotonic "
    f"n, events, AUROC (95% CI), Brier, O:E, intercept, slope, mean predicted risk = main_performance.csv: max |diff| "
    f"{MP_DIFF:.1e}")

# ------------------------------------------------------------------------------- interpretation
ci = lambda e, k, d=2: f"{e[k]:.{d}f} ({e[k + '_lo']:.{d}f} to {e[k + '_hi']:.{d}f})"
ext_all = [E[(t, "external", m)] for t in TARGETS for m in METHODS]
PR, MR = {}, {}
log("\nInterpretation (written from the numbers of this run)")
log("(1) External over-prediction: a calibration-in-the-large difference between the settings")
for tkey, cfg in TARGETS.items():
    t = cfg["short"]; ri = E[(tkey, "internal", ISO)]; rx = E[(tkey, "external", ISO)]
    prev_tr = fold_info[tkey]["train_events"] / fold_info[tkey]["train_n"]
    PR[t] = rx["prevalence"] / ri["prevalence"]; MR[t] = rx["mean_pred"] / ri["mean_pred"]
    log(f"  {t:>6} observed prevalence KNHANES {rx['prevalence']:.4f} vs internal test {ri['prevalence']:.4f} "
        f"(external/development ratio {PR[t]:.3f}; vs development training persons {prev_tr:.4f}: {rx['prevalence'] / prev_tr:.3f}); "
        f"mean predicted risk KNHANES {rx['mean_pred']:.4f} vs internal {ri['mean_pred']:.4f} (ratio {MR[t]:.3f})")
    for m in METHODS:
        e = E[(tkey, "external", m)]
        log(f"         {m:17} external O:E {ci(e, 'oe')}; intercept {ci(e, 'intercept')}; slope {ci(e, 'slope')}")
    rr = rec[rec.target == t].iloc[0]
    log(f"         intercept update fitted on one KNHANES half, evaluated on the other (100 splits): offset "
        f"{rr.intercept_fit:+.2f}; O:E {rr.oe_raw:.2f} -> {rr.oe_after_intercept:.2f}; scaled Brier {rr.sbrier_raw:.3f} -> "
        f"{rr.sbrier_intercept:.3f} (adding a slope: {rr.sbrier_slope:.3f})")
oes = [e["oe"] for e in ext_all]; ints = [e["intercept"] for e in ext_all]
# slope statement: the isotonic model and the log-odds calibrator; sklearn's sigmoid refit is reported apart, because it
# calibrates the probability (not the log-odds) and cannot fall below expit(-b), which bends its slope on logit(p)
sls = [E[(t, "external", m)]["slope"] for t in TARGETS for m in (ISO, "logistic_logodds")]
sls_sig = [E[(t, "external", "sigmoid_refit")]["slope"] for t in TARGETS]
rec_oe = rec["oe_after_intercept"]
slope_gain = (rec["sbrier_slope"] - rec["sbrier_intercept"]).to_numpy()
slope_txt = ("adding a slope gives no further gain (scaled Brier change "
             f"{slope_gain.min():+.3f} to {slope_gain.max():+.3f})" if slope_gain.max() < 0.005 else
             f"adding a slope changes the scaled Brier by {slope_gain.min():+.3f} to {slope_gain.max():+.3f}")
log(f"  Reading: in KNHANES every calibrator over-predicts (external O:E {min(oes):.2f}-{max(oes):.2f}, calibration "
    f"intercept {min(ints):+.2f} to {max(ints):+.2f} across thresholds and calibrators), while the calibration slope of "
    f"the isotonic model and of the log-odds calibrator is {min(sls):.2f}-{max(sls):.2f} (sklearn's sigmoid refit, which "
    f"calibrates the probability and cannot fall below expit(-b): {min(sls_sig):.2f}-{max(sls_sig):.2f}): the ranking and "
    f"spread of the predicted log-odds transfer, the overall level does not. An intercept update fitted on local KNHANES "
    f"data removes it (O:E after update {rec_oe.min():.2f}-{rec_oe.max():.2f}); {slope_txt}. The external/development "
    f"prevalence ratios are " + ", ".join(f"{t} {PR[t]:.3f}" for t in PR) + " (KNHANES vs internal test set).")
if any(e["oe_hi"] >= 1 or e["intercept_hi"] >= 0 for e in ext_all):
    log("  WARNING: not every calibrator over-predicts in KNHANES with a CI excluding 1; revisit the statement above.")
if min(sls) < 0.85 or max(sls) > 1.15:
    log("  WARNING: an external calibration slope of the isotonic model or the log-odds calibrator lies outside "
        "0.85-1.15; revisit 'the ranking and spread transfer'.")
lower = [t for t in MR if MR[t] < 0.9]
log("  Case mix: mean predicted risk KNHANES/internal " + ", ".join(f"{t} {MR[t]:.2f}" for t in MR)
    + "; observed risk KNHANES/internal " + ", ".join(f"{t} {PR[t]:.2f}" for t in PR) + ". "
    + (f"At {', '.join(lower)} the predicted risk is also lower in KNHANES, so the predictor distribution accounts for "
       f"part of the lower observed risk; " if lower else "")
    + "the observed risk is lower than the predicted risk at every threshold (O:E < 1). If KNHANES differed only in the "
      "distribution of the predictors, predicted and observed risk would fall together and O:E would stay near 1; "
      "instead the observed risk at a given predicted risk is lower.")
if any(PR[t] >= MR[t] for t in MR):
    log("  WARNING: observed risk does not fall further than predicted risk for "
        + ", ".join(t for t in MR if PR[t] >= MR[t]) + "; revisit the case-mix statement.")
ubg_dev = (d["ua_urobilinogen"] == 0.5).mean(); ubg_ext = (ext["ua_urobilinogen"] == 0.5).mean()
nit_neg = ", ".join(f"{t} {NIT[t]['neg']['oe']:.2f} (all {E[(k, 'external', ISO)]['oe']:.2f})"
                    for k, t in ((k, c["short"]) for k, c in TARGETS.items()))
log(f"  Possible contributors, which these data cannot separate: (i) a different outcome risk at the same predictor "
    f"values in the two settings; (ii) measurement differences in the inputs (different dipstick products and reading "
    f"procedures). The KNHANES dipstick codes are mapped to the development grades with the same label (mapping X1; "
    f"urobilinogen 'normal' = 0.5 in {ubg_dev:.1%} of development persons and {ubg_ext:.1%} of KNHANES adults), and "
    f"leukocyte esterase, which KNHANES does not measure, is not an input. Nitrite = 1 in {(nit_dev == 1).mean():.2%} of "
    f"development persons and {(nit_ext == 1).mean():.2%} of KNHANES adults; KNHANES O:E among nitrite-negative adults "
    f"{nit_neg}. Input-coding sensitivity analyses are in calc_shift_encoding_log.txt.")
log("(2) Isotonic vs parametric calibration (held-out data; same fold learners and calibration folds)")
for tkey, cfg in TARGETS.items():
    for cohort in ("internal", "external"):
        r = pvi[(pvi.target == cfg["short"]) & (pvi.cohort == cohort) & (pvi.method == "logistic_logodds")].iloc[0]
        log(f"  {cfg['short']:>6} {cohort:8} log-odds logistic minus isotonic: dAUROC {r.d_auroc_vs_isotonic:+.4f} "
            f"({r.d_auroc_lo:+.4f} to {r.d_auroc_hi:+.4f}); dBrier {r.d_brier_vs_isotonic:+.5f} ({r.d_brier_lo:+.5f} to "
            f"{r.d_brier_hi:+.5f}); Brier isotonic {E[(tkey, cohort, ISO)]['brier']:.5f} vs {r.brier:.5f}")
lg = pvi[pvi.method == "logistic_logodds"]
max_dauc, max_dbr = float(lg.d_auroc_vs_isotonic.abs().max()), float(lg.d_brier_vs_isotonic.abs().max())
sig_in = pvi[(pvi.cohort == "internal") & (pvi.method == "sigmoid_refit")]
same = max_dauc < 0.002 and max_dbr < 0.0005
log(("  Reading: isotonic and the log-odds logistic calibrator give practically identical discrimination and Brier "
     if same else "  Reading: isotonic and the log-odds logistic calibrator differ ")
    + f"on held-out internal and external data (|dAUROC| <= {max_dauc:.4f}, |dBrier| <= {max_dbr:.5f}). "
    + ("This shows no held-out loss from isotonic; it is not evidence either way on whether isotonic regression "
       "overfits its calibration folds. " if same else "")
    + "(sklearn's sigmoid refit calibrates the probability, not the log-odds, and has an output floor expit(-b); "
      "internal dBrier vs isotonic "
    + ", ".join(f"{r.target} {r.d_brier_vs_isotonic:+.5f} ({r.d_brier_lo:+.5f} to {r.d_brier_hi:+.5f})"
                for _, r in sig_in.iterrows()) + "; see (d).)")
log("  Limitation, exact-zero isotonic outputs (predicted risk exactly 0):")
for tkey, cfg in TARGETS.items():
    for cohort in ("internal", "external"):
        e = E[(tkey, cohort, ISO)]; el = E[(tkey, cohort, "logistic_logodds")]
        log(f"  {cfg['short']:>6} {cohort:8} p=0 for {e['n_p0']:,} of {e['n']:,} ({e['share_p0']:.1%}); true cases at p=0 "
            f"{e['events_p0']} of {e['events']:,} ({e['share_events_p0']:.1%}); observed risk at p=0 "
            + (f"{e['events_p0'] / e['n_p0']:.4f} (95% upper {e['obs_risk_p0_upper95']:.4f}); " if e['n_p0'] else "n/a; ")
            + f"slope {e['slope']:.2f} vs {e['slope_excl_p0']:.2f} excluding p=0 | log-odds calibrator p=0 {el['n_p0']}, "
            f"min p {el['p_min']:.4f}")
log("  A predicted risk of exactly 0 is not a valid individual probability: a case given p=0 is missed at every risk "
    "threshold, and log-loss is unbounded. Slopes on logit(p) clip p=0 to 1e-9 (logit -20.7); slopes excluding p=0 "
    "are in calibration.csv (slope_excl_p0).")
log(f"\nTotal runtime {(time.time() - t_start) / 60:.1f} min")
with open(os.path.join(HERE, "calc_calibration_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
