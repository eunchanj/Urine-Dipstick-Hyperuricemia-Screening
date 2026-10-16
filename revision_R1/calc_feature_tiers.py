"""Feature tiers, per-set tuning and the retrained ensemble vs simpler models (Reviewer AW #5 and #18).

Retrained models (decision of 2026-10-16): bestmodel_v5 (cohort.load_model), 12 inputs (FMODEL: the 9 dipstick fields
without leukocyte esterase, age, male, BMI), Optuna-tuned LightGBM + XGBoost + CatBoost soft vote wrapped in an isotonic
CalibratedClassifierCV(cv=5), fitted on the training persons of the corrected development cohort (cohort.main_cohort:
dev_v2, adults, one record per person; ONE person-level 80/20 split shared by the three thresholds: training persons
d.iloc[tr], internal test persons d.iloc[te]). External = KNHANES adults aged >= 19 (ext), dipstick fields coded by the
label-matched mapping X1 (the primary coding). Every refit below is fitted on the training persons d.iloc[tr] and
evaluated on the internal test persons d.iloc[te] (setting "internal") and on KNHANES coded by X1 (setting "external").

(a) Equal-budget nested ablation, 5 input sets x 4 model classes.
    Tiers: 2F (age, male), 3F (age, male, BMI), 9F (the 9 dipstick fields), 11F (dipstick + age + male), 12F (the 12
    model inputs, in FMODEL order).
    Model classes: LightGBM with one fixed configuration (LGBM_FIXED); main-effects penalised logistic regression
    (StandardScaler + LogisticRegression C=1); the same LR with cubic B-splines for age and BMI (5 quantile knots, linear
    extrapolation) plus an age x male term (the tiers that contain age: 2F, 3F, 11F, 12F; 9F has no age); LightGBM tuned
    separately for each input set (the GRID-tuned models of (b)).
    AUROC, AP (AUPRC as sklearn average precision, step-wise) and scaled Brier with bootstrap CIs, internally and
    externally.
    Contrasts (AUROC difference, paired bootstrap CI, paired DeLong p), per model class:
        12F-3F (dipstick added to age+sex+BMI), 12F-11F (BMI added to dipstick+age+sex), 11F-9F (age+sex added to
        dipstick), 11F-2F (dipstick added to age+sex), 3F-2F (BMI added to age+sex);
    and LightGBM (fixed) minus LR and minus the spline LR within each tier.
    -> feature_tiers.csv
(b) Per-set tuning of LightGBM for every tier: best GRID configuration by 3-fold stratified CV AUROC on the training
    persons (n_estimators 300), refitted on all training persons (n_estimators 300), compared with the fixed
    configuration of (a) (tuning_gain; for the contrasts increment_change_vs_fixed).
    -> tuned_per_set.csv (row_type cv_grid, tuned, contrast; the tuned and contrast metrics are the same computations
       as the "LightGBM (GRID-tuned)" rows of feature_tiers.csv)
(c) Retrained ensemble vs simpler 12-input models fitted on the same training persons and evaluated on the same persons:
    logistic regression, spline LR, LightGBM (fixed) and LightGBM (GRID-tuned), which are the 12F models of (a) and (b),
    and the retrained ensemble's own LightGBM member configuration (the 'lgb' learner of clone(load_model(t).estimator),
    Optuna-tuned) wrapped in the same isotonic CalibratedClassifierCV on the retrained model's own 5 calibration folds
    (cohort.cal_folds, verified against the saved isotonic calibrators), fitted on cohort.train_xy(threshold). Its 5 fold
    learners reproduce the ensemble's LightGBM members exactly (asserted on the internal test persons), so "Retrained
    ensemble minus this arm" isolates what adding XGBoost and CatBoost contributes. The retrained ensemble's predictions
    are read from preds_internal_clean.csv / preds_external_corrected.csv (calc_main.py; alignment asserted, spot-checked
    against the model, AUROC asserted equal to main_performance.csv). The cv_grid rows behind the GRID-tuned arm (tier
    12F of (b)) are repeated in this file.
    -> ensemble_vs_simple.csv
(d) Coding dependence (diagnostic, point estimates): every external arm with dipstick inputs (tiers 9F, 11F and 12F of
    (a); the retrained ensemble and the v5-config arm of (c)) scored with the KNHANES dipstick fields coded by
        X1  label-matched mapping (primary; cohort.knhanes_x1): glucose, protein, ketone, blood and bilirubin
            0, 0.5, 1, 2, 3, 3 for -, +/-, +, ++, +++, ++++; urobilinogen normal and trace -> 0.5 (the development normal
            code), + -> 1, ++ -> 2, >= +++ -> 3;
        M1  the coding of the submission (calc_shift_encoding.py MAPPINGS["M1"], from 05_src/prepare_knhanes_uric.py):
            bilirubin 0, 1, 2, 3, 3, 3; urobilinogen 0, 0.2, 0.5, 1, 2, 3;
        M2  the uniform printed map 0, 0.5, 1, 2, 3, 3 for every semi-quantitative field, bilirubin and urobilinogen
            included (calc_shift_encoding.py MAPPINGS["M2"]);
        X1_sgclip  X1 with specific gravity clipped to the range of the training persons (KNHANES records up to 1.050).
    Nitrite (raw code >= 1 -> 1), pH, specific gravity and the other fields are identical under X1, M1 and M2. X1 is
    rebuilt from the raw KNHANES files (cohort.knhanes_raw) and asserted equal to ext. The mean predicted risk among
    nitrite-positive KNHANES adults is reported next to their observed risk.
    -> feature_tiers_coding_sensitivity.csv

Removed with the retraining (2026-10-16):
  * The setting "external_harmonised" (KNHANES urobilinogen set to 0.5 and nitrite to 0 on the submission's coding),
    the "as coded" artefact flag and the bilirubin 2.0 -> 1.0 diagnostic recoding. X1 codes urobilinogen and bilirubin
    with the development labels, and development nitrite is now parsed correctly (dev_v2: 1,035 nitrite-positive persons
    instead of 2), so a linear model no longer extrapolates the KNHANES nitrite and urobilinogen codes. The columns
    primary and note are dropped (one external setting). HARM is kept, empty, only because calc_followup.py imports it:
    ext.assign(**HARM) returns the X1 inputs unchanged.
  * Every comparison with the original (submitted) models, the rebuild of their split and calibration folds and the
    bookkeeping of their training records of persons aged < 19 years: the comparators of (c) are fitted on exactly the
    training persons of the retrained models.

DeLong p: common.delong_paired returns 2*(1-cdf(|z|)), which underflows to exactly 0 for |z| > ~8.3. This script
recomputes the same statistic with 2*sf(|z|) (asserted equal to common.delong_paired wherever that is non-zero) and also
stores z and log10 p (sf itself underflows beyond |z| ~ 37). Where DeLong and the paired bootstrap CI disagree,
`delong_ci_disagree` is True and the bootstrap CI should be read as primary.

Scaled Brier: the retrained ensemble and the "v5 config, isotonic CV5" arm are isotonic-calibrated; the other refits are
not. Brier contrasts are like-for-like only where `both_isotonic` is True (part (c)).

Bootstrap: 1000 resamples at full size, percentile CI, seed 42 (common.py helpers). Every distinct evaluation is computed
once and shared by the rows of the four output files that report it.
Threads are capped at 4 (LightGBM n_jobs=4; ensemble prediction threads 4; bootstrap workers = 4 processes).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import time, functools
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
# Import order (workflow rule of 2026-10-11): cohort and common first, then the boosting libraries (cohort imports
# 05_src/run_uric_module_v4, which loads lightgbm, xgboost and catboost itself).
from cohort import (HERE, TARGETS, F3, F9, F11, FMODEL, ID, KN_RAW_COL, KN_UNIFORM, labels, main_cohort, load_model,
                    train_xy, cal_folds, knhanes_raw, knhanes_x1)
from common import (LGBM_FIXED, GRID, SEED, auroc_ci, auprc_ci, boot_ci, paired_boot, delong_paired, scaled_brier,
                    _placements)
import xgboost  # noqa: F401,E402
from lightgbm import LGBMClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, FunctionTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

N_THREADS = 4
# Execution-only settings added to every LightGBM fit: thread cap and reproducible histogram construction.
LGBM_RUN = dict(n_jobs=N_THREADS, deterministic=True, force_col_wise=True)
# Cohort sizes of the retrained models (decision of 2026-10-16; cohort_info_v2.csv), asserted below.
N_TRAIN, N_TEST, N_EXT = 103_628, 25_907, 49_987
F2 = [c for c in F3 if c != "bmi_unified"]                    # age, male
TIERS = {"2F": F2, "3F": F3, "9F": F9, "11F": F11, "12F": FMODEL}
SPLINE_TIERS = ("2F", "3F", "11F", "12F")                     # tiers that contain age (the spline LR needs age and male)
DIP_TIERS = ("9F", "11F", "12F")                              # tiers with dipstick inputs (coding diagnostics)
TIER_CONTRASTS = [("12F", "3F", "dipstick added to age+sex+BMI"),
                  ("12F", "11F", "BMI added to dipstick+age+sex"),
                  ("11F", "9F", "age+sex added to dipstick"),
                  ("11F", "2F", "dipstick added to age+sex"),
                  ("3F", "2F", "BMI added to age+sex")]
LGBM_NAME, LR_NAME = "LightGBM (fixed)", "Logistic regression"
LRS_NAME = "LR (age/BMI splines + age x male)"
GBT_NAME, GBI_NAME = "LightGBM (GRID-tuned)", "LightGBM (v5 config, isotonic CV5)"
ENS_NAME = "Retrained ensemble (v5)"
TIER_MODELS = {LGBM_NAME: tuple(TIERS), LR_NAME: tuple(TIERS), LRS_NAME: SPLINE_TIERS, GBT_NAME: tuple(TIERS)}
SHORT = {LGBM_NAME: "LightGBM", LR_NAME: "LR", LRS_NAME: "spline LR", GBT_NAME: "tuned LightGBM"}
ENS_ARMS = (ENS_NAME, LR_NAME, LRS_NAME, LGBM_NAME, GBT_NAME, GBI_NAME)
ISOTONIC_ARMS = (ENS_NAME, GBI_NAME)
SETTINGS = ("internal", "external")                           # external = KNHANES adults, X1 coding

# KNHANES codings. X1 (primary) is cohort.knhanes_x1; M1 and M2 differ from it only in bilirubin and urobilinogen.
# The maps are those of calc_shift_encoding.py MAPPINGS (M1: BIL_USED, URO_USED; M2: UNIFORM for both).
HARM = {}   # former harmonisation (urobilinogen 0.5, nitrite 0) is obsolete under X1; kept empty for calc_followup.py
CODINGS = ("X1", "M1", "M2")
SGCLIP = "X1_sgclip"
CODING_MAPS = {"M1": dict(bil={0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}, uro={0: 0, 1: 0.2, 2: 0.5, 3: 1, 4: 2, 5: 3}),
               "M2": dict(bil=KN_UNIFORM, uro=KN_UNIFORM)}
LOG_PATH = os.path.join(HERE, "calc_feature_tiers_log.txt")
_LOG = []


def log(msg=""):
    print(msg, flush=True)
    _LOG.append(msg)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")


# ------------------------------------------------------------------ models
def lgbm(**over):
    return LGBMClassifier(**{**LGBM_FIXED, **over, **LGBM_RUN})


def logreg(feats=None):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def _product(Z):
    return Z[:, [0]] * Z[:, [1]]


def logreg_spline(feats):
    """Main-effects LR plus cubic B-splines for age and BMI (5 quantile knots, linear extrapolation) and age x male."""
    ia, im = feats.index("age"), feats.index("male")
    ib = feats.index("bmi_unified") if "bmi_unified" in feats else None
    spline = dict(n_knots=5, degree=3, knots="quantile", extrapolation="linear", include_bias=False)
    parts = [("age_spline", SplineTransformer(**spline), [ia])]
    if ib is not None:
        parts.append(("bmi_spline", SplineTransformer(**spline), [ib]))
    linear = [i for i in range(len(feats)) if i not in (ia, ib)]          # includes male and the dipstick fields
    parts += [("linear", "passthrough", linear), ("age_x_male", FunctionTransformer(_product), [ia, im])]
    return make_pipeline(ColumnTransformer(parts), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def cap_threads(cal):
    """Limit the retrained ensemble's prediction threads to N_THREADS (in memory only; nothing is saved)."""
    for cc in cal.calibrated_classifiers_:
        for est in cc.estimator.estimators_:
            if isinstance(est, CatBoostClassifier):
                est.predict_proba = functools.partial(est.predict_proba, thread_count=N_THREADS)
            else:
                est.set_params(n_jobs=N_THREADS)
    return cal


def lgbm_v5_iso(ens, folds):
    """Single LightGBM with the retrained ensemble's (Optuna-tuned) LightGBM hyperparameters, cloned from the unfitted
    template clone(ens.estimator), in the same calibration wrapper (isotonic CalibratedClassifierCV on `folds`, the
    retrained model's own calibration folds; see cohort.cal_folds)."""
    member = dict(clone(ens.estimator).estimators)["lgb"]
    est = clone(member).set_params(**LGBM_RUN)
    params = member.get_params()
    return CalibratedClassifierCV(est, method="isotonic", cv=folds), {k: params[k] for k in (
        "n_estimators", "max_depth", "learning_rate", "num_leaves", "min_child_samples", "reg_alpha", "reg_lambda")}


def grid_tune(X_tr, y_tr, tag):
    """Best GRID configuration by 3-fold stratified CV AUROC (n_estimators 300). Returns (best, best_cv, rows)."""
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y_tr)), y_tr))
    best, best_cv, rows = None, -np.inf, []
    for k, g in enumerate(GRID):
        aucs = [roc_auc_score(y_tr[b], lgbm(**g, n_estimators=300).fit(X_tr[a], y_tr[a]).predict_proba(X_tr[b])[:, 1])
                for a, b in folds]
        rows.append(dict(row_type="cv_grid", **tag, config=k, **g,
                         cv_auroc=float(np.mean(aucs)), cv_auroc_sd=float(np.std(aucs, ddof=1))))
        if np.mean(aucs) > best_cv:
            best_cv, best = float(np.mean(aucs)), g
    return best, best_cv, rows


# ------------------------------------------------------------------ KNHANES codings
def _code(s, table):
    return s.map(lambda x: np.nan if pd.isna(x) else table.get(int(x), np.nan)).astype(float)


def ext_codings(ext, sg_range):
    """KNHANES 12-input frames (FMODEL order, rows in ext order) under X1 (= ext), M1, M2 and X1 with specific gravity
    clipped to sg_range. X1 is rebuilt from the raw files and asserted equal to ext (dipstick fields)."""
    raw = knhanes_raw()
    assert raw["id"].is_unique
    r = raw.set_index("id").loc[ext["id"].to_numpy()].reset_index()
    x1 = knhanes_x1(r)
    assert np.array_equal(x1["id"].to_numpy(), ext["id"].to_numpy())
    for c in F9:
        assert np.array_equal(x1[c].to_numpy(float), ext[c].to_numpy(float)), f"X1 rebuilt from raw differs from ext: {c}"
    E = {"X1": ext[FMODEL].copy()}
    for key, m in CODING_MAPS.items():
        e = ext[FMODEL].copy()
        e["ua_bilirubin"] = _code(r[KN_RAW_COL["ua_bilirubin"]], m["bil"]).to_numpy()
        e["ua_urobilinogen"] = _code(r[KN_RAW_COL["ua_urobilinogen"]], m["uro"]).to_numpy()
        assert e.notna().all().all(), key
        E[key] = e
    e = ext[FMODEL].copy()
    e["ua_sg"] = e["ua_sg"].clip(*sg_range)
    E[SGCLIP] = e
    return E


# ------------------------------------------------------------------ metrics (run in worker processes)
def tier_metrics(y, p):
    auc, auc_lo, auc_hi = auroc_ci(y, p)
    ap, ap_lo, ap_hi = auprc_ci(y, p)
    sb_lo, sb_hi = boot_ci(y, p, scaled_brier)
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()), mean_pred=float(p.mean()),
                auroc=auc, auroc_lo=auc_lo, auroc_hi=auc_hi, auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi,
                sbrier=scaled_brier(y, p), sbrier_lo=sb_lo, sbrier_hi=sb_hi, brier=brier_score_loss(y, p))


def delong_sf(y, p1, p2):
    """Paired DeLong (same statistic as common.delong_paired) with p = 2*sf(|z|), which does not underflow."""
    a10, a01, m, n = _placements(y, p1); b10, b01, _, _ = _placements(y, p2)
    S = np.cov(np.vstack([a10, b10])) / m + np.cov(np.vstack([a01, b01])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; d = a10.mean() - b10.mean()
    z = d / np.sqrt(var) if var > 0 else np.nan
    return d, z, (2 * stats.norm.sf(abs(z)) if var > 0 else np.nan)


def contrast_metrics(y, p_a, p_b):
    """Model a minus model b on the same records."""
    lo, hi = paired_boot(y, p_a, p_b)
    d, z, p_delong = delong_sf(y, p_a, p_b)
    d_common, p_common = delong_paired(y, p_a, p_b)
    assert abs(d - d_common) < 1e-12 and (p_common == 0 or abs(p_delong - p_common) < 1e-9), (d, d_common, p_common)
    auc_a, auc_b = roc_auc_score(y, p_a), roc_auc_score(y, p_b)
    ci_excl = bool(lo > 0 or hi < 0)
    return dict(n=len(y), positives=int(y.sum()), auroc_a=auc_a, auroc_b=auc_b, delta_auroc=auc_a - auc_b,
                delta_lo=lo, delta_hi=hi, delong_z=z, delong_p=p_delong,
                delong_log10_p=(stats.norm.logsf(abs(z)) + np.log(2)) / np.log(10), ci_excludes_0=ci_excl,
                delong_ci_disagree=bool(ci_excl != (p_delong < 0.05)),
                delta_auprc=average_precision_score(y, p_a) - average_precision_score(y, p_b),
                delta_sbrier=scaled_brier(y, p_a) - scaled_brier(y, p_b))


class Evaluations:
    """Registry of the bootstrap evaluations: each distinct (arm, setting) or (arm a, arm b, setting) is computed once.
    An arm is (target, model, tier); predictions P[(target, model, tier, setting)], labels Y[(target, setting)]."""

    def __init__(self, P, Y):
        self.P, self.Y, self.jobs, self.res = P, Y, {}, {}

    def tier(self, t, model, tier, setting):
        key = ("tier", t, model, tier, setting)
        if key not in self.res:
            self.jobs[key] = (tier_metrics, (self.Y[(t, setting)], self.P[(t, model, tier, setting)]))
        return key

    def contrast(self, t, a, b, setting):
        """a, b = (model, tier)."""
        key = ("contrast", t, a, b, setting)
        if key not in self.res:
            self.jobs[key] = (contrast_metrics, (self.Y[(t, setting)], self.P[(t, *a, setting)], self.P[(t, *b, setting)]))
        return key

    def run(self):
        keys = [k for k in self.jobs if k not in self.res]
        t0 = time.time()
        out = Parallel(n_jobs=N_THREADS)(delayed(self.jobs[k][0])(*self.jobs[k][1]) for k in keys)
        self.res.update(zip(keys, out))
        self.jobs = {}
        log(f"  bootstrap for {len(keys)} distinct evaluations in {time.time() - t0:.0f}s")

    def __getitem__(self, key):
        return self.res[key]


def fmt_p(p, log10_p=None):
    if p == 0:                      # |z| > ~37: even the survival function underflows; report log10 p
        return f"1e{log10_p:.0f}" if log10_p is not None else "<1e-300"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def ci(r, k="auroc"):
    return f"{r[k]:.4f} ({r[k + '_lo']:.4f}-{r[k + '_hi']:.4f})"


def dci(r):
    flag = "  [DeLong/CI disagree]" if r["delong_ci_disagree"] else ""
    return f"{r['delta_auroc']:+.4f} ({r['delta_lo']:+.4f} to {r['delta_hi']:+.4f}), DeLong p={fmt_p(r['delong_p'], r['delong_log10_p'])}{flag}"


# ------------------------------------------------------------------ fits
def predict_all(P, t, mname, tier, model, X_te, E, as_frame=False):
    """Internal and external (X1) predictions of a fitted arm, plus the coding diagnostics for dipstick tiers."""
    feats = TIERS[tier]
    conv = (lambda f: f[feats]) if as_frame else (lambda f: f[feats].to_numpy(float))
    P[(t, mname, tier, "internal")] = model.predict_proba(X_te)[:, 1]
    P[(t, mname, tier, "external")] = model.predict_proba(conv(E["X1"]))[:, 1]
    if tier in DIP_TIERS:
        for c in ("M1", "M2", SGCLIP):
            P[(t, mname, tier, c)] = model.predict_proba(conv(E[c]))[:, 1]


def fit_tiers(d, tr, te, E, P, Y):
    """(a) LightGBM (fixed), main-effects LR and spline LR for every tier (the GRID-tuned class is fitted in (b))."""
    log("=" * 100 + "\n(a) Equal-budget nested ablation: fits on the training persons (internal = test persons; external = "
        "KNHANES adults, X1; M1/M2 = coding diagnostics)")
    for t, cfg in TARGETS.items():
        y = labels(d["urate"], cfg["threshold"])
        y_tr = y[tr]
        log(f"{cfg['short']}: train {len(tr)} (pos {int(y_tr.sum())}), internal test {len(te)} "
            f"(pos {int(Y[(t, 'internal')].sum())}), external {len(Y[(t, 'external')])} (pos {int(Y[(t, 'external')].sum())})")
        for mname, make in ((LGBM_NAME, lambda f: lgbm()), (LR_NAME, logreg), (LRS_NAME, logreg_spline)):
            for tier in TIER_MODELS[mname]:
                feats = TIERS[tier]
                X = d[feats].to_numpy(float)
                t0 = time.time()
                model = make(feats).fit(X[tr], y_tr)
                predict_all(P, t, mname, tier, model, X[te], E)
                cod = (f"  M1 {roc_auc_score(Y[(t, 'external')], P[(t, mname, tier, 'M1')]):.4f}  M2 "
                       f"{roc_auc_score(Y[(t, 'external')], P[(t, mname, tier, 'M2')]):.4f}" if tier in DIP_TIERS else "")
                log(f"  fitted {mname:34s} {tier:4s} in {time.time() - t0:5.1f}s  AUROC int "
                    f"{roc_auc_score(Y[(t, 'internal')], P[(t, mname, tier, 'internal')]):.4f}  ext X1 "
                    f"{roc_auc_score(Y[(t, 'external')], P[(t, mname, tier, 'external')]):.4f}{cod}")


def tune_tiers(d, tr, te, E, P, Y):
    """(b) GRID-tuned LightGBM for every tier. Returns ({(t, tier): (best, best_cv)}, cv_grid rows)."""
    log("=" * 100 + "\n(b) Per-set tuning, every tier: GRID by 3-fold stratified CV AUROC on the training persons "
        "(n_estimators 300), refit on all training persons (n_estimators 300)")
    tuned, grid_rows = {}, []
    for t, cfg in TARGETS.items():
        y_tr = labels(d["urate"], cfg["threshold"])[tr]
        for tier, feats in TIERS.items():
            X = d[feats].to_numpy(float)
            t0 = time.time()
            best, best_cv, rows = grid_tune(X[tr], y_tr, dict(target=cfg["short"], threshold=cfg["threshold"],
                                                             role=cfg["role"], tier=tier))
            grid_rows += rows
            tuned[(t, tier)] = (best, best_cv)
            model = lgbm(**best, n_estimators=300).fit(X[tr], y_tr)
            predict_all(P, t, GBT_NAME, tier, model, X[te], E)
            log(f"  {cfg['short']:6s} {tier:4s} best {best} CV AUROC {best_cv:.4f} (grid {time.time() - t0:.0f}s)  "
                f"AUROC int {roc_auc_score(Y[(t, 'internal')], P[(t, GBT_NAME, tier, 'internal')]):.4f} "
                f"ext {roc_auc_score(Y[(t, 'external')], P[(t, GBT_NAME, tier, 'external')]):.4f} (fixed config int "
                f"{roc_auc_score(Y[(t, 'internal')], P[(t, LGBM_NAME, tier, 'internal')]):.4f} ext "
                f"{roc_auc_score(Y[(t, 'external')], P[(t, LGBM_NAME, tier, 'external')]):.4f})")
    return tuned, grid_rows


def fit_ensemble_arms(d, tr, te, ext, E, P, Y, tuned):
    """(c) Retrained-ensemble predictions (saved files, checked) and the v5-config isotonic LightGBM arm.
    Returns {target: config strings}."""
    log("=" * 100 + "\n(c) Retrained ensemble vs simpler 12-input models fitted on the same training persons (LR, spline "
        "LR, LightGBM fixed and GRID-tuned = the 12F models of (a) and (b))")
    pint = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))
    pext = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"))
    mp = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
    dte = d.iloc[te]
    configs = {}
    for t, cfg in TARGETS.items():
        thr = cfg["threshold"]
        t0 = time.time()
        ens = cap_threads(load_model(t))
        assert list(ens.feature_names_in_) == FMODEL, "retrained model inputs differ from FMODEL"
        X_tr, y_tr = train_xy(thr)
        assert len(y_tr) == len(tr) and np.array_equal(y_tr, labels(d["urate"], thr)[tr])
        folds = cal_folds(ens, X_tr, y_tr)                  # verified against the saved isotonic calibrators

        # retrained-ensemble predictions: saved files, aligned and spot-checked against the model
        di, de = pint[pint.target == t], pext[pext.target == t]
        assert len(di) == len(te) == N_TEST and np.array_equal(di.y.values, Y[(t, "internal")])
        assert np.array_equal(di["id"].values, dte[ID].values) and np.allclose(di.age.values, dte["age"].values)
        assert np.array_equal(di.male.values, dte["male"].values)
        assert len(de) == len(ext) == N_EXT and np.array_equal(de.y.values, Y[(t, "external")])
        assert np.array_equal(de["id"].astype(str).values, ext["id"].astype(str).values)
        assert np.allclose(de.age.values, ext["age"].values)
        P[(t, ENS_NAME, "12F", "internal")] = di.p.to_numpy(float)
        P[(t, ENS_NAME, "12F", "external")] = de.p.to_numpy(float)
        chk_i = ens.predict_proba(dte[FMODEL].iloc[:2000])[:, 1]
        chk_e = ens.predict_proba(E["X1"].iloc[:2000])[:, 1]
        err = max(np.abs(chk_i - P[(t, ENS_NAME, "12F", "internal")][:2000]).max(),
                  np.abs(chk_e - P[(t, ENS_NAME, "12F", "external")][:2000]).max())
        assert err < 1e-9, err
        for c in ("M1", "M2", SGCLIP):
            P[(t, ENS_NAME, "12F", c)] = ens.predict_proba(E[c])[:, 1]          # in memory; the model is not refitted
        for setting, cohort in (("internal", "internal test"), ("external", "KNHANES adults")):
            ref = mp[(mp.target == cfg["short"]) & (mp.cohort == cohort)]
            assert len(ref) == 1
            a = roc_auc_score(Y[(t, setting)], P[(t, ENS_NAME, "12F", setting)])
            assert round(a, 4) == round(float(ref.auroc.iloc[0]), 4) and abs(a - float(ref.auroc.iloc[0])) < 1e-9, \
                (t, setting, a, float(ref.auroc.iloc[0]))

        # the ensemble's LightGBM member configuration, isotonic CV5 on the retrained model's own folds
        gbi, v5_params = lgbm_v5_iso(ens, folds)
        gbi.fit(X_tr, y_tr)
        member_diff = max(float(np.abs(
            gbi.calibrated_classifiers_[k].estimator.predict_proba(dte[FMODEL])[:, 1]
            - ens.calibrated_classifiers_[k].estimator.named_estimators_["lgb"].predict_proba(dte[FMODEL])[:, 1]).max())
            for k in range(len(folds)))
        assert member_diff < 1e-12, f"v5-config fold learners do not reproduce the ensemble's LightGBM members: {member_diff}"
        predict_all(P, t, GBI_NAME, "12F", gbi, dte[FMODEL], E, as_frame=True)
        del ens
        best, best_cv = tuned[(t, "12F")]
        configs[t] = {
            ENS_NAME: "v5: Optuna-tuned LGBM+XGB+CatBoost soft vote, isotonic CalibratedClassifierCV cv=5 (bestmodel_v5)",
            LR_NAME: "StandardScaler + LogisticRegression C=1 (12F model of feature_tiers.csv)",
            LRS_NAME: "splines(age, BMI) + age x male + StandardScaler + LogisticRegression C=1 (12F model of "
                      "feature_tiers.csv)",
            LGBM_NAME: "LGBM_FIXED (12F model of feature_tiers.csv)",
            GBT_NAME: f"GRID best {best}, n_estimators 300 (3-fold CV AUROC {best_cv:.4f}; 12F model of tuned_per_set.csv)",
            GBI_NAME: f"retrained ensemble's LightGBM member {v5_params}, isotonic CalibratedClassifierCV on the retrained "
                      "model's 5 calibration folds (fold learners identical to the ensemble's LightGBM members)"}
        log(f"{cfg['short']}: training persons {len(tr)} (pos {int(y_tr.sum())}); internal test {len(te)}; external "
            f"{len(ext)}; retrained-ensemble spot-check max |diff| {err:.1e}; AUROC int "
            f"{roc_auc_score(Y[(t, 'internal')], P[(t, ENS_NAME, '12F', 'internal')]):.4f} ext "
            f"{roc_auc_score(Y[(t, 'external')], P[(t, ENS_NAME, '12F', 'external')]):.4f} (= main_performance.csv)")
        log(f"  calibration folds (cal_folds, verified against the saved isotonic calibrators): rows "
            f"{[len(b) for _, b in folds]}, positives {[int(y_tr[b].sum()) for _, b in folds]}")
        log(f"  v5 config : {v5_params}; fold learners vs the ensemble's LightGBM members on the internal test persons: "
            f"max |diff| {member_diff:.1e}")
        log(f"  v5 config, isotonic CV5: AUROC int {roc_auc_score(Y[(t, 'internal')], P[(t, GBI_NAME, '12F', 'internal')]):.4f} "
            f"ext {roc_auc_score(Y[(t, 'external')], P[(t, GBI_NAME, '12F', 'external')]):.4f} ({time.time() - t0:.0f}s)")
    return configs


# ------------------------------------------------------------------ output tables
C_CONTRASTS = [(ENS_NAME, LR_NAME, "ensemble vs main-effects LR"),
               (ENS_NAME, LRS_NAME, "ensemble vs spline LR"),
               (ENS_NAME, LGBM_NAME, "ensemble (tuned, CV-bagged, calibrated) vs untuned single booster"),
               (ENS_NAME, GBT_NAME, "ensemble vs GRID-tuned single booster"),
               (ENS_NAME, GBI_NAME, "ensembling alone (adding XGBoost and CatBoost): same LightGBM member, CV-bagging "
                                    "folds and isotonic calibration"),
               (LGBM_NAME, LR_NAME, "untuned booster vs main-effects LR"),
               (GBT_NAME, LR_NAME, "tuned booster vs main-effects LR")]


def register(ev):
    """Register every evaluation needed by the four output files."""
    for t in TARGETS:
        for setting in SETTINGS:
            for mname, tiers in TIER_MODELS.items():
                for tier in tiers:
                    ev.tier(t, mname, tier, setting)
                for a, b, _ in TIER_CONTRASTS:
                    if a in tiers and b in tiers:
                        ev.contrast(t, (mname, a), (mname, b), setting)
            for other, tiers in ((LR_NAME, tuple(TIERS)), (LRS_NAME, SPLINE_TIERS)):
                for tier in tiers:
                    ev.contrast(t, (LGBM_NAME, tier), (other, tier), setting)
            for mname in ENS_ARMS:
                ev.tier(t, mname, "12F", setting)
            for a, b, _ in C_CONTRASTS:
                ev.contrast(t, (a, "12F"), (b, "12F"), setting)


def table_tiers(ev, base):
    rows = []
    for t, cfg in TARGETS.items():
        for setting in SETTINGS:
            for mname, tiers in TIER_MODELS.items():
                for tier in tiers:
                    rows.append(dict(row_type="tier", **base[t], model=mname, setting=setting, tier=tier,
                                     features=len(TIERS[tier]), **ev[ev.tier(t, mname, tier, setting)]))
                for a, b, meaning in TIER_CONTRASTS:
                    if a in tiers and b in tiers:
                        rows.append(dict(row_type="contrast_tier", **base[t], model=mname, setting=setting,
                                         tier=f"{a}-{b}", comparison=f"{mname}: {a} vs {b}", meaning=meaning,
                                         **ev[ev.contrast(t, (mname, a), (mname, b), setting)]))
            for other, tiers in ((LR_NAME, tuple(TIERS)), (LRS_NAME, SPLINE_TIERS)):
                for tier in tiers:
                    rows.append(dict(row_type="contrast_model", **base[t], model=f"LightGBM - {other}", setting=setting,
                                     tier=tier, comparison=f"{tier}: {LGBM_NAME} vs {other}",
                                     meaning=f"boosting over {other.lower()}, same inputs",
                                     **ev[ev.contrast(t, (LGBM_NAME, tier), (other, tier), setting)]))
    return pd.DataFrame(rows)


def table_tuned(ev, tuned, grid_rows, fa):
    rows = list(grid_rows)
    for t, cfg in TARGETS.items():
        b = dict(target=cfg["short"], threshold=cfg["threshold"], role=cfg["role"])
        for tier in TIERS:
            best, best_cv = tuned[(t, tier)]
            for setting in SETTINGS:
                fixed = fa[(fa.target == cfg["short"]) & (fa.setting == setting) & (fa.row_type == "tier") &
                           (fa.model == LGBM_NAME) & (fa.tier == tier)]
                assert len(fixed) == 1
                rows.append(dict(row_type="tuned", **b, tier=tier, setting=setting, best_config=str(best),
                                 **{f"best_{k}": v for k, v in best.items()}, cv_auroc=best_cv,
                                 auroc_fixed_config=float(fixed.auroc.iloc[0]),
                                 **ev[ev.tier(t, GBT_NAME, tier, setting)]))
        for a, bb, meaning in TIER_CONTRASTS:
            for setting in SETTINGS:
                fixed = fa[(fa.target == cfg["short"]) & (fa.setting == setting) & (fa.row_type == "contrast_tier") &
                           (fa.model == LGBM_NAME) & (fa.tier == f"{a}-{bb}")]
                assert len(fixed) == 1
                rows.append(dict(row_type="contrast", **b, tier=f"{a}-{bb}", setting=setting,
                                 comparison=f"tuned {a} vs tuned {bb}", meaning=meaning,
                                 delta_fixed_config=float(fixed.delta_auroc.iloc[0]),
                                 **ev[ev.contrast(t, (GBT_NAME, a), (GBT_NAME, bb), setting)]))
    order = {"tuned": 0, "contrast": 1, "cv_grid": 2}
    out = pd.DataFrame(rows)
    out["tuning_gain"] = out["auroc"] - out["auroc_fixed_config"]
    out["increment_change_vs_fixed"] = out["delta_auroc"] - out["delta_fixed_config"]
    return out.sort_values(["row_type", "threshold"], key=lambda c: c.map(order) if c.name == "row_type" else c,
                           kind="stable").reset_index(drop=True)


def table_ensemble(ev, base, configs, grid_rows):
    rows = []
    for t, cfg in TARGETS.items():
        for setting in SETTINGS:
            for mname in ENS_ARMS:
                rows.append(dict(row_type="model", **base[t], setting=setting, model=mname, features=len(FMODEL),
                                 isotonic_calibrated=mname in ISOTONIC_ARMS, config=configs[t][mname],
                                 **ev[ev.tier(t, mname, "12F", setting)]))
            for a, b, meaning in C_CONTRASTS:
                rows.append(dict(row_type="contrast", **base[t], setting=setting, model=f"{a} - {b}",
                                 comparison=f"{a} vs {b}", meaning=meaning,
                                 both_isotonic=a in ISOTONIC_ARMS and b in ISOTONIC_ARMS,
                                 **ev[ev.contrast(t, (a, "12F"), (b, "12F"), setting)]))
    grid12 = [r for r in grid_rows if r["tier"] == "12F"]           # the grid behind the GRID-tuned arm
    return pd.DataFrame(rows + grid12)


def coding_row(part, t, model, tier, y, P, nit_pos):
    """Diagnostic row: one external arm under the KNHANES codings (point estimates)."""
    pr = {c: P[(t, model, tier, "external" if c == "X1" else c)] for c in (*CODINGS, SGCLIP)}
    r = dict(part=part, target=TARGETS[t]["short"], model=model, tier=tier, n=len(y), positives=int(y.sum()))
    r.update({f"auroc_{c}": roc_auc_score(y, pr[c]) for c in (*CODINGS, SGCLIP)})
    r.update({f"sbrier_{c}": scaled_brier(y, pr[c]) for c in CODINGS})
    r.update({f"mean_pred_{c}": float(pr[c].mean()) for c in CODINGS})
    r.update(observed=float(y.mean()), nitrite_pos_n=int(nit_pos.sum()), nitrite_pos_observed=float(y[nit_pos].mean()))
    r.update({f"nitrite_pos_mean_pred_{c}": float(pr[c][nit_pos].mean()) for c in CODINGS})
    return r


def write_coding(P, Y, ext):
    nit_pos = ext["ua_nitrite"].to_numpy() == 1
    rows = []
    for t in TARGETS:
        for mname, tiers in TIER_MODELS.items():
            for tier in tiers:
                if tier in DIP_TIERS:
                    rows.append(coding_row("a", t, mname, tier, Y[(t, "external")], P, nit_pos))
        for mname in ISOTONIC_ARMS:
            rows.append(coding_row("c", t, mname, "12F", Y[(t, "external")], P, nit_pos))
    cs = pd.DataFrame(rows)
    cs.to_csv(os.path.join(HERE, "feature_tiers_coding_sensitivity.csv"), index=False, encoding="utf-8")
    log("=" * 100 + "\nCoding diagnostics (external point estimates): AUROC X1 (primary) | M1 | M2 | X1 with SG clipped; "
        "mean predicted risk among nitrite-positive KNHANES adults X1 | M1 | M2 (observed)")
    for _, r in cs.iterrows():
        log(f"  ({r.part}) {r.target:6s} {r.model:34s} {r.tier:4s} AUROC {r.auroc_X1:.4f} | {r.auroc_M1:.4f} | "
            f"{r.auroc_M2:.4f} | {r[f'auroc_{SGCLIP}']:.4f}  sBrier {r.sbrier_X1:+.4f} | {r.sbrier_M1:+.4f} | "
            f"{r.sbrier_M2:+.4f}  mean pred {r.mean_pred_X1:.4f} | {r.mean_pred_M1:.4f} | {r.mean_pred_M2:.4f} (obs "
            f"{r.observed:.4f})  nitrite+ {r.nitrite_pos_mean_pred_X1:.3f} | {r.nitrite_pos_mean_pred_M1:.3f} | "
            f"{r.nitrite_pos_mean_pred_M2:.3f} (obs {r.nitrite_pos_observed:.3f})")
    for c in ("M1", "M2", SGCLIP):
        dd = (cs[f"auroc_{c}"] - cs.auroc_X1).abs()
        log(f"  |AUROC {c} - AUROC X1|: max {dd.max():.4f} ({cs.loc[dd.idxmax(), ['part', 'target', 'model', 'tier']].tolist()})")
    return cs


def log_tables(fa, fb, fc):
    for t, cfg in TARGETS.items():
        s = fa[fa.target == cfg["short"]]
        log(f"\n[{cfg['short']}] (a) tier AUROC (95% CI) | AP (average precision) | scaled Brier")
        for setting in SETTINGS:
            for _, r in s[(s.row_type == "tier") & (s.setting == setting)].iterrows():
                log(f"  {setting:8s} {r.model:34s} {r.tier:4s} AUROC {ci(r)}  AP {r.auprc:.4f}  sBrier {r.sbrier:+.4f}")
        log(f"[{cfg['short']}] (a) contrasts (AUROC difference, paired bootstrap 95% CI)")
        for setting in SETTINGS:
            for _, r in s[(s.row_type != "tier") & (s.setting == setting)].iterrows():
                log(f"  {setting:8s} {r.comparison:55s} {dci(r)}")
    log("")
    for t, cfg in TARGETS.items():
        s = fb[fb.target == cfg["short"]]
        for _, r in s[s.row_type == "tuned"].iterrows():
            log(f"  [{cfg['short']}] (b) tuned {r.tier:4s} {r.setting:8s} AUROC {ci(r)}  (fixed config "
                f"{r.auroc_fixed_config:.4f}; tuning gain {r.tuning_gain:+.4f})")
        for _, r in s[s.row_type == "contrast"].iterrows():
            log(f"  [{cfg['short']}] (b) tuned {r.tier:7s} {r.setting:8s} {dci(r)}  (fixed config "
                f"{r.delta_fixed_config:+.4f}; change {r.increment_change_vs_fixed:+.4f})")
    tu = fb[fb.row_type == "tuned"]
    log(f"  tuning gain (tuned - fixed AUROC) ranges {tu.tuning_gain.min():+.4f} to {tu.tuning_gain.max():+.4f}; "
        f"largest: {tu.loc[tu.tuning_gain.idxmax(), ['target', 'tier', 'setting']].tolist()}")
    log("")
    for t, cfg in TARGETS.items():
        s = fc[fc.target == cfg["short"]]
        for setting in SETTINGS:
            for _, r in s[(s.row_type == "model") & (s.setting == setting)].iterrows():
                log(f"  [{cfg['short']}] (c) {setting:8s} {r.model:34s} AUROC {ci(r)}  AP {r.auprc:.4f}  sBrier {r.sbrier:+.4f}")
            for _, r in s[(s.row_type == "contrast") & (s.setting == setting)].iterrows():
                log(f"  [{cfg['short']}] (c) {setting:8s} {r.comparison:60s} {dci(r)}")
            iso = s[(s.row_type == "contrast") & (s.setting == setting) & (s.both_isotonic == True)]  # noqa: E712
            for _, r in iso.iterrows():
                log(f"  [{cfg['short']}] (c) {setting:8s} {r.comparison:60s} delta scaled Brier {r.delta_sbrier:+.4f} "
                    "(both isotonic-calibrated)")


def key_results(fa, fb, fc):
    """Bottom-line numbers, all read back from the written outputs."""
    log("=" * 100 + "\nKEY RESULTS (external = KNHANES adults, X1 coding; bootstrap CI primary where DeLong disagrees)")

    def row(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        assert m.sum() == 1, kw
        return df[m].iloc[0]
    for cfg in TARGETS.values():
        tg = cfg["short"]
        log(f"[{tg}]")
        for a, b, _ in (TIER_CONTRASTS[0], TIER_CONTRASTS[3], TIER_CONTRASTS[4]):
            for setting in SETTINGS:
                parts = []
                for mname, tiers in TIER_MODELS.items():
                    if a in tiers and b in tiers:
                        r = row(fa, target=tg, setting=setting, row_type="contrast_tier", model=mname, tier=f"{a}-{b}")
                        parts.append(f"{SHORT[mname]} {r.delta_auroc:+.4f} ({r.delta_lo:+.4f} to {r.delta_hi:+.4f})")
                log(f"  (a) {a}-{b} {setting:8s} " + "; ".join(parts))
        for setting in SETTINGS:
            r = row(fa, target=tg, setting=setting, row_type="contrast_model", model=f"LightGBM - {LR_NAME}", tier="12F")
            r2 = row(fa, target=tg, setting=setting, row_type="contrast_model", model=f"LightGBM - {LRS_NAME}", tier="12F")
            log(f"  (a) 12F LightGBM-LR {setting:8s} {r.delta_auroc:+.4f} ({r.delta_lo:+.4f} to {r.delta_hi:+.4f}); "
                f"LightGBM-spline LR {r2.delta_auroc:+.4f} ({r2.delta_lo:+.4f} to {r2.delta_hi:+.4f})")
        for setting in SETTINGS:
            r = row(fb, target=tg, setting=setting, row_type="contrast", tier="12F-3F")
            log(f"  (b) tuned 12F-3F {setting:8s} {r.delta_auroc:+.4f} ({r.delta_lo:+.4f} to {r.delta_hi:+.4f}) vs fixed "
                f"{r.delta_fixed_config:+.4f} (change {r.increment_change_vs_fixed:+.4f})")
        for setting in SETTINGS:
            r = row(fc, target=tg, setting=setting, row_type="model", model=ENS_NAME)
            log(f"  (c) {setting:8s} {ENS_NAME} AUROC {ci(r)}")
            for a, b, _ in C_CONTRASTS[:5]:
                r = row(fc, target=tg, setting=setting, row_type="contrast", model=f"{a} - {b}")
                log(f"  (c) {setting:8s} Retrained - {b:34s} {dci(r)}")


def main():
    t_start = time.time()
    d, tr, te, ext = main_cohort()
    assert len(tr) == N_TRAIN and len(te) == N_TEST and len(ext) == N_EXT, (len(tr), len(te), len(ext))
    assert d[ID].is_unique and not np.intersect1d(d[ID].values[tr], d[ID].values[te]).size
    assert ext["id"].is_unique and (ext["age"] >= 19).all()
    dtr = d.iloc[tr]
    sg_range = (float(dtr["ua_sg"].min()), float(dtr["ua_sg"].max()))
    E = ext_codings(ext, sg_range)
    log(f"calc_feature_tiers.py  retrained models (bestmodel_v5, {len(FMODEL)} inputs); dev_v2 {len(d)} persons: training "
        f"{len(tr)}, internal test {len(te)} (one person-level split shared by the three thresholds); KNHANES adults "
        f"(X1) {len(ext)}")
    log("Tiers: " + "; ".join(f"{k} = {v}" for k, v in TIERS.items()))
    log(f"LightGBM = LGBM_FIXED + {LGBM_RUN}; LR = StandardScaler + LogisticRegression(C=1, max_iter=5000); "
        f"spline LR = cubic B-splines (5 quantile knots) for age and BMI + age x male; GRID = {len(GRID)} configurations")
    m1u, m2u = E["M1"]["ua_urobilinogen"], E["M2"]["ua_urobilinogen"]
    log(f"Coding: development nitrite+ {int((d['ua_nitrite'] == 1).sum())}/{len(d)} (training "
        f"{int((dtr['ua_nitrite'] == 1).sum())}), KNHANES nitrite+ {int((ext['ua_nitrite'] == 1).sum())}/{len(ext)}; "
        f"urobilinogen == 0.5: development {(d['ua_urobilinogen'] == 0.5).mean():.4f}, KNHANES X1 "
        f"{(ext['ua_urobilinogen'] == 0.5).mean():.4f}; KNHANES urobilinogen == 0 under M1 {(m1u == 0).mean():.4f}, "
        f"M2 {(m2u == 0).mean():.4f}; bilirubin 1.0 / 2.0: development {(d['ua_bilirubin'] == 1).mean():.4f} / "
        f"{(d['ua_bilirubin'] == 2).mean():.4f}, KNHANES X1 {(ext['ua_bilirubin'] == 1).mean():.4f} / "
        f"{(ext['ua_bilirubin'] == 2).mean():.4f}, M1 {(E['M1']['ua_bilirubin'] == 1).mean():.4f} / "
        f"{(E['M1']['ua_bilirubin'] == 2).mean():.4f}; KNHANES SG > training max {sg_range[1]:.3f}: "
        f"{(ext['ua_sg'] > sg_range[1]).mean():.4f}")
    log(f"Primary external coding X1 (cohort.knhanes_x1; rebuilt from the raw KNHANES files and equal to ext). "
        f"Diagnostics: M1 (submission), M2 (uniform printed map), {SGCLIP} (SG clipped to [{sg_range[0]:.3f}, "
        f"{sg_range[1]:.3f}]). HARM = {HARM} (no harmonisation)")

    P, Y = {}, {}
    for t, cfg in TARGETS.items():
        Y[(t, "internal")] = labels(d["urate"], cfg["threshold"])[te]
        Y[(t, "external")] = labels(ext["urate"], cfg["threshold"])
    fit_tiers(d, tr, te, E, P, Y)
    tuned, grid_rows = tune_tiers(d, tr, te, E, P, Y)
    configs = fit_ensemble_arms(d, tr, te, ext, E, P, Y, tuned)

    log("=" * 100 + "\nBootstrap (1000 resamples, seed 42) for every distinct evaluation")
    ev = Evaluations(P, Y)
    register(ev)
    ev.run()
    base = {t: dict(target=cfg["short"], threshold=cfg["threshold"], role=cfg["role"], n_train=len(tr),
                    pos_train=int(labels(d["urate"], cfg["threshold"])[tr].sum())) for t, cfg in TARGETS.items()}
    fa = table_tiers(ev, base)
    fb = table_tuned(ev, tuned, grid_rows, fa)
    fc = table_ensemble(ev, base, configs, grid_rows)
    # sanity: full internal test set and full KNHANES cohort everywhere
    for name, df in (("feature_tiers", fa), ("tuned_per_set", fb), ("ensemble_vs_simple", fc)):
        ev_rows = df[df.row_type != "cv_grid"]
        assert (ev_rows[ev_rows.setting == "internal"].n == N_TEST).all(), name
        assert (ev_rows[ev_rows.setting == "external"].n == N_EXT).all(), name
    # the 12F refits of (a)/(b) are the comparators of (c): identical rows by construction
    for mname in (LR_NAME, LRS_NAME, LGBM_NAME, GBT_NAME):
        a_ = fa[(fa.row_type == "tier") & (fa.model == mname) & (fa.tier == "12F")].set_index(["target", "setting"]).auroc
        c_ = fc[(fc.row_type == "model") & (fc.model == mname)].set_index(["target", "setting"]).auroc
        assert np.allclose(a_.sort_index().values, c_.sort_index().values, rtol=0, atol=0), mname
    fa.to_csv(os.path.join(HERE, "feature_tiers.csv"), index=False, encoding="utf-8")
    fb.to_csv(os.path.join(HERE, "tuned_per_set.csv"), index=False, encoding="utf-8")
    fc.to_csv(os.path.join(HERE, "ensemble_vs_simple.csv"), index=False, encoding="utf-8")
    log(f"  feature_tiers.csv {len(fa)} rows; tuned_per_set.csv {len(fb)} rows; ensemble_vs_simple.csv {len(fc)} rows")
    log_tables(fa, fb, fc)
    cs = write_coding(P, Y, ext)
    log(f"  feature_tiers_coding_sensitivity.csv {len(cs)} rows")
    for name, df in (("feature_tiers", fa), ("tuned_per_set", fb), ("ensemble_vs_simple", fc)):
        c = df[df["delong_p"].notna()]
        bad = c[c.delong_ci_disagree == True]  # noqa: E712
        log(f"  {name}: contrasts {len(c)}, delong_p == 0: {int((c.delong_p == 0).sum())}, min p {c.delong_p.min():.1e}, "
            f"DeLong/bootstrap-CI disagreements: {len(bad)}")
        for _, r in bad.iterrows():
            log(f"    disagree: {r.target} {r.setting} {r.comparison}: {dci(r)}")
    key_results(fa, fb, fc)
    log(f"\nDone in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
