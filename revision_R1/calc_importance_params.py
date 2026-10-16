"""Hyperparameters, whole-ensemble permutation importance, ensemble SHAP, partial dependence and the specific-gravity
analysis of the retrained models (bestmodel_v5).

Models: the three calibrated soft-voting ensembles retrained on 2026-10-16 (train_v5.py: Optuna-tuned LightGBM +
XGBoost + CatBoost, soft voting, isotonic CalibratedClassifierCV cv=5) with 12 inputs (cohort.FMODEL = 9 dipstick fields
+ age, male, BMI; leukocyte esterase is not an input because KNHANES does not measure it).
Cohorts (cohort.main_cohort): internal test = the 25,907 test persons of the one person-level 80/20 split of dev_v2
(the same persons for every threshold; no test person is in the training split); external = the 49,987 KNHANES adults
aged >= 19 (ext_v2, dipstick inputs coded by the label-matched mapping X1).  The predictions are recomputed and checked
against preds_internal_clean.csv / preds_external_corrected.csv (calc_main.py) and the AUROCs against
main_performance.csv.  Cohort labels in the outputs: "external KNHANES >=19" and "internal person-clean test" (the
internal test persons are disjoint from the training persons by construction).

(a) hyperparameters.csv      tuned + fixed hyperparameters of every base learner, read from the retrained models
                             (calibrated_classifiers_[k].estimator.named_estimators_, all 5 calibration folds), with
                             a check that they are identical across the folds and that the saved values equal the
                             Optuna best parameters written by train_v5.py (bestmodel_v5/tune_<target>.json).
(b) permutation_importance.csv
                             drop in AUROC of the WHOLE calibrated ensemble when one input is permuted (10 repeats,
                             seed 42) on the KNHANES adults and on the internal test persons; <=20,000 records per
                             evaluation (all cases kept, non-cases sampled at random, seed 42 -- AUROC does not depend
                             on the case:non-case ratio).
(c) shap_ensemble.csv        TreeSHAP of all 15 base learners (3 learners x 5 calibration folds) on 1,000 KNHANES
                             records (seed 42; primary) and 1,000 internal test records (secondary).  Primary
                             importance = mean|SHAP| (log-odds, tree-path-dependent) averaged over the 15 learners.
                             Soft voting averages PROBABILITIES, so averaging log-odds SHAP only approximates the
                             ensemble before isotonic calibration; the exact SHAP of the pre-calibration ensemble
                             (interventional TreeSHAP on the probability scale, 200 background records, averaged over
                             the 15 learners -- exact by linearity) is reported alongside.  The same log-odds TreeSHAP
                             is also computed on the FULL cohorts (all 49,987 KNHANES adults; all 25,907 internal test
                             persons) as a stability check of the ranking (columns *_fullcohort).  `direction` does
                             not report only the sign of Spearman(value, SHAP): it flags a non-monotone partial
                             dependence (pdp_shape) and a direction that is opposite in men and women.
     pdp_calibrated.csv      partial dependence of the calibrated ensemble (and of the pre-calibration vote) on SG,
                             pH, age and BMI, averaged over the FULL cohort (KNHANES adults; internal test persons),
                             overall and within men and within women (sex kept at its observed value), with a shape
                             label per curve (evaluated on grid values within half a grid step of the cohort's
                             1st-99th percentile range, column within_support; a step counts when >= 5% of the curve's
                             range and >= 5% relative).
(d) shap_by_sg.csv           per threshold, cohort (KNHANES adults; internal test) and subgroup (all/men/women): by SG
                             level (1.005 ... 1.030) the mean learner-averaged log-odds SHAP of SG (full cohort;
                             per-learner range and sign agreement; 1,000-record sample and exact probability-scale SHAP
                             alongside), the observed prevalence, and the calibrated partial dependence with SG set to
                             that level.  SG is rounded to the nearest development dipstick level (0.005 steps) and
                             clipped to 1.005-1.030, i.e. bins [level-0.0025, level+0.0025) (KNHANES records SG in
                             steps of 0.001, 1.000-1.050).
     shap_age_by_sex.csv     mean learner-averaged log-odds SHAP of age by age band within men and within women (full
                             cohorts), to show the sex-specific direction of the age effect.
(e) sg_univariate.csv        model-free: SG vs serum urate in dev_v2 (all 129,535 persons; cohort label
                             "development (dev_v2)") and in the KNHANES adults ("external (ext_v2, KNHANES >=19)"):
                             prevalence by SG level (overall and by sex; development and KNHANES side by side, all
                             three thresholds, with share of men and mean age per level), Spearman(SG, urate), odds
                             ratios per +0.010 SG (crude, then sequentially adjusted for sex, age, BMI and the other
                             model inputs; within sex), and odds ratios with SG as a CATEGORICAL term (reference
                             1.010), crude and adjusted for sex, age+sex, age+sex+BMI (primary), sex x 10-year age band
                             + BMI, and the other model inputs, for every threshold in both cohorts (within-sex models
                             at >=7.0).  Likelihood-ratio test of SG (5 df) for every categorical model.
The log ends with an interpretation section generated from these numbers (top-5 inputs by every ranking; SG shape
and confounding; age shape by sex).

Removed with the retraining (decision of 2026-10-16), because they concerned only the original (submitted) models:
the reproduction of the submitted SHAP figure (LightGBM member of the first calibration fold on the row-level internal
test: shap_ensemble.csv rows "submitted figure computation ..." and columns submitted_fig_fixed_order_rank /
adult_recomputed_fixed_order_rank); the leukocyte esterase rows (constant 0.321 imputed in KNHANES; no longer an
input); and the notes that attributed the near-zero nitrite importance to the nitrite parsing error (nitrite is parsed
correctly in dev_v2, so the retrained models could learn a nitrite effect).
Log: calc_importance_params_log.txt
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"                                   # cap BLAS/OpenMP threads before any import

import functools, json, time, warnings
import numpy as np, pandas as pd

# cohort and common are imported BEFORE shap / lightgbm / xgboost: the reverse order can deadlock while the
# XGBoost learners are unpickled.
from cohort import HERE, TARGETS, FMODEL, ID, ADULT_AGE, MODEL_DIR_V5, labels, load_model, main_cohort
import common  # noqa: F401,E402  (import order only; no calibration statistic is computed in this script)

import shap  # noqa: E402
import shap.explainers._tree as _shap_tree  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.stats import spearmanr, chi2  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 42
N_THREADS = 4
N_REPEATS = 10
MAX_PERM = 20_000
N_SHAP = 1_000
N_BG = 200
N_INTERNAL, N_EXTERNAL = 25_907, 49_987                    # main analysis (main_performance.csv)
LEARNERS = ("lgb", "xgb", "cat")
LABEL = {"ua_glucose": "Glucose", "ua_protein": "Protein", "ua_ketone": "Ketone", "ua_bilirubin": "Bilirubin",
         "ua_blood": "Occult blood", "ua_urobilinogen": "Urobilinogen", "ua_nitrite": "Nitrite",
         "ua_pH": "pH", "ua_sg": "Specific gravity", "age": "Age", "male": "Male sex", "bmi_unified": "BMI"}
assert list(LABEL) == FMODEL, "LABEL must list the model inputs in FMODEL order"
# Optuna search space of train_v5.py (the recipe of 05_src/run_uric_module_v4.py, copied; 25 TPE trials per learner,
# 3-fold CV AUROC).
TUNED = {"lgb": {"n_estimators", "max_depth", "learning_rate", "num_leaves", "min_child_samples", "reg_alpha",
                 "reg_lambda"},
         "xgb": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"},
         "cat": {"iterations", "depth", "learning_rate", "l2_leaf_reg"}}
XGB_EFFECTIVE = {"reg_lambda": "lambda", "reg_alpha": "alpha", "gamma": "gamma", "max_bin": "max_bin",
                 "grow_policy": "grow_policy", "max_leaves": "max_leaves", "colsample_bylevel": "colsample_bylevel",
                 "colsample_bynode": "colsample_bynode", "max_delta_step": "max_delta_step",
                 "sampling_method": "sampling_method"}
CAT_EFFECTIVE = ("loss_function", "border_count", "bootstrap_type", "subsample", "random_strength", "rsm",
                 "grow_policy", "min_data_in_leaf", "leaf_estimation_method", "leaf_estimation_iterations",
                 "boost_from_average", "nan_mode")
PDP_GRID = {"ua_sg": np.round(np.arange(1.005, 1.0301, 0.005), 3), "ua_pH": np.arange(5.0, 8.51, 0.5),
            "age": np.arange(20, 81, 10), "bmi_unified": np.arange(16, 36.1, 2)}
KEY = ("ua_sg", "ua_pH", "age", "bmi_unified")
# Specific gravity: development dipstick levels (KNHANES rounded to the nearest level, clipped to 1.005-1.030).
SG_ORDER = ["<=1.005", "1.010", "1.015", "1.020", "1.025", ">=1.030"]
SG_VALUE = dict(zip(SG_ORDER, (1.005, 1.010, 1.015, 1.020, 1.025, 1.030)))
SG_REF = "1.010"
# both cohorts are adults (age >= ADULT_AGE), so the first band starts at 19
AGE_EDGES = [ADULT_AGE, 30, 40, 50, 60, 70, 80, np.inf]
AGE_BANDS = ["19-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"]
EXT, INT = "external KNHANES >=19", "internal person-clean test"
DEV_NAME, EXT_NAME = "development (dev_v2)", "external (ext_v2, KNHANES >=19)"     # sg_univariate.csv cohorts
SUBGROUPS = ("all", "men", "women")

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ shap / XGBoost 3.1 compatibility
# XGBoost 3.x stores base_score as a one-element vector string ("[1.55E-1]") that shap 0.48 cannot parse.
# The patch only strips the brackets for single-output models; SHAP values are checked below against
# the raw margin (additivity).
_orig_decode = _shap_tree.decode_ubjson_buffer


def _decode_scalar_base_score(fd):
    j = _orig_decode(fd)
    lmp = j.get("learner", {}).get("learner_model_param", {})
    bs = lmp.get("base_score")
    if isinstance(bs, str) and bs.startswith("[") and "," not in bs:
        lmp["base_score"] = bs.strip("[]")
    return j


_shap_tree.decode_ubjson_buffer = _decode_scalar_base_score


# ------------------------------------------------------------------ model helpers
def members(model):
    """(fold, learner name, fitted estimator) for the 15 base learners."""
    for k, cc in enumerate(model.calibrated_classifiers_):
        for name in LEARNERS:
            yield k, name, cc.estimator.named_estimators_[name]


def cap_threads(model):
    """In-memory only (the saved files are untouched): 4 threads for every learner."""
    for _, name, est in members(model):
        if name in ("lgb", "xgb"):
            est.set_params(n_jobs=N_THREADS)
        else:
            est.predict_proba = functools.partial(type(est).predict_proba, est, thread_count=N_THREADS)
            est.get_feature_importance = functools.partial(type(est).get_feature_importance, est,
                                                           thread_count=N_THREADS)
    return model


def p_cal(model, X):
    return model.predict_proba(pd.DataFrame(np.asarray(X, float), columns=FMODEL))[:, 1]


def p_precal(model, X):
    """Soft-voting probability averaged over the 5 folds, before isotonic calibration."""
    X = pd.DataFrame(np.asarray(X, float), columns=FMODEL)
    return np.mean([cc.estimator.predict_proba(X)[:, 1] for cc in model.calibrated_classifiers_], axis=0)


def p_both(model, X):
    """(calibrated, pre-calibration) ensemble probability in one pass over the 15 learners.
    Calibrated = mean over folds of isotonic(soft vote), as CalibratedClassifierCV.predict_proba (checked in main)."""
    X = pd.DataFrame(np.asarray(X, float), columns=FMODEL); cal, raw = [], []
    for cc in model.calibrated_classifiers_:
        r = cc.estimator.predict_proba(X)[:, 1]
        raw.append(r); cal.append(cc.calibrators[0].predict(r))
    return np.mean(cal, axis=0), np.mean(raw, axis=0)


def raw_margin(name, est, X):
    if name == "lgb":
        return est.predict(X, raw_score=True)
    if name == "xgb":
        return est.predict(X, output_margin=True)
    return est.predict(X, prediction_type="RawFormulaVal", thread_count=N_THREADS)


def fmt(v):
    return repr(v) if isinstance(v, float) else str(v)


def pct(v, d=1):
    return "n/a" if v is None or pd.isna(v) else f"{100 * v:.{d}f}%"


def sgn(v, d=3):
    """Signed number for the log; n/a when undefined (e.g. Spearman with an input constant within a sex)."""
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{d}f}"


# ------------------------------------------------------------------ (a) hyperparameters
def hyperparameter_rows(target, model):
    vals = {}                                  # (learner, param, source) -> [value per fold]
    for k, name, est in members(model):
        d = {}
        for p, v in est.get_params().items():
            if v is not None:
                d[(p, "get_params (as saved)")] = v
        if name == "xgb":
            cfg = json.loads(est.get_booster().save_config())
            gb = cfg["learner"]["gradient_booster"]
            tp = gb.get("tree_train_param", {})
            saved = est.get_params()
            for sk, ck in XGB_EFFECTIVE.items():
                if saved.get(sk) is None and ck in tp:
                    d[(sk, "booster config (library default in effect)")] = tp[ck]
            tm = gb.get("gbtree_train_param", {}).get("tree_method")
            if saved.get("tree_method") is None and tm is not None:
                d[("tree_method", "booster config (library default in effect)")] = tm
            spw = cfg["learner"].get("objective", {}).get("reg_loss_param", {}).get("scale_pos_weight")
            if saved.get("scale_pos_weight") is None and spw is not None:
                d[("scale_pos_weight", "booster config (library default in effect)")] = spw
            d[("n_trees_fitted", "fitted model")] = est.get_booster().num_boosted_rounds()
        elif name == "cat":
            allp = est.get_all_params()
            for p in CAT_EFFECTIVE:
                if p in allp and p not in est.get_params():
                    d[(p, "get_all_params (library default in effect)")] = allp[p]
            d[("n_trees_fitted", "fitted model")] = est.tree_count_
        else:
            d[("n_trees_fitted", "fitted model")] = est.booster_.num_trees()
        for (p, src), v in d.items():
            vals.setdefault((name, p, src), {})[k] = v
    rows = []
    for (name, p, src), per_fold in vals.items():
        reprs = [fmt(per_fold.get(k)) for k in range(len(model.calibrated_classifiers_))]
        rows.append(dict(target=target, threshold=TARGETS[target]["threshold"], learner=name, param=p,
                         value=reprs[0], tuned=p in TUNED[name], source=src,
                         n_folds=len(reprs), n_distinct_across_folds=len(set(reprs)),
                         identical_across_folds=len(set(reprs)) == 1))
    return rows


def tune_json_check(target, model):
    """Values saved by train_v5.py (bestmodel_v5/tune_<target>.json, best_params: the Optuna best values plus the fixed
    seeds/verbosity/threads) against get_params of every base learner in every calibration fold.  Call before
    cap_threads.  Returns (values compared, mismatches, saved keys that get_params does not return)."""
    with open(os.path.join(MODEL_DIR_V5, f"tune_{target}.json"), encoding="utf-8") as fh:
        best = json.load(fh)["best_params"]
    n, bad, absent = 0, [], set()
    for k, name, est in members(model):
        gp = est.get_params()
        assert TUNED[name] <= set(best[name]), f"{target} {name}: a tuned parameter is missing from tune_{target}.json"
        for p, v in best[name].items():
            if p not in gp:
                absent.add(f"{name}.{p}")
                continue
            n += 1
            if gp[p] != v:
                bad.append(f"fold {k + 1} {name}.{p}: model {gp[p]!r} vs tune json {v!r}")
    return n, bad, sorted(absent)


# ------------------------------------------------------------------ (b) permutation importance
def subsample_idx(y, cap, seed=SEED):
    """All records if <= cap; otherwise every case plus randomly chosen non-cases up to `cap`."""
    n = len(y)
    if n <= cap:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) > cap // 2:                                  # not reached here; stratified fallback
        from sklearn.model_selection import train_test_split
        return np.sort(train_test_split(np.arange(n), train_size=cap, random_state=seed, stratify=y)[0])
    return np.sort(np.concatenate([pos, rng.choice(neg, cap - len(pos), replace=False)]))


def permutation_importance(model, X, y, n_repeats=N_REPEATS, seed=SEED):
    Xv = np.asarray(X, float); n, m = Xv.shape
    base = roc_auc_score(y, p_cal(model, Xv))
    rng = np.random.RandomState(seed)
    drops = np.zeros((n_repeats, m))
    for r in range(n_repeats):
        blocks = []
        for j in range(m):
            Xp = Xv.copy(); Xp[:, j] = Xv[rng.permutation(n), j]; blocks.append(Xp)
        P = p_cal(model, np.vstack(blocks)).reshape(m, n)
        drops[r] = base - np.array([roc_auc_score(y, P[j]) for j in range(m)])
    return base, drops


# ------------------------------------------------------------------ (c) SHAP and partial dependence
def tree_shap_logodds(name, est, X):
    ex = shap.TreeExplainer(est)
    sv = np.asarray(ex.shap_values(X))
    if sv.ndim == 3:
        sv = sv[..., -1]
    ev = float(np.ravel(ex.expected_value)[-1])
    err = float(np.max(np.abs(sv.sum(1) + ev - raw_margin(name, est, X))))
    return sv, ev, err


def tree_shap_prob(est, X, bg):
    ex = shap.TreeExplainer(est, data=bg, model_output="probability", feature_perturbation="interventional")
    sv = np.asarray(ex.shap_values(X, check_additivity=False))
    if sv.ndim == 3:
        sv = sv[..., -1]
    ev = float(np.ravel(ex.expected_value)[-1])
    err = float(np.max(np.abs(sv.sum(1) + ev - est.predict_proba(X)[:, 1])))
    return sv, ev, err


def shap_full(model, X):
    """Tree-path-dependent log-odds SHAP of the 15 learners on every record: array (15, n, 12), max additivity error."""
    Xs = X.reset_index(drop=True); L, errs = [], []
    for _, name, est in members(model):
        sv, _, err = tree_shap_logodds(name, est, Xs); L.append(sv); errs.append(err)
    return np.stack(L), max(errs)


def spearman(x, s):
    x = np.asarray(x, float); s = np.asarray(s, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(s) == 0:
        return np.nan
    return float(spearmanr(x, s).correlation)


def rank_desc(v):
    return pd.Series(v).rank(ascending=False, method="min").astype(int).values


def pdp_shape(grid, pred, tol=0.05):
    """Shape of a partial-dependence curve.  A step between neighbouring grid values counts only when it is
    >= `tol` of the curve's range AND >= `tol` relative to the preceding value; smaller steps count as flat."""
    pred = np.asarray(pred, float); rng = pred.max() - pred.min()
    if rng <= 0.05 * np.mean(pred):
        return "flat (range < 5% of mean)"
    d = np.diff(pred)
    s = [int(np.sign(v)) for v, p0 in zip(d, pred[:-1]) if abs(v) >= tol * rng and abs(v) >= tol * abs(p0)]
    seq = [v for i, v in enumerate(s) if i == 0 or v != s[i - 1]]
    lo, hi = grid[int(np.argmin(pred))], grid[int(np.argmax(pred))]
    if seq == [1]:
        return "monotone increasing"
    if seq == [-1]:
        return "monotone decreasing"
    if seq == [-1, 1]:
        return f"U-shaped (minimum at {lo:g})"
    if seq == [1, -1]:
        return f"inverted U (maximum at {hi:g})"
    return f"non-monotone (minimum at {lo:g}, maximum at {hi:g})"


def largest_step(grid, pred):
    d = np.diff(pred); k = int(np.argmax(np.abs(d)))
    return f"largest step {grid[k]:g} -> {grid[k + 1]:g} ({100 * d[k] / pred[k]:+.0f}%)"


def pdp_full(target, model, X, cohort):
    """Calibrated (and pre-calibration) partial dependence on every record of the cohort, overall and within sex."""
    Xv = np.asarray(X, float); n = len(Xv); men = X["male"].values == 1
    masks = {"all": np.ones(n, bool), "men": men, "women": ~men}; rows = []
    for f, grid in PDP_GRID.items():
        j = FMODEL.index(f)
        Z = np.tile(Xv, (len(grid), 1)); Z[:, j] = np.repeat(grid, n)
        pc, pu = p_both(model, Z); pc = pc.reshape(len(grid), n); pu = pu.reshape(len(grid), n)
        # shape labels use only grid values within half a grid step of the cohort's 1st-99th percentile range
        # (a grid value stands for the bin around it; points further out are extrapolation)
        lo, hi = np.percentile(Xv[:, j], [1, 99]); h = float(np.min(np.diff(grid))) / 2
        sup = (grid >= lo - h - 1e-9) & (grid <= hi + h + 1e-9)
        for sub, m in masks.items():
            a, b = pc[:, m].mean(1), pu[:, m].mean(1)
            shape = pdp_shape(grid[sup], a[sup]); rho = spearman(grid[sup], a[sup])
            for g, s_, va, vb in zip(grid, sup, a, b):
                rows.append(dict(target=target, threshold=TARGETS[target]["threshold"], cohort=cohort, subgroup=sub,
                                 feature=f, label=LABEL[f], value=float(g), mean_pred_calibrated=va,
                                 mean_pred_precalibration=vb, ratio_to_first_grid_value=va / a[0],
                                 n_records=int(m.sum()), within_support=bool(s_), pdp_shape=shape,
                                 pdp_spearman_grid=rho, pdp_largest_step=largest_step(grid[sup], a[sup])))
    return rows


def pdp_summary(pdp_rows):
    """{(feature, subgroup): (shape, Spearman(grid, PDP), largest step, {grid value: PDP})} for one target x cohort."""
    d = pd.DataFrame(pdp_rows)
    return {(f, s): (g.pdp_shape.iloc[0], g.pdp_spearman_grid.iloc[0], g.pdp_largest_step.iloc[0],
                     dict(zip(g.value.round(3), g.mean_pred_calibrated)))
            for (f, s), g in d.groupby(["feature", "subgroup"])}


# segment over which the sex-specific direction is compared (age: mid-life, where the curves diverge)
SEGMENT = {"age": (40.0, 70.0)}


def segment_change(f, curve):
    a, b = SEGMENT.get(f, (min(curve), max(curve)))
    return a, b, curve[b] / curve[a] - 1


def monotone_text(r):
    return "higher value -> higher predicted risk" if r > 0 else "higher value -> lower predicted risk"


def direction_text(f, rho, rho_m_full, rho_w_full, pdps):
    """Direction of an input's effect: monotone only when neither the partial dependence nor the sexes disagree."""
    if np.isnan(rho):
        return "n/a (input constant in sample)"
    parts = []
    if (f, "all") in pdps:
        sh, _, step, curve = pdps[(f, "all")]
        if f == "ua_sg" and not sh.startswith(("monotone", "flat")):
            g = sorted(curve); k = int(np.argmin(np.diff([curve[v] for v in g])))
            parts.append(f"non-monotone{', threshold-like' if (g[k], g[k + 1]) == (1.015, 1.02) else ''}: "
                         f"partial dependence {sh}, {step}")
        elif not sh.startswith(("monotone", "flat")):
            parts.append(f"non-monotone: partial dependence {sh}")
        (sm_, rm, _, cm), (sw_, rw, _, cw) = pdps[(f, "men")], pdps[(f, "women")]
        a, b, chm = segment_change(f, cm); _, _, chw = segment_change(f, cw)
        detail = (f"partial dependence men {sm_}, Spearman {rm:+.2f}; women {sw_}, Spearman {rw:+.2f}; "
                  f"{a:g} -> {b:g}: men {chm:+.0%}, women {chw:+.0%}")
        if f in SEGMENT:        # age: judged on the mid-life segment (same rule as the log interpretation)
            if np.sign(chm) != np.sign(chw) and min(abs(chm), abs(chw)) >= 0.10:
                parts.append(f"opposite direction by sex ({detail})")
            elif np.sign(rm) != np.sign(rw):
                parts.append(f"direction differs by sex, not opposite over {a:g}-{b:g} ({detail})")
        elif np.sign(rm) != np.sign(rw) and min(abs(rm), abs(rw)) >= 0.25:
            parts.append(f"opposite direction by sex ({detail})")
    elif (not np.isnan(rho_m_full) and not np.isnan(rho_w_full) and np.sign(rho_m_full) != np.sign(rho_w_full)
          and min(abs(rho_m_full), abs(rho_w_full)) >= 0.10):
        parts.append(f"opposite direction by sex (full-cohort SHAP Spearman men {rho_m_full:+.2f}, "
                     f"women {rho_w_full:+.2f})")
    if not parts:
        return monotone_text(rho)
    return "; ".join(parts) + f"; overall Spearman(value, SHAP) {rho:+.2f} summarises only the net trend"


def shap_note(x, f, dev_mode):
    """Flags where SHAP reflects a constant input or a coding/distribution difference from the development data."""
    notes = []
    if pd.Series(x).nunique() == 1:
        notes.append("input constant in sample: tree-path-dependent SHAP only measures the offset of this value "
                     "from the training distribution (interventional SHAP = 0)")
    else:
        mode = pd.Series(x).mode().iloc[0]
        if f in dev_mode and not np.isclose(mode, dev_mode[f]):
            notes.append(f"modal value {mode:g} differs from the development data ({dev_mode[f]:g}): SHAP partly "
                         "reflects a coding/distribution difference")
    return "; ".join(notes)


def shap_block(target, model, X, bg, analysis, dev_mode, full):
    """All 15 learners on the 1,000-record sample; full-cohort log-odds SHAP alongside.
    Returns per-feature rows, QC numbers and the sample's learner-averaged log-odds and probability SHAP."""
    Xs = X.reset_index(drop=True)
    men = Xs["male"].values == 1
    L, Lp, names, ev_p, errs, errs_p = [], [], [], [], [], []
    for k, name, est in members(model):
        sv, _, err = tree_shap_logodds(name, est, Xs)
        svp, evp, errp = tree_shap_prob(est, Xs, bg)
        L.append(sv); Lp.append(svp); names.append(name); ev_p.append(evp); errs.append(err); errs_p.append(errp)
    L, Lp, names = np.stack(L), np.stack(Lp), np.array(names)            # (15, n, 12)
    A = np.abs(L).mean(1)                                                  # (15, 12) mean|SHAP| per learner
    ranks_per_learner = np.vstack([rank_desc(a) for a in A])
    S_avg = L.mean(0); P_avg = Lp.mean(0)
    # exact decomposition check: pre-calibration ensemble prob = mean(expected) + sum(prob SHAP)
    ens_err = float(np.max(np.abs(P_avg.sum(1) + np.mean(ev_p) - p_precal(model, Xs))))
    main = A.mean(0); rk = rank_desc(main)
    prob_imp = np.abs(P_avg).mean(0); rk_p = rank_desc(prob_imp)
    Lf, Xf = full["L"], full["X"].reset_index(drop=True)
    Af = np.abs(Lf).mean(1).mean(0); rkf = rank_desc(Af); Sf = Lf.mean(0); menf = Xf["male"].values == 1
    pdps = full["pdps"]
    rows = []
    for j, f in enumerate(FMODEL):
        x = Xs[f].values; xf = Xf[f].values
        rho_l = np.array([spearman(x, L[i][:, j]) for i in range(len(L))])
        rho = spearman(x, S_avg[:, j]); rho_p = spearman(x, P_avg[:, j])
        rho_mf, rho_wf = spearman(xf[menf], Sf[menf, j]), spearman(xf[~menf], Sf[~menf, j])
        rows.append(dict(
            target=target, threshold=TARGETS[target]["threshold"], analysis=analysis, n_records=len(Xs),
            n_learners=len(L), feature=f, label=LABEL[f], n_distinct_values_in_sample=int(pd.Series(x).nunique()),
            mean_abs_shap_logodds=main[j], rank=int(rk[j]),
            mean_abs_shap_lgb=A[names == "lgb", j].mean(), mean_abs_shap_xgb=A[names == "xgb", j].mean(),
            mean_abs_shap_cat=A[names == "cat", j].mean(),
            rank_min_across_learners=int(ranks_per_learner[:, j].min()),
            rank_max_across_learners=int(ranks_per_learner[:, j].max()),
            mean_abs_of_learner_avg_shap_logodds=np.abs(S_avg[:, j]).mean(),
            spearman_value_vs_shap=rho,
            direction=direction_text(f, rho, rho_mf, rho_wf, pdps),
            pdp_shape=pdps.get((f, "all"), ("",))[0], pdp_shape_men=pdps.get((f, "men"), ("",))[0],
            pdp_shape_women=pdps.get((f, "women"), ("",))[0],
            n_learners_rho_pos=int(np.nansum(rho_l > 0)), n_learners_rho_neg=int(np.nansum(rho_l < 0)),
            spearman_men=spearman(x[men], S_avg[men, j]), spearman_women=spearman(x[~men], S_avg[~men, j]),
            mean_abs_shap_prob_precal_ensemble=prob_imp[j], rank_prob=int(rk_p[j]), spearman_prob=rho_p,
            n_fullcohort=len(Xf), mean_abs_shap_logodds_fullcohort=Af[j], rank_fullcohort=int(rkf[j]),
            spearman_fullcohort=spearman(xf, Sf[:, j]), spearman_men_fullcohort=rho_mf,
            spearman_women_fullcohort=rho_wf,
            note=shap_note(x, f, dev_mode)))
    qc = dict(max_additivity_err_logodds=max(errs), max_additivity_err_prob=max(errs_p), ensemble_prob_err=ens_err)
    return rows, qc, S_avg, P_avg


# ------------------------------------------------------------------ (d) SHAP by SG level, age by sex
def sg_level(sg):
    """Nearest development dipstick level (0.005 steps), clipped to 1.005-1.030."""
    v = np.clip(np.round(np.asarray(sg, float) / 0.005) * 0.005, 1.005, 1.030)
    lab = {1.005: "<=1.005", 1.030: ">=1.030"}
    return pd.Series([lab.get(round(x, 3), f"{x:.3f}") for x in v], dtype="object")


def shap_by_sg_rows(target, c, Xs, S_s, P_s):
    """Mean SG SHAP by SG level (full cohort and 1,000-record sample) with the calibrated PDP at that level."""
    j = FMODEL.index("ua_sg"); X = c["X"].reset_index(drop=True); y = c["y"]
    Lj = c["L"][:, :, j]; S = Lj.mean(0)
    lvl = sg_level(X["ua_sg"]).values; men = X["male"].values == 1
    Xs = Xs.reset_index(drop=True); lvl_s = sg_level(Xs["ua_sg"]).values; men_s = Xs["male"].values == 1
    pdp = pd.DataFrame(c["pdp_rows"]); pdp = pdp[pdp.feature == "ua_sg"]
    rows = []
    for sub, m, ms in (("all", np.ones(len(X), bool), np.ones(len(Xs), bool)), ("men", men, men_s),
                       ("women", ~men, ~men_s)):
        pq = pdp[pdp.subgroup == sub].set_index(pdp[pdp.subgroup == sub].value.round(3))
        ref_shap = S[m & (lvl == SG_REF)].mean(); ref_pdp = pq.loc[SG_VALUE[SG_REF], "mean_pred_calibrated"]
        ref_per = Lj[:, m & (lvl == SG_REF)].mean(1)                # each learner's own mean at 1.010
        for l in SG_ORDER:
            k = m & (lvl == l); ks = ms & (lvl_s == l); per = Lj[:, k].mean(1)
            p = pq.loc[SG_VALUE[l]]
            rows.append(dict(
                target=target, threshold=TARGETS[target]["threshold"], cohort=c["name"], subgroup=sub, sg_level=l,
                sg_value=SG_VALUE[l], sg_raw_min=float(X["ua_sg"].values[k].min()),
                sg_raw_max=float(X["ua_sg"].values[k].max()), n_records=int(k.sum()), cases=int(y[k].sum()),
                observed_prevalence=float(y[k].mean()),
                mean_shap_sg_logodds=float(S[k].mean()), sd_shap_sg_logodds=float(S[k].std(ddof=1)),
                mean_shap_sg_minus_ref_1010=float(S[k].mean() - ref_shap),
                learner_level_mean_min=float(per.min()), learner_level_mean_max=float(per.max()),
                n_learners_mean_pos=int((per > 0).sum()), n_learners_mean_neg=int((per < 0).sum()),
                n_learners_below_own_1010=int((per < ref_per).sum()) if l != SG_REF else np.nan,
                n_sample1000=int(ks.sum()),
                mean_shap_sg_logodds_sample1000=float(S_s[ks, j].mean()) if ks.any() else np.nan,
                mean_shap_sg_prob_precal_sample1000=float(P_s[ks, j].mean()) if ks.any() else np.nan,
                pdp_calibrated_sg_set_to_level=float(p.mean_pred_calibrated),
                pdp_precalibration_sg_set_to_level=float(p.mean_pred_precalibration),
                pdp_calibrated_ratio_vs_1010=float(p.mean_pred_calibrated / ref_pdp),
                pdp_calibrated_diff_vs_1010=float(p.mean_pred_calibrated - ref_pdp),
                pdp_n_records=int(p.n_records)))
    return rows


def age_bands(age):
    band = pd.cut(age, AGE_EDGES, labels=AGE_BANDS, right=False)
    assert band.notna().all(), "age outside the adult age bands"
    return band.astype(str)


def age_by_sex_rows(target, c):
    j = FMODEL.index("age"); X = c["X"].reset_index(drop=True); y = c["y"]
    La = c["L"][:, :, j]; S = La.mean(0)
    band = age_bands(X["age"]).values
    men = X["male"].values == 1; rows = []
    for sub, m in (("men", men), ("women", ~men), ("all", np.ones(len(X), bool))):
        rho = spearman(X["age"].values[m], S[m])
        for b in AGE_BANDS:
            k = m & (band == b)
            if not k.any():
                continue
            per = La[:, k].mean(1)
            rows.append(dict(target=target, threshold=TARGETS[target]["threshold"], cohort=c["name"], subgroup=sub,
                             age_band=b, n_records=int(k.sum()), mean_age=float(X["age"].values[k].mean()),
                             cases=int(y[k].sum()), observed_prevalence=float(y[k].mean()),
                             mean_shap_age_logodds=float(S[k].mean()), sd_shap_age_logodds=float(S[k].std(ddof=1))
                             if k.sum() > 1 else np.nan,
                             learner_level_mean_min=float(per.min()), learner_level_mean_max=float(per.max()),
                             n_learners_mean_pos=int((per > 0).sum()), n_learners_mean_neg=int((per < 0).sum()),
                             spearman_age_vs_shap_in_subgroup=rho))
    return rows


# ------------------------------------------------------------------ (e) specific gravity vs serum urate
def fit_logit(y, X):
    """Unpenalised maximum-likelihood logistic regression (statsmodels Newton-Raphson)."""
    Xc = X.astype(float).copy(); Xc.insert(0, "const", 1.0)
    return sm.Logit(np.asarray(y, float), Xc).fit(disp=0, maxiter=200)


def covariates(d, spec):
    if spec == "sexage":
        cell = d["male"].astype(int).astype(str) + "_" + age_bands(d["age"])
        return pd.concat([pd.get_dummies(cell, prefix="sexage", drop_first=True, dtype=float), d[["bmi_unified"]]],
                         axis=1)
    if spec == "others":        # the other model inputs (FMODEL without SG) with >= 20 records off their modal value
        return d[[f for f in FMODEL if f != "ua_sg" and d[f].value_counts().iloc[1:].sum() >= 20]]
    return d[list(spec)]


CONT_MODELS = [("crude", []), ("adjusted for sex", ["male"]), ("adjusted for age, sex", ["age", "male"]),
               ("adjusted for age, sex, BMI", ["age", "male", "bmi_unified"]),
               ("adjusted for the other inputs", "others")]
CAT_MODELS = [("crude", []), ("adjusted for sex", ["male"]), ("adjusted for age, sex", ["age", "male"]),
              ("adjusted for age, sex, BMI (primary)", ["age", "male", "bmi_unified"]),
              ("sensitivity: adjusted for sex x 10-year age band, BMI", "sexage"),
              ("sensitivity: adjusted for the other inputs", "others")]
PRIMARY = "adjusted for age, sex, BMI (primary)"


def prep_sg(d):
    d = d.reset_index(drop=True).copy()
    d["lvl"] = sg_level(d["ua_sg"]).values
    d["sg10"] = (d["ua_sg"] - 1.0) / 0.010                       # per +0.010 SG
    for t, cfg in TARGETS.items():
        d[f"y_{cfg['threshold']:g}"] = labels(d["urate"], cfg["threshold"])
    return d


def categorical_or_rows(name, d, thr, model_name, spec):
    y = d[f"y_{thr:g}"].values
    D = pd.DataFrame({f"sg_{l}": (d["lvl"] == l).astype(float) for l in SG_ORDER if l != SG_REF}, index=d.index)
    C = covariates(d, spec)
    base = dict(cohort=name, row_type="logistic OR by SG level (reference 1.010)", threshold=thr, model=model_name,
                n_model=len(d), cases_model=int(y.sum()), covariates=", ".join(map(str, C.columns)) if len(C.columns)
                <= len(FMODEL) else f"{len(C.columns)} terms (sex x age-band cells + BMI)")
    try:
        fit = fit_logit(y, pd.concat([D, C], axis=1)); fit0 = fit_logit(y, C)
        lr = 2 * (fit.llf - fit0.llf); p_joint = float(chi2.sf(lr, D.shape[1]))
        conv = bool(fit.mle_retvals.get("converged", True)); ci = fit.conf_int()
    except Exception as e:                                       # e.g. separation in a sparse stratum
        return [dict(base, sg_level="all", note=f"not estimable: {type(e).__name__}: {e}")]
    rows = []
    for l in SG_ORDER:
        k = d["lvl"] == l
        r = dict(base, sg_level=l, n=int(k.sum()), cases=int(y[k.values].sum()), lr_chi2_sg=float(lr),
                 p_sg_joint=p_joint, converged=conv)
        if l == SG_REF:
            r.update(estimate=1.0, lo=1.0, hi=1.0, p=np.nan, note="reference")
        else:
            c = f"sg_{l}"
            r.update(estimate=float(np.exp(fit.params[c])), lo=float(np.exp(ci.loc[c, 0])),
                     hi=float(np.exp(ci.loc[c, 1])), p=float(fit.pvalues[c]))
        rows.append(r)
    return rows


def sg_rows(name, d):
    """Per-cohort rows: prevalence by level (>=7.0), Spearman, OR per +0.010 SG (>=7.0), categorical ORs."""
    rows = []
    for lvl in SG_ORDER:
        g = d[d["lvl"] == lvl]; gm = g[g["male"] == 1]; gf = g[g["male"] == 0]
        rows.append(dict(cohort=name, row_type="prevalence by SG level", threshold=7.0, sg_level=lvl, n=len(g),
                         cases_7=int(g["y_7"].sum()), prev_7=g["y_7"].mean(),
                         n_men=len(gm), prev_7_men=gm["y_7"].mean(), n_women=len(gf), prev_7_women=gf["y_7"].mean(),
                         pct_men=g["male"].mean(), mean_age=g["age"].mean(), mean_bmi=g["bmi_unified"].mean(),
                         prev_9=g["y_9"].mean(), prev_10=g["y_10"].mean()))
    for grp, g in (("all", d), ("men", d[d["male"] == 1]), ("women", d[d["male"] == 0])):
        rows.append(dict(cohort=name, row_type="Spearman(SG, serum urate)", sg_level=grp, n=len(g),
                         estimate=float(spearmanr(g["ua_sg"], g["urate"]).correlation)))
    y = d["y_7"].values
    for model_name, spec in CONT_MODELS:
        fit = fit_logit(y, pd.concat([d[["sg10"]], covariates(d, spec)], axis=1))
        lo, hi = fit.conf_int().loc["sg10"]
        rows.append(dict(cohort=name, row_type="logistic OR for urate >=7.0 per +0.010 SG", threshold=7.0,
                         sg_level="per +0.010 SG", model=model_name, n=len(d), estimate=float(np.exp(fit.params["sg10"])),
                         lo=float(np.exp(lo)), hi=float(np.exp(hi)), p=float(fit.pvalues["sg10"]),
                         covariates=", ".join(map(str, covariates(d, spec).columns))))
    for sex, g in (("men", d[d["male"] == 1]), ("women", d[d["male"] == 0])):
        for label, cov in (("crude", []), ("adjusted for age, BMI", ["age", "bmi_unified"])):
            fit = fit_logit(g["y_7"].values, g[["sg10"] + cov])
            lo, hi = fit.conf_int().loc["sg10"]
            rows.append(dict(cohort=name, row_type="logistic OR for urate >=7.0 per +0.010 SG", threshold=7.0,
                             sg_level="per +0.010 SG", model=f"{sex}: {label}", n=len(g),
                             estimate=float(np.exp(fit.params["sg10"])), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                             p=float(fit.pvalues["sg10"]), covariates=", ".join(cov)))
    for t, cfg in TARGETS.items():
        thr = cfg["threshold"]
        for model_name, spec in CAT_MODELS:
            rows += categorical_or_rows(name, d, thr, model_name, spec)
        if thr == 7.0:                     # within-sex models: >=9.0 / >=10.0 have few female cases per level
            for sex, g in (("men", d[d["male"] == 1]), ("women", d[d["male"] == 0])):
                rows += categorical_or_rows(name, g.reset_index(drop=True), thr, f"{sex}: adjusted for age, BMI",
                                            ["age", "bmi_unified"])
    return rows


def sg_side_by_side(dev, ext):
    """Prevalence by SG level within sex, development and KNHANES side by side, every threshold."""
    rows = []
    for t, cfg in TARGETS.items():
        thr = cfg["threshold"]; yc = f"y_{thr:g}"
        for sex in SUBGROUPS:
            for lvl in SG_ORDER:
                r = dict(cohort="development (dev_v2) vs KNHANES >=19 (ext_v2)",
                         row_type="prevalence by SG level and sex, both cohorts", threshold=thr, sex=sex, sg_level=lvl)
                for tag, d in (("dev", dev), ("ext", ext)):
                    g = d[d["lvl"] == lvl]
                    if sex != "all":
                        g = g[g["male"] == (1 if sex == "men" else 0)]
                    r.update({f"n_{tag}": len(g), f"cases_{tag}": int(g[yc].sum()), f"prev_{tag}": g[yc].mean(),
                              f"mean_age_{tag}": g["age"].mean(), f"pct_men_{tag}": g["male"].mean()})
                rows.append(r)
    return rows


# ------------------------------------------------------------------ interpretation (generated from the outputs)
def top5(df, rank_col, value_col):
    d = df.sort_values([rank_col, value_col], ascending=[True, False]).head(5)
    return list(d.label)


def interpretation(perm, shp, pdp, sgb, agb, sgu):
    log("\n" + "=" * 100 + "\nINTERPRETATION (generated from the numbers above)\n" + "=" * 100)
    # ---- top-5 inputs
    log("\n[1] Top-5 inputs per threshold")
    for t, cfg in TARGETS.items():
        rk = {}
        for cohort in (EXT, INT):
            rk[f"permutation (AUROC drop), {cohort}"] = top5(perm[(perm.target == t) & (perm.cohort == cohort)],
                                                              "rank", "auroc_drop_mean")
        for tag, an in (("KNHANES", "ensemble, external KNHANES >=19 (primary)"),
                        ("internal", "ensemble, internal person-clean test (secondary)")):
            s = shp[(shp.target == t) & (shp.analysis == an)]
            rk[f"ensemble SHAP mean|log-odds| (1,000 {tag})"] = top5(s, "rank", "mean_abs_shap_logodds")
            rk[f"ensemble SHAP mean|log-odds| (full cohort {tag})"] = top5(s, "rank_fullcohort",
                                                                           "mean_abs_shap_logodds_fullcohort")
            rk[f"exact pre-calibration probability SHAP (1,000 {tag})"] = top5(s, "rank_prob",
                                                                               "mean_abs_shap_prob_precal_ensemble")
        log(f"{cfg['short']}:")
        for k, v in rk.items():
            log(f"   {k}: {', '.join(v)}")
        every = [l for l in rk[next(iter(rk))] if all(l in v for v in rk.values())]
        some = sorted({l for v in rk.values() for l in v} - set(every))
        log(f"   -> in the top 5 of every ranking: {', '.join(every)}; in some rankings only: {', '.join(some) or '-'}")

    # ---- specific gravity
    log("\n[2] Specific gravity: shape of the model's effect (conditional on the other inputs)")
    for t, cfg in TARGETS.items():
        thr_like_ext = None
        for cohort in (EXT, INT):
            a = sgb[(sgb.target == t) & (sgb.cohort == cohort) & (sgb.subgroup == "all")].set_index("sg_level").loc[SG_ORDER]
            p = a.pdp_calibrated_sg_set_to_level.values; s = a.mean_shap_sg_minus_ref_1010.values
            step = np.diff(p); k = int(np.argmin(step)); low, high = p[:3], p[3:]
            rel = p / p[1] - 1                                          # relative to 1.010
            thr_like = bool(k == 2 and high.max() < low.min())
            if cohort == EXT:
                thr_like_ext = thr_like
            if rel[0] >= -0.10 and rel[2] >= -0.10:
                lowdesc = "flat or slightly higher up to 1.015"
            elif rel[2] >= -0.10:
                lowdesc = f"lower at <=1.005 ({rel[0]:+.0%} vs 1.010), plateau/peak at 1.010-1.015"
            else:
                lowdesc = f"<=1.005 {rel[0]:+.0%} and 1.015 {rel[2]:+.0%} vs 1.010"
            shape_lab = pdp[(pdp.target == t) & (pdp.cohort == cohort) & (pdp.subgroup == "all") &
                            (pdp.feature == "ua_sg")].pdp_shape.iloc[0]
            shape = (f"threshold-like: {lowdesc}; lower from 1.020" if thr_like else
                     f"not threshold-like at 1.015/1.020: partial dependence {shape_lab}, largest drop "
                     f"{SG_ORDER[k]} -> {SG_ORDER[k + 1]}")
            log(f"{cfg['short']} {cohort}: {shape}")
            log("   calibrated PDP: " + ", ".join(f"{l} {pct(v, 2)}" for l, v in zip(SG_ORDER, p)) +
                "; relative to 1.010: " + ", ".join(f"{l} {r:+.0%}" for l, r in zip(SG_ORDER, rel)) +
                f"; mean <=1.015 {pct(low.mean(), 2)} vs >=1.020 {pct(high.mean(), 2)} (ratio {high.mean() / low.mean():.2f}); "
                f"step 1.015 -> 1.020 {step[2] / p[2]:+.0%}")
            for sub in ("men", "women"):
                b = sgb[(sgb.target == t) & (sgb.cohort == cohort) & (sgb.subgroup == sub)].set_index("sg_level").loc[SG_ORDER]
                log(f"   PDP {sub}: " + ", ".join(f"{l} {pct(v, 2)}" for l, v in zip(SG_ORDER, b.pdp_calibrated_sg_set_to_level)))
            log("   mean SG SHAP by observed level, log-odds relative to 1.010 (full cohort): " +
                ", ".join(f"{l} {v:+.3f}" for l, v in zip(SG_ORDER, s)) +
                "; learners whose mean at 1.020/1.025/>=1.030 is below their own 1.010 mean: " +
                "/".join(str(int(v)) for v in a.n_learners_below_own_1010.values[3:]) + " of 15")
        r = shp[(shp.target == t) & (shp.feature == "ua_sg") & shp.analysis.str.startswith("ensemble, external")].iloc[0]
        if thr_like_ext:
            what = "reflects the step above 1.015, not a linear dose-response"
        elif str(r.pdp_shape).startswith("monotone"):
            what = f"is consistent with the {r.pdp_shape} partial dependence"
        else:
            what = f"summarises a non-linear effect only as a net trend (partial dependence {r.pdp_shape})"
        log(f"   Spearman(SG, SHAP) {r.spearman_value_vs_shap:+.2f} (KNHANES) {what}")

    log("\n[3] Specific gravity: crude vs within-sex vs adjusted (serum urate >=7.0 unless stated)")
    sb = sgu[sgu.row_type == "prevalence by SG level and sex, both cohorts"]
    for thr in (7.0, 9.0, 10.0):
        s = sb[sb.threshold == thr]
        log(f"prevalence >= {thr:g} by SG level (development | KNHANES):")
        for sex in SUBGROUPS:
            g = s[s.sex == sex].set_index("sg_level").loc[SG_ORDER]
            log(f"   {sex:5s}: " + ", ".join(f"{l} {pct(a.prev_dev)} | {pct(a.prev_ext)}" for l, a in g.iterrows()))
        if thr == 7.0:
            g = s[s.sex == "all"].set_index("sg_level").loc[SG_ORDER]
            log("   share of men: " + ", ".join(f"{l} {pct(a.pct_men_dev, 0)} | {pct(a.pct_men_ext, 0)}"
                                                for l, a in g.iterrows()))
            for sex in ("men", "women"):
                g = s[s.sex == sex].set_index("sg_level").loc[SG_ORDER]
                log(f"   mean age, {sex}: " + ", ".join(f"{l} {a.mean_age_dev:.1f} | {a.mean_age_ext:.1f}"
                                                        for l, a in g.iterrows()))
    cont = sgu[sgu.row_type == "logistic OR for urate >=7.0 per +0.010 SG"]
    for name in (DEV_NAME, EXT_NAME):
        c = cont[cont.cohort == name].set_index("model")
        o = c.estimate
        log(f"{name}: OR per +0.010 SG " + "; ".join(f"{m} {a.estimate:.2f} ({a.lo:.2f}-{a.hi:.2f})" for m, a in c.iterrows()))
        verdict = ("the positive crude gradient reverses once sex is adjusted for (confounded by sex)"
                   if o["crude"] > 1 > o["adjusted for sex"] else
                   "the crude gradient is attenuated by sex adjustment" if abs(np.log(o["adjusted for sex"])) <
                   abs(np.log(o["crude"])) else "sex adjustment does not attenuate the crude gradient")
        log(f"   crude {o['crude']:.2f} -> sex {o['adjusted for sex']:.2f} -> age, sex {o['adjusted for age, sex']:.2f}"
            f" -> age, sex, BMI {o['adjusted for age, sex, BMI']:.2f}: {verdict}; within sex (crude) men "
            f"{o['men: crude']:.2f}, women {o['women: crude']:.2f}")
    cat = sgu[sgu.row_type == "logistic OR by SG level (reference 1.010)"]
    for thr in (7.0, 9.0, 10.0):
        log(f"categorical OR vs 1.010, urate >= {thr:g}:")
        for name in (DEV_NAME, EXT_NAME):
            for model in ("crude", "adjusted for sex", PRIMARY, "sensitivity: adjusted for sex x 10-year age band, BMI",
                          "sensitivity: adjusted for the other inputs", "men: adjusted for age, BMI",
                          "women: adjusted for age, BMI"):
                g = cat[(cat.threshold == thr) & (cat.cohort == name) & (cat.model == model)]
                if g.empty:
                    continue
                if "estimate" not in g or g.estimate.isna().all():
                    log(f"   {name} [{model}]: {g.note.iloc[0]}"); continue
                g = g.set_index("sg_level").loc[SG_ORDER]
                log(f"   {name} [{model}]: " + ", ".join(
                    f"{l} {a.estimate:.2f} ({a.lo:.2f}-{a.hi:.2f})" if l != SG_REF else f"{l} 1" for l, a in g.iterrows())
                    + f"; LR test p={g.p_sg_joint.iloc[0]:.1e}")

    # ---- age
    log("\n[4] Age: shape of the model's effect")
    for t, cfg in TARGETS.items():
        for cohort in (EXT, INT):
            q = pdp[(pdp.target == t) & (pdp.cohort == cohort) & (pdp.feature == "age")]
            parts = {}
            for sub in SUBGROUPS:
                g = q[q.subgroup == sub].sort_values("value"); v = g.mean_pred_calibrated.values; x = g.value.values
                i = int(np.argmin(v))
                parts[sub] = (g.pdp_shape.iloc[0], g.pdp_spearman_grid.iloc[0],
                              ", ".join(f"{a:g} y {pct(b, 2)}" for a, b in zip(x, v)), x[i])
            r = shp[(shp.target == t) & (shp.feature == "age") &
                    (shp.analysis == ("ensemble, external KNHANES >=19 (primary)" if cohort == EXT
                                      else "ensemble, internal person-clean test (secondary)"))].iloc[0]
            curves = {sub: dict(zip(q[q.subgroup == sub].value, q[q.subgroup == sub].mean_pred_calibrated))
                      for sub in SUBGROUPS}

            def seg(sub, a, b):
                r = curves[sub][b] / curves[sub][a] - 1
                return ("flat" if abs(r) < 0.10 else "rises" if r > 0 else "falls"), r
            mid = {sub: seg(sub, 40, 70) for sub in ("men", "women")}
            opp = {mid["men"][0], mid["women"][0]} == {"falls", "rises"}
            log(f"{cfg['short']} {cohort}: overall PDP {parts['all'][0]}; men {parts['men'][0]} "
                f"(Spearman {parts['men'][1]:+.2f}), women {parts['women'][0]} (Spearman {parts['women'][1]:+.2f}); "
                f"mid-life 40 -> 70 y: men {mid['men'][0]} ({mid['men'][1]:+.0%}), women {mid['women'][0]} "
                f"({mid['women'][1]:+.0%}) -> " + ("OPPOSITE direction by sex" if opp else
                                                  f"not opposite (men {mid['men'][0]}, women {mid['women'][0]})"))
            for sub in SUBGROUPS:
                log(f"   PDP {sub}: {parts[sub][2]}")
            for sub in ("men", "women"):
                log(f"   {sub} segments: " + ", ".join(f"{a}->{b} y {seg(sub, a, b)[0]} ({seg(sub, a, b)[1]:+.0%})"
                                                     for a, b in ((20, 40), (40, 70), (70, 80))))
            log(f"   SHAP Spearman(age, SHAP): overall {r.spearman_value_vs_shap:+.2f} (not a valid single direction), "
                f"men {r.spearman_men:+.2f}, women {r.spearman_women:+.2f} (1,000 sample); full cohort men "
                f"{r.spearman_men_fullcohort:+.2f}, women {r.spearman_women_fullcohort:+.2f}")
            a = agb[(agb.target == t) & (agb.cohort == cohort)]
            for sub in ("men", "women"):
                g = a[a.subgroup == sub]
                log(f"   mean age SHAP (log-odds) {sub}: " + ", ".join(f"{b} {v:+.2f}" for b, v in
                                                                      zip(g.age_band, g.mean_shap_age_logodds)))


# ================================================================== main
def main():
    d, tr, te, ext = main_cohort()
    dte = d.iloc[te].reset_index(drop=True)          # internal test persons, in the order of preds_internal_clean.csv
    Xi, Xe = dte[FMODEL], ext[FMODEL]
    assert len(dte) == N_INTERNAL and len(ext) == N_EXTERNAL, "internal test / KNHANES sizes differ from the main analysis"
    assert d["age"].min() >= ADULT_AGE and ext["age"].min() >= ADULT_AGE, "cohorts are not adults only"
    assert d[ID].is_unique and not set(d[ID].iloc[tr]) & set(dte[ID]), "the split is not person-level"
    ref_int = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))
    ref_ext = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"))
    mperf = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
    log(f"retrained models (bestmodel_v5), {len(FMODEL)} inputs: {', '.join(FMODEL)}")
    log(f"dev_v2 {len(d):,} persons: training {len(tr):,}, internal test {len(te):,} (one person-level split, shared by "
        f"the three thresholds); KNHANES adults >= {ADULT_AGE} (ext_v2, X1 coding) {len(ext):,}")
    nit = dict(dev=int((d["ua_nitrite"] > 0).sum()), train=int((d["ua_nitrite"].iloc[tr] > 0).sum()),
               test=int((dte["ua_nitrite"] > 0).sum()), ext=int((ext["ua_nitrite"] > 0).sum()))
    log(f"Nitrite (parsed correctly in dev_v2): positive in {nit['dev']:,} of {len(d):,} development persons (training "
        f"{nit['train']:,}, internal test {nit['test']:,}); KNHANES adults {nit['ext']:,} of {len(ext):,}")
    sgv = d["ua_sg"]
    odd = sgv[~np.isclose(sgv / 0.005, np.round(sgv / 0.005)) | (sgv < 1.005 - 1e-9) | (sgv > 1.030 + 1e-9)]
    log("development SG values that are not a dipstick level 1.005-1.030: " +
        (", ".join(f"{v:.3f} ({n:,})" for v, n in odd.value_counts().sort_index().items())
         + " (assigned to the nearest level within 1.005-1.030 for the SG levels)" if len(odd) else "none"))
    log(f"KNHANES SG recorded in 0.001 steps ({ext['ua_sg'].min():.3f}-{ext['ua_sg'].max():.3f}); outside the development "
        f"dipstick range 1.005-1.030: {int((ext['ua_sg'] < 1.0025).sum()):,} below 1.0025 and "
        f"{int((ext['ua_sg'] >= 1.0325).sum()):,} at >= 1.0325 (rounded into <=1.005 / >=1.030)")

    shap_sample = Xe.sample(N_SHAP, random_state=SEED)
    shap_bg = Xe.drop(shap_sample.index).sample(N_BG, random_state=SEED)
    shap_sample = shap_sample.reset_index(drop=True); shap_bg = shap_bg.reset_index(drop=True)
    # modal value of each ordinal dipstick field in the development data (all dev_v2 persons): flags coding or
    # distribution differences in the SHAP samples
    dev_mode = {f: float(d[f].mode().iloc[0]) for f in FMODEL if d[f].nunique() <= 6 and f != "male"}
    log("development modal values (ordinal dipstick fields): " + ", ".join(f"{LABEL[f]} {v:g}" for f, v in dev_mode.items())
        + "; KNHANES: " + ", ".join(f"{LABEL[f]} {ext[f].mode().iloc[0]:g}" for f in dev_mode))

    hp, perm, shp, pdp, sgb, agb = [], [], [], [], [], []
    for t, cfg in TARGETS.items():
        thr = cfg["threshold"]; t1 = time.time()
        log(f"\n=== {cfg['short']} ({t}) ===")
        model = load_model(t, "v5")
        assert list(model.feature_names_in_) == FMODEL, "model inputs differ from FMODEL"
        hp += hyperparameter_rows(t, model)                    # read before capping threads (saved values)
        n_cmp, bad, absent = tune_json_check(t, model)
        assert not bad, f"saved hyperparameters differ from tune_{t}.json: {bad}"
        log(f"hyperparameters: {n_cmp} values (5 folds x 3 learners) equal tune_{t}.json best_params; saved keys that "
            f"get_params does not return: {', '.join(absent) or 'none'}")
        cap_threads(model)

        # internal test persons and KNHANES adults: predictions must equal those of calc_main.py
        yi, ye = labels(dte["urate"], thr), labels(ext["urate"], thr)
        pi, pe = p_cal(model, Xi), p_cal(model, Xe)
        ri, re_ = ref_int[ref_int.target == t], ref_ext[ref_ext.target == t]
        same_ids = (len(ri) == len(dte) and len(re_) == len(ext)
                    and np.array_equal(ri["id"].astype(str).values, dte[ID].astype(str).values)
                    and np.array_equal(re_["id"].astype(str).values, ext["id"].astype(str).values))
        d_i = float(np.max(np.abs(pi - ri.p.values))) if len(ri) == len(pi) else np.inf
        d_e = float(np.max(np.abs(pe - re_.p.values))) if len(re_) == len(pe) else np.inf
        assert same_ids and np.array_equal(yi, ri.y.values) and np.array_equal(ye, re_.y.values) and max(d_i, d_e) < 1e-9, \
            "rebuilt cohorts/predictions do not match preds_internal_clean.csv / preds_external_corrected.csv"
        d_b = max(float(np.max(np.abs(p_both(model, Xi)[0] - pi))), float(np.max(np.abs(p_both(model, Xe)[0] - pe))))
        assert d_b < 1e-12, "one-pass calibrated probability does not match CalibratedClassifierCV.predict_proba"
        auc_i, auc_e = roc_auc_score(yi, pi), roc_auc_score(ye, pe)
        for coh, a, n in (("internal test", auc_i, len(yi)), ("KNHANES adults", auc_e, len(ye))):
            r = mperf[(mperf.target == cfg["short"]) & (mperf.cohort == coh)]
            assert len(r) == 1 and int(r.n.iloc[0]) == n and f"{a:.4f}" == f"{r.auroc.iloc[0]:.4f}", \
                f"AUROC {cfg['short']} {coh} differs from main_performance.csv"
        log(f"predictions reproduced (max |diff| internal {d_i:.1e}, external {d_e:.1e}; one-pass {d_b:.1e}); AUROC "
            f"internal {auc_i:.4f} (n={len(yi):,}, events {int(yi.sum()):,}), external {auc_e:.4f} (n={len(ye):,}, events "
            f"{int(ye.sum()):,}) = main_performance.csv")

        # (b) permutation importance of the whole calibrated ensemble
        for cohort, X, yy, pfull in ((EXT, Xe, ye, pe), (INT, Xi, yi, pi)):
            idx = subsample_idx(yy, MAX_PERM)
            base, drops = permutation_importance(model, X.values[idx], yy[idx])
            mean = drops.mean(0); rk = rank_desc(mean)
            for j, f in enumerate(FMODEL):
                note = "input constant in this cohort: permutation cannot change predictions" if X[f].nunique() == 1 else ""
                perm.append(dict(target=t, threshold=thr, cohort=cohort, n_cohort=len(yy), cases_cohort=int(yy.sum()),
                                 auroc_cohort=roc_auc_score(yy, pfull), n_used=len(idx), cases_used=int(yy[idx].sum()),
                                 auroc_base_used=base, feature=f, label=LABEL[f], n_repeats=N_REPEATS,
                                 auroc_drop_mean=mean[j], auroc_drop_sd=drops[:, j].std(ddof=1),
                                 auroc_drop_min=drops[:, j].min(), auroc_drop_max=drops[:, j].max(),
                                 pct_of_discrimination=100 * mean[j] / (base - 0.5), rank=int(rk[j]), note=note))
            top = np.argsort(-mean)[:5]
            log(f"permutation [{cohort}] n={len(idx):,} of {len(yy):,} (cases {int(yy[idx].sum()):,}) base AUROC "
                f"{base:.4f}; top-5: " + ", ".join(f"{LABEL[FMODEL[j]]} {mean[j]:.4f}" for j in top))
            jn = FMODEL.index("ua_nitrite")
            log("   key inputs: " + ", ".join(f"{LABEL[f]} {mean[FMODEL.index(f)]:.4f} (rank {rk[FMODEL.index(f)]})"
                                             for f in KEY)
                + f"; nitrite {mean[jn]:.4f} (rank {rk[jn]}; nitrite-positive records in cohort: "
                  f"{nit['ext'] if cohort == EXT else nit['test']:,})")

        # (c) full-cohort partial dependence and log-odds SHAP
        full = {"external": dict(name=EXT, X=Xe, y=ye), "internal": dict(name=INT, X=Xi, y=yi)}
        for c in full.values():
            c["pdp_rows"] = pdp_full(t, model, c["X"], c["name"]); pdp += c["pdp_rows"]
            c["pdps"] = pdp_summary(c["pdp_rows"])
            c["L"], err = shap_full(model, c["X"])
            log(f"full-cohort log-odds SHAP [{c['name']}] n={len(c['X']):,}, 15 learners, additivity {err:.1e}")
            for f in KEY:
                log(f"   PDP {LABEL[f]}: " + "; ".join(f"{s} {c['pdps'][(f, s)][0]} (rho {c['pdps'][(f, s)][1]:+.2f})"
                                                     for s in SUBGROUPS))

        # (c) SHAP on the 1,000-record samples: external (primary) and internal (secondary)
        Xi_s = Xi.sample(min(N_SHAP, len(Xi)), random_state=SEED)
        Xi_bg = Xi.drop(Xi_s.index).sample(N_BG, random_state=SEED).reset_index(drop=True)
        for analysis, X, bg, key in (("ensemble, external KNHANES >=19 (primary)", shap_sample, shap_bg, "external"),
                                     ("ensemble, internal person-clean test (secondary)", Xi_s.reset_index(drop=True),
                                      Xi_bg, "internal")):
            rows, qc, S_s, P_s = shap_block(t, model, X, bg, analysis, dev_mode, full[key])
            shp += rows
            sgb += shap_by_sg_rows(t, full[key], X, S_s, P_s)
            agb += age_by_sex_rows(t, full[key])
            r = pd.DataFrame(rows).sort_values("rank")
            log(f"SHAP [{analysis}] QC: additivity log-odds {qc['max_additivity_err_logodds']:.1e}, "
                f"prob {qc['max_additivity_err_prob']:.1e}, pre-calibration ensemble {qc['ensemble_prob_err']:.1e}")
            log("   rank (mean|SHAP| log-odds): " + ", ".join(f"{a.label} {a.mean_abs_shap_logodds:.3f}" for a in r.itertuples()))
            log("   rank (prob-scale exact pre-cal ensemble): " +
                ", ".join(f"{a.label}" for a in r.sort_values("rank_prob").itertuples()))
            log("   rank (mean|SHAP| log-odds, full cohort): " +
                ", ".join(f"{a.label} {a.mean_abs_shap_logodds_fullcohort:.3f}" for a in r.sort_values("rank_fullcohort").itertuples()))
            for f in KEY + ("ua_nitrite",):
                a = r[r.feature == f].iloc[0]
                log(f"   {a.label}: rank {a['rank']} (learners {a.rank_min_across_learners}-{a.rank_max_across_learners}), "
                    f"rho {sgn(a.spearman_value_vs_shap)} [{a.n_learners_rho_pos}+/{a.n_learners_rho_neg}- of 15], "
                    f"(men {sgn(a.spearman_men)}, women {sgn(a.spearman_women)}), "
                    f"prob-scale rank {a.rank_prob} rho {sgn(a.spearman_prob)}; full-cohort rank {a.rank_fullcohort}")
                log(f"      direction: {a.direction}")
            notes = r[r.note.astype(str).str.len() > 0]
            if len(notes):
                log("   notes: " + "; ".join(f"{a.label}: {a.note}" for a in notes.itertuples()))
            g = pd.DataFrame(sgb); g = g[(g.target == t) & (g.cohort == full[key]["name"]) & (g.subgroup == "all")]
            log("   SG level: n / mean SG SHAP (full cohort, log-odds) / calibrated PDP: " +
                ", ".join(f"{a.sg_level} {a.n_records:,} / {a.mean_shap_sg_logodds:+.3f} / "
                          f"{pct(a.pdp_calibrated_sg_set_to_level, 2)}" for a in g.itertuples()))
        for c in full.values():
            del c["L"]
        log(f"({time.time() - t1:.0f} s)")

    hp, perm, shp, pdp, sgb, agb = (pd.DataFrame(x) for x in (hp, perm, shp, pdp, sgb, agb))
    assert set(perm.feature) == set(FMODEL) and set(shp.feature) == set(FMODEL), "importance rows differ from FMODEL"
    hp.to_csv(os.path.join(HERE, "hyperparameters.csv"), index=False, encoding="utf-8")
    perm.to_csv(os.path.join(HERE, "permutation_importance.csv"), index=False, encoding="utf-8")
    shp.to_csv(os.path.join(HERE, "shap_ensemble.csv"), index=False, encoding="utf-8")
    pdp.to_csv(os.path.join(HERE, "pdp_calibrated.csv"), index=False, encoding="utf-8")
    sgb.to_csv(os.path.join(HERE, "shap_by_sg.csv"), index=False, encoding="utf-8")
    agb.to_csv(os.path.join(HERE, "shap_age_by_sex.csv"), index=False, encoding="utf-8")

    log("\n=== (a) hyperparameters ===")
    log(f"rows {len(hp)}; all identical across the 5 calibration folds: {bool(hp.identical_across_folds.all())}")
    for (t, lr), g in hp[hp.tuned | (hp.param == "n_trees_fitted")].groupby(["target", "learner"], sort=False):
        log(f"{t} {lr}: " + ", ".join(f"{a.param}={a.value}" for a in g.itertuples()))

    # (e) specific gravity vs serum urate (model-free): all dev_v2 persons and the KNHANES adults
    dev, ext_sg = prep_sg(d), prep_sg(ext)
    rows = sg_rows(DEV_NAME, dev) + sg_rows(EXT_NAME, ext_sg)
    rows += sg_side_by_side(dev, ext_sg)
    sgu = pd.DataFrame(rows)
    lead = ["cohort", "row_type", "threshold", "model", "sex", "sg_level"]
    sgu = sgu[lead + [c for c in sgu.columns if c not in lead]]
    sgu.to_csv(os.path.join(HERE, "sg_univariate.csv"), index=False, encoding="utf-8")
    bad = sgu[(sgu.get("converged") == False) | sgu.get("note", pd.Series(dtype=str)).astype(str).str.startswith("not estimable")]  # noqa: E712
    log(f"\n=== (e) specific gravity vs serum urate: {len(sgu)} rows; categorical models not converged/estimable: "
        f"{len(bad)}" + (" (" + "; ".join(sorted({f'{a.cohort} >={a.threshold:g} [{a.model}]' for a in bad.itertuples()}))
                         + ")" if len(bad) else ""))
    for name, g in sgu[sgu.row_type == "prevalence by SG level"].groupby("cohort", sort=False):
        log(f"{name}: prevalence >=7.0 by SG " + ", ".join(f"{a.sg_level} {pct(a.prev_7)} (n={int(a.n):,}, men {pct(a.pct_men, 0)}, "
                                                          f"age {a.mean_age:.1f})" for a in g.itertuples()))
    for a in sgu[sgu.row_type == "Spearman(SG, serum urate)"].itertuples():
        log(f"   {a.cohort} Spearman(SG, urate) [{a.sg_level}]: {a.estimate:.3f}")
    for a in sgu[(sgu.row_type == "logistic OR for urate >=7.0 per +0.010 SG") &
                 sgu.model.isin(["crude", "adjusted for age, sex, BMI"])].itertuples():
        log(f"   {a.cohort} OR for urate >=7.0 per +0.010 SG, {a.model}: {a.estimate:.2f} ({a.lo:.2f}-{a.hi:.2f}), "
            f"n={int(a.n):,}")

    interpretation(perm, shp, pdp, sgb, agb, sgu)
    log("\noutputs: " + ", ".join(f"{fn} {len(x):,} rows" for fn, x in (
        ("hyperparameters.csv", hp), ("permutation_importance.csv", perm), ("shap_ensemble.csv", shp),
        ("pdp_calibrated.csv", pdp), ("shap_by_sg.csv", sgb), ("shap_age_by_sex.csv", agb), ("sg_univariate.csv", sgu))))
    log(f"\nTotal runtime {(time.time() - T0) / 60:.1f} min")
    with open(os.path.join(HERE, "calc_importance_params_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
