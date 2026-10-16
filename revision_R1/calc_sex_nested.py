"""Subgroup AUROC, female-male gap, nested-threshold coherence and a continuous-urate model (retrained models).

Reviewer AW #8, #12, #19 and BZ #2 (JMIR Medical Informatics #107515, R1).

Models: the retrained calibrated soft-voting ensembles (bestmodel_v5, decision of 2026-10-16: Optuna-tuned LightGBM +
XGBoost + CatBoost, soft voting, isotonic CalibratedClassifierCV cv=5; 12 inputs = cohort.FMODEL: 9 dipstick fields,
age, male, BMI), loaded read-only with cohort.load_model. Data: cohort.main_cohort() -> d = the corrected development
cohort dev_v2 (adults >= 19 y, one record per person) with ONE person-level 80/20 split shared by the three thresholds
(training persons d.iloc[tr], internal test persons d.iloc[te]: the same test persons for every threshold), and ext =
KNHANES adults (>= 19 y; dipstick inputs coded by the label-matched mapping X1). The ensembles' predictions are
recomputed here and checked against preds_internal_clean.csv / preds_external_corrected.csv (calc_main.py), and their
AUROC against main_performance.csv.

(a) subgroups.csv        AUROC with a percentile bootstrap drawn WITHIN each subgroup (1000 resamples; age 19-39,
                         40-59, 60-79, >=80; male; female) on KNHANES adults (primary) and on the internal test persons
                         (secondary). The 'All' rows equal main_performance.csv (same bootstrap, checked).
(b) sex_difference.csv   female minus male AUROC: DeLong for two independent ROC curves (common.delong_unpaired)
                         and a bootstrap CI of the difference (each sex resampled independently, 2000 resamples);
                         sensitivity restricted to age 40-69 (similar case-mix of age in both sexes).
(c) nested_coherence.csv proportion of persons whose predicted risks violate the nesting of the outcomes
                         (P(>=9) > P(>=7), P(>=10) > P(>=9), P(>=10) > P(>=7)) and the size of the violations: the
                         retrained ensembles, the separately fitted binary LightGBM comparators (one per threshold) and
                         the continuous model, on KNHANES adults (primary) and on the internal test persons (the same
                         persons for the three thresholds, so every internal test person enters every comparison).
(d) continuous.csv       A simple homoscedastic continuous LightGBM: regressor for log(urate) (LGBM_FIXED, objective
                         regression). P(urate >= t) = share of 5-fold out-of-fold training residuals r with
                         pred + r >= log t (one residual distribution for every person, i.e. homoscedastic; monotone in
                         t by construction). To separate the continuous formulation from the tuning and ensembling of
                         the retrained models, an equal-budget binary LightGBM (LGBM_FIXED, n_jobs=4, no tuning, no
                         calibration) is fitted per threshold. Both refits use the 12 model inputs and the training
                         persons of the shared split exactly as the retrained ensembles were trained (cohort.train_xy),
                         so all three models are compared on persons none of them was trained on.
                         Blocks (row_type classification / regression / comparison; paired DeLong with 95% CI):
                           1. External KNHANES >=19 y (primary).
                           2. Internal test persons of the shared split.

Removed with the retraining (2026-10-16): leak_demo_internal.csv, which showed that the original models' record-level
split had put most development test persons into their training rows (the shared person-level split has no such
overlap, asserted below; the former file is deleted when this script runs and kept in _snapshot_original_models_261016);
the refits on the original models' training rows (former continuous.csv blocks 3 and 4 and the per-threshold regressor
rows) and the nested-coherence rows on the intersection of the original models' three test sets.

Calibration intercept/slope come from common.calib_stats (exact maximum likelihood, Newton-Raphson). Threads are
capped at 4 (OMP, LightGBM, XGBoost, CatBoost). The refit models live only in memory.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import functools, time
import numpy as np, pandas as pd
# Import order matters on macOS: importing lightgbm before sklearn/cohort makes XGBoost pick up a second libomp and
# deadlock while unpickling the saved ensembles. cohort/common first.
from cohort import HERE, TARGETS, FMODEL, ID, ADULT_AGE, main_cohort, train_xy, labels, load_model, is_adult
from common import (SEED, LGBM_FIXED, auroc_ci, auprc_ci, delong_unpaired, delong_paired, calib_stats, scaled_brier,
                    _placements)
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score, mean_absolute_error

N_BOOT, N_BOOT_DIFF, N_JOBS = 1000, 2000, 4
T_KEYS = list(TARGETS)                                # uric_hyper_70, uric_severe_9, uric_severe_10
THR = {t: TARGETS[t]["threshold"] for t in T_KEYS}
SHORT = {t: TARGETS[t]["short"] for t in T_KEYS}
LOG = []
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def out(name):
    return os.path.join(HERE, name)


def cap_threads(model, n=N_JOBS):
    """Limit prediction threads of every base learner inside the calibrated voting ensemble (in memory only)."""
    for cc in model.calibrated_classifiers_:
        for est in cc.estimator.estimators_:
            kind = type(est).__name__
            if kind.startswith(("LGBM", "XGB")):
                est.set_params(n_jobs=n)
            elif kind.startswith("CatBoost"):
                est.predict_proba = functools.partial(type(est).predict_proba, est, thread_count=n)
    return model


def boot_auc_within(y, p, n=N_BOOT, seed=SEED):
    """Percentile bootstrap of AUROC resampling only the persons of this subgroup (as common.boot_ci)."""
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def boot_diff_indep(yf, pf, ym, pm, n=N_BOOT_DIFF, seed=SEED):
    """Bootstrap of AUROC(female) - AUROC(male), each sex resampled independently at its own size."""
    rng = np.random.RandomState(seed); af, am, d = [], [], []
    for _ in range(n):
        i = rng.randint(0, len(yf), len(yf)); j = rng.randint(0, len(ym), len(ym))
        if yf[i].min() == yf[i].max() or ym[j].min() == ym[j].max(): continue
        a1 = roc_auc_score(yf[i], pf[i]); a2 = roc_auc_score(ym[j], pm[j])
        af.append(a1); am.append(a2); d.append(a1 - a2)
    af, am, d = map(np.asarray, (af, am, d))
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    p_boot = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return dict(female_boot=pct(af), male_boot=pct(am), diff_boot=pct(d), p_boot=p_boot, n_valid=len(d))


# ------------------------------------------------------------------ data, retrained ensembles and their predictions
d, tr, te, ext = main_cohort()
dte = d.iloc[te]
ids = d[ID].to_numpy()
assert d[ID].is_unique and is_adult(d["age"]).all() and is_adult(ext["age"]).all()   # dev_v2 / ext_v2: adults, one row per person
assert not np.isin(ids[te], ids[tr]).any()                                            # internal test persons never trained on
assert (d["urate"] > 0).all() and (ext["urate"] > 0).all()                            # log(urate) in (d)
MP = pd.read_csv(out("main_performance.csv"))
MP_COH = {"internal": "internal test", "external": "KNHANES adults"}


def mp_row(t, cohort):
    r = MP[(MP.target == SHORT[t]) & (MP.cohort == MP_COH[cohort])]
    assert len(r) == 1, (t, cohort)
    return r.iloc[0]


for t in T_KEYS:
    assert int(mp_row(t, "internal").n) == len(te) and int(mp_row(t, "external").n) == len(ext), t
log(f"dev_v2 persons {len(d):,} (adults >={ADULT_AGE} y, one record per person); shared person-level split: training "
    f"{len(tr):,}, internal test {len(te):,} (the same test persons for every threshold); KNHANES adults {len(ext):,}")

models = {t: cap_threads(load_model(t)) for t in T_KEYS}
for t in T_KEYS:
    assert list(models[t].feature_names_in_) == FMODEL, t
y_int = {t: labels(dte["urate"], THR[t]) for t in T_KEYS}
y_ext = {t: labels(ext["urate"], THR[t]) for t in T_KEYS}
p_int = {t: models[t].predict_proba(dte[FMODEL])[:, 1] for t in T_KEYS}
p_ext = {t: models[t].predict_proba(ext[FMODEL])[:, 1] for t in T_KEYS}
age_i, male_i = dte["age"].to_numpy(float), dte["male"].to_numpy().astype(int)
age_e, male_e = ext["age"].to_numpy(float), ext["male"].to_numpy().astype(int)
ref_i = pd.read_csv(out("preds_internal_clean.csv"), dtype={"id": str})
ref_e = pd.read_csv(out("preds_external_corrected.csv"), dtype={"id": str})
for t in T_KEYS:
    for cohort, ref, y, p, age, male, idv in (
            ("internal", ref_i, y_int[t], p_int[t], age_i, male_i, dte[ID].astype(str).to_numpy()),
            ("external", ref_e, y_ext[t], p_ext[t], age_e, male_e, ext["id"].astype(str).to_numpy())):
        r = ref[ref.target == t]
        assert (len(r) == len(y) and (r.id.to_numpy() == idv).all() and (r.y.to_numpy() == y).all()
                and np.allclose(r.p.to_numpy(), p, rtol=0, atol=1e-12) and (r.male.to_numpy() == male).all()
                and np.allclose(r.age.to_numpy(), age)), (t, cohort)
        auc = roc_auc_score(y, p)
        assert abs(auc - mp_row(t, cohort).auroc) < 5e-5, (t, cohort, auc)
log("retrained ensembles (bestmodel_v5, 12 inputs): predictions reproduce preds_internal_clean.csv and "
    "preds_external_corrected.csv (same persons and order, same labels, |dp| <= 1e-12); AUROC equals main_performance.csv")

# ------------------------------------------------------------------ (a) subgroup AUROC, within-subgroup bootstrap
C_EXT = "External KNHANES >=19 y (primary)"
C_INT = "Internal test (shared person-level split; secondary)"
log("\n(a) Subgroup AUROC (retrained ensembles; bootstrap within subgroup, 1000 resamples)")


def subgroup_masks(age, male):
    assert is_adult(age).all()                        # both cohorts are adults: no age band below 19
    return [("All", np.ones(len(age), bool)),
            ("Age 19-39", (age >= 19) & (age <= 39)), ("Age 40-59", (age >= 40) & (age <= 59)),
            ("Age 60-79", (age >= 60) & (age <= 79)), ("Age >=80", age >= 80),
            ("Male", male == 1), ("Female", male == 0)]


SOURCES = [(C_EXT, "external", {t: (y_ext[t], p_ext[t], age_e, male_e) for t in T_KEYS}),
           (C_INT, "internal", {t: (y_int[t], p_int[t], age_i, male_i) for t in T_KEYS})]
rows = []
for cohort_name, _, data in SOURCES:
    for t in T_KEYS:
        y, p, age, male = data[t]; y = y.astype(int); male = male.astype(int)
        masks = subgroup_masks(age, male)
        assert sum(m.sum() for s, m in masks if s.startswith("Age")) == len(y) == sum(
            m.sum() for s, m in masks if s in ("Male", "Female"))
        for sg, mask in masks:
            ys, ps = y[mask], p[mask]; pos = int(ys.sum())
            if pos == 0 or pos == len(ys):
                rows.append(dict(cohort=cohort_name, target=SHORT[t], subgroup=sg, n=int(mask.sum()), positives=pos,
                                 prevalence=ys.mean() if len(ys) else np.nan, auroc=np.nan, auroc_lo=np.nan,
                                 auroc_hi=np.nan, n_boot_valid=0, note="AUROC undefined (one class)"))
                continue
            lo, hi, nv = boot_auc_within(ys, ps)
            rows.append(dict(cohort=cohort_name, target=SHORT[t], subgroup=sg, n=int(mask.sum()), positives=pos,
                             prevalence=ys.mean(), auroc=roc_auc_score(ys, ps), auroc_lo=lo, auroc_hi=hi,
                             n_boot_valid=nv, note="fewer than 20 positives: imprecise" if pos < 20 else ""))
sub = pd.DataFrame(rows)
# the 'All' rows are the headline estimates of calc_main.py (same bootstrap: seed 42, 1000 resamples)
for cohort_name, cohort, _ in SOURCES:
    for t in T_KEYS:
        r, m = sub[(sub.cohort == cohort_name) & (sub.target == SHORT[t]) & (sub.subgroup == "All")].iloc[0], mp_row(t, cohort)
        assert (r.n == m.n and r.positives == m.events and abs(r.auroc - m.auroc) < 5e-5
                and abs(r.auroc_lo - m.auroc_lo) < 1e-12 and abs(r.auroc_hi - m.auroc_hi) < 1e-12), (t, cohort)
sub.to_csv(out("subgroups.csv"), index=False)
for _, r in sub.iterrows():
    tag = "EXT" if r.cohort == C_EXT else "INT"
    log(f"  {tag} {r.target:>6} {r.subgroup:<10} n={r.n:>6} pos={r.positives:>5} AUROC {r.auroc:.3f} "
        f"({r.auroc_lo:.3f}-{r.auroc_hi:.3f}) [{r.n_boot_valid} valid resamples]")

# ------------------------------------------------------------------ (b) female - male AUROC difference
log("\n(b) Female minus male AUROC (DeLong unpaired; bootstrap within each sex, 2000 resamples)")
rows = []
for cohort_name, _, data in SOURCES:
    for t in T_KEYS:
        y, p, age, male = data[t]; y = y.astype(int); male = male.astype(int)
        for analysis, amask in (("All ages", np.ones(len(y), bool)), ("Age 40-69 (sensitivity)", (age >= 40) & (age <= 69))):
            f, m = amask & (male == 0), amask & (male == 1)
            base = dict(cohort=cohort_name, target=SHORT[t], analysis=analysis, n_female=int(f.sum()),
                        pos_female=int(y[f].sum()), n_male=int(m.sum()), pos_male=int(y[m].sum()))
            if min(y[f].sum(), y[m].sum()) == 0 or y[f].sum() == f.sum() or y[m].sum() == m.sum():
                rows.append(dict(base, n_boot_valid=0, note="AUROC undefined in one sex (one class)"))
                continue
            dd, dlo, dhi, dp = delong_unpaired(y[f], p[f], y[m], p[m])
            b = boot_diff_indep(y[f], p[f], y[m], p[m])
            rows.append(dict(base, auroc_female=roc_auc_score(y[f], p[f]),
                             auroc_female_lo=b["female_boot"][0], auroc_female_hi=b["female_boot"][1],
                             auroc_male=roc_auc_score(y[m], p[m]),
                             auroc_male_lo=b["male_boot"][0], auroc_male_hi=b["male_boot"][1],
                             diff_f_minus_m=dd, delong_lo=dlo, delong_hi=dhi, delong_p=dp,
                             boot_lo=b["diff_boot"][0], boot_hi=b["diff_boot"][1], boot_p_twosided=b["p_boot"],
                             n_boot_valid=b["n_valid"],
                             note="fewer than 20 positives in one sex: imprecise" if min(y[f].sum(), y[m].sum()) < 20 else ""))
SEX_COLS = ["cohort", "target", "analysis", "n_female", "pos_female", "auroc_female", "auroc_female_lo", "auroc_female_hi",
            "n_male", "pos_male", "auroc_male", "auroc_male_lo", "auroc_male_hi", "diff_f_minus_m", "delong_lo",
            "delong_hi", "delong_p", "boot_lo", "boot_hi", "boot_p_twosided", "n_boot_valid", "note"]
sexd = pd.DataFrame(rows).reindex(columns=SEX_COLS)
sexd.to_csv(out("sex_difference.csv"), index=False)
for _, r in sexd.iterrows():
    tag = "EXT" if r.cohort == C_EXT else "INT"
    log(f"  {tag} {r.target:>6} {r.analysis:<24} F {r.auroc_female:.3f} (pos {r.pos_female}) M {r.auroc_male:.3f} "
        f"(pos {r.pos_male}) diff {r.diff_f_minus_m:+.3f} DeLong ({r.delong_lo:+.3f},{r.delong_hi:+.3f}) "
        f"p={r.delong_p:.4f} boot ({r.boot_lo:+.3f},{r.boot_hi:+.3f})")

# ------------------------------------------------------------------ (d) continuous vs equal-budget binary vs retrained ensemble
# (fitted before (c) so the refit models can be included there)
log("\n(d) Simple homoscedastic continuous LightGBM vs equal-budget binary LightGBM vs retrained ensemble")
M_CONT = "Simple homoscedastic continuous LightGBM -> P(urate>=t)"
M_BIN = "Equal-budget binary LightGBM (LGBM_FIXED)"
M_ENS = "Retrained calibrated soft-voting ensemble"
SHORTM = {M_CONT: "Continuous", M_BIN: "Binary LGBM", M_ENS: "Ensemble"}
REG_PARAMS = {**LGBM_FIXED, "objective": "regression", "n_jobs": N_JOBS}
CLF_PARAMS = {**LGBM_FIXED, "n_jobs": N_JOBS}
COMPS = [(M_CONT, M_BIN, "continuous minus binary LightGBM (formulation effect)"),
         (M_BIN, M_ENS, "binary LightGBM minus retrained ensemble (ensemble/tuning effect)"),
         (M_CONT, M_ENS, "continuous minus retrained ensemble (total)")]


class ContinuousLGBM:
    """log(urate) LightGBM regressor plus ONE pooled 5-fold OOF residual distribution (homoscedastic)."""

    def __init__(self, X, z):
        oof = np.full(len(z), np.nan)
        for a, b in KFold(n_splits=5, shuffle=True, random_state=SEED).split(X):
            oof[b] = LGBMRegressor(**REG_PARAMS).fit(X.iloc[a], z[a]).predict(X.iloc[b])
        self.resid = np.sort(z - oof)
        self.reg = LGBMRegressor(**REG_PARAMS).fit(X, z)
        self.smear = float(np.mean(np.exp(self.resid)))          # Duan smearing for the mean on the mg/dL scale

    def mu(self, X):
        return self.reg.predict(X)

    def p_exceed(self, mu, t):
        """P(urate >= t) = share of OOF residuals r with mu + r >= log t."""
        r = self.resid
        return (len(r) - np.searchsorted(r, np.log(t) - mu, side="left")) / len(r)

    def describe(self):
        r = self.resid
        return (f"OOF residual (log scale) mean {r.mean():+.4f} SD {r.std():.4f}, 2.5/50/97.5 pct "
                f"{np.percentile(r, 2.5):+.3f}/{np.median(r):+.3f}/{np.percentile(r, 97.5):+.3f}")


def clf_metrics(y, p):
    auc, alo, ahi = auroc_ci(y, p); ap, plo, phi = auprc_ci(y, p)
    ci, cs, oe = calib_stats(y, p)
    return dict(n=len(y), positives=int(y.sum()), prevalence=y.mean(), auroc=auc, auroc_lo=alo, auroc_hi=ahi,
                auprc=ap, auprc_lo=plo, auprc_hi=phi, share_p_eq_0=float(np.mean(p == 0)),
                positives_at_p_eq_0=int(((p == 0) & (y == 1)).sum()), mean_p=float(np.mean(p)),
                calib_intercept=ci, calib_slope=cs, oe_ratio=oe, brier=brier_score_loss(y, p), scaled_brier=scaled_brier(y, p))


def delong_ci(y, pa, pb):
    """Paired DeLong for AUROC(a) - AUROC(b) on the same persons: difference, 95% CI, two-sided p
    (identical statistic to common.delong_paired, which supplies the p-value)."""
    a10, a01, m, n = _placements(y, pa); b10, b01, _, _ = _placements(y, pb)
    S = np.cov(np.vstack([a10, b10])) / m + np.cov(np.vstack([a01, b01])) / n
    se = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 0.0)); dd = a10.mean() - b10.mean()
    d0, p = delong_paired(y, pa, pb)
    assert abs(d0 - dd) < 1e-12
    return dd, dd - 1.96 * se, dd + 1.96 * se, p


def evaluate_block(cohort_name, trained_on, Y, P):
    """Classification rows for every model in P ({model: {target: p}}, aligned with Y[target]) and paired
    DeLong comparison rows for every pair in COMPS."""
    res = []
    for t in T_KEYS:
        y = np.asarray(Y[t]).astype(int)
        for model, pt in P.items():
            res.append(dict(cohort=cohort_name, row_type="classification", target=SHORT[t], model=model,
                            trained_on=trained_on[model], **clf_metrics(y, pt[t])))
        for a, b, lab in COMPS:
            dd, lo, hi, p = delong_ci(y, P[a][t], P[b][t])
            res.append(dict(cohort=cohort_name, row_type="comparison", target=SHORT[t], model=lab,
                            model_a=a, model_b=b, n=len(y), positives=int(y.sum()), prevalence=y.mean(),
                            dauroc=dd, dauroc_lo=lo, dauroc_hi=hi, delong_p=p))
    return res


def reg_row(cohort_name, trained_on, target, mu, u, smear):
    lz = np.log(u)
    return dict(cohort=cohort_name, row_type="regression", target=target, model=M_CONT.split(" ->")[0] + " (log urate)",
                trained_on=trained_on, n=len(u), mae_log=mean_absolute_error(lz, mu), r2_log=r2_score(lz, mu),
                mae_mgdl=mean_absolute_error(u, np.exp(mu)), r2_mgdl=r2_score(u, np.exp(mu) * smear),
                pearson_r_mgdl=float(np.corrcoef(u, np.exp(mu))[0, 1]))


# --- refits on the training persons of the shared split, in the order the retrained ensembles were trained on them
XY = {t: train_xy(THR[t]) for t in T_KEYS}
X_tr = XY[T_KEYS[0]][0]
assert all(XY[t][0].equals(X_tr) for t in T_KEYS) and list(X_tr.columns) == FMODEL and len(X_tr) == len(tr)
u_tr = d["urate"].to_numpy(float)[tr]
assert all((XY[t][1] == (u_tr >= THR[t])).all() for t in T_KEYS)
z_tr = np.log(u_tr)
cont = ContinuousLGBM(X_tr, z_tr)
bin_ = {t: LGBMClassifier(**CLF_PARAMS).fit(X_tr, XY[t][1]) for t in T_KEYS}
log(f"  shared training split (adults >={ADULT_AGE} y; the retrained ensembles' training persons): n={len(tr):,}; "
    + ", ".join(f"{SHORT[t]} training positives {int(XY[t][1].sum()):,}" for t in T_KEYS) + f"; internal test n={len(te):,}")
log(f"  continuous LightGBM (shared training split): {cont.describe()}")
TRAINED = {M_CONT: f"shared training split (n={len(tr)}; one regressor for all thresholds)",
           M_BIN: f"shared training split (n={len(tr)}; one classifier per threshold)",
           M_ENS: f"shared training split (n={len(tr)}; retrained v5 ensemble: Optuna-tuned LightGBM + XGBoost + "
                  f"CatBoost, isotonic-calibrated soft voting)"}
mu_int, mu_ext = cont.mu(dte[FMODEL]), cont.mu(ext[FMODEL])
P_ext = {M_CONT: {t: cont.p_exceed(mu_ext, THR[t]) for t in T_KEYS},
         M_BIN: {t: bin_[t].predict_proba(ext[FMODEL])[:, 1] for t in T_KEYS},
         M_ENS: p_ext}
P_int = {M_CONT: {t: cont.p_exceed(mu_int, THR[t]) for t in T_KEYS},
         M_BIN: {t: bin_[t].predict_proba(dte[FMODEL])[:, 1] for t in T_KEYS},
         M_ENS: p_int}

CC_EXT = "External KNHANES >=19 y (primary; all models trained on the shared training split)"
CC_INT = "Internal test (shared person-level split; all models trained on its training persons)"
rows = evaluate_block(CC_EXT, TRAINED, y_ext, P_ext) + evaluate_block(CC_INT, TRAINED, y_int, P_int)
reg_rows = [reg_row(CC_EXT, TRAINED[M_CONT], "log(urate)", mu_ext, ext["urate"].to_numpy(float), cont.smear),
            reg_row(CC_INT, TRAINED[M_CONT], "log(urate)", mu_int, dte["urate"].to_numpy(float), cont.smear)]
for r in reg_rows:
    log(f"  REG {r['cohort'][:40]:<40} n={r['n']:>6} MAE {r['mae_mgdl']:.3f} mg/dL (log {r['mae_log']:.4f}); "
        f"R2 {r['r2_mgdl']:.3f} (log {r['r2_log']:.3f}); r {r['pearson_r_mgdl']:.3f}")

COLS = ["cohort", "row_type", "target", "model", "trained_on", "model_a", "model_b", "n", "positives", "prevalence",
        "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "calib_intercept", "calib_slope", "oe_ratio",
        "brier", "scaled_brier", "mean_p", "share_p_eq_0", "positives_at_p_eq_0",
        "dauroc", "dauroc_lo", "dauroc_hi", "delong_p", "mae_log", "r2_log", "mae_mgdl", "r2_mgdl", "pearson_r_mgdl"]


def tidy(df, cols):
    for c in ("positives", "positives_at_p_eq_0"):
        if c in df:
            df[c] = df[c].astype("Int64")
    return df.reindex(columns=cols)


def log_rows(df, tagger):
    for _, r in df.iterrows():
        tag = tagger(r.cohort)
        if r.row_type == "classification":
            log(f"  {tag:<4} {r.target:>6} {SHORTM[r.model]:<11} n={r.n} pos={r.positives} AUROC {r.auroc:.3f} "
                f"({r.auroc_lo:.3f}-{r.auroc_hi:.3f}) AUPRC {r.auprc:.3f} int {r.calib_intercept:+.3f} slope "
                f"{r.calib_slope:.3f} O:E {r.oe_ratio:.3f} sBrier {r.scaled_brier:.3f} [p=0: {100 * r.share_p_eq_0:.1f}%, "
                f"{int(r.positives_at_p_eq_0)} positives]")
        elif r.row_type == "comparison":
            log(f"  {tag:<4} {r.target:>6}   dAUROC {SHORTM[r.model_a]} - {SHORTM[r.model_b]}: {r.dauroc:+.4f} "
                f"({r.dauroc_lo:+.4f} to {r.dauroc_hi:+.4f}) DeLong p={r.delong_p:.4f}")


cont_df = tidy(pd.concat([pd.DataFrame(rows), pd.DataFrame(reg_rows)], ignore_index=True), COLS)
# the retrained ensemble rows equal main_performance.csv (same predictions, bootstrap and calibration code)
for coh, cohort in ((CC_EXT, "external"), (CC_INT, "internal")):
    for t in T_KEYS:
        r = cont_df[(cont_df.cohort == coh) & (cont_df.target == SHORT[t]) & (cont_df.model == M_ENS)].iloc[0]
        m = mp_row(t, cohort)
        assert r.n == m.n and r.positives == m.events and abs(r.auroc - m.auroc) < 5e-5, (t, cohort)
        for a, b in (("auroc_lo", "auroc_lo"), ("auroc_hi", "auroc_hi"), ("oe_ratio", "oe"), ("brier", "brier"),
                     ("calib_intercept", "cal_intercept"), ("calib_slope", "cal_slope")):
            assert abs(r[a] - m[b]) < 1e-9, (t, cohort, a)
cont_df.to_csv(out("continuous.csv"), index=False)
log_rows(cont_df, lambda c: {CC_EXT: "EXT", CC_INT: "INT"}[c])

# ------------------------------------------------------------------ (c) nested-threshold coherence
log("\n(c) Nested-threshold coherence (share of persons with a higher predicted risk for a stricter threshold)")
log(f"  internal: all {len(te):,} test persons of the shared split (the same persons for the three thresholds); "
    f"external: all {len(ext):,} KNHANES adults")
M_BIN_NC = M_BIN + "; shared training split"
SHORTM[M_BIN_NC] = "Binary LGBM"
PAIRS = [("P(>=9.0) > P(>=7.0)", "uric_severe_9", "uric_hyper_70"),
         ("P(>=10.0) > P(>=9.0)", "uric_severe_10", "uric_severe_9"),
         ("P(>=10.0) > P(>=7.0)", "uric_severe_10", "uric_hyper_70")]


def coherence(P, cohort_name, model):
    res = []
    anyv = np.zeros(len(P[T_KEYS[0]]), bool)
    for label, hi, lo in PAIRS:
        v = P[hi] > P[lo]; anyv |= v; size = (P[hi] - P[lo])[v]
        ratio = (P[hi][v] / P[lo][v])[P[lo][v] > 0]
        res.append(dict(cohort=cohort_name, model=model, comparison=label, n=len(v), n_violations=int(v.sum()),
                        prop_violations=v.mean(),
                        median_violation=float(np.median(size)) if v.any() else np.nan,
                        p75_violation=float(np.percentile(size, 75)) if v.any() else np.nan,
                        p95_violation=float(np.percentile(size, 95)) if v.any() else np.nan,
                        max_violation=float(size.max()) if v.any() else np.nan,
                        prop_violation_gt_0_01=float(np.mean((P[hi] - P[lo]) > 0.01)),
                        median_ratio_hi_over_lo=float(np.median(ratio)) if len(ratio) else np.nan,
                        median_p_lower_threshold_in_violators=float(np.median(P[lo][v])) if v.any() else np.nan,
                        share_violators_p_lower_eq_0=float(np.mean(P[lo][v] == 0)) if v.any() else np.nan))
    res.append(dict(cohort=cohort_name, model=model, comparison="Any of the three", n=len(anyv),
                    n_violations=int(anyv.sum()), prop_violations=anyv.mean()))
    return res


def log_coherence(df):
    for _, r in df.iterrows():
        med = (f" median size {r.median_violation:.2e} (p95 {r.p95_violation:.2e}, max {r.max_violation:.3f}; "
               f">0.01 in {100 * r.prop_violation_gt_0_01:.3f}%; P_lower=0 in {100 * r.share_violators_p_lower_eq_0:.0f}% of violators)"
               if pd.notna(r.get("median_violation")) else "")
        log(f"  {r.cohort[:34]:<34} {SHORTM.get(r.model, r.model)[:11]:<11} {r.comparison:<22} {r.n_violations:>6}/{r.n:<6} "
            f"= {100 * r.prop_violations:6.2f}%{med}")


rows = []
rows += coherence(p_ext, C_EXT, M_ENS)
rows += coherence(p_int, C_INT, M_ENS)
rows += coherence(P_ext[M_BIN], C_EXT, M_BIN_NC)
rows += coherence(P_int[M_BIN], C_INT, M_BIN_NC)
rows += coherence(P_ext[M_CONT], C_EXT, M_CONT)
rows += coherence(P_int[M_CONT], C_INT, M_CONT)
coh = pd.DataFrame(rows)
assert (coh.loc[coh.model == M_CONT, "n_violations"] == 0).all()        # coherent by construction
assert set(coh.loc[coh.cohort == C_INT, "n"]) == {len(te)} and set(coh.loc[coh.cohort == C_EXT, "n"]) == {len(ext)}
coh.to_csv(out("nested_coherence.csv"), index=False)
log_coherence(coh)

# ------------------------------------------------------------------ removed output
# leak_demo_internal.csv is no longer produced: it isolated the original models' metrics on development test persons
# most of whom were in their record-level training rows. The shared person-level split has no such overlap (asserted
# at the top), so the stale file is deleted (a copy is kept in _snapshot_original_models_261016/working).
stale = out("leak_demo_internal.csv")
if os.path.exists(stale):
    os.remove(stale)
    log("\nremoved the former output leak_demo_internal.csv (no longer produced)")

log(f"\nRuntime {(time.time() - T0) / 60:.1f} min")
with open(out("calc_sex_nested_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
