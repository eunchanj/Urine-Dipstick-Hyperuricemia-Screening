"""Shared metrics for the uric revision (ported from the liver JBI audit pipeline).

Conventions
  * Every bootstrap resamples with replacement at the original sample size; percentile 95% CI;
    resamples containing one class are skipped; seed 42 unless stated.
  * Development evaluations use the corrected cohort of the retrained models (cohort.dev_v2, one record per
    person) and its single person-level 80/20 split shared by the three thresholds (cohort.split_v2).
  * External evaluations use KNHANES participants aged >= 19 with the X1 coding (cohort.ext_v2).
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split

SEED = 42
EPS = 1e-9
# Fixed LightGBM configuration used for every feature tier (equal modelling budget), as in the liver audit.
LGBM_FIXED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
GRID = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc)
        for nl in (15, 31, 63) for lr in (0.03, 0.1) for mc in (20, 100)]
# Decision-curve threshold grids suited to each outcome's prevalence.
NB_GRID = {7.0: np.round(np.arange(0.02, 0.51, 0.01), 3),
           9.0: np.round(np.arange(0.002, 0.101, 0.002), 4),
           10.0: np.round(np.arange(0.001, 0.051, 0.001), 4)}
REFER_FRACS = (0.01, 0.02, 0.05, 0.10, 0.20)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def boot_ci(y, p, fn, n=1000, seed=SEED):
    rng = np.random.RandomState(seed); out = []
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        out.append(fn(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def auroc_ci(y, p, n=1000, seed=SEED):
    return roc_auc_score(y, p), *boot_ci(y, p, roc_auc_score, n, seed)


def auprc_ci(y, p, n=1000, seed=SEED):
    return average_precision_score(y, p), *boot_ci(y, p, average_precision_score, n, seed)


def paired_boot(y, p_hi, p_lo, n=1000, seed=SEED):
    """Paired bootstrap CI for an AUROC difference (identical indices for both arms)."""
    rng = np.random.default_rng(seed); y = np.asarray(y); out = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if y[s].min() == y[s].max(): continue
        out.append(roc_auc_score(y[s], p_hi[s]) - roc_auc_score(y[s], p_lo[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _placements(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum()
    xp, xn = p[pos], p[neg]; xn_s = np.sort(xn); xp_s = np.sort(xp)
    v10 = (np.searchsorted(xn_s, xp, "right") + np.searchsorted(xn_s, xp, "left")) / (2.0 * n)
    v01 = 1.0 - (np.searchsorted(xp_s, xn, "right") + np.searchsorted(xp_s, xn, "left")) / (2.0 * m)
    return v10, v01, m, n


def delong_paired(y, p1, p2):
    """DeLong test for two correlated ROC curves on the same records. Returns (diff, p)."""
    a10, a01, m, n = _placements(y, p1); b10, b01, _, _ = _placements(y, p2)
    S = np.cov(np.vstack([a10, b10])) / m + np.cov(np.vstack([a01, b01])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; d = a10.mean() - b10.mean()
    return d, (2 * (1 - stats.norm.cdf(abs(d) / np.sqrt(var))) if var > 0 else np.nan)


def delong_var(y, p):
    v10, v01, m, n = _placements(y, p)
    return v10.mean(), np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n


def delong_unpaired(y1, p1, y2, p2):
    """Difference of two independent AUROCs (e.g. women vs men) with DeLong variances."""
    a1, v1 = delong_var(y1, p1); a2, v2 = delong_var(y2, p2)
    se = np.sqrt(v1 + v2); z = (a1 - a2) / se
    return a1 - a2, a1 - a2 - 1.96 * se, a1 - a2 + 1.96 * se, 2 * (1 - stats.norm.cdf(abs(z)))


def fit_logistic_1d(x, y):
    """Exact maximum-likelihood logistic regression of y on one covariate (Newton-Raphson).
    Returns (intercept, slope). Replaces sklearn lbfgs, which stops early on logit(p) with
    extreme values (isotonic zeros give logit about -20.7)."""
    x = np.asarray(x, float); y = np.asarray(y, float); X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
    for _ in range(100):
        pr = expit(X @ b); W = pr * (1 - pr)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - pr)); b += step
        if np.max(np.abs(step)) < 1e-10: break
    return float(b[0]), float(b[1])


def fit_offset_intercept(y, lp):
    """Calibration-in-the-large by maximum likelihood: logit(p') = a + logit(p), slope fixed at 1."""
    a = 0.0
    for _ in range(100):
        pn = expit(a + lp); g = float((y - pn).sum()); h = float(-(pn * (1 - pn)).sum())
        step = -g / h; a += step
        if abs(step) < 1e-10: break
    return a


def calib_stats(y, p):
    """Calibration intercept (offset MLE), calibration slope (unpenalised logistic on logit p), O:E."""
    y = np.asarray(y); lp = logit(p)
    a = fit_offset_intercept(y, lp)
    slope = fit_logistic_1d(lp, y)[1]
    return a, slope, y.mean() / np.mean(p)


def scaled_brier(y, p):
    prev = np.mean(y); return 1 - brier_score_loss(y, p) / (prev * (1 - prev))


def recalibrate(y, p, n_rep=100, seed=SEED):
    """Repeated half-splits: recalibration fitted on one half, evaluated on the other."""
    y = np.asarray(y); lp = logit(p); idx = np.arange(len(y)); rows = []
    for r in range(n_rep):
        tr, te = train_test_split(idx, test_size=0.5, random_state=seed + r, stratify=y)
        a = fit_offset_intercept(y[tr], lp[tr]); p_i = expit(a + lp[te])
        b0, b1 = fit_logistic_1d(lp[tr], y[tr])
        p_s = expit(b0 + b1 * lp[te])
        ci, cs, oe = calib_stats(y[te], p_i)
        rows.append(dict(intercept_fit=a, slope_fit=b1,
                         sbrier_raw=scaled_brier(y[te], np.asarray(p)[te]), sbrier_intercept=scaled_brier(y[te], p_i),
                         sbrier_slope=scaled_brier(y[te], p_s), oe_after_intercept=oe))
    d = pd.DataFrame(rows); out = {"n_splits": n_rep}
    for c in d.columns:
        out[c] = d[c].mean(); out[c + "_lo"] = d[c].quantile(0.025); out[c + "_hi"] = d[c].quantile(0.975)
    return out


def decile_curve(y, p, bins=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    return pd.DataFrame([dict(bin=k + 1, n=int((b == k).sum()), mean_pred=float(np.mean(np.asarray(p)[b == k])),
                              observed=float(np.mean(np.asarray(y)[b == k]))) for k in range(len(q) - 1) if (b == k).any()])


def confusion_at(y, p, thr):
    y = np.asarray(y); yh = np.asarray(p) >= thr
    tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum())
    fn = int((~yh & (y == 1)).sum()); tn = int((~yh & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return dict(threshold=float(thr), sens=sens, spec=spec, ppv=ppv, npv=npv, pos_rate=(tp + fp) / len(y),
                tp=tp, fp=fp, fn=fn, tn=tn, referrals_per_case=(1 / ppv if ppv > 0 else np.nan))


def threshold_for_sensitivity(y, p, s):
    fpr, tpr, th = roc_curve(y, p); k = min(np.searchsorted(tpr, s), len(th) - 1); return th[k]


def youden_threshold(y, p):
    fpr, tpr, th = roc_curve(y, p); return th[np.argmax(tpr - fpr)]


def split_half_operating(y, p, rule, n_rep=100, seed=SEED):
    """Threshold chosen on one random half by `rule(y, p)`, evaluated on the other half."""
    y = np.asarray(y); p = np.asarray(p); idx = np.arange(len(y)); rows = []
    for r in range(n_rep):
        a, b = train_test_split(idx, test_size=0.5, random_state=seed + r, stratify=y)
        rows.append(confusion_at(y[b], p[b], rule(y[a], p[a])))
    d = pd.DataFrame(rows)[["sens", "spec", "ppv", "npv", "pos_rate"]]
    return {k: d[k].mean() for k in d} | {k + "_lo": d[k].quantile(0.025) for k in d} | {k + "_hi": d[k].quantile(0.975) for k in d}


def net_benefit(y, p, thresholds):
    y = np.asarray(y); p = np.asarray(p); n = len(y); prev = y.mean(); out = []
    for pt in thresholds:
        yh = p >= pt; tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum())
        out.append(dict(threshold=float(pt), nb_model=tp / n - fp / n * pt / (1 - pt),
                        nb_all=prev - (1 - prev) * pt / (1 - pt), nb_none=0.0))
    return out


def psi(a, b, bins=10):
    a = np.asarray(a); b = np.asarray(b); cats = np.union1d(np.unique(a), np.unique(b))
    if len(cats) <= 20:
        pa = np.array([(a == c).mean() for c in cats]); pb = np.array([(b == c).mean() for c in cats])
    else:
        qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        pa = np.histogram(a, bins=qs)[0] / len(a); pb = np.histogram(b, bins=qs)[0] / len(b)
    pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def dev_split(d, threshold, seed=SEED):
    """Person-disjoint 80/20 split of the corrected development cohort, stratified on the outcome."""
    y = (d["urate"].values >= threshold).astype(int); idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    return tr, te, y
