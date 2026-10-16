"""Clinical-utility metrics for the retrained uric-acid ensembles (Reviewer AW #7).

The reviewer's point: at 0.2-0.9% prevalence AUROC flatters the model; show AUPRC, sensitivity,
PPV and net benefit. Everything in (a)-(d) uses the retrained calibrated ensembles (bestmodel_v5; 12 inputs =
cohort.FMODEL: 9 dipstick fields, age, sex, BMI) through their stored predictions (preds_internal_clean.csv,
preds_external_corrected.csv, written by calc_main.py); nothing is refitted there. Only (e) fits new models
(LGBM_FIXED, in memory only).

COHORTS (decision of 2026-10-16): the corrected development cohort dev_v2 (cohort.main_cohort: adults, one record
per person) with ONE person-level 80/20 split shared by the three thresholds. "internal" = the internal test persons
(d.iloc[te], the same persons for every threshold); "external" = KNHANES adults aged >= 19 with the dipstick inputs
coded by the label-matched mapping X1. The internally chosen thresholds in (b), their transfer to KNHANES and the
internal decision curves in (d) therefore come from the internal test persons. (e) fits on the training persons
d.iloc[tr] and evaluates on d.iloc[te] and on KNHANES. The cohort counts are checked against cohort_info_v2.csv and
the AUROCs of (a) against main_performance.csv.

  (a) discrimination.csv        AUROC and AUPRC (1000-bootstrap CI), prevalence; internal and external.
  (b) operating_points.csv      thresholds chosen on the INTERNAL test persons (Youden; sensitivity 80%
                                and 90%), applied unchanged to external; sens, spec, PPV, NPV,
                                positive rate, referrals per detected case, cases detected per 1,000,
                                with 1000-bootstrap CIs (threshold held fixed). Plus top-k referral
                                policies at REFER_FRACS (flag the top fraction by predicted risk).
                                threshold_source keeps the labels "internal clean test" (external rows) and
                                "internal clean test (apparent)" (internal rows): the internal test persons are
                                disjoint from the training persons (make_tables.py checks the label).
  (c) splithalf_operating.csv   external only: threshold chosen on one random half, applied to the
                                other, 100 repeats (common.split_half_operating).
  (d) decision_curve.csv        net benefit of the model vs treat-all / treat-none over NB_GRID,
      decision_curve_ranges.csv and the threshold ranges where the model beats both defaults.
  (e) utility_12F_vs_3F.csv     LGBM_FIXED 12F (the retrained models' 12 inputs) vs 3F (age, sex, BMI), both fitted on
                                the training persons of the shared split; discrimination and O:E on the internal test
                                persons and on KNHANES; on KNHANES also paired net-benefit differences (400-resample
                                paired bootstrap) and additional cases detected per 1,000 under top-k referral.
                                Column `scenario` = "primary" (one scenario; the former leukocyte esterase scenarios
                                were removed because leukocyte esterase is not an input of the retrained models).
                                value_12F / beats_defaults_12F / delta_12F_vs_default hold the 12F arm (model_A).
                                Net-benefit rows (analysis == "net_benefit") also carry:
                                  best_default          max(treat-all, treat-none) = max(nb_all, 0);
                                  delta_*_vs_default    arm net benefit minus best_default (paired CI);
                                  beats_defaults_*      point estimate > best_default (_ci: CI lower > 0);
                                  nb_region             where the comparison sits relative to the defaults.
                                Rows with analysis == "net_benefit_summary" give, per target, the threshold
                                ranges (`ranges`) meeting each `criterion`; the 12F advantage is stated only where
                                12F beats both defaults; elsewhere both arms are below the best default and 12F is
                                at most less harmful. The log also gives the number of nitrite splits of the 12F fit.

Bootstrap: resampling with replacement at full size, percentile 95% CI, RandomState(42).randint --
the same draws as common.boot_ci, so every CI in a cohort x target cell uses identical resamples.
Top-k with tied predictions: the records tied at the cut-off are flagged fractionally (the expected
count under random tie-breaking), so exactly fraction f of the screened are referred.
Run from the working folder:  /opt/anaconda3/bin/python calc_utility.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
import time  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
# cohort and common are imported BEFORE lightgbm (import-order rule for the revision scripts)
from cohort import HERE, TARGETS, F3, FMODEL, ID, main_cohort, labels  # noqa: E402
from common import (SEED, LGBM_FIXED, NB_GRID, REFER_FRACS, auroc_ci, auprc_ci, confusion_at,  # noqa: E402
                    net_benefit, threshold_for_sensitivity, youden_threshold, split_half_operating,
                    delong_paired, paired_boot)
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

N_BOOT = 1000          # CIs on the retrained models' metrics
N_BOOT_PAIRED = 400    # paired bootstrap for 12F vs 3F utility differences
N_SPLIT = 100          # split-half repeats
N_JOBS = 4             # thread cap
METRICS = ["sens", "spec", "ppv", "npv", "pos_rate", "referrals_per_case", "detected_per_1000"]
RULES = {"youden": youden_threshold,
         "sens80": lambda y, p: threshold_for_sensitivity(y, p, 0.80),
         "sens90": lambda y, p: threshold_for_sensitivity(y, p, 0.90)}

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def pct(v):
    return f"{100 * v:.1f}%"


# ------------------------------------------------------------------ bootstrap machinery
def boot_weights(n, n_boot, seed=SEED):
    """Bootstrap resamples as per-record multiplicities (identical draws to common.boot_ci)."""
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.randint(0, n, n), minlength=n).astype(float)


def ci(samples):
    a = np.asarray(samples, float)
    return np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)


def op_metrics(tp, fp, fn, tn):
    """Operating metrics from (possibly weighted / fractional) confusion counts; arrays allowed."""
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / (tp + fp), npv=tn / (tn + fn),
                    pos_rate=(tp + fp) / n, referrals_per_case=(tp + fp) / tp, detected_per_1000=1000 * tp / n)


class Scorer:
    """Confusion counts for one prediction vector under fixed thresholds or top-k referral, for any weights."""

    def __init__(self, y, p):
        self.y = np.asarray(y, float); self.p = np.asarray(p, float); self.n = len(self.y)
        u, inv = np.unique(self.p, return_inverse=True)       # ascending unique predictions
        self.grp = len(u) - 1 - inv                             # group 0 = highest predicted risk
        self.ug = u[::-1]; self.G = len(u); self._F = {}

    def fixed(self, thrs, w=None):
        """(tp, fp, fn, tn) arrays for rule p >= thr at each threshold."""
        w = np.ones(self.n) if w is None else w
        key = tuple(np.asarray(thrs, float))
        if key not in self._F:
            self._F[key] = (self.p[:, None] >= np.asarray(key)[None, :]).astype(float)
        F = self._F[key]
        wp, wn = w * self.y, w * (1 - self.y)
        tp, fp = wp @ F, wn @ F
        return tp, fp, wp.sum() - tp, wn.sum() - fp

    def topk(self, fracs, w=None):
        """(tp, fp, fn, tn) when exactly fraction f of the (weighted) sample with highest risk is flagged;
        the group tied at the cut-off is flagged fractionally."""
        w = np.ones(self.n) if w is None else w
        pos_g = np.bincount(self.grp, weights=w * self.y, minlength=self.G)
        tot_g = np.bincount(self.grp, weights=w, minlength=self.G)
        before = np.cumsum(tot_g) - tot_g
        n, P = w.sum(), (w * self.y).sum()
        k = np.asarray(fracs, float) * n
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip((k[:, None] - before[None, :]) / tot_g[None, :], 0, 1)
        frac = np.nan_to_num(frac)                              # empty groups (weight 0) flag nothing
        tp = frac @ pos_g; fp = k - tp
        return tp, fp, P - tp, (n - P) - fp

    def topk_cutoff(self, frac):
        """Predicted risk at the top-k boundary (unweighted)."""
        tot = np.bincount(self.grp, minlength=self.G)
        return float(self.ug[min(np.searchsorted(np.cumsum(tot), frac * self.n), self.G - 1)])

    def net_benefit(self, grid, w=None):
        tp, fp, fn, tn = self.fixed(grid, w)
        n = tp + fp + fn + tn; odds = np.asarray(grid) / (1 - np.asarray(grid)); prev = (tp + fn) / n
        return tp / n - fp / n * odds, prev - (1 - prev) * odds


def metrics_with_ci(sc, counts_fn, arg, n_boot=N_BOOT):
    """Point estimates and bootstrap CIs of op_metrics for each threshold / fraction in `arg`."""
    point = op_metrics(*counts_fn(arg))
    boots = {m: [] for m in METRICS}
    for w in boot_weights(sc.n, n_boot):
        b = op_metrics(*counts_fn(arg, w))
        for m in METRICS: boots[m].append(b[m])
    out = {}
    for m in METRICS:
        lo, hi = ci(boots[m]); out[m], out[m + "_lo"], out[m + "_hi"] = point[m], lo, hi
    counts = counts_fn(arg)
    return out, counts


def ranges(grid, flag):
    """Contiguous runs of grid points where flag is True, as 'a-b' strings."""
    runs, start, prev = [], None, None
    for g, f in zip(grid, flag):
        if f and start is None: start = g
        if not f and start is not None: runs.append((start, prev)); start = None
        prev = g
    if start is not None: runs.append((start, prev))
    return "; ".join(f"{a:g}-{b:g}" if a != b else f"{a:g}" for a, b in runs) or "none"


# ------------------------------------------------------------------ data
t0 = time.time()
pi = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))
pe = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"))
COH = {"internal": pi, "external": pe}
MP = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
MP_COHORT = {"internal": "internal test", "external": "KNHANES adults"}     # main_performance.csv cohort labels
CINFO = {k: int(v) for k, v in pd.read_csv(os.path.join(HERE, "cohort_info_v2.csv"))[["item", "value"]].values}
d, tr, te, ext = main_cohort()
dte = d.iloc[te]
assert len(d) == CINFO["analytic cohort persons"] and len(tr) == CINFO["training persons"] \
    and len(te) == CINFO["test persons"] and len(ext) == CINFO["KNHANES adults with 12 inputs and urate"]
log("=== cohorts (retrained models; dev_v2 and the shared person-level split; KNHANES adults, X1 coding) ===")
log(f"dev_v2 {len(d):,} persons: training {len(tr):,}, internal test {len(te):,} (the same persons for every "
    f"threshold); KNHANES adults {len(ext):,} (counts = cohort_info_v2.csv)")
for t, cfg in TARGETS.items():
    s, thr = cfg["short"], cfg["threshold"]
    gi, ge = pi[pi.target == t], pe[pe.target == t]
    assert len(gi) == len(te) and np.array_equal(gi.id.values, dte[ID].values), f"{s}: internal rows differ"
    assert np.array_equal(gi.y.values, labels(dte["urate"], thr)) and int(gi.y.sum()) == CINFO[f"test events {s}"]
    assert len(ge) == len(ext) and np.array_equal(ge.id.astype(str).values, ext["id"].astype(str).values)
    assert np.array_equal(ge.y.values, labels(ext["urate"], thr)) and int(ge.y.sum()) == CINFO[f"KNHANES events {s}"]
    log(f"{s:>6} internal test persons {len(gi):,} ({int(gi.y.sum()):,} events); KNHANES adults {len(ge):,} "
        f"({int(ge.y.sum()):,} events)")

# ================================================================== (a) discrimination
log("\n=== (a) discrimination (retrained ensembles) ===")
rows = []
MP_DIFF = 0.0
for t, cfg in TARGETS.items():
    for coh, df in COH.items():
        g = df[df.target == t]; y, p = g.y.values, g.p.values
        auc, auc_lo, auc_hi = auroc_ci(y, p, N_BOOT)
        ap, ap_lo, ap_hi = auprc_ci(y, p, N_BOOT)
        prev = y.mean()
        mp = MP[(MP.target == cfg["short"]) & (MP.cohort == MP_COHORT[coh])]
        assert len(mp) == 1 and int(mp.n.iloc[0]) == len(y) and int(mp.events.iloc[0]) == int(y.sum())
        for v, ref in ((auc, mp.auroc.iloc[0]), (auc_lo, mp.auroc_lo.iloc[0]), (auc_hi, mp.auroc_hi.iloc[0])):
            assert round(float(v), 4) == round(float(ref), 4), (cfg["short"], coh, v, ref)
            MP_DIFF = max(MP_DIFF, abs(float(v) - float(ref)))
        rows.append(dict(target=cfg["short"], role=cfg["role"], cohort=coh, n=len(y), positives=int(y.sum()),
                         prevalence=prev, auroc=auc, auroc_lo=auc_lo, auroc_hi=auc_hi,
                         auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi, auprc_lift=ap / prev,
                         auprc_lift_lo=ap_lo / prev, auprc_lift_hi=ap_hi / prev))
        log(f"{cfg['short']:>6} {coh:8s} n={len(y)} pos={int(y.sum())} prev={pct(prev)} "
            f"AUROC {auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})  AUPRC {ap:.3f} ({ap_lo:.3f}-{ap_hi:.3f}) = {ap / prev:.1f}x prevalence")
disc = pd.DataFrame(rows)
disc.to_csv(os.path.join(HERE, "discrimination.csv"), index=False)
log(f"AUROC and 95% CI = main_performance.csv (calc_main.py): max |diff| {MP_DIFF:.1e}")
log(f"[{time.time() - t0:.0f}s]")

# ================================================================== (b) operating points
log("\n=== (b) operating points: thresholds from the internal test persons, applied to external ===")
rows = []
for t, cfg in TARGETS.items():
    gi = pi[pi.target == t]; ge = pe[pe.target == t]
    thr = {r: float(f(gi.y.values, gi.p.values)) for r, f in RULES.items()}
    for coh, g in (("internal", gi), ("external", ge)):
        sc = Scorer(g.y.values, g.p.values); prev = g.y.mean()
        base = dict(target=cfg["short"], cohort=coh, n=sc.n, positives=int(g.y.sum()), prevalence=prev)
        # fixed thresholds chosen on internal
        names = list(RULES)
        res = [metrics_with_ci(sc, lambda a, w=None: sc.fixed(a, w), [thr[r]]) for r in names]
        for r, (m, (tp, fp, fn, tn)) in zip(names, res):
            chk = confusion_at(g.y.values, g.p.values, thr[r])       # cross-check against common.py
            assert chk["tp"] == tp[0] and chk["fp"] == fp[0] and abs(chk["sens"] - m["sens"][0]) < 1e-12
            rows.append(base | dict(policy=r, threshold_source="internal clean test" if coh == "external"
                                    else "internal clean test (apparent)", threshold=thr[r],
                                    tp=tp[0], fp=fp[0], fn=fn[0], tn=tn[0])
                        | {k: float(np.atleast_1d(v)[0]) for k, v in m.items()})
        # top-k referral policies
        m, (tp, fp, fn, tn) = metrics_with_ci(sc, lambda a, w=None: sc.topk(a, w), list(REFER_FRACS))
        for j, f in enumerate(REFER_FRACS):
            rows.append(base | dict(policy=f"top_{f * 100:g}%", threshold_source=f"top {f * 100:g}% of screened",
                                    threshold=sc.topk_cutoff(f), tp=tp[j], fp=fp[j], fn=fn[j], tn=tn[j])
                        | {k: float(v[j]) for k, v in m.items()})
op = pd.DataFrame(rows)
op.to_csv(os.path.join(HERE, "operating_points.csv"), index=False)
for t, cfg in TARGETS.items():
    for _, r in op[(op.target == cfg["short"])].iterrows():
        if r.cohort == "internal" and not r.policy.startswith("top"):
            log(f"{r.target:>6} internal {r.policy:7s} thr={r.threshold:.4f} sens {pct(r.sens)} spec {pct(r.spec)} "
                f"PPV {pct(r.ppv)} pos {pct(r.pos_rate)}")
        if r.cohort == "external":
            log(f"{r.target:>6} external {r.policy:7s} thr={r.threshold:.4f} sens {pct(r.sens)} ({pct(r.sens_lo)}-{pct(r.sens_hi)}) "
                f"spec {pct(r.spec)} PPV {pct(r.ppv)} ({pct(r.ppv_lo)}-{pct(r.ppv_hi)}) NPV {100 * r.npv:.2f}% "
                f"pos {pct(r.pos_rate)} refer/case {r.referrals_per_case:.1f} ({r.referrals_per_case_lo:.1f}-{r.referrals_per_case_hi:.1f}) "
                f"det/1000 {r.detected_per_1000:.2f}")

# external vs internal at the sensitivity-target rules (descriptive only; no causal attribution)
log("\n-- external vs internal at the sensitivity-target thresholds (nominal targets 80% and 90%) --")
SENS_TARGET = {"sens80": 0.80, "sens90": 0.90}
above, below, ppv_fell, prev_txt = [], [], [], []
for t, cfg in TARGETS.items():
    s = cfg["short"]; rel = []
    pv_i = op[(op.target == s) & (op.cohort == "internal")].prevalence.iloc[0]
    pv_e = op[(op.target == s) & (op.cohort == "external")].prevalence.iloc[0]
    prev_txt.append(f"{s} {pct(pv_i)} -> {pct(pv_e)}")
    for r, st in SENS_TARGET.items():
        ri = op[(op.target == s) & (op.cohort == "internal") & (op.policy == r)].iloc[0]
        rx = op[(op.target == s) & (op.cohort == "external") & (op.policy == r)].iloc[0]
        side = "above" if rx.sens > st else ("below" if rx.sens < st else "at")
        inc = bool(rx.sens_lo <= st <= rx.sens_hi)
        rel.append((st, rx.sens, side, inc)); ppv_fell.append(rx.ppv < ri.ppv)
        log(f"{s:>6} {r}: target {pct(st)} (internal achieved {pct(ri.sens)}) -> external {pct(rx.sens)} "
            f"({pct(rx.sens_lo)}-{pct(rx.sens_hi)}), {side} target, CI "
            f"{'includes' if inc else 'excludes'} target; PPV {pct(ri.ppv)} -> "
            f"{pct(rx.ppv)}; positive rate {pct(ri.pos_rate)} -> {pct(rx.pos_rate)}; prevalence {pct(pv_i)} -> {pct(pv_e)}")
    if all(x[2] == "above" for x in rel): above.append((s, rel))
    else: below.append((s, rel))
ci_txt = lambda rel: ("95% CI includes the target" if all(x[3] for x in rel) else "95% CI excludes the target"
                      if not any(x[3] for x in rel) else "95% CI includes the target for "
                      + " and ".join(f"the {pct(x[0])} target" for x in rel if x[3]))
msg = [f"External sensitivity (point estimate) exceeded the internal target only at "
       + (" and ".join(f"{s} ({ci_txt(rel)})" for s, rel in above) if above else "no threshold")]
for s, rel in below:
    gap = max(100 * (st - v) for st, v, _, _ in rel)
    msg.append(f"at {s} it was below the target ({', '.join(f'{pct(v)} for the {pct(st)} target' for st, v, _, _ in rel)}; "
               f"shortfall at most {gap:.1f} percentage points; {ci_txt(rel)})")
msg = "; ".join(msg) + ". "
msg += ("External PPV was lower than internal for every target and rule, alongside lower external prevalence ("
        if all(ppv_fell) else "External PPV was not uniformly lower than internal; prevalence internal -> external (")
msg += "; ".join(prev_txt) + ")."
log("SUMMARY: " + msg)
log(f"[{time.time() - t0:.0f}s]")

# ================================================================== (c) split-half operating points
log("\n=== (c) split-half operating points on external (threshold on one half, applied to the other; 100 repeats) ===")
rows = []
for t, cfg in TARGETS.items():
    g = pe[pe.target == t]; y, p = g.y.values, g.p.values
    for r, rule in RULES.items():
        chosen = []                                             # thresholds picked on each selection half

        def recording_rule(ya, pa, rule=rule):
            chosen.append(float(rule(ya, pa))); return chosen[-1]

        res = split_half_operating(y, p, recording_rule, N_SPLIT, SEED)
        ch = np.asarray(chosen, float)
        rows.append(dict(target=cfg["short"], rule=r, n=len(y), positives=int(y.sum()), n_repeats=N_SPLIT)
                    | {k: float(v) for k, v in res.items()}
                    | dict(threshold_mean=ch.mean(), threshold_lo=np.quantile(ch, 0.025), threshold_hi=np.quantile(ch, 0.975)))
sh = pd.DataFrame(rows)
cols = ["target", "rule", "n", "positives", "n_repeats"] + \
       [c for m in ["sens", "spec", "ppv", "npv", "pos_rate", "threshold"] for c in
        ((m + "_mean", m + "_lo", m + "_hi") if m == "threshold" else (m, m + "_lo", m + "_hi"))]
sh = sh[cols]
sh.to_csv(os.path.join(HERE, "splithalf_operating.csv"), index=False)
for _, r in sh.iterrows():
    log(f"{r.target:>6} {r.rule:7s} thr {r.threshold_mean:.4f} ({r.threshold_lo:.4f}-{r.threshold_hi:.4f}) "
        f"sens {pct(r.sens)} ({pct(r.sens_lo)}-{pct(r.sens_hi)}) spec {pct(r.spec)} PPV {pct(r.ppv)} "
        f"({pct(r.ppv_lo)}-{pct(r.ppv_hi)}) pos {pct(r.pos_rate)}")
log(f"[{time.time() - t0:.0f}s]")

# ================================================================== (d) decision curves
log("\n=== (d) decision curves (retrained ensembles) ===")
rows, rng_rows = [], []
for t, cfg in TARGETS.items():
    grid = NB_GRID[cfg["threshold"]]; odds = grid / (1 - grid)
    for coh, df in COH.items():
        g = df[df.target == t]; y, p = g.y.values, g.p.values; sc = Scorer(y, p)
        nb_m, nb_a = sc.net_benefit(grid)
        ref = pd.DataFrame(net_benefit(y, p, grid))                 # cross-check against common.py
        assert np.allclose(ref.nb_model, nb_m) and np.allclose(ref.nb_all, nb_a)
        delta = nb_m - np.maximum(nb_a, 0)
        bm, bd = [], []
        for w in boot_weights(sc.n, N_BOOT):
            m_, a_ = sc.net_benefit(grid, w); bm.append(m_); bd.append(m_ - np.maximum(a_, 0))
        m_lo, m_hi = ci(bm); d_lo, d_hi = ci(bd)
        tp, fp, _, _ = sc.fixed(grid); prev = y.mean()
        for j, pt in enumerate(grid):
            rows.append(dict(target=cfg["short"], cohort=coh, n=sc.n, prevalence=prev, threshold=pt,
                             nb_model=nb_m[j], nb_model_lo=m_lo[j], nb_model_hi=m_hi[j], nb_all=nb_a[j], nb_none=0.0,
                             delta_vs_best_default=delta[j], delta_lo=d_lo[j], delta_hi=d_hi[j],
                             beats_both=bool(delta[j] > 0), beats_both_ci=bool(d_lo[j] > 0),
                             standardized_nb=nb_m[j] / prev,
                             interventions_avoided_per_100=100 * (nb_m[j] - nb_a[j]) / odds[j],
                             tp=int(tp[j]), fp=int(fp[j]), pos_rate=(tp[j] + fp[j]) / sc.n))
        for crit, flag in (("point estimate > max(treat-all, treat-none)", delta > 0),
                           ("95% CI lower bound > 0", d_lo > 0)):
            rng_rows.append(dict(target=cfg["short"], cohort=coh, criterion=crit, grid_min=grid[0], grid_max=grid[-1],
                                 grid_step=round(grid[1] - grid[0], 4), n_grid=len(grid),
                                 n_grid_beating=int(flag.sum()), ranges=ranges(grid, flag)))
        log(f"{cfg['short']:>6} {coh:8s} grid {grid[0]:g}-{grid[-1]:g}: beats both at "
            f"{ranges(grid, delta > 0)} (CI>0: {ranges(grid, d_lo > 0)})")
dca = pd.DataFrame(rows)
dca.to_csv(os.path.join(HERE, "decision_curve.csv"), index=False)
pd.DataFrame(rng_rows).to_csv(os.path.join(HERE, "decision_curve_ranges.csv"), index=False)
# a few anchor thresholds for the log
for t, cfg in TARGETS.items():
    sub = dca[(dca.target == cfg["short"]) & (dca.cohort == "external")]
    for pt in sub.threshold.values[[0, len(sub) // 4, len(sub) // 2]]:
        r = sub[sub.threshold == pt].iloc[0]
        log(f"{cfg['short']:>6} external pt={pt:g}: NB model {r.nb_model:.5f} ({r.nb_model_lo:.5f}-{r.nb_model_hi:.5f}) "
            f"treat-all {r.nb_all:.5f}; delta vs best default {r.delta_vs_best_default:.5f} ({r.delta_lo:.5f}-{r.delta_hi:.5f})")
log(f"[{time.time() - t0:.0f}s]")

# ================================================================== (e) 12F vs 3F utility
log("\n=== (e) LGBM_FIXED 12F vs 3F (training persons of the shared split -> internal test persons and KNHANES adults) ===")
NIT = "ua_nitrite"
SCENARIOS = {   # scenario -> (label of the richer-input arm, description)
    "primary": ("12F", "12F LGBM_FIXED (9 dipstick fields, age, sex, BMI: the inputs of the retrained models)"),
}
# Removed with the retraining: scenarios sens_LE0 (KNHANES leukocyte esterase set to 0) and sens_12F_noLE (refit
# without leukocyte esterase), and the leukocyte esterase split diagnostics; leukocyte esterase is not an input.
params = {**LGBM_FIXED, "n_jobs": N_JOBS}


def split_thresholds(mdl, feats, feat):
    """Split thresholds (x <= thr goes left) on `feat` over all trees of a fitted LGBMClassifier."""
    dm = mdl.booster_.dump_model(); j = feats.index(feat); out = []

    def walk(node):
        if "split_index" in node:
            if node["split_feature"] == j:
                assert node["decision_type"] == "<="; out.append(node["threshold"])
            walk(node["left_child"]); walk(node["right_child"])
    for tr_ in dm["tree_info"]: walk(tr_["tree_structure"])
    return np.asarray(out, float)


def region_label(arm, nbA, nb3, best, nba, d, lo, hi):
    """Where the arm comparison sits relative to the best default at one threshold probability."""
    if nbA > best:
        sig = (f"{arm} > 3F (CI excludes 0)" if lo > 0 else f"3F > {arm} (CI excludes 0)" if hi < 0
               else f"{arm} vs 3F CI includes 0")
        return f"{arm} beats both defaults; {sig}"
    if nb3 > best:
        return "only 3F beats both defaults"
    ref = "treat-none" if nba <= 0 else "treat-all"
    pos = "below" if (nbA < best and nb3 < best) else "at or below"
    rel = f"{arm} less harmful" if d > 0 else ("3F less harmful" if d < 0 else f"{arm} = 3F")
    return f"both {pos} {ref}; {rel}"


def compare(t, cfg, scn, ye, pA_ext, p3_ext, internal):
    """All (e) rows for one scenario: discrimination/calibration (internal test persons and external), net benefit
    with per-arm default comparisons, net-benefit range summary, and top-k referral differences (external)."""
    arm, desc = SCENARIOS[scn]; grid = NB_GRID[cfg["threshold"]]
    base = dict(target=cfg["short"], scenario=scn, model_A=desc)
    rows = []
    cohorts = [("internal",) + internal, ("external", ye, pA_ext, p3_ext)]
    for coh, yy, a, b in cohorts:
        d, pval = delong_paired(yy, a, b)
        lo, hi = paired_boot(yy, a, b, N_BOOT)
        aA, a3 = roc_auc_score(yy, a), roc_auc_score(yy, b)
        pA, p3 = average_precision_score(yy, a), average_precision_score(yy, b)
        rows.append(base | dict(analysis="discrimination", cohort=coh, x=np.nan, metric="auroc", value_12F=aA,
                                value_3F=a3, diff=d, diff_lo=lo, diff_hi=hi, p_value=pval,
                                ci_method="paired bootstrap 1000; DeLong p"))
        rows.append(base | dict(analysis="discrimination", cohort=coh, x=np.nan, metric="auprc",
                                value_12F=pA, value_3F=p3, diff=pA - p3))
        rows.append(base | dict(analysis="calibration", cohort=coh, x=np.nan, metric="observed_to_expected",
                                value_12F=yy.mean() / a.mean(), value_3F=yy.mean() / b.mean()))
        log(f"{cfg['short']:>6} [{scn}] {coh}: AUROC {arm} {aA:.3f} vs 3F {a3:.3f} (diff {d:+.3f}, {lo:+.3f} to "
            f"{hi:+.3f}; DeLong p={pval:.2g}); AUPRC {pA:.3f} vs {p3:.3f}; O:E {arm} {yy.mean() / a.mean():.2f}, "
            f"3F {yy.mean() / b.mean():.2f}")
    # paired net-benefit and top-k differences on external, identical resamples for both arms
    sA, s3 = Scorer(ye, pA_ext), Scorer(ye, p3_ext)
    nbA, nba = sA.net_benefit(grid); nb3, _ = s3.net_benefit(grid)
    best = np.maximum(nba, 0.0)
    mA_, m3_ = op_metrics(*sA.topk(REFER_FRACS)), op_metrics(*s3.topk(REFER_FRACS))
    bnb, bdA, bd3, bdet, bsen, bppv = [], [], [], [], [], []
    for w in boot_weights(len(ye), N_BOOT_PAIRED):
        wA, wall = sA.net_benefit(grid, w); w3, _ = s3.net_benefit(grid, w); wbest = np.maximum(wall, 0.0)
        bnb.append(wA - w3); bdA.append(wA - wbest); bd3.append(w3 - wbest)
        mA, m3 = op_metrics(*sA.topk(REFER_FRACS, w)), op_metrics(*s3.topk(REFER_FRACS, w))
        bdet.append(mA["detected_per_1000"] - m3["detected_per_1000"])
        bsen.append(mA["sens"] - m3["sens"]); bppv.append(mA["ppv"] - m3["ppv"])
    nb_lo, nb_hi = ci(bnb); dA_lo, dA_hi = ci(bdA); d3_lo, d3_hi = ci(bd3)
    d = nbA - nb3
    region = np.array([region_label(arm, nbA[j], nb3[j], best[j], nba[j], d[j], nb_lo[j], nb_hi[j])
                       for j in range(len(grid))])
    for j, pt in enumerate(grid):
        rows.append(base | dict(analysis="net_benefit", cohort="external", x=pt, metric="net_benefit",
                                value_12F=nbA[j], value_3F=nb3[j], diff=d[j], diff_lo=nb_lo[j], diff_hi=nb_hi[j],
                                nb_all=nba[j], best_default=best[j],
                                delta_12F_vs_default=nbA[j] - best[j], delta_12F_vs_default_lo=dA_lo[j],
                                delta_12F_vs_default_hi=dA_hi[j],
                                delta_3F_vs_default=nb3[j] - best[j], delta_3F_vs_default_lo=d3_lo[j],
                                delta_3F_vs_default_hi=d3_hi[j],
                                beats_defaults_12F=bool(nbA[j] > best[j]), beats_defaults_3F=bool(nb3[j] > best[j]),
                                beats_defaults_12F_ci=bool(dA_lo[j] > 0), beats_defaults_3F_ci=bool(d3_lo[j] > 0),
                                nb_region=region[j], ci_method=f"paired bootstrap {N_BOOT_PAIRED}"))
    # range summary
    beatA = nbA > best
    crit = [(f"{arm} beats both defaults (point estimate > max(treat-all, treat-none))", beatA),
            (f"{arm} beats both defaults (CI lower bound > 0)", dA_lo > 0),
            ("3F beats both defaults (point estimate)", nb3 > best),
            ("3F beats both defaults (CI lower bound > 0)", d3_lo > 0),
            (f"{arm} advantage over 3F: {arm} beats both defaults and {arm}-3F CI excludes 0", beatA & (nb_lo > 0)),
            (f"{arm}-3F difference CI excludes 0 in favour of {arm} (any region)", nb_lo > 0),
            ("3F significantly better than " + arm + " (difference CI upper bound < 0)", nb_hi < 0)]
    crit += [(f"region: {lab}", region == lab) for lab in dict.fromkeys(region)]
    summ = {}
    for c, flag in crit:
        summ[c] = ranges(grid, flag)
        rows.append(base | dict(analysis="net_benefit_summary", cohort="external", metric="threshold_range",
                                criterion=c, ranges=summ[c], n_grid=int(np.sum(flag)),
                                grid_min=grid[0], grid_max=grid[-1], grid_step=round(grid[1] - grid[0], 4)))
    adv = summ[crit[4][0]]
    elsewhere = [(lab, summ[f"region: {lab}"]) for lab in dict.fromkeys(region) if not lab.startswith(arm + " beats")]
    stmt = (f"{arm} beats both defaults at pt {summ[crit[0][0]]}; {arm} advantage over 3F (CI excludes 0) stated only "
            f"there: {adv}. Elsewhere: " + ("; ".join(f"{r} -> {lab}" for lab, r in elsewhere) if elsewhere else "none")
            + f". 3F significantly better: {summ[crit[6][0]]}.")
    rows.append(base | dict(analysis="net_benefit_summary", cohort="external", metric="statement",
                            criterion="summary statement", ranges=stmt, n_grid=len(grid),
                            grid_min=grid[0], grid_max=grid[-1], grid_step=round(grid[1] - grid[0], 4)))
    log(f"{cfg['short']:>6} [{scn}] NB grid {grid[0]:g}-{grid[-1]:g}: {stmt}")
    log(f"{'':>6} [{scn}]   {arm} beats defaults with CI>0 at {summ[crit[1][0]]}; 3F beats defaults at "
        f"{summ[crit[2][0]]} (CI>0: {summ[crit[3][0]]}); max {arm}-3F diff {d.max():.5f} at pt={grid[np.argmax(d)]:g}")
    for metric, vA, v3, bs in (("detected_per_1000", 1000 * sA.topk(REFER_FRACS)[0] / sA.n,
                                1000 * s3.topk(REFER_FRACS)[0] / s3.n, bdet),
                               ("sensitivity", mA_["sens"], m3_["sens"], bsen), ("ppv", mA_["ppv"], m3_["ppv"], bppv)):
        lo, hi = ci(bs)
        for j, f in enumerate(REFER_FRACS):
            rows.append(base | dict(analysis="topk_referral", cohort="external", x=f, metric=metric,
                                    value_12F=vA[j], value_3F=v3[j], diff=vA[j] - v3[j], diff_lo=lo[j], diff_hi=hi[j],
                                    ci_method=f"paired bootstrap {N_BOOT_PAIRED}"))
        if metric == "detected_per_1000":
            det = (vA, v3, lo, hi)
    vA, v3, lo, hi = det
    for j, f in enumerate(REFER_FRACS):
        log(f"{'':>6} [{scn}]   top {f * 100:g}%: detected/1000 {arm} {vA[j]:.2f} vs 3F {v3[j]:.2f} "
            f"({vA[j] - v3[j]:+.2f}, {lo[j]:.2f} to {hi[j]:.2f}); sens {pct(mA_['sens'][j])} vs {pct(m3_['sens'][j])}; "
            f"PPV {pct(mA_['ppv'][j])} vs {pct(m3_['ppv'][j])}")
    sig3 = [f"{f * 100:g}%" for j, f in enumerate(REFER_FRACS) if hi[j] < 0]
    log(f"{'':>6} [{scn}]   top-k: 3F significantly more detected at {', '.join(sig3) if sig3 else 'none'}")
    return rows


rows = []
for t, cfg in TARGETS.items():
    thr_u = cfg["threshold"]
    y = labels(d["urate"], thr_u); ytr, yi = y[tr], y[te]
    log(f"{cfg['short']:>6} shared split: training persons {len(tr)} ({int(ytr.sum())} events), internal test persons "
        f"{len(te)} ({int(yi.sum())} events)")
    ye = labels(ext["urate"], thr_u)
    assert np.array_equal(ye, pe[pe.target == t].y.values)       # same external rows/order as the stored predictions
    assert np.array_equal(yi, pi[pi.target == t].y.values)       # same internal test persons/order
    m12 = LGBMClassifier(**params).fit(d[FMODEL].iloc[tr], ytr)
    m3 = LGBMClassifier(**params).fit(d[F3].iloc[tr], ytr)
    p12_i, p12_e = m12.predict_proba(d[FMODEL].iloc[te])[:, 1], m12.predict_proba(ext[FMODEL])[:, 1]
    p3_i, p3_e = m3.predict_proba(d[F3].iloc[te])[:, 1], m3.predict_proba(ext[F3])[:, 1]
    nit_tr = int(np.nansum(d[NIT].values[tr])); nit_ext = int(np.nansum(ext[NIT].values))
    log(f"{'':>6} 12F nitrite splits: {len(split_thresholds(m12, FMODEL, NIT))} over {m12.n_estimators} trees "
        f"(nitrite-positive training persons: {nit_tr}; KNHANES adults nitrite-positive: {nit_ext})")
    rows += compare(t, cfg, "primary", ye, p12_e, p3_e, internal=(yi, p12_i, p3_i))
    log(f"[{time.time() - t0:.0f}s]")

ut = pd.DataFrame(rows)
lead = ["target", "scenario", "model_A", "analysis", "cohort", "x", "metric"]
ut = ut[lead + [c for c in ut.columns if c not in lead]]
ut.to_csv(os.path.join(HERE, "utility_12F_vs_3F.csv"), index=False)

# compact overview (external AUROC difference and 12F advantage range) for the log
log("\n-- (e) overview, external --")
for t, cfg in TARGETS.items():
    for scn, (arm, _) in SCENARIOS.items():
        s = ut[(ut.target == cfg["short"]) & (ut.scenario == scn)]
        a = s[(s.analysis == "discrimination") & (s.cohort == "external") & (s.metric == "auroc")].iloc[0]
        sm = s[s.analysis == "net_benefit_summary"].set_index("criterion").ranges
        log(f"{cfg['short']:>6} {scn:14s} AUROC {arm} {a.value_12F:.3f} vs 3F {a.value_3F:.3f} "
            f"(diff {a['diff']:+.3f}, {a.diff_lo:+.3f} to {a.diff_hi:+.3f}); {arm} beats defaults at "
            f"{sm[f'{arm} beats both defaults (point estimate > max(treat-all, treat-none))']}; advantage over 3F "
            f"stated at {sm[f'{arm} advantage over 3F: {arm} beats both defaults and {arm}-3F CI excludes 0']}; "
            f"3F significantly better at {sm['3F significantly better than ' + arm + ' (difference CI upper bound < 0)']}")
log(f"[{time.time() - t0:.0f}s] done")

with open(os.path.join(HERE, "calc_utility_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
