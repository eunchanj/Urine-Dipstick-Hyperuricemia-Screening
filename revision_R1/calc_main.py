"""Predictions and headline performance of the retrained models (bestmodel_v5; decision of 2026-10-16).

Internal test set = the split_v2 test persons of the corrected cohort dev_v2 (one person-level split shared by the
three thresholds). External = ext_v2 (KNHANES adults aged >= 19, inputs coded by the label-matched mapping X1).
Writes, under the file names and columns that the downstream analysis scripts read (the same names the R1 evaluation of the original models used
read them), plus an id column:
  preds_internal_clean.csv     target, y, p, age, male, id   (internal test persons, dev_v2 order)
  preds_external_corrected.csv target, y, p, age, male, year, wt, id   (ext_v2 order)
  main_performance.csv         per target and cohort: n, events, AUROC (1,000-resample percentile CI, seed 42),
                               Brier, O:E, calibration intercept and slope
For the response letter only (original_vs_retrained.csv): the original (submitted, bestmodel_v4) models on the same
KNHANES adults with the inputs they were validated with (cached KNHANES coding M1, leukocyte esterase 0.321), and on
the internal test persons of split_v2 who were not in their own training split (inputs as in dev_v2, plus the
observed leukocyte esterase), next to the retrained models on the same persons.
"""
import os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from cohort import (HERE, TARGETS, F13, FMODEL, ID, LE, cached, cached_v2, dev_v2, split_v2, ext_v2, ext_corrected,
                    dev_submitted, split_submitted, labels, load_model)
from common import auroc_ci, calib_stats

T0 = time.time()
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def perf(y, p):
    auc, lo, hi = auroc_ci(y, p)
    a, s, oe = calib_stats(y, p)
    return dict(n=len(y), events=int(np.sum(y)), auroc=auc, auroc_lo=lo, auroc_hi=hi, brier=brier_score_loss(y, p),
                oe=oe, cal_intercept=a, cal_slope=s, mean_pred=float(np.mean(p)), prevalence=float(np.mean(y)))


win2, kn2 = cached_v2()
d, steps = dev_v2(win2)
tr, te = split_v2(d)
ext = ext_v2(kn2).reset_index(drop=True)
dte = d.iloc[te]
log(f"dev_v2 {len(d):,} persons; internal test {len(te):,}; KNHANES adults {len(ext):,}")

rows, pint, pext = [], [], []
for t, cfg in TARGETS.items():
    m = load_model(t, "v5")
    assert list(m.feature_names_in_) == FMODEL, "v5 model inputs differ from FMODEL"
    y_i = labels(dte["urate"], cfg["threshold"]); p_i = m.predict_proba(dte[FMODEL])[:, 1]
    y_e = labels(ext["urate"], cfg["threshold"]); p_e = m.predict_proba(ext[FMODEL])[:, 1]
    for name, yy, pp in (("internal test", y_i, p_i), ("KNHANES adults", y_e, p_e)):
        r = perf(yy, pp)
        rows.append(dict(target=cfg["short"], cohort=name, **r))
        log(f"{cfg['short']:>6} {name:<15} n {r['n']:,} events {r['events']:,}  AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f})  O:E {r['oe']:.2f}  slope {r['cal_slope']:.2f}  "
            f"Brier {r['brier']:.4f}  exact-0 predictions {int((pp == 0).sum()):,}")
    pint.append(pd.DataFrame(dict(target=t, y=y_i, p=p_i, age=dte["age"].to_numpy(), male=dte["male"].to_numpy(),
                                  id=dte[ID].to_numpy())))
    pext.append(pd.DataFrame(dict(target=t, y=y_e, p=p_e, age=ext["age"].to_numpy(), male=ext["male"].to_numpy(),
                                  year=ext["year"].to_numpy(), wt=ext["wt_itvex"].to_numpy(), id=ext["id"].to_numpy())))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "main_performance.csv"), index=False)
pd.concat(pint).to_csv(os.path.join(HERE, "preds_internal_clean.csv"), index=False)
pd.concat(pext).to_csv(os.path.join(HERE, "preds_external_corrected.csv"), index=False)

# ------------------------------------------------------------------ original vs retrained (response letter only)
log("\noriginal (v4) vs retrained (v5) on the same persons")
win, kn = cached()
ds = dev_submitted(win)
ec = ext_corrected(kn).set_index("id").loc[ext["id"]].reset_index()      # same persons and order as ext
le_by_id = win.drop_duplicates(ID).set_index(ID)[LE]
cmp_rows = []
for t, cfg in TARGETS.items():
    m4, m5 = load_model(t, "v4"), load_model(t, "v5")
    y_e = labels(ext["urate"], cfg["threshold"])
    p4 = m4.predict_proba(ec[F13])[:, 1]; p5 = m5.predict_proba(ext[FMODEL])[:, 1]
    tr4, _, _ = split_submitted(ds, cfg["threshold"])
    unseen = ~dte[ID].isin(set(ds[ID].iloc[tr4])).to_numpy()
    x4 = dte[FMODEL].copy(); x4[LE] = le_by_id.reindex(dte[ID]).to_numpy()
    ok = unseen & x4[LE].notna().to_numpy()
    y_i = labels(dte["urate"], cfg["threshold"])[ok]
    p4i = m4.predict_proba(x4[F13][ok])[:, 1]
    p5i = m5.predict_proba(dte[FMODEL][ok])[:, 1]
    for cohort, yy, pa, pb in (("KNHANES adults", y_e, p4, p5),
                               ("internal test persons not in the original model's training split", y_i, p4i, p5i)):
        ra, rb = perf(yy, pa), perf(yy, pb)
        cmp_rows.append(dict(target=cfg["short"], cohort=cohort, n=ra["n"], events=ra["events"],
                             **{f"original_{k}": ra[k] for k in ("auroc", "auroc_lo", "auroc_hi", "oe", "cal_slope", "brier")},
                             **{f"retrained_{k}": rb[k] for k in ("auroc", "auroc_lo", "auroc_hi", "oe", "cal_slope", "brier")}))
        log(f"{cfg['short']:>6} {cohort[:40]:<40} n {ra['n']:,}: AUROC original {ra['auroc']:.3f} vs retrained "
            f"{rb['auroc']:.3f}; O:E {ra['oe']:.2f} vs {rb['oe']:.2f}")
pd.DataFrame(cmp_rows).to_csv(os.path.join(HERE, "original_vs_retrained.csv"), index=False)
log(f"\ncalc_main done ({time.time() - T0:.0f} s)")
with open(os.path.join(HERE, "calc_main_log.txt"), "w") as f:
    f.write("\n".join(LOG) + "\n")
