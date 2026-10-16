"""Reconcile the submitted results with the corrected cohorts, using the deployed v4 ensembles.

For each target:
  A. submitted internal test (row-level split, seed 42)     -> must reproduce performance.csv
  B. submitted external (KNHANES incl. ages 10-18)          -> must reproduce performance.csv
  C. person-clean internal test, adults (drop test rows of persons also in training, implausible ages,
     and, since 2026-10-11, rows aged < 19 at the urine test: adult target population (age >= 19))
  D. corrected external (KNHANES aged >= 19)
A and B reproduce the submitted all-ages estimates and are unchanged; C and D are adults only.
Outputs: reconciliation.csv, preds_internal_clean.csv (adult rows only), preds_external_corrected.csv,
calc_original_log.txt
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from cohort import (HERE, TARGETS, F13, ID, AGE_MAX, ADULT_AGE, load_dev_window, load_knhanes, dev_submitted,
                    ext_submitted, ext_corrected, labels, split_submitted, load_model, is_adult)

N_BOOT = 1000
LOG = []


def log(s=""):
    print(s); LOG.append(s)


def boot_auc(y, p, n=N_BOOT, seed=42):
    rng = np.random.RandomState(seed); out = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


win = load_dev_window(); kn = load_knhanes()
ds = dev_submitted(win); es = ext_submitted(kn); ec = ext_corrected(kn)
# performance.csv as written by src/train_uric.py (original folder 04_results/module_uric_v4; PROJ_U_RESULTS overrides)
ref = pd.read_csv(os.path.join(os.environ.get("PROJ_U_RESULTS", "/Users/chan/Desktop/Proj_U/04_results/module_uric_v4"),
                               "performance.csv"))
rows, pint, pext = [], [], []
for k, (t, cfg) in enumerate(TARGETS.items()):
    m = load_model(t); thr = cfg["threshold"]
    tr, te, y = split_submitted(ds, thr)
    p_te = m.predict_proba(ds[F13].iloc[te])[:, 1]; y_te = y[te]
    ids = ds[ID].values; ages = ds["age"].values
    overlap = np.isin(ids[te], ids[tr]); implausible = ages[te] > AGE_MAX; minor = ages[te] < ADULT_AGE
    keep = ~overlap & ~implausible & is_adult(ages[te])          # person-clean AND adult (adult target population)
    ye_s = labels(es["urate"], thr); pe_s = m.predict_proba(es[F13])[:, 1]
    ye_c = labels(ec["urate"], thr); pe_c = m.predict_proba(ec[F13])[:, 1]
    r = ref.iloc[k]
    for name, yy, pp, reported_auc, reported_brier in (
            ("A internal, submitted split", y_te, p_te, r.Internal_AUROC, r.Internal_Brier),
            ("B external, submitted (incl. 10-18 y)", ye_s, pe_s, r.External_AUROC, r.External_Brier),
            ("C internal, person-clean adults (>=19 y)", y_te[keep], p_te[keep], np.nan, np.nan),
            ("D external, corrected (>=19 y)", ye_c, pe_c, np.nan, np.nan)):
        lo, hi = boot_auc(yy, pp)
        rows.append(dict(target=cfg["short"], evaluation=name, n=len(yy), positives=int(yy.sum()),
                         auroc=roc_auc_score(yy, pp), lo=lo, hi=hi, brier=brier_score_loss(yy, pp),
                         reported_auroc=reported_auc, reported_brier=reported_brier))
    pint.append(pd.DataFrame(dict(target=t, y=y_te[keep], p=p_te[keep], age=ds["age"].values[te][keep],
                                  male=ds["male"].values[te][keep])))
    pext.append(pd.DataFrame(dict(target=t, y=ye_c, p=pe_c, age=ec["age"].values, male=ec["male"].values,
                                  year=ec["year"].values, wt=ec["wt_itvex"].values)))
    n_ov = int(overlap.sum()); n_im = int((~overlap & implausible).sum())
    n_mi = int((~overlap & ~implausible & minor).sum())
    log(f"{cfg['short']}: internal test rows {len(te):,}; removed: person also in training {n_ov:,}, implausible age "
        f"{n_im:,}, aged < {ADULT_AGE} (minor, outside the adult target population) {n_mi:,}; kept (person-clean "
        f"adults) {int(keep.sum()):,} (positives {int(y_te[keep].sum()):,}); minor rows in the locked model's training "
        f"split {int((ages[tr] < ADULT_AGE).sum()):,} of {len(tr):,} (locked model, not refitted)")
out = pd.DataFrame(rows)
out["reproduced"] = np.where(out.reported_auroc.notna(), (out.auroc - out.reported_auroc).abs() < 1e-6, np.nan)
out.to_csv(os.path.join(HERE, "reconciliation.csv"), index=False)
pd.concat(pint).to_csv(os.path.join(HERE, "preds_internal_clean.csv"), index=False)
pd.concat(pext).to_csv(os.path.join(HERE, "preds_external_corrected.csv"), index=False)
pd.set_option("display.width", 200)
log(out.round(4).to_string(index=False))
with open(os.path.join(HERE, "calc_original_log.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
