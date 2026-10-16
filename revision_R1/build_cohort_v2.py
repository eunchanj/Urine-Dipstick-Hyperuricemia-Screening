"""Build and check the corrected cohort of the retrained models (decision of 2026-10-16).

Writes cache/dev_window_v2.parquet and cache/knhanes_v2.parquet (cohort.write_cache_v2), then cohort_info_v2.csv:
the exclusion steps of dev_v2 (Figure 1), the split_v2 counts, the KNHANES ext_v2 counts and the checks below.
Checks (assertions): all window rows of a person carry the same model inputs and same-day urate; no person younger
than 19 by exact completed years (atlas cache lab_day_wide.adult on the urine-test day); the training and test
persons are disjoint; the external cohort is the adult KNHANES cohort of the revision (ext_corrected ids).
"""
import os, time
import numpy as np, pandas as pd
from cohort import (HERE, TARGETS, ID, DATE, FMODEL, F13, ATLAS_V2, write_cache_v2, cached_v2, dev_v2, split_v2,
                    ext_v2, ext_corrected, cached, labels)

T0 = time.time()
LOG, ROWS = [], []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def row(item, value, note=""):
    ROWS.append(dict(item=item, value=value, note=note))


log("write_cache_v2:")
for item, value in write_cache_v2(log=lambda s: None):
    row(item, value, "write_cache_v2 check")
    log(f"  {item}: {value:,}" if isinstance(value, (int, np.integer)) else f"  {item}: {value}")

win, kn2 = cached_v2()
# all rows of a person repeat the same first-visit urine test: inputs and same-day values must be constant
w = win.copy(); w["ua_nitrite"] = w["ua_nitrite_v2"]; w["bmi_unified"] = w["bmi_v2"]
cols = FMODEL + ["urate_day"]
nun = w.groupby(ID)[cols].nunique(dropna=False).max()
assert (nun <= 1).all(), f"inputs vary within person: {nun[nun > 1].to_dict()}"
log("all window rows of a person carry identical inputs and same-day urate")

d, steps = dev_v2(win)
log("\ndev_v2 exclusion steps (persons):")
prev = None
for desc, n in steps:
    log(f"  {desc}: {n:,}" + (f"  (excluded {prev - n:,})" if prev is not None else ""))
    row(f"step: {desc}", n, "" if prev is None else f"excluded {prev - n:,}")
    prev = n
assert d[ID].is_unique

# exact completed-years age on the urine-test day (atlas cache) -> no minors
import pyarrow.parquet as pq  # noqa: E402
lab = pq.read_table(f"{ATLAS_V2}/derived/lab_day_wide.parquet", columns=["id", "date", "age", "adult"]).to_pandas()
lab["id"] = lab["id"].astype("int64"); lab["date"] = pd.to_datetime(lab["date"])
j = pd.DataFrame({"id": d[ID].astype("int64").to_numpy(),
                  "date": pd.to_datetime(d[DATE]).dt.normalize().to_numpy()}).merge(lab, on=["id", "date"], how="left",
                                                                                     validate="one_to_one")
n_minor_exact = int((j["adult"] == False).sum())  # noqa: E712
n_no_age = int(j["adult"].isna().sum())
log(f"exact completed-years age (atlas): minors {n_minor_exact}, not determinable {n_no_age}; "
    f"max |legacy age - exact age| {np.nanmax(np.abs(d['age'].to_numpy(float) - j['age'].to_numpy(float))):.2f}")
assert n_minor_exact == 0
row("cohort persons younger than 19 by exact completed years (atlas)", n_minor_exact)
row("cohort persons with exact age not determinable (atlas)", n_no_age)

n = len(d)
row("analytic cohort persons", n)
row("nitrite positive persons", int((d["ua_nitrite"] == 1).sum()), f"{100 * (d['ua_nitrite'] == 1).mean():.2f}%")
row("serum urate '<1.0' persons (counted as 0.5)", int(d["urate_day_below"].sum()))
row("serum urate same-time repeats averaged (persons)", int((d["urate_day_nrep"] > 1).sum()))
for src, k in d["bmi_v2_source"].value_counts().items():
    row(f"BMI source {src} (persons)", int(k))
row("processed days_difference != 0 among cohort persons", int((d["days_difference"] != 0).sum()),
    "their same-day urate comes from the atlas cache")

tr, te = split_v2(d)
assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == n
assert not set(d[ID].iloc[tr]) & set(d[ID].iloc[te])
row("training persons", len(tr)); row("test persons", len(te))
log(f"\nsplit_v2: training {len(tr):,}, test {len(te):,} persons (disjoint)")
for t, cfg in TARGETS.items():
    y = labels(d["urate"], cfg["threshold"])
    for name, idx in (("cohort", np.arange(n)), ("training", tr), ("test", te)):
        row(f"{name} events {cfg['short']}", int(y[idx].sum()), f"{100 * y[idx].mean():.2f}%")
    log(f"  {cfg['short']}: events cohort {int(y.sum()):,} ({100 * y.mean():.2f}%), training {int(y[tr].sum()):,}, "
        f"test {int(y[te].sum()):,}")

# external
e2 = ext_v2(kn2)
_, kn = cached()
ec = ext_corrected(kn)
assert set(e2["id"]) == set(ec["id"]), "ext_v2 differs from the adult KNHANES cohort of the revision"
row("KNHANES adults with 12 inputs and urate", len(e2))
for t, cfg in TARGETS.items():
    y = labels(e2["urate"], cfg["threshold"])
    row(f"KNHANES events {cfg['short']}", int(y.sum()), f"{100 * y.mean():.2f}%")
    row(f"KNHANES female events {cfg['short']}", int(y[e2["male"].to_numpy() == 0].sum()))
changed = (e2.set_index("id")[FMODEL[:9]].sort_index() != ec.set_index("id")[FMODEL[:9]].sort_index()).any(axis=1)
row("KNHANES adults whose dipstick inputs change from the submitted coding (M1) to X1", int(changed.sum()))
log(f"KNHANES ext_v2: {len(e2):,} adults (same ids as ext_corrected); inputs changed by X1: {int(changed.sum()):,}")

out = pd.DataFrame(ROWS)
out.to_csv(os.path.join(HERE, "cohort_info_v2.csv"), index=False)
log(f"\ncohort_info_v2.csv: {len(out)} rows ({time.time() - T0:.0f} s)")
with open(os.path.join(HERE, "build_cohort_v2_log.txt"), "w") as f:
    f.write("\n".join(LOG) + "\n")
