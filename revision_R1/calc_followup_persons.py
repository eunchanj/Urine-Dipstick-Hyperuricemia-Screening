"""Persons versus records in the internal test set of the retrained models (follow-up review item: Table 2 labels the
internal test set in records; how many distinct persons are they?).

The corrected development cohort of the retrained models (cohort.dev_v2) holds adults only (age 19-110 y at the urine
test) with one record per person, and split_v2 is ONE person-level 80/20 split shared by the three thresholds. The
internal test set is therefore the same persons for every threshold and records = persons. The script checks this
(no repeated person, no person in both training and test, no person younger than 19, test persons and labels identical
to preds_internal_clean.csv) and writes one row per threshold so the table generators can look the counts up by target.

Removed with the retraining (decision of 2026-10-16; the previous version is in _snapshot_original_models_261016/):
the columns test_records_submitted, under19_records, minor_records_removed and training_records_minor_excluded, which
described the row-level split of the original models (submitted test rows, rows of persons also in training, rows of
persons younger than 19); none applies to the person-level adult split of the retrained models.
Writes fu_int_persons.csv and calc_followup_persons_log.txt. Run: /opt/anaconda3/bin/python calc_followup_persons.py"""
import os
import numpy as np, pandas as pd
from cohort import HERE, TARGETS, ID, ADULT_AGE, labels, main_cohort, is_adult

N_TEST, N_TRAIN = 25907, 103628
LOG = []


def log(s=""):
    print(s); LOG.append(s)


d, tr, te, _ = main_cohort()
dtr, dte = d.iloc[tr], d.iloc[te]
assert len(dte) == N_TEST and len(dtr) == N_TRAIN and len(d) == N_TEST + N_TRAIN
assert d[ID].is_unique and not set(dtr[ID]) & set(dte[ID]), "a person is in both the training and the test set"
assert is_adult(d["age"]).all() and int((dte["age"] < ADULT_AGE).sum()) == 0
ref = pd.read_csv(os.path.join(HERE, "preds_internal_clean.csv"))
rows = []
for t, cfg in TARGETS.items():
    r = ref[ref["target"] == t]
    y_te = labels(dte["urate"], cfg["threshold"])
    y_tr = labels(dtr["urate"], cfg["threshold"])
    assert len(r) == len(dte) and (r["id"].to_numpy() == dte[ID].to_numpy()).all() and (r["y"].to_numpy() == y_te).all()
    vc = dte[ID].value_counts()
    rows.append(dict(target=cfg["short"], records=len(dte), persons=int(len(vc)),
                     persons_with_2plus_records=int((vc > 1).sum()), training_records=len(dtr),
                     training_persons=int(dtr[ID].nunique())))
    o = rows[-1]
    assert o["records"] == o["persons"] and o["persons_with_2plus_records"] == 0
    assert o["training_records"] == o["training_persons"]
    log(f"{cfg['short']}: internal test {o['records']:,} records = {o['persons']:,} persons (one record per person; the "
        f"same persons for every threshold; matches preds_internal_clean.csv), events {int(y_te.sum()):,}; training "
        f"{o['training_records']:,} records = {o['training_persons']:,} persons, events {int(y_tr.sum()):,}; persons in "
        f"both sets 0; persons aged < {ADULT_AGE} 0")
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "fu_int_persons.csv"), index=False)
pd.set_option("display.width", 200)
log(out.to_string(index=False))
with open(os.path.join(HERE, "calc_followup_persons_log.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
