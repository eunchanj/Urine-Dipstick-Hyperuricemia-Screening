"""Small tabulations needed by Multimedia Appendix 1 that no other analysis script writes to a CSV, for the corrected
cohort of the retrained models (bestmodel_v5, 12 inputs; decision of 2026-10-16).

Everything is computed from the v2 analysis cache with the cohort definitions in cohort.py (cached_v2, persons_v2,
main_cohort, cohort_steps). Nothing is modelled; nothing is refitted. The development data are one first-visit urine
test per person (every window row of a person repeats the same urine test), so records = persons.

  section window_gap              interval (days) between the first-visit urine test and the nearest blood test of the
                                  +/-14-day window ('days_difference'; the nearest over the person's window rows), per
                                  adult person: all adult window persons, adult window persons with a serum urate of
                                  the urine-test day, and the development cohort (dev_v2). Serum urate is taken from
                                  the urine-test day, so the last two rows are same-day by definition (asserted).
  section knhanes_flow            sequential attrition of the harmonised KNHANES 2016-2024 file (dipstick inputs coded
                                  by the label-matched mapping X1) to the records with the 12 inputs and urate (all
                                  ages; the same records as the external cohort of the submission) and to the external
                                  validation cohort (aged >= 19; identical to cohort.main_cohort() ext).
  section dipstick_levels_by_year distribution of every level of the semi-quantitative dipstick inputs by calendar year:
                                  development cohort (dev_v2, year of the urine test) and KNHANES adults (external
                                  cohort, survey year; model scale under X1). Documents the 2022 reporting change level
                                  by level.
  section dev_flow                sequential development flow in persons: window persons (all ages) -> aged >= 19 ->
                                  plausible age (<= 110 y) = adult window persons -> serum urate of the urine-test day
                                  -> BMI of the urine-test day -> 9 dipstick fields and sex = development cohort.
                                  Steps 0 and 2-5 equal cohort.cohort_steps() (Figure 1) and steps 2-5 the attrition of
                                  calc_missing_bias.py; 'n' (records) equals 'n_persons'. Rows with order >= 90 are
                                  reference counts, not flow steps.

ADULT RESTRICTION (decision of 2026-10-11; adult target population, age >= 19 at the urine test, as in KNHANES)
  Every development statistic uses adults (cohort.persons_v2 'adult' flag: age 19-110 y at the urine test); dev_flow
  shows the restriction as steps 1-2. KNHANES rows use participants aged >= 19 except the all-ages flow steps 0-4 and
  the reference rows.

Removed with the retraining (decision of 2026-10-16; the previous version is in _snapshot_original_models_261016/):
  the window_gap row and the two dev_flow reference rows of the submitted row-level frame (they described the frame
  of the original models), the dev_flow step 'one record per person' (the v2 frame is one record per person from the
  start), and leukocyte esterase in dipstick_levels_by_year (not a model input; not part of dev_v2).

Output: supplement_derived.csv, calc_supplement_derived_log.txt
Run from the working folder:  /opt/anaconda3/bin/python calc_supplement_derived.py
"""
import os
import numpy as np
import pandas as pd

import cohort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "supplement_derived.csv")
LOG = os.path.join(HERE, "calc_supplement_derived_log.txt")
LEVEL_FIELDS = ["ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen"]
N_TEST, N_EXT = 25907, 49987


def gap_row(label, days):
    d = np.abs(np.asarray(days, float))
    d = d[~np.isnan(d)]
    return dict(section="window_gap", population=label, n=len(d),
                n_same_day=int((d == 0).sum()), pct_same_day=100 * float((d == 0).mean()),
                n_1_7=int(((d >= 1) & (d <= 7)).sum()), pct_1_7=100 * float(((d >= 1) & (d <= 7)).mean()),
                n_8_14=int(((d >= 8) & (d <= 14)).sum()), pct_8_14=100 * float(((d >= 8) & (d <= 14)).mean()),
                max_days=float(d.max()))


def main():
    ID = cohort.ID
    win2, kn2 = cohort.cached_v2()
    d, tr, te, ext = cohort.main_cohort()
    assert len(te) == N_TEST and len(ext) == N_EXT
    pa = cohort.persons_v2()                                       # one row per window person (all ages)
    rows, log = [], []

    # ---- adult restriction (adult target population): every development statistic below uses adult persons
    ax = cohort.age_exclusion_counts(pa, col="age_at_urine_test")  # raw age keeps the implausible values
    adult = pa["adult"]
    assert int(adult.sum()) == ax["adult"]
    log.append(f"window: {len(win2):,} rows of {len(pa):,} persons (one first-visit urine test per person)")
    log.append(f"adult restriction (age at the urine test {cohort.ADULT_AGE}-{cohort.AGE_MAX} y): window persons "
               f"{ax['rows']:,} -> {ax['adult']:,}; excluded {ax['minor']:,} aged < {cohort.ADULT_AGE} y, "
               f"{ax['implausible']:,} aged > {cohort.AGE_MAX} y, {ax['missing']:,} with age not determinable")

    # ---- interval between the urine test and the nearest blood test of the window (adult persons)
    near = win2.assign(_a=win2["days_difference"].abs()).groupby(ID)["_a"].min()
    pa = pa.assign(gap_days=near.reindex(pa[ID]).to_numpy())
    assert pa["gap_days"].notna().all()
    ad = pa[adult]
    coh = pa.set_index(ID).loc[d[ID]]
    assert (coh["gap_days"].to_numpy() == d["days_difference"].abs().to_numpy()).all()
    rows.append(gap_row("all adult window persons (first-visit urine test with a blood test within 14 days)",
                        ad["gap_days"]))
    rows.append(gap_row("adult window persons with serum urate measured on the urine-test day",
                        ad.loc[ad["has_urate"], "gap_days"]))
    rows.append(gap_row("development cohort (one record per person)", coh["gap_days"]))
    assert all(r["n_same_day"] == r["n"] for r in rows[1:]), "same-day urate with a nearest blood test on another day"
    assert rows[2]["n"] == len(d)
    for r in rows:
        log.append(f"gap (adults)  {r['population']:<90s} n={r['n']:>7,} same day {r['n_same_day']:>7,} "
                   f"({r['pct_same_day']:.2f}%)  1-7 d {r['n_1_7']:>5,}  8-14 d {r['n_8_14']:>5,}  max {r['max_days']:.0f}")

    # ---- KNHANES flow (sequential; inputs coded by X1)
    steps = []
    k0 = kn2
    steps.append(("0. records in the harmonised KNHANES 2016-2024 file (all ages)", len(k0)))
    k1 = k0[k0["urate"].notna()]
    steps.append(("1. serum urate measured", len(k1)))
    k2 = k1.dropna(subset=cohort.F9)
    steps.append(("2. dipstick fields recorded (9 fields; leukocyte esterase is not measured in KNHANES and is not a "
                  "model input)", len(k2)))
    k3 = k2.dropna(subset=["age", "male"])
    steps.append(("3. age and sex recorded", len(k3)))
    k4 = k3.dropna(subset=["bmi_unified"])
    steps.append(("4. BMI available = submitted external cohort (all ages)", len(k4)))
    k5 = k4[k4["age"] >= cohort.ADULT_AGE]
    steps.append(("5. aged 19 years or older = corrected external cohort", len(k5)))
    assert len(k4) == len(k3.dropna(subset=cohort.FMODEL))
    assert len(k5) == len(ext) and set(k5["id"]) == set(ext["id"]), "KNHANES flow does not end at the external cohort"
    prev = None
    for i, (lab, n) in enumerate(steps):
        rows.append(dict(section="knhanes_flow", step=lab, order=i, n=n,
                         n_excluded=(prev - n) if prev is not None else np.nan))
        log.append(f"knhanes  {lab:<110s} {n:>7,}" + (f"  excluded {prev - n:,}" if prev is not None else ""))
        prev = n
    rows.append(dict(section="knhanes_flow", step="aged < 19 years in the harmonised file", order=90,
                     n=int((k0["age"] < 19).sum()), n_excluded=np.nan))
    rows.append(dict(section="knhanes_flow", step="aged < 19 years in the submitted external cohort", order=91,
                     n=int((k4["age"] < 19).sum()), n_excluded=np.nan))
    rows.append(dict(section="knhanes_flow", step="minimum age in the submitted external cohort", order=92,
                     n=float(k4["age"].min()), n_excluded=np.nan))

    # ---- dipstick level distribution by year (development cohort dev_v2; KNHANES adults under X1)
    dev = d.copy()
    dev["year"] = pd.to_datetime(dev[cohort.DATE], errors="coerce").dt.year
    ex = ext.copy()
    ex["year"] = pd.to_numeric(ex["year"], errors="coerce")
    assert dev["year"].notna().all() and ex["year"].notna().all()
    for label, frame in (("development (corrected cohort)", dev), ("KNHANES adults (corrected external cohort)", ex)):
        for f in LEVEL_FIELDS:
            tab = pd.crosstab(frame["year"], frame[f])
            assert int(tab.values.sum()) == len(frame), (label, f)
            for yr, r in tab.iterrows():
                tot = int(r.sum())
                for lev, cnt in r.items():
                    rows.append(dict(section="dipstick_levels_by_year", cohort=label, feature=f, year=int(yr),
                                     level=float(lev), n=int(cnt), n_year=tot, pct=100 * cnt / tot))
    lv = pd.DataFrame([r for r in rows if r["section"] == "dipstick_levels_by_year"])
    for coh_lab, f, lev in (("development", "ua_ketone", 0.5), ("development", "ua_bilirubin", 1.0),
                            ("KNHANES", "ua_bilirubin", 1.0), ("KNHANES", "ua_urobilinogen", 0.5)):
        s = lv[(lv.cohort.str.startswith(coh_lab)) & (lv.feature == f) & (lv.level == lev)].set_index("year")["pct"]
        log.append(f"levels {coh_lab} {f} level {lev}: " + "; ".join(f"{y} {v:.2f}%" for y, v in s.items()))

    # ---- development flow: sequential, one first-visit urine test per person (records = persons). Steps 1-2 are the
    # adult restriction (age < 19 as its own step, then age > 110); steps 0 and 2-5 are cohort.cohort_steps().
    age = pd.to_numeric(pa["age_at_urine_test"], errors="coerce")
    fs = [("0. persons with a first-visit urine test and a blood test within 14 days (+/-14-day window), all ages",
           pd.Series(True, index=pa.index))]
    fs.append((f"1. aged {cohort.ADULT_AGE} years or older at the urine test (persons younger than {cohort.ADULT_AGE} "
               "years excluded as outside the adult target population)"
               + (f"; {ax['missing']:,} persons with age not determinable also excluded" if ax["missing"] else ""),
               fs[-1][1] & (age >= cohort.ADULT_AGE)))
    fs.append((f"2. plausible age (<= {cohort.AGE_MAX} y) = adult window persons (age {cohort.ADULT_AGE}-"
               f"{cohort.AGE_MAX} y at the urine test)", fs[-1][1] & (age <= cohort.AGE_MAX)))
    fs.append(("3. serum urate measured on the urine-test day (incl. '<1.0')", fs[-1][1] & pa["has_urate"]))
    fs.append(("4. BMI recorded on the urine-test day", fs[-1][1] & pa["has_bmi"]))
    fs.append(("5. all 9 dipstick fields and sex recorded = development cohort (adults, one record per person)",
               fs[-1][1] & pa[cohort.F9].notna().all(axis=1) & pa["male"].notna()))
    cs = cohort.cohort_steps()
    assert [int(fs[k][1].sum()) for k in (0, 2, 3, 4, 5)] == [n for _, n in cs], cs
    assert int((~fs[1][1]).sum()) == ax["minor"] + ax["missing"]
    assert (fs[2][1] == adult).all() and (fs[-1][1] == pa["in_cohort"]).all()
    assert set(pa.loc[fs[-1][1], ID]) == set(d[ID])
    prev = None
    for i, (lab, m) in enumerate(fs):
        n = int(m.sum())
        assert int(pa.loc[m, ID].nunique()) == n                    # one record per person
        rows.append(dict(section="dev_flow", step=lab, order=i, n=n, n_excluded=(prev - n) if prev is not None else np.nan,
                         n_persons=n, n_persons_excluded=(prev - n) if prev is not None else np.nan))
        prev = n
    minor_cc = (age < cohort.ADULT_AGE) & pa["complete"]
    rows.append(dict(section="dev_flow", step=f"reference: persons younger than {cohort.ADULT_AGE} years with complete "
                     "inputs and a serum urate of the urine-test day (excluded by the age restriction)", order=90,
                     n=int(minor_cc.sum()), n_excluded=np.nan, n_persons=int(pa.loc[minor_cc, ID].nunique()),
                     n_persons_excluded=np.nan))
    rows.append(dict(section="dev_flow", step="reference: window rows behind step 0 (a person's first-visit urine test "
                     "appears once for each blood test within 14 days)", order=91, n=len(win2), n_excluded=np.nan,
                     n_persons=int(win2[ID].nunique()), n_persons_excluded=np.nan))
    assert int(win2[ID].nunique()) == len(pa)
    for r in rows:
        if r["section"] == "dev_flow":
            log.append(f"dev flow  {r['step']:<140s} records {r['n']:>7,}  persons {r['n_persons']:>7,}" +
                       ("" if pd.isna(r["n_excluded"]) else
                        f"  excluded {int(r['n_excluded']):,} records / {int(r['n_persons_excluded']):,} persons"))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    with open(LOG, "w") as fh:
        fh.write(__doc__.strip() + "\n\n" + "\n".join(log) + "\n")
    print("\n".join(log))
    print(f"-> {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
