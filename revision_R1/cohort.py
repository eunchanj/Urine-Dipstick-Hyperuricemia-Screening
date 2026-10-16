"""Cohorts for the uric-acid revision (JMIR Medical Informatics #107515, R1).

RETRAINED MODELS (decision of 2026-10-16; bestmodel_v5, see train_v5.py)
  dev_v2()    corrected development cohort: adults (age >= 19 at the urine test), first urine test of each person,
              serum urate and BMI recorded on the urine-test day, nitrite parsed correctly, '<1.0 mg/dL' urate
              counted as 0.5, the 12 model inputs (FMODEL) complete, one record per person (129,535 persons).
  split_v2()  one person-level 80/20 split shared by the three thresholds, stratified on urate bands, seed 42.
  ext_v2()    KNHANES 2016-2024 adults aged >= 19, dipstick inputs coded by the label-matched mapping X1.
  main_cohort(), train_xy(), cal_folds(), persons_v2(): helpers for the analysis scripts.
  build_cohort_v2.py builds the v2 caches and cohort_info_v2.csv. The frozen copy of this module used for training
  is bestmodel_v5/frozen_code/cohort.py (persons_v2 was added after the freeze; nothing used in training changed).

SUBMITTED ANALYSIS (kept only to reproduce the submitted models and for the response letter's comparison)
  dev_submitted(), split_submitted(), ext_submitted(), dev_corrected(), ext_corrected() rebuild the frames of the
  v2.1 submission (run_uric_module_v4.py: row-level +/-14-day window, 13 inputs, per-threshold 80/20 splits, no age
  limit) and the R1 adult evaluation cohort. train_v5.py (mode reproduce) uses them to show that the copied training
  code reproduces bestmodel_v4 exactly; calc_main.py uses them for original_vs_retrained.csv.
  The age column is age_at_urine_test: whole years, floor((urine test date - birth date) / 365.25).
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Release (revision R1): the training code is imported from this repository's src/ folder, where
# train_uric.py is the verbatim 05_src/run_uric_module_v4.py. PROJ_U_SRC overrides the folder; the original
# value was "/Users/chan/Desktop/Proj_U/05_src", where the module is named run_uric_module_v4. The data path,
# the KNHANES path and the folder of the submitted models come from that module (PROJ_U_DATA, PROJ_U_KNHANES,
# PROJ_U_MODELS).
SRC = os.environ.get("PROJ_U_SRC", os.path.normpath(os.path.join(HERE, "..", "src")))
sys.path.insert(0, SRC)
from data_utils import DIPSTICK_FEATURES, DEMO_FEATURES, load_data  # noqa: E402
try:
    from train_uric import add_bmi_unified, DATA_PATH, KNHANES_PATH, MODEL_DIR  # noqa: E402
except ModuleNotFoundError:  # original analysis layout (05_src)
    from run_uric_module_v4 import add_bmi_unified, DATA_PATH, KNHANES_PATH, MODEL_DIR  # noqa: E402

TIME_WINDOW = 14
AGE_MAX = 110
# Adult target population (age >= 19): age >= 19 years at the urine test in international (full) age,
# i.e. Korean age 20 ("20세 이상"), as in the KNHANES external validation cohort. Hospital rows below ADULT_AGE
# (or with missing / implausible age) are excluded from every new analysis (decision of 2026-10-11). The age
# column is age_at_urine_test: whole years, floor((urine test date - birth date) / 365.25).
ADULT_AGE = 19
ID, DATE = "일련번호", "소변검사일자"
F3 = DEMO_FEATURES + ["bmi_unified"]
F10 = list(DIPSTICK_FEATURES)
F12 = list(DIPSTICK_FEATURES) + DEMO_FEATURES
F13 = list(DIPSTICK_FEATURES) + DEMO_FEATURES + ["bmi_unified"]
SEED = 42

# Retrained models (decision of 2026-10-16): 12 inputs, leukocyte esterase removed because KNHANES does not
# measure it. FMODEL is their input order; F9/F11 are the dipstick and dipstick + age + sex sets without it.
# Folder of the retrained model files (bestmodel_v5; not distributed): PROJ_U_MODELS_V5 overrides; the fallback
# is the original value.
MODEL_DIR_V5 = os.environ.get("PROJ_U_MODELS_V5", "/Users/chan/Desktop/Proj_U/03_models/module_uric/bestmodel_v5")
MODEL_SET = os.environ.get("URIC_MODEL_SET", "v5")
LE = "ua_wbc(s)"
F9 = [c for c in DIPSTICK_FEATURES if c != LE]
F11 = F9 + DEMO_FEATURES
FMODEL = F9 + DEMO_FEATURES + ["bmi_unified"]

TARGETS = {
    "uric_hyper_70":  dict(label="Hyperuricemia (>=7.0 mg/dL)", short=">=7.0", threshold=7.0, role="primary"),
    "uric_severe_9":  dict(label="Markedly elevated (>=9.0 mg/dL)", short=">=9.0", threshold=9.0, role="secondary"),
    "uric_severe_10": dict(label="ULT threshold (>=10.0 mg/dL)", short=">=10.0", threshold=10.0, role="exploratory"),
}


def urate_numeric(s, keep_below_detection):
    """Serum urate as float. '<1.0' -> 0.5 when keep_below_detection, else NaN (as submitted)."""
    s = s.astype(str).str.strip()
    below = s.str.startswith("<")
    v = pd.to_numeric(s, errors="coerce")
    if keep_below_detection:
        v = v.where(~below, 0.5)
    return v


def is_adult(age):
    """True where the age at the urine test is a number in [ADULT_AGE, AGE_MAX] (adult target population).
    Missing, non-numeric and implausible (> AGE_MAX) ages are False. Takes a scalar, an array or a
    Series; a Series keeps its index, so the result can index the frame it came from."""
    if isinstance(age, pd.Series):
        a = pd.to_numeric(age, errors="coerce")
        return (a >= ADULT_AGE) & (a <= AGE_MAX)
    a = np.asarray(pd.to_numeric(np.ravel(age), errors="coerce"), dtype=float)
    ok = (a >= ADULT_AGE) & (a <= AGE_MAX)
    return bool(ok[0]) if np.ndim(age) == 0 else ok.reshape(np.shape(age))


def age_exclusion_counts(frame, col="age"):
    """Row counts behind adult_window(frame): rows, adult, minor (< ADULT_AGE), implausible (> AGE_MAX),
    missing (age not determinable). Log these wherever a hospital row frame is restricted to adults."""
    a = pd.to_numeric(frame[col], errors="coerce")
    return dict(rows=len(frame), adult=int(is_adult(a).sum()), minor=int((a < ADULT_AGE).sum()),
                implausible=int((a > AGE_MAX).sum()), missing=int(a.isna().sum()))


def adult_window(win, col="age"):
    """Adult rows (age at the urine test in [ADULT_AGE, AGE_MAX]) of a hospital row frame: the +/-14-day
    window, the raw first-visit file or the processed data file (pass col="age_at_urine_test" when the
    frame has no 'age' column). Rows with missing or implausible age are dropped as well."""
    return win[is_adult(win[col])].copy()


def load_dev_window():
    df = add_bmi_unified(load_data(DATA_PATH))
    win = df[df["days_difference"].abs() <= TIME_WINDOW].copy()
    win["urate_sub"] = urate_numeric(win["uric_acid"], keep_below_detection=False)
    win["urate"] = urate_numeric(win["uric_acid"], keep_below_detection=True)
    return win


def load_knhanes():
    kn = pd.read_csv(KNHANES_PATH, low_memory=False)
    bmi_demo = pd.to_numeric(kn.get("BMI_demo", np.nan), errors="coerce")
    bmi_calc = pd.to_numeric(kn.get("bmi_calc", np.nan), errors="coerce")
    kn["bmi_unified"] = bmi_demo.fillna(bmi_calc)
    kn.loc[(kn["bmi_unified"] < 10) | (kn["bmi_unified"] > 70), "bmi_unified"] = np.nan
    kn["urate"] = pd.to_numeric(kn["HE_Uacid"], errors="coerce")
    return kn


def dev_submitted(win):
    """Row-level complete-case frame exactly as prepare_dev() built it (same row order)."""
    d = win[F13 + ["urate_sub", ID, DATE]].dropna(subset=F13 + ["urate_sub"]).copy()
    d["urate"] = d["urate_sub"]
    return d


def dev_corrected(win, *, adults_only=True):
    """Corrected person-level development cohort -> (frame, n_bad_age, n_repeat).

    adults_only=True (default, adult target population): rows outside [ADULT_AGE, AGE_MAX] are removed BEFORE one record
    per person is chosen, so the result is identical whether or not `win` went through adult_window().
    n_bad_age = rows of `win` with age > AGE_MAX (set to missing, never adult); n_repeat = repeat rows of the
    same person among the rows considered (adult rows by default). adults_only=False reproduces the former
    all-ages cohort and is used only to report the exclusion count in cohort_info.csv."""
    w = win.copy()
    bad = w["age"] > AGE_MAX
    w.loc[bad, "age"] = np.nan
    if adults_only:
        w = w[is_adult(w["age"])].copy()
    w["_ok"] = w[F13 + ["urate"]].notna().all(axis=1)
    w["_date"] = pd.to_datetime(w[DATE], errors="coerce")
    one = (w.sort_values(["_ok", "_date"], ascending=[False, True])
             .drop_duplicates(ID, keep="first").sort_index())
    d = one[one["_ok"]][F13 + ["urate", ID, DATE, "BMI_demo", "bmi_calc"]].copy()
    return d, int(bad.sum()), len(w) - len(one)


def ext_submitted(kn):
    return kn.dropna(subset=F13 + ["urate"]).copy()


def ext_corrected(kn):
    e = ext_submitted(kn)
    return e[e["age"] >= 19].copy()


def labels(urate, threshold):
    return (np.asarray(urate) >= threshold).astype(int)


def split_submitted(d, threshold):
    """Reproduce prepare_dev(): train_test_split(X, y, 0.2, seed 42, stratify=y) on the row frame."""
    from sklearn.model_selection import train_test_split
    y = labels(d["urate"], threshold)
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    return tr, te, y


def load_model(target, model_set=None):
    """Calibrated ensemble for a target. model_set 'v5' (default, URIC_MODEL_SET) = the retrained models of
    2026-10-16 (12 inputs, bestmodel_v5); 'v4' = the submitted (original) models."""
    import joblib
    ms = model_set or MODEL_SET
    return joblib.load(os.path.join({"v4": MODEL_DIR, "v5": MODEL_DIR_V5}[ms], f"{target}_calibrated_voting.joblib"))


if __name__ == "__main__":
    raise SystemExit("cohort.py is a module; run build_cohort_v2.py to build the v2 caches and cohort_info_v2.csv")


# ---------------------------------------------------------------- cache for parallel analyses
CACHE = os.path.join(HERE, "cache")
KEEP = F13 + ["urate", "urate_sub", ID, DATE, "days_difference", "BMI_demo", "bmi_calc", "키", "몸무게",
              "수진일자", "검사일자", "age_at_urine_test"]


def write_cache():
    os.makedirs(CACHE, exist_ok=True)
    win = load_dev_window()
    win[[c for c in KEEP if c in win.columns]].to_parquet(os.path.join(CACHE, "dev_window.parquet"))
    load_knhanes().to_parquet(os.path.join(CACHE, "knhanes.parquet"))


def cached():
    """(win, kn) from the parquet cache: the +/-14-day development window and KNHANES, with urate columns."""
    return (pd.read_parquet(os.path.join(CACHE, "dev_window.parquet")),
            pd.read_parquet(os.path.join(CACHE, "knhanes.parquet")))


# ================================================================ v2: corrected cohort of the retrained models
# Decision of 2026-10-16 (models retrained, bestmodel_v5). Starting from the cached +/-14-day window of first-visit
# urine tests, dev_v2() keeps adults (is_adult, as before) and, per person (all window rows of a person repeat the same
# first-visit urine test):
#   * serum urate measured ON THE URINE-TEST DAY, taken from the validated atlas cache (lab_day_wide.uric_acid:
#     earliest draw of the day, same-time repeats averaged; '<1.0' -> 0.5), because the processed file's
#     days_difference belongs to the nearest lab of any code, not to urate (22 adult cohort records had urate
#     measured 15-296 days away);
#   * nitrite parsed correctly from the raw first-visit text ('Positive'/'POSITIVE' -> 1; checked against the atlas
#     cache dipstick_episode_wide.nitrite_pos);
#   * BMI from the urine-test day: checkup BMI only when the checkup was on that day (every other checkup BMI came
#     from a LATER checkup), otherwise height and weight from the nursing record, which is always of that day; the
#     plausibility rules of run_uric_module_v4.add_bmi_unified are kept;
#   * complete 12 inputs (FMODEL) and urate.
# split_v2() is ONE person-level 80/20 split shared by the three thresholds, stratified on urate bands.
# ext_v2() is KNHANES adults with the 12 inputs coded by the label-matched mapping X1 (knhanes_x1).
# Raw first-visit urine file (hospital data, not shareable): PROJ_U_DATA_ROOT overrides the folder; the fallback is
# the original value.
RAW_FIRST = os.path.join(os.environ.get("PROJ_U_DATA_ROOT", "/Users/chan/Desktop/Proj_U/00_data"),
                         "urine_wide_first_visit_raw.csv")
RAW_DT = "소변검사일시"
# Validated data cache of the hospital data, version 2 (parquet; not shareable): PROJ_U_ATLAS overrides; the
# fallback is the original value.
ATLAS_V2 = os.environ.get("PROJ_U_ATLAS", "/Users/chan/Desktop/Proj_U/00_data/rawdata/parquet_cache/v2")
# Raw KNHANES SAS files hn16_all ... hn24_all (public, KDCA): PROJ_U_KNHANES_RAW overrides the folder.
KN_DIR = os.environ.get("PROJ_U_KNHANES_RAW", "/Users/chan/Desktop/Proj_U/00_data/KNHANES")
URATE_CUTS = [7.0, 9.0, 10.0]
KN_RAW_COL = {"ua_glucose": "he_uglu", "ua_protein": "he_upro", "ua_ketone": "he_uket", "ua_bilirubin": "he_ubil",
              "ua_blood": "he_ubld", "ua_urobilinogen": "he_uro", "ua_nitrite": "he_unitr",
              "ua_pH": "he_uph", "ua_sg": "he_usg"}
KN_UNIFORM = {0: 0, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}          # -, +/-, +, ++, +++, ++++ -> same-label grade
KN_URO_ALIGNED = {0: 0.5, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}     # urobilinogen normal and trace -> development normal 0.5


def recover_nitrite(text):
    """Raw nitrite text -> 1 / 0 / NaN ('Positive', 'POSITIVE' and '+' -> 1; 'Negative', '-' and '0' -> 0, as the
    atlas parser reads them)."""
    s = pd.Series(text).astype("string").str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index)
    out[s.isin(["POSITIVE", "+"]).fillna(False).to_numpy()] = 1.0
    out[s.isin(["NEGATIVE", "-", "0"]).fillna(False).to_numpy()] = 0.0
    return out


def knhanes_raw():
    """KNHANES 2016-2024 raw SAS columns (as 05_src/prepare_knhanes_uric.py and calc_shift_encoding.read_raw)."""
    import pyreadstat
    frames = []
    want = ["id", "sex", "age", "he_uacid"] + list(KN_RAW_COL.values())
    for year in range(2016, 2025):
        path = f"{KN_DIR}/hn{str(year)[-2:]}_all.sas7bdat"
        _, meta = pyreadstat.read_sas7bdat(path, metadataonly=True)
        name = {c.lower(): c for c in meta.column_names}
        df, _ = pyreadstat.read_sas7bdat(path, usecols=[name[c] for c in want], encoding="latin1")
        df.columns = [c.lower() for c in df.columns]
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def knhanes_x1(raw):
    """KNHANES dipstick inputs under the label-matched mapping X1: each code to the development grade with the same
    label, 4+ merged into 3 (bilirubin '+' -> 1), urobilinogen normal and trace -> 0.5 (the development normal code),
    nitrite >= 1 -> 1, pH and specific gravity as recorded; leukocyte esterase is not an input."""
    def cm(s, t):
        return s.map(lambda x: np.nan if pd.isna(x) else t.get(int(x), np.nan)).astype(float)
    out = pd.DataFrame({"id": raw["id"].to_numpy()}, index=raw.index)
    for c in ("ua_glucose", "ua_protein", "ua_ketone", "ua_blood", "ua_bilirubin"):
        out[c] = cm(raw[KN_RAW_COL[c]], KN_UNIFORM)
    out["ua_urobilinogen"] = cm(raw["he_uro"], KN_URO_ALIGNED)
    out["ua_nitrite"] = raw["he_unitr"].map(lambda x: np.nan if pd.isna(x) else (1.0 if int(x) >= 1 else 0.0)).astype(float)
    out["ua_pH"] = pd.to_numeric(raw["he_uph"], errors="coerce")
    out["ua_sg"] = pd.to_numeric(raw["he_usg"], errors="coerce")
    return out


def write_cache_v2(log=print):
    """cache/dev_window_v2.parquet (cached window + ua_nitrite_v2, urate_day, bmi_v2) and cache/knhanes_v2.parquet
    (KNHANES with the X1 inputs). Returns a list of (item, value) checks."""
    import pyarrow.parquet as pq
    win, kn = cached()
    chk = []
    # ---- (1) nitrite from the raw first-visit text
    raw = pd.read_csv(RAW_FIRST, usecols=[ID, RAW_DT, "ua_nitrite"], encoding="utf-8-sig", dtype=str)
    raw[ID] = raw[ID].astype(int)
    assert raw[ID].is_unique, "raw first-visit file has repeated 일련번호"
    raw = raw.set_index(ID)
    ids = win[ID].astype("int64").to_numpy()
    assert np.isin(ids, raw.index).all(), "window person without a raw first-visit record"
    raw_dt = pd.to_datetime(raw.loc[ids, RAW_DT].to_numpy())
    assert (raw_dt.normalize() == pd.to_datetime(win[DATE]).dt.normalize().to_numpy()).all(), \
        "window row is not the person's raw first visit"
    nit = recover_nitrite(raw.loc[ids, "ua_nitrite"].to_numpy()).to_numpy(float)
    proc = win["ua_nitrite"].to_numpy(float)
    have = ~np.isnan(proc)
    assert (proc[have] == nit[have]).all(), "recovered nitrite disagrees with the processed value"
    win["ua_nitrite_v2"] = np.where(have, proc, nit)
    chk += [("window rows", len(win)), ("window persons", win[ID].nunique()),
            ("window rows nitrite processed missing -> recovered 1", int((~have & (nit == 1)).sum())),
            ("window rows nitrite still missing", int(np.isnan(win["ua_nitrite_v2"].to_numpy(float)).sum()))]
    dip = pq.read_table(f"{ATLAS_V2}/derived/dipstick_episode_wide.parquet", columns=["id", "dt", "nitrite_pos"]).to_pandas()
    dip["id"] = dip["id"].astype("int64")
    n_dup = int(dip.duplicated(["id", "dt"]).sum())
    dip = dip.drop_duplicates(["id", "dt"], keep="first")
    j = pd.DataFrame({"id": ids, "dt": raw_dt.astype("datetime64[ms]")}).merge(dip, on=["id", "dt"], how="left",
                                                                                validate="many_to_one")
    cache_nit = j["nitrite_pos"].to_numpy(float)
    v2 = win["ua_nitrite_v2"].to_numpy(float)
    both = ~np.isnan(cache_nit) & ~np.isnan(v2)
    chk += [("atlas dipstick episodes duplicated on (id, dt), dropped", n_dup),
            ("window rows with atlas nitrite", int((~np.isnan(cache_nit)).sum())),
            ("window rows recovered nitrite == atlas nitrite (both present)", int((cache_nit[both] == v2[both]).sum())),
            ("window rows recovered nitrite != atlas nitrite (both present)", int((cache_nit[both] != v2[both]).sum())),
            ("window rows recovered nitrite missing, atlas present", int((np.isnan(v2) & ~np.isnan(cache_nit)).sum()))]
    # ---- (2) serum urate of the urine-test day from the atlas cache
    lab = pq.read_table(f"{ATLAS_V2}/derived/lab_day_wide.parquet",
                        columns=["id", "date", "uric_acid", "uric_acid_cmp", "uric_acid_nrep"]).to_pandas()
    lab = lab[lab["uric_acid"].notna()].copy()
    lab["id"] = lab["id"].astype("int64"); lab["date"] = pd.to_datetime(lab["date"])
    j = pd.DataFrame({"id": ids, "date": pd.to_datetime(win[DATE]).dt.normalize().to_numpy()}).merge(
        lab, on=["id", "date"], how="left", validate="many_to_one")
    cmp_ = j["uric_acid_cmp"].fillna("").to_numpy()
    for k, v in pd.Series(cmp_[~j["uric_acid"].isna().to_numpy()]).value_counts().items():
        chk.append((f"window rows same-day urate comparator '{k}'", int(v)))
    below = cmp_ == "<"
    win["urate_day"] = np.where(below, 0.5, j["uric_acid"].to_numpy(float))
    win["urate_day_below"] = below
    win["urate_day_nrep"] = j["uric_acid_nrep"].to_numpy(float)
    leg = win["urate"].to_numpy(float)
    has_day = ~np.isnan(win["urate_day"].to_numpy(float))
    dd0 = (win["days_difference"] == 0).to_numpy()
    eq = has_day & ~np.isnan(leg) & (np.abs(leg - win["urate_day"].to_numpy(float)) < 1e-9)
    chk += [("window rows with a same-day urate (atlas)", int(has_day.sum())),
            ("window rows days_difference == 0", int(dd0.sum())),
            ("window rows days_difference == 0 without a same-day urate", int((dd0 & ~has_day).sum())),
            ("window rows days_difference != 0 with a same-day urate", int((~dd0 & has_day).sum())),
            ("window rows same-day urate == processed urate", int(eq.sum())),
            ("window rows same-day urate != processed urate (both present)", int((has_day & ~np.isnan(leg) & ~eq).sum()))]
    # ---- (3) BMI of the urine-test day
    ck_day = pd.to_datetime(win["수진일자"].astype(str), errors="coerce", format="mixed").dt.normalize()
    same_ck = (ck_day == pd.to_datetime(win[DATE]).dt.normalize()).to_numpy()
    demo = pd.to_numeric(win["BMI_demo"], errors="coerce").where(same_ck)
    bmi = demo.fillna(win["bmi_calc"])
    bmi[(bmi < 10) | (bmi > 70)] = np.nan
    win["bmi_v2"] = bmi
    win["bmi_v2_source"] = np.select([demo.notna().to_numpy(), win["bmi_calc"].notna().to_numpy()],
                                     ["checkup", "nursing"], "none")
    later = pd.to_numeric(win["BMI_demo"], errors="coerce").notna().to_numpy() & ~same_ck
    gap = (ck_day - pd.to_datetime(win[DATE]).dt.normalize()).dt.days.to_numpy()
    chk += [("window rows checkup BMI from another day", int(later.sum())),
            ("window rows checkup BMI from an EARLIER day", int((later & (gap < 0)).sum())),
            ("window rows BMI of the urine-test day", int(win["bmi_v2"].notna().sum()))]
    os.makedirs(CACHE, exist_ok=True)
    win.to_parquet(os.path.join(CACHE, "dev_window_v2.parquet"))
    # ---- (4) KNHANES inputs under X1
    rk = knhanes_raw()
    x1 = knhanes_x1(rk)
    k2 = kn.drop(columns=[c for c in F10 if c in kn.columns]).merge(x1, on="id", how="left", validate="one_to_one")
    assert len(k2) == len(kn) and np.array_equal(k2["id"].to_numpy(), kn["id"].to_numpy())
    k2.to_parquet(os.path.join(CACHE, "knhanes_v2.parquet"))
    chk.append(("KNHANES records (2016-2024)", len(k2)))
    for item, value in chk:
        log(f"  {item}: {value:,}" if isinstance(value, (int, np.integer)) else f"  {item}: {value}")
    return chk


def cached_v2():
    """(win_v2, kn_v2) from cache/dev_window_v2.parquet and cache/knhanes_v2.parquet (write_cache_v2)."""
    return (pd.read_parquet(os.path.join(CACHE, "dev_window_v2.parquet")),
            pd.read_parquet(os.path.join(CACHE, "knhanes_v2.parquet")))


def dev_v2(win):
    """Corrected person-level development cohort of the retrained models -> (frame, steps).
    win must come from cached_v2(). steps = [(description, persons)] in the order the exclusions are applied."""
    w = win.copy()
    w.loc[w["age"] > AGE_MAX, "age"] = np.nan
    w["ua_nitrite"] = w["ua_nitrite_v2"]
    w["bmi_unified"] = w["bmi_v2"]
    w["urate"] = w["urate_day"]
    one = w.drop_duplicates(ID, keep="first")
    steps = [("persons with a first-visit urine test and a blood test within 14 days", len(one))]
    one = one[is_adult(one["age"])]
    steps.append(("aged 19 years or older at the urine test", len(one)))
    one = one[one["urate"].notna()]
    steps.append(("serum urate measured on the urine-test day", len(one)))
    one = one[one["bmi_unified"].notna()]
    steps.append(("BMI recorded on the urine-test day", len(one)))
    one = one[one[FMODEL].notna().all(axis=1)]
    steps.append(("all 9 dipstick fields and sex recorded (analytic cohort)", len(one)))
    cols = FMODEL + ["urate", "urate_day_below", "urate_day_nrep", ID, DATE, "BMI_demo", "bmi_calc", "bmi_v2_source",
                     "days_difference"]
    return one[cols].copy(), steps


def split_v2(d, seed=SEED):
    """One person-level 80/20 split of dev_v2 shared by the three thresholds, stratified on urate bands
    (<7, 7-<9, 9-<10, >=10 mg/dL) -> (train positions, test positions)."""
    from sklearn.model_selection import train_test_split
    band = np.digitize(d["urate"].to_numpy(float), URATE_CUTS)
    tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=seed, stratify=band)
    return tr, te


def ext_v2(kn2):
    """KNHANES adults (>= 19 years) with the 12 inputs (X1 coding) and serum urate."""
    e = kn2.dropna(subset=FMODEL + ["urate"]).copy()
    return e[e["age"] >= ADULT_AGE].copy()


# ---------------------------------------------------------------- adapter for the analysis scripts (retrained models)
_MAIN = {}


def main_cohort():
    """(d, tr, te, ext) of the retrained models, cached per process: d = dev_v2 frame (FMODEL, urate, ids), tr/te =
    split_v2 positions in d (the same internal test persons for every threshold), ext = ext_v2 with a fresh index.
    The internal test frame is d.iloc[te]; preds_internal_clean.csv rows follow that order for every target and
    preds_external_corrected.csv rows follow ext."""
    if not _MAIN:
        win2, kn2 = cached_v2()
        d, steps = dev_v2(win2)
        tr, te = split_v2(d)
        _MAIN.update(d=d, tr=tr, te=te, ext=ext_v2(kn2).reset_index(drop=True), steps=steps)
    return _MAIN["d"], _MAIN["tr"], _MAIN["te"], _MAIN["ext"]


def cohort_steps():
    """Exclusion steps of dev_v2 [(description, persons)] (Figure 1)."""
    main_cohort()
    return list(_MAIN["steps"])


def train_xy(threshold):
    """Training inputs and labels exactly as train_v5.py passed them to the recipe (split order, fresh index)."""
    d, tr, _, _ = main_cohort()
    X = d[FMODEL].iloc[tr].reset_index(drop=True)
    return X, labels(d["urate"].iloc[tr], threshold)


def cal_folds(model, X_tr, y_tr):
    """The 5 isotonic calibration folds of a fitted CalibratedClassifierCV(cv=5) ensemble: StratifiedKFold(5) without
    shuffling on the training rows in the order they were passed (train_xy), checked against the model's isotonic
    calibrators (an isotonic fit on each fold's calibration rows must reproduce the saved thresholds)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.isotonic import IsotonicRegression
    y_tr = np.asarray(y_tr)
    folds = list(StratifiedKFold(n_splits=5).split(X_tr, y_tr))
    assert len(folds) == len(model.calibrated_classifiers_)
    for k, (_, cal_idx) in enumerate(folds):
        cc = model.calibrated_classifiers_[k]
        iso = IsotonicRegression(out_of_bounds="clip").fit(cc.estimator.predict_proba(X_tr.iloc[cal_idx])[:, 1],
                                                           y_tr[cal_idx])
        assert np.allclose(iso.y_thresholds_, cc.calibrators[0].y_thresholds_), f"calibration fold {k} does not match"
    return folds


def persons_v2():
    """One row per person of the cached v2 window (every window person, BEFORE the dev_v2 exclusions) with the v2
    fields: FMODEL inputs (nitrite recovered, BMI of the urine-test day), urate (urine-test day, '<1.0' = 0.5), the
    observed leukocyte esterase (LE, not a model input), and flags adult, has_urate, has_bmi, complete (12 inputs +
    urate) and in_cohort (exactly the dev_v2 persons). For missing-data and verification-bias analyses."""
    win2, _ = cached_v2()
    w = win2.copy()
    w.loc[w["age"] > AGE_MAX, "age"] = np.nan
    w["ua_nitrite"] = w["ua_nitrite_v2"]
    w["bmi_unified"] = w["bmi_v2"]
    w["urate"] = w["urate_day"]
    one = w.drop_duplicates(ID, keep="first").copy()
    one["adult"] = is_adult(one["age"])
    one["has_urate"] = one["urate"].notna()
    one["has_bmi"] = one["bmi_unified"].notna()
    one["complete"] = one[FMODEL + ["urate"]].notna().all(axis=1)
    one["in_cohort"] = one["adult"] & one["complete"]
    d = main_cohort()[0]
    assert set(one.loc[one["in_cohort"], ID]) == set(d[ID]), "persons_v2 in_cohort differs from dev_v2"
    return one
