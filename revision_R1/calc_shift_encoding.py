"""Input encoding and KNHANES dipstick mapping for the retrained models (R1: AW #10, #13, #17; BZ #3, #4).

Retrained models (bestmodel_v5, decision of 2026-10-16): 12 inputs (cohort.FMODEL = 9 dipstick fields + age, sex, BMI;
leukocyte esterase is not an input because KNHANES does not measure it), trained on the corrected development cohort
dev_v2 (cohort.main_cohort: adults aged >= 19, first-visit urine test, serum urate and BMI of the urine-test day, nitrite
parsed correctly, one record per person). External cohort: KNHANES adults aged >= 19 with the 12 inputs and serum urate
(cohort.main_cohort ext), dipstick inputs coded by the label-matched mapping X1 (cohort.knhanes_x1), as fed to the models.

(a) encoding.csv
      codebook   distinct values (counts, %) of every model input in the development cohort (dev_v2, all persons) and in
                 the KNHANES adults (X1); age and BMI are shown in bands because they are continuous (age band [0, 19)
                 kept with n = 0).
      summary    per input: distinct levels, mean/SD, range, PSI (common.psi, development = expected), an alternative PSI
                 (open outer bins for age/BMI; development grid for SG), positivity (> 0) and above-normal rates for the
                 dipstick fields (urobilinogen above normal = > 0.5 on the model scale in both cohorts, because X1 codes
                 the KNHANES normal and trace results 0.5 like the development normal).
      raw_codes  KNHANES raw codes (0 -, 1 +/-, 2 +, 3 ++, 4 +++, 5 ++++) among the KNHANES adults and the value each
                 mapping assigns.
(b) mapping_sensitivity.csv
      KNHANES dipstick inputs rebuilt from the raw hn16-hn24 SAS files (cohort.knhanes_raw) under alternative mappings,
      linked to the KNHANES adults by the KNHANES 'id', and scored with the retrained models (leukocyte esterase is not
      an input under any mapping):
        X1 primary, label-matched: GLU/PRO/KET/BLD/BIL 0, 0.5, 1, 2, 3, 3 (bilirubin + -> 1); urobilinogen normal and
           trace -> 0.5, + -> 1, ++ -> 2, >= +++ -> 3; verified to equal cohort.knhanes_x1 and the cached model inputs on
           every record, and its predictions to reproduce preds_external_corrected.csv exactly
        M1 the submission's coding as used (verified against the mapping functions of 05_src/prepare_knhanes_uric.py):
           bilirubin 0, 1, 2, 3, 3, 3; urobilinogen 0, 0.2, 0.5, 1, 2, 3
        M2 uniform printed map (0, 0.5, 1, 2, 3, 3) for bilirubin and urobilinogen as well
        M3 as M1 but the urobilinogen normal code aligned to the development code (raw 0 -> 0.5)
        X2 (extra) as X1 but specific gravity snapped to the development grid (0.005 steps, <= 1.030)
      in_spec is True for X1 and for the three codings of the submission (M1 as used, M2 as printed, M3), False for the
      extra check X2. Deltas (changed inputs and predictions; AUROC difference with paired bootstrap CI and DeLong p) are
      against X1.
Removed on 2026-10-16: le_sensitivity.csv (leukocyte-esterase substitution in the original 13-input models and a refit
without it) and the refit_noLE/ models; leukocyte esterase is no longer a model input and the original models are no
longer reported.
Bootstrap: 1000 resamples, percentile CI, seed 42 (common.auroc_ci / common.paired_boot).
Run from the working folder:  /opt/anaconda3/bin/python calc_shift_encoding.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
import ast, functools, time, warnings  # noqa: E402
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd  # noqa: E402

# cohort and common are imported BEFORE lightgbm/xgboost/catboost (unpickling the saved XGBoost learners can otherwise
# deadlock); import order does not affect any result.
from cohort import (HERE, TARGETS, FMODEL, F9, main_cohort, knhanes_raw, knhanes_x1, KN_UNIFORM,  # noqa: E402
                    KN_URO_ALIGNED, KN_RAW_COL, labels, load_model)
from common import auroc_ci, paired_boot, delong_paired, calib_stats, psi  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

THREADS = 4
# The submission's KNHANES coding. Release (revision R1): the preparation script is read (parsed, not run) from this
# repository's data/knhanes_processing.py, the verbatim 05_src/prepare_knhanes_uric.py with configurable paths.
# PROJ_U_KNHANES_PREP overrides the file; the original value was
# "/Users/chan/Desktop/Proj_U/05_src/prepare_knhanes_uric.py".
SUBMITTED_PREP = os.environ.get("PROJ_U_KNHANES_PREP",
                                os.path.normpath(os.path.join(HERE, "..", "data", "knhanes_processing.py")))
T0 = time.time()
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ KNHANES raw codes and mappings
RAW_LABEL = {0: "-", 1: "+/-", 2: "+", 3: "++", 4: "+++", 5: "++++"}
UNIFORM = dict(KN_UNIFORM)                                 # {0: 0, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}; GLU/PRO/KET/BLD always
BIL_USED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}           # prepare_knhanes_uric.map_ua_bilirubin (submission)
URO_USED = {0: 0, 1: 0.2, 2: 0.5, 3: 1, 4: 2, 5: 3}       # prepare_knhanes_uric.map_ua_urobilinogen (submission)
URO_M3 = {**URO_USED, 0: 0.5}
URO_ALIGNED = dict(KN_URO_ALIGNED)                         # {0: 0.5, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}
REF = "X1"
# Dict order = column order of the map_* columns of encoding.csv (raw_codes), as before.
MAPPINGS = {
    "M1": dict(in_spec=True, bil=BIL_USED, uro=URO_USED, sg="raw",
               desc="submission's coding as used: GLU/PRO/KET/BLD 0,0.5,1,2,3,3; BIL 0,1,2,3,3,3; URO 0,0.2,0.5,1,2,3; "
                    "NIT>=1->1"),
    "M2": dict(in_spec=True, bil=UNIFORM, uro=UNIFORM, sg="raw",
               desc="uniform printed map 0,0.5,1,2,3,3 for every semi-quantitative field incl. BIL and URO"),
    "M3": dict(in_spec=True, bil=BIL_USED, uro=URO_M3, sg="raw",
               desc="as M1 but URO normal (raw 0) -> 0.5, the development normal code"),
    "X1": dict(in_spec=True, bil=UNIFORM, uro=URO_ALIGNED, sg="raw",
               desc="primary (label-matched): GLU/PRO/KET/BLD/BIL 0,0.5,1,2,3,3 (BIL + -> 1); URO normal/trace -> 0.5, "
                    "+ -> 1, ++ -> 2, >=+++ -> 3; NIT>=1->1"),
    "X2": dict(in_spec=False, bil=UNIFORM, uro=URO_ALIGNED, sg="grid",
               desc="extra: as X1 but SG snapped to the development grid (nearest 0.005, capped at 1.030)"),
}
ORDER = [REF] + [k for k in MAPPINGS if k != REF]       # mapping_sensitivity.csv rows: reference first


def code_map(s, table):
    return s.map(lambda x: np.nan if pd.isna(x) else table.get(int(x), np.nan)).astype(float)


def snap_sg(s):
    return np.minimum(np.round(np.asarray(s, float) / 0.005) * 0.005, 1.030).round(3)


def build_inputs(raw, key):
    """The 9 dipstick inputs (F9) under mapping `key`; leukocyte esterase is not an input."""
    m = MAPPINGS[key]; out = pd.DataFrame(index=raw.index)
    for c in ("ua_glucose", "ua_protein", "ua_ketone", "ua_blood"):
        out[c] = code_map(raw[KN_RAW_COL[c]], UNIFORM)
    out["ua_bilirubin"] = code_map(raw["he_ubil"], m["bil"])
    out["ua_urobilinogen"] = code_map(raw["he_uro"], m["uro"])
    out["ua_nitrite"] = raw["he_unitr"].map(lambda x: np.nan if pd.isna(x) else (1 if int(x) >= 1 else 0)).astype(float)
    out["ua_pH"] = pd.to_numeric(raw["he_uph"], errors="coerce")
    sg = pd.to_numeric(raw["he_usg"], errors="coerce")
    out["ua_sg"] = snap_sg(sg) if m["sg"] == "grid" else sg
    return out[F9]


def submitted_coding():
    """The submission's KNHANES dipstick coding, read from 05_src/prepare_knhanes_uric.py without running it:
    ({map function: dict literal it applies}, {model input: (raw KNHANES column, map function)})."""
    tree = ast.parse(open(SUBMITTED_PREP, encoding="utf-8").read())
    maps, applied = {}, {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("map_"):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get"
                        and isinstance(sub.func.value, ast.Dict)):
                    maps[node.name] = ast.literal_eval(sub.func.value)
        # sub['ua_x'] = sub['he_y'].apply(map_z)
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "apply" and isinstance(node.value.func.value, ast.Subscript)
                and node.value.args and isinstance(node.value.args[0], ast.Name)):
            applied[ast.literal_eval(node.targets[0].slice)] = (ast.literal_eval(node.value.func.value.slice),
                                                               node.value.args[0].id)
    return maps, applied


# ------------------------------------------------------------------ model helpers
def cap_predict_threads(cal):
    """Cap prediction threads of an in-memory calibrated ensemble (predictions are unaffected by thread count)."""
    for cc in cal.calibrated_classifiers_:
        for est in cc.estimator.estimators_:
            if isinstance(est, (LGBMClassifier, XGBClassifier)):
                est.set_params(n_jobs=THREADS)
            elif isinstance(est, CatBoostClassifier):
                est.predict_proba = functools.partial(CatBoostClassifier.predict_proba, est, thread_count=THREADS)
    return cal


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate(y, p):
    y = np.asarray(y); p = np.asarray(p)
    auc, lo, hi = auroc_ci(y, p)
    a, s, oe = calib_stats(y, p)
    return dict(n=len(y), positives=int(y.sum()), auroc=auc, auroc_lo=lo, auroc_hi=hi, cal_intercept=a,
                cal_slope=s, oe_ratio=oe, brier=brier_score_loss(y, p), mean_pred=float(p.mean()), obs_rate=float(y.mean()))


def compare(y, p, p_ref):
    """AUROC difference (p minus reference) on the same records: paired bootstrap CI and DeLong p."""
    y = np.asarray(y); p = np.asarray(p); p_ref = np.asarray(p_ref)
    diff = np.abs(p - p_ref)
    out = dict(n_pred_changed=int((diff > 1e-12).sum()), mean_abs_dp=float(diff.mean()), max_abs_dp=float(diff.max()))
    if out["n_pred_changed"] == 0:
        return out | dict(delta_auroc=0.0, delta_lo=np.nan, delta_hi=np.nan, delong_p=np.nan)
    d, pv = delong_paired(y, p, p_ref); lo, hi = paired_boot(y, p, p_ref)
    return out | dict(delta_auroc=d, delta_lo=lo, delta_hi=hi, delong_p=pv)


# ------------------------------------------------------------------ data
dv, _, _, ext = main_cohort()                  # dev_v2 (all persons) and the KNHANES adults (X1), as the models see them
log(f"development cohort dev_v2 {len(dv):,} persons (age {dv['age'].min():g}-{dv['age'].max():g}) | KNHANES adults "
    f"{len(ext):,} (age {ext['age'].min():g}-{ext['age'].max():g}) | inputs ({len(FMODEL)}): {', '.join(FMODEL)}")

raw = knhanes_raw()
assert raw["id"].is_unique, "duplicated KNHANES ids in the raw SAS files"
log(f"raw KNHANES rows {len(raw):,}; duplicated ids {int(raw['id'].duplicated().sum())}")
raw_ext = ext[["id"]].merge(raw, on="id", how="left", validate="one_to_one")
assert len(raw_ext) == len(ext) and raw_ext[list(KN_RAW_COL.values())].notna().all().all()
assert np.array_equal(raw_ext["age"].to_numpy(float), ext["age"].to_numpy(float))
assert np.array_equal((raw_ext["sex"] == 1).astype(int).to_numpy(), ext["male"].astype(int).to_numpy())
assert np.array_equal(raw_ext["he_uacid"].to_numpy(float), ext["urate"].to_numpy(float))
log(f"every KNHANES adult links to one raw record by id; age, sex and serum urate agree on {len(ext):,} records")

# --- M1 is the submission's coding: mapping functions of 05_src/prepare_knhanes_uric.py, field by field
sub_maps, sub_applied = submitted_coding()
for fn_name, table in (("map_urine_standard", UNIFORM), ("map_ua_bilirubin", BIL_USED), ("map_ua_urobilinogen", URO_USED)):
    assert sub_maps[fn_name] == table, f"{fn_name} differs from the M1 table"
want = {c: (KN_RAW_COL[c], "map_urine_standard") for c in ("ua_glucose", "ua_protein", "ua_ketone", "ua_blood")}
want |= {"ua_bilirubin": ("he_ubil", "map_ua_bilirubin"), "ua_urobilinogen": ("he_uro", "map_ua_urobilinogen"),
         "ua_nitrite": ("he_unitr", "map_ua_nitrite")}
assert all(sub_applied.get(c) == v for c, v in want.items()), f"submitted field coding differs: {sub_applied}"
log("M1 equals the submission's KNHANES coding (05_src/prepare_knhanes_uric.py: map functions and the raw column each "
    "field is built from, read from the source)")

inputs = {k: pd.concat([raw[["id"]], build_inputs(raw, k)], axis=1) for k in MAPPINGS}
# --- X1 rebuilt here equals cohort.knhanes_x1 on every raw record (two implementations of the primary coding)
x1_api = knhanes_x1(raw)
for c in F9:
    assert np.array_equal(inputs["X1"][c].to_numpy(float), x1_api[c].to_numpy(float), equal_nan=True), c
log(f"X1 rebuilt from the raw SAS files equals cohort.knhanes_x1 for all 9 dipstick inputs on {len(raw):,} records")

ext_in = {}
for k, frame in inputs.items():
    e = ext[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
    assert (e["id"].values == ext["id"].values).all() and e[F9].notna().all().all()
    X = ext[FMODEL].copy()
    X[F9] = e[F9].values
    ext_in[k] = X
assert np.array_equal(ext_in[REF][FMODEL].to_numpy(float), ext[FMODEL].to_numpy(float))
log(f"X1 reproduces the cached model inputs of all {len(ext):,} KNHANES adults exactly")

# ================================================================== (a) encoding.csv
log("\n(a) encoding (development dev_v2 vs KNHANES adults, X1 coding)")
AGE_BANDS = [0, 19, 30, 40, 50, 60, 70, 80, np.inf]   # band [0, 19) kept with n = 0 (adults only; make_tables reads it)
BMI_BANDS = [0, 18.5, 23, 25, 30, 35, np.inf]
POS_FIELDS = ["ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen", "ua_nitrite"]
n_nit = int((dv["ua_nitrite"] > 0).sum())
NITRITE_NOTE = (f"parsed from the raw result text in the corrected cohort ('Positive'/'POSITIVE' -> 1; the submission's "
                f"processed file had mapped these results to missing): development {n_nit:,} of {len(dv):,} positive "
                f"({100 * n_nit / len(dv):.2f}%)")
UROB_NOTE = ("normal coded 0.5 in both cohorts (development normal; KNHANES normal and trace under X1), so >0 is ~100%; "
             "above-normal = >0.5 on the model scale (development 1+ or more; KNHANES raw code >=2, + or more)")
BIL_NOTE = "KNHANES + (raw code 2) coded 1 under X1, the development 1+ code (the submission's coding M1 used 2)"
rows = []
for c in FMODEL:
    a, b = dv[c].to_numpy(float), ext[c].to_numpy(float)
    if c in ("age", "bmi_unified"):
        edges = AGE_BANDS if c == "age" else BMI_BANDS
        ca = pd.cut(a, edges, right=False); cb = pd.cut(b, edges, right=False)
        for iv in ca.categories:
            na, nb = int((ca == iv).sum()), int((cb == iv).sum())
            rows.append(dict(section="codebook", feature=c, value=f"[{iv.left:g}, {iv.right:g})", value_num=np.nan,
                             n_dev=na, pct_dev=100 * na / len(a), n_ext=nb, pct_ext=100 * nb / len(b)))
    else:
        for v in np.union1d(np.unique(a), np.unique(b)):
            na, nb = int((a == v).sum()), int((b == v).sum())
            rows.append(dict(section="codebook", feature=c, value=format(v, "g"), value_num=v,
                             n_dev=na, pct_dev=100 * na / len(a), n_ext=nb, pct_ext=100 * nb / len(b)))


def psi_alt(c, a, b):
    """Alternative PSI where common.psi is fragile: open outer bins (age, BMI); development grid (SG)."""
    if c == "ua_sg":
        return psi(snap_sg(a), snap_sg(b)), "SG snapped to the 0.005 development grid (cap 1.030), categorical"
    if c in ("age", "bmi_unified"):
        inner = np.unique(np.quantile(a, np.linspace(0, 1, 11)))[1:-1]
        ia, ib = np.searchsorted(inner, a, side="right"), np.searchsorted(inner, b, side="right")
        k = np.arange(len(inner) + 1)
        pa = np.clip([(ia == j).mean() for j in k], 1e-6, None); pb = np.clip([(ib == j).mean() for j in k], 1e-6, None)
        return float(np.sum((pb - pa) * np.log(pb / pa))), "development deciles with open outer bins"
    return np.nan, ""


sg_out = int((ext["ua_sg"] > dv["ua_sg"].max()).sum())
for c in FMODEL:
    a, b = dv[c].to_numpy(float), ext[c].to_numpy(float)
    r = dict(section="summary", feature=c, n_dev=len(a), n_ext=len(b), n_levels_dev=len(np.unique(a)),
             n_levels_ext=len(np.unique(b)), mean_dev=a.mean(), sd_dev=a.std(ddof=1), mean_ext=b.mean(),
             sd_ext=b.std(ddof=1), min_dev=a.min(), max_dev=a.max(), min_ext=b.min(), max_ext=b.max(),
             psi=psi(a, b))
    r["psi_alt"], r["psi_alt_method"] = psi_alt(c, a, b)
    note = []
    if c in POS_FIELDS:
        r["pos_dev"] = float((a > 0).mean())
        r["pos_ext"] = float((b > 0).mean())
        if c == "ua_urobilinogen":   # normal is 0.5 on the model scale in both cohorts (X1)
            r["abn_dev"] = float((a > 0.5).mean()); r["abn_ext"] = float((b > 0.5).mean())
            assert r["abn_ext"] == float((raw_ext["he_uro"] >= 2).mean())
            note.append(UROB_NOTE)
        else:
            r["abn_dev"], r["abn_ext"] = r["pos_dev"], r["pos_ext"]
    if c == "ua_bilirubin":
        note.append(BIL_NOTE)
    if c == "ua_sg":
        note.append(f"development recorded in 0.005 steps top-coded at 1.030; KNHANES in 0.001 steps up to "
                    f"{ext['ua_sg'].max():g} ({sg_out:,} ext values >1.030 fall outside the development range and are "
                    "dropped by common.psi's quantile branch)")
    if c == "ua_nitrite":
        note.append(NITRITE_NOTE)
    r["note"] = "; ".join(note)
    rows.append(r)

for c in ["ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen", "ua_nitrite"]:
    s = raw_ext[KN_RAW_COL[c]]
    for code in range(6):
        n = int((s == code).sum())
        if c == "ua_nitrite" and code > 1:
            continue
        r = dict(section="raw_codes", feature=c, value=f"{code} ({'neg' if code == 0 else 'pos'})" if c == "ua_nitrite"
                 else f"{code} ({RAW_LABEL[code]})", value_num=code, n_ext=n, pct_ext=100 * n / len(s))
        for k, m in MAPPINGS.items():
            if c == "ua_nitrite":
                r[f"map_{k}"] = float(code >= 1)
            elif c == "ua_bilirubin":
                r[f"map_{k}"] = m["bil"][code]
            elif c == "ua_urobilinogen":
                r[f"map_{k}"] = m["uro"][code]
            else:
                r[f"map_{k}"] = UNIFORM[code]
        rows.append(r)

enc = pd.DataFrame(rows)
enc.to_csv(os.path.join(HERE, "encoding.csv"), index=False, encoding="utf-8")
summ = enc[enc.section == "summary"].set_index("feature")
for c in FMODEL:
    s = summ.loc[c]
    pos = f" pos dev {100 * s.pos_dev:.2f}% ext {100 * s.pos_ext:.2f}%" if c in POS_FIELDS else ""
    log(f"  {c:16s} PSI {s.psi:8.4f}  alt {s.psi_alt:8.4f}{pos}")
log(f"  urobilinogen above-normal (>0.5): dev {100 * summ.loc['ua_urobilinogen', 'abn_dev']:.3f}% "
    f"ext {100 * summ.loc['ua_urobilinogen', 'abn_ext']:.3f}%; KNHANES SG > 1.030: {sg_out:,}")
log(f"  ua_nitrite: {NITRITE_NOTE}")
log(f"  -> encoding.csv ({len(enc)} rows: " + ", ".join(f"{k} {v}" for k, v in enc.section.value_counts().items()) + ")")

# ================================================================== (b) mapping_sensitivity.csv
log("\n(b) mapping sensitivity (retrained models, KNHANES adults; deltas against X1)")
# round_trip: pandas' default float parser can be off by ~1e-16, which would break the exact reproduction check
pext = pd.read_csv(os.path.join(HERE, "preds_external_corrected.csv"), float_precision="round_trip")
mperf = pd.read_csv(os.path.join(HERE, "main_performance.csv"))
models = {}
for t in TARGETS:
    models[t] = cap_predict_threads(load_model(t))
    assert list(models[t].feature_names_in_) == FMODEL, "retrained model inputs differ from FMODEL"
psi_map = {k: {c: psi(dv[c].to_numpy(float), X[c].to_numpy(float)) for c in ("ua_bilirubin", "ua_urobilinogen", "ua_sg")}
           for k, X in ext_in.items()}
rows = []
for t, cfg in TARGETS.items():
    y = labels(ext["urate"], cfg["threshold"])
    p = {k: proba(models[t], X[FMODEL]) for k, X in ext_in.items()}
    ref = pext[pext.target == t]
    assert np.array_equal(ref["id"].to_numpy(), ext["id"].to_numpy()) and np.array_equal(ref["y"].to_numpy(), y)
    assert np.array_equal(ref["p"].to_numpy(), p[REF]), f"X1 predictions do not reproduce preds_external_corrected.csv for {t}"
    for k in ORDER:
        m = MAPPINGS[k]
        changed = int((ext_in[k][F9].to_numpy(float) != ext_in[REF][F9].to_numpy(float)).any(axis=1).sum())
        r = dict(target=cfg["short"], mapping=k, in_spec=m["in_spec"], description=m["desc"],
                 n_participants_inputs_changed=changed) | evaluate(y, p[k])
        r |= compare(y, p[k], p[REF]) if k != REF else dict(n_pred_changed=0, delta_auroc=0.0)
        r |= {f"psi_{c.replace('ua_', '')}": v for c, v in psi_map[k].items()}
        rows.append(r)
        log(f"  {cfg['short']:6s} {k}: AUROC {r['auroc']:.4f} ({r['auroc_lo']:.4f}-{r['auroc_hi']:.4f}) "
            f"dAUROC {r['delta_auroc']:+.4f} int {r['cal_intercept']:+.3f} slope {r['cal_slope']:.3f} "
            f"O:E {r['oe_ratio']:.3f} changed inputs {changed:,} preds {r['n_pred_changed']:,}")
    mp = mperf[(mperf.target == cfg["short"]) & (mperf.cohort == "KNHANES adults")].iloc[0]
    rx = rows[-len(ORDER)]
    assert rx["mapping"] == REF and rx["n"] == int(mp["n"]) == len(ext) and round(rx["auroc"], 4) == round(mp["auroc"], 4)
mapping = pd.DataFrame(rows)
mapping.to_csv(os.path.join(HERE, "mapping_sensitivity.csv"), index=False, encoding="utf-8")
log(f"  X1 predictions reproduce preds_external_corrected.csv exactly for all targets (n = {len(ext):,}; AUROC equals "
    f"main_performance.csv)")
log(f"  -> mapping_sensitivity.csv ({len(mapping)} rows)")
log(f"\nDone in {(time.time() - T0) / 60:.1f} min")
with open(os.path.join(HERE, "calc_shift_encoding_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
