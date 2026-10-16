"""Retrain the uric-acid ensembles for the JMIR Medical Informatics #107515 revision (bestmodel_v5).

Decision of 2026-10-16: the submitted (original, bestmodel_v4) models are retrained on the corrected development
cohort (cohort.dev_v2: adults, nitrite parsed correctly, serum urate and BMI from the urine-test day, one record per
person) with 12 inputs (leukocyte esterase removed, because KNHANES does not measure it) and one person-level 80/20
split shared by the three thresholds (cohort.split_v2).

The recipe is 05_src/run_uric_module_v4.tune_and_build, copied line for line (search spaces, Optuna TPE sampler with
seed 42, 25 trials per learner, 3-fold shuffled StratifiedKFold AUROC objective with seed 42, learner seeds 42,
soft voting, isotonic CalibratedClassifierCV with cv=5). The only change is that each Optuna study is returned, so
that its trials and best parameters can be saved (the original kept them only in memory).

Modes
  reproduce  run the copied recipe on the submitted frame (cohort.dev_submitted + split_submitted, 13 inputs) and
             check that the tuned parameters and the calibrated predictions equal bestmodel_v4 exactly. Nothing is
             saved except train_v5_reproduce_log.txt.
  train      fit the three v5 models on the training part of split_v2 (12 inputs) and save them, with config.json
             (inputs, search space, seeds, data/split/code sha256, library versions), tune_<target>.json and
             optuna_trials_<target>_<learner>.csv, to 03_models/module_uric/bestmodel_v5. No test or external
             performance is computed here: the recipe is frozen before any evaluation.
"""
import os, sys, json, time, hashlib, argparse, platform, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from cohort import (HERE, TARGETS, F13, ID, cached, dev_submitted, split_submitted, labels, load_model)  # noqa: E402
import joblib  # noqa: E402
import sklearn, lightgbm as lgb, xgboost as xgb, catboost as cb, optuna  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.ensemble import VotingClassifier  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25
V5_DIR = "/Users/chan/Desktop/Proj_U/03_models/module_uric/bestmodel_v5"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ recipe (copied from run_uric_module_v4.py L113-181)
SPACES = {
    "LightGBM": "n_estimators 100-300 step 50; max_depth 4-8; learning_rate 0.02-0.10 log; num_leaves 15-63; "
                "min_child_samples 10-100; reg_alpha 1e-3-1.0 log; reg_lambda 1e-3-1.0 log; random_state 42",
    "XGBoost": "n_estimators 100-300 step 50; max_depth 4-8; learning_rate 0.02-0.10 log; subsample 0.7-1.0; "
               "colsample_bytree 0.7-1.0; min_child_weight 1-10; random_state 42; eval_metric logloss",
    "CatBoost": "iterations 100-300 step 50; depth 4-8; learning_rate 0.02-0.10 log; l2_leaf_reg 1e-2-10 log; "
                "random_seed 42",
}


def tune_and_build(X_train, y_train, n_trials=N_TRIALS):
    studies = {}

    def _tune(model_cls, param_fn, name):
        def objective(trial):
            params = param_fn(trial)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            aucs = []
            for tr, va in skf.split(X_train, y_train):
                m = model_cls(**params)
                m.fit(X_train.iloc[tr], y_train.iloc[tr])
                aucs.append(roc_auc_score(y_train.iloc[va], m.predict_proba(X_train.iloc[va])[:, 1]))
            return np.mean(aucs)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log(f'    {name}: CV AUC = {study.best_value:.4f}')
        studies[name] = study
        return study.best_params, study.best_value

    log('    Tuning LightGBM...')
    lgb_p, lgb_a = _tune(
        lgb.LGBMClassifier,
        lambda t: {'n_estimators': t.suggest_int('n_estimators', 100, 300, step=50),
                   'max_depth':     t.suggest_int('max_depth', 4, 8),
                   'learning_rate': t.suggest_float('learning_rate', 0.02, 0.10, log=True),
                   'num_leaves':    t.suggest_int('num_leaves', 15, 63),
                   'min_child_samples': t.suggest_int('min_child_samples', 10, 100),
                   'reg_alpha': t.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                   'reg_lambda': t.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                   'random_state': 42, 'verbose': -1, 'n_jobs': -1},
        'LightGBM')

    log('    Tuning XGBoost...')
    xgb_p, xgb_a = _tune(
        xgb.XGBClassifier,
        lambda t: {'n_estimators': t.suggest_int('n_estimators', 100, 300, step=50),
                   'max_depth':     t.suggest_int('max_depth', 4, 8),
                   'learning_rate': t.suggest_float('learning_rate', 0.02, 0.10, log=True),
                   'subsample':     t.suggest_float('subsample', 0.7, 1.0),
                   'colsample_bytree': t.suggest_float('colsample_bytree', 0.7, 1.0),
                   'min_child_weight': t.suggest_int('min_child_weight', 1, 10),
                   'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
                   'use_label_encoder': False, 'eval_metric': 'logloss'},
        'XGBoost')

    log('    Tuning CatBoost...')
    cb_p, cb_a = _tune(
        cb.CatBoostClassifier,
        lambda t: {'iterations':   t.suggest_int('iterations', 100, 300, step=50),
                   'depth':        t.suggest_int('depth', 4, 8),
                   'learning_rate': t.suggest_float('learning_rate', 0.02, 0.10, log=True),
                   'l2_leaf_reg':  t.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                   'random_seed': 42, 'verbose': False, 'thread_count': -1},
        'CatBoost')

    lgb_p.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    xgb_p.update({'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
                  'use_label_encoder': False, 'eval_metric': 'logloss'})
    cb_p.update({'random_seed': 42, 'verbose': False, 'thread_count': -1})

    voting = VotingClassifier(
        estimators=[('lgb', lgb.LGBMClassifier(**lgb_p)),
                    ('xgb', xgb.XGBClassifier(**xgb_p)),
                    ('cat', cb.CatBoostClassifier(**cb_p))],
        voting='soft', n_jobs=1)

    log('    Fitting calibrated ensemble (isotonic, 5-fold)...')
    cal = CalibratedClassifierCV(voting, method='isotonic', cv=5, n_jobs=1)
    cal.fit(X_train, y_train)
    return cal, {'lgb': lgb_p, 'xgb': xgb_p, 'cat': cb_p,
                 'cv_aucs': {'lgb': lgb_a, 'xgb': xgb_a, 'cat': cb_a}}, studies


# ------------------------------------------------------------------ helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def template_params(cal):
    """Parameters of the unfitted soft-voting template (learner__param), plain values only."""
    out = {}
    for k, v in cal.estimator.get_params(deep=True).items():
        if "__" in k and (v is None or isinstance(v, (int, float, str, bool, np.integer, np.floating))):
            out[k] = v
    return out


def same_value(a, b):
    """Equality that treats NaN == NaN (e.g. XGBoost missing=nan)."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def versions():
    return dict(python=platform.python_version(), sklearn=sklearn.__version__, lightgbm=lgb.__version__,
                xgboost=xgb.__version__, catboost=cb.__version__, optuna=optuna.__version__,
                numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__,
                machine=platform.machine(), system=platform.system())


# ------------------------------------------------------------------ modes
def reproduce():
    """Copied recipe on the submitted frame must give bestmodel_v4 exactly."""
    win, kn = cached()
    ds = dev_submitted(win)
    log(f"reproduce: submitted frame {len(ds):,} rows (13 inputs), as in run_uric_module_v4.prepare_dev")
    ok_all = True
    for t, cfg in TARGETS.items():
        t0 = time.time()
        tr, te, y = split_submitted(ds, cfg["threshold"])
        X_tr = ds[F13].iloc[tr]; y_tr = pd.Series(y[tr], index=X_tr.index)
        X_te = ds[F13].iloc[te]
        log(f"\n[{cfg['short']}] train {len(tr):,} (events {int(y_tr.sum()):,}), test {len(te):,}")
        cal, info, _ = tune_and_build(X_tr, y_tr)
        old = load_model(t, "v4")
        p_new, p_old = template_params(cal), template_params(old)
        diff_params = {k: (p_old.get(k), p_new.get(k)) for k in sorted(set(p_new) | set(p_old))
                       if not same_value(p_old.get(k), p_new.get(k))}
        d_test = float(np.max(np.abs(cal.predict_proba(X_te)[:, 1] - old.predict_proba(X_te)[:, 1])))
        ke = kn.dropna(subset=F13)
        d_ext = float(np.max(np.abs(cal.predict_proba(ke[F13])[:, 1] - old.predict_proba(ke[F13])[:, 1])))
        ok = (not diff_params) and d_test == 0.0 and d_ext == 0.0
        ok_all &= ok
        log(f"  template parameters that differ from bestmodel_v4: {diff_params if diff_params else 'none'}")
        log(f"  max |p_new - p_v4|: internal test {d_test:.3g} ({len(te):,} rows), KNHANES {d_ext:.3g} "
            f"({len(ke):,} rows) -> {'REPRODUCED' if ok else 'NOT REPRODUCED'} ({(time.time() - t0) / 60:.1f} min)")
    log(f"\nreproduce: {'ALL THREE MODELS REPRODUCED EXACTLY' if ok_all else 'MISMATCH - stop and investigate'}")
    with open(os.path.join(HERE, "train_v5_reproduce_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    return ok_all


def train():
    from cohort import cached_v2, dev_v2, split_v2, FMODEL
    t_start = time.time()
    win, kn = cached_v2()
    d, steps = dev_v2(win)
    tr, te = split_v2(d)
    os.makedirs(V5_DIR, exist_ok=True)
    existing = [f for f in os.listdir(V5_DIR) if f.endswith(".joblib")]
    assert not existing, f"{V5_DIR} already holds models {existing}; refusing to overwrite"
    X_tr = d[FMODEL].iloc[tr].reset_index(drop=True)
    log(f"train: dev_v2 {len(d):,} persons; training split {len(tr):,}; test split {len(te):,} (not touched here)")
    log(f"inputs ({len(FMODEL)}): {FMODEL}")
    code = {os.path.basename(p): sha256_file(p) for p in
            (os.path.abspath(__file__), os.path.join(HERE, "cohort.py"),
             "/Users/chan/Desktop/Proj_U/05_src/run_uric_module_v4.py")}
    config = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), purpose="JMIR MI #107515 R1 retrained models (decision 2026-10-16)",
        inputs=FMODEL, n_inputs=len(FMODEL), targets={t: cfg["threshold"] for t, cfg in TARGETS.items()},
        cohort="cohort.dev_v2 (adults, first visit, same-day serum urate and BMI, nitrite recovered, '<1.0' = 0.5, "
               "complete 12 inputs + urate, one record per person)",
        cohort_steps=steps, n_persons=len(d), n_train=len(tr), n_test=len(te),
        split="cohort.split_v2: person-level 80/20, stratified on urate bands <7, 7-<9, 9-<10, >=10; seed 42; "
              "shared by the three thresholds",
        recipe=dict(source="05_src/run_uric_module_v4.tune_and_build (copied)", n_trials_per_learner=N_TRIALS,
                    sampler="optuna TPESampler(seed=42)", objective="mean AUROC, StratifiedKFold(3, shuffle, 42)",
                    search_spaces=SPACES, ensemble="VotingClassifier(soft, equal weights)",
                    calibration="CalibratedClassifierCV(method='isotonic', cv=5) = StratifiedKFold(5) without "
                                "shuffling on the training rows in split order"),
        sha256=dict(train_inputs=sha256_frame(X_tr),
                    train_ids=hashlib.sha256(",".join(map(str, d[ID].iloc[tr])).encode()).hexdigest(),
                    test_ids=hashlib.sha256(",".join(map(str, d[ID].iloc[te])).encode()).hexdigest(),
                    code=code),
        versions=versions())
    with open(os.path.join(V5_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=1, ensure_ascii=False, default=str)
    log("config.json written (recipe frozen) at", config["created"])
    for t, cfg in TARGETS.items():
        t0 = time.time()
        y_tr = pd.Series(labels(d["urate"].iloc[tr], cfg["threshold"]), index=X_tr.index)
        log(f"\n[{cfg['short']}] training events {int(y_tr.sum()):,} of {len(y_tr):,}")
        cal, info, studies = tune_and_build(X_tr, y_tr)
        joblib.dump(cal, os.path.join(V5_DIR, f"{t}_calibrated_voting.joblib"))
        with open(os.path.join(V5_DIR, f"tune_{t}.json"), "w") as f:
            json.dump(dict(target=t, threshold=cfg["threshold"], n_train=len(y_tr), events=int(y_tr.sum()),
                           best_params=dict(lgb=info["lgb"], xgb=info["xgb"], cat=info["cat"]),
                           cv_auroc=info["cv_aucs"],
                           calibration_fold_events=[int(y_tr.iloc[c].sum()) for _, c in
                                                    StratifiedKFold(5).split(X_tr, y_tr)],
                           minutes=round((time.time() - t0) / 60, 2)), f, indent=1, default=float)
        for name, st in studies.items():
            st.trials_dataframe().to_csv(os.path.join(V5_DIR, f"optuna_trials_{t}_{name}.csv"), index=False)
        log(f"  saved {t}_calibrated_voting.joblib ({(time.time() - t0) / 60:.1f} min)")
    log(f"\ntrain: done in {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(V5_DIR, "train_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    with open(os.path.join(HERE, "train_v5_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reproduce", "train"])
    mode = ap.parse_args().mode
    if mode == "reproduce":
        sys.exit(0 if reproduce() else 1)
    train()
