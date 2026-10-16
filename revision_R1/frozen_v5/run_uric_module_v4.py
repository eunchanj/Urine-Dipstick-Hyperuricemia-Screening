"""
Uric Acid / Hyperuricemia Module — v3 (13F, unified threshold updated to ≥7.0)
─────────────────────────────────────────────────────────────────────
  - ±14-day urine-blood matching
  - 13 base features (10 urine dipstick + age + sex + BMI)
  - Bayesian hyperparameter tuning (Optuna, 25 trials × 3 learners)
  - Soft-voting ensemble: LightGBM + XGBoost + CatBoost
  - Isotonic calibration (5-fold CV)
  - KNHANES external validation (2016–2024)

Targets (v3 — aligned to SNUBH clinical practice and ACR/EULAR guidelines):
  - uric_hyper_sex : Hyperuricemia, sex-specific (M ≥ 7.0 / F ≥ 6.0 mg/dL)  ← primary
  - uric_hyper_70  : Hyperuricemia, unified ≥ 7.0 mg/dL                       ← secondary (updated from 6.5)
  - uric_severe    : Markedly elevated hyperuricemia ≥ 10.0 mg/dL              ← exploratory
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import VotingClassifier

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from data_utils import load_data, BASE_FEATURES, FEATURE_LABELS
from eval_utils import bootstrap_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, '00_data/processed/final_urine_lab_mr_demo_v2.csv')
KNHANES_PATH = os.path.join(PROJECT_ROOT, '00_data/processed/knhanes_uric_validation.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, '03_models/module_uric/bestmodel_v4')
RESULT_DIR = os.path.join(PROJECT_ROOT, '04_results/module_uric_v4')

TIME_WINDOW = 14
N_TRIALS = 25
N_BOOTSTRAP = 1000

FEATURES_13 = BASE_FEATURES + ['bmi_unified']

TARGETS = {
    'uric_hyper_70': {
        'label': 'Hyperuricemia (unified ≥7.0 mg/dL)',
        'short': 'Hyperuricemia (≥7.0)',
        'knhanes_binary': 'uric_hyper_70',
    },
    'uric_severe_9': {
        'label': 'Markedly elevated hyperuricemia (≥9.0 mg/dL)',
        'short': 'Markedly elevated (≥9.0)',
        'knhanes_binary': 'uric_severe',
    },
    'uric_severe_10': {
        'label': 'Institutional ULT threshold (≥10.0 mg/dL)',
        'short': 'Severe (≥10.0)',
        'knhanes_binary': 'uric_severe_10',
    },
}


def make_uric_labels(df):
    df = df.copy()
    df['uric_acid'] = pd.to_numeric(df['uric_acid'], errors='coerce')
    df['uric_hyper_70']  = np.where(df['uric_acid'].notna(),
                                     (df['uric_acid'] >= 7.0).astype(float), np.nan)
    df['uric_severe_9']    = np.where(df['uric_acid'].notna(),
                                     (df['uric_acid'] >= 9.0).astype(float), np.nan)
    df['uric_severe_10']    = np.where(df['uric_acid'].notna(),
                                     (df['uric_acid'] >= 10.0).astype(float), np.nan)
    return df


def add_bmi_unified(df):
    """Dual-source BMI: BMI_demo first, fallback to calculated from height/weight."""
    df = df.copy()
    df['키_num'] = pd.to_numeric(df.get('키', np.nan), errors='coerce')
    df['몸_num'] = pd.to_numeric(df.get('몸무게', np.nan), errors='coerce')
    df.loc[(df['키_num'] < 50) | (df['키_num'] > 220), '키_num'] = np.nan
    df.loc[(df['몸_num'] < 10) | (df['몸_num'] > 300), '몸_num'] = np.nan
    df['bmi_calc'] = df['몸_num'] / (df['키_num'] / 100) ** 2
    bmi_demo = pd.to_numeric(df.get('BMI_demo', np.nan), errors='coerce')
    df['bmi_unified'] = bmi_demo.fillna(df['bmi_calc'])
    df.loc[(df['bmi_unified'] < 10) | (df['bmi_unified'] > 70), 'bmi_unified'] = np.nan
    return df


def prepare_dev(df, target_col):
    df_f = df[df['days_difference'].abs() <= TIME_WINDOW].copy()
    cols = FEATURES_13 + [target_col]
    data = df_f[cols].dropna().copy()
    data[target_col] = data[target_col].astype(int)
    y = data[target_col]
    X = data[FEATURES_13]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def prepare_knhanes(knhanes, target_col):
    cols = FEATURES_13 + [target_col]
    data = knhanes[cols].dropna().copy()
    y = data[target_col].astype(int)
    X = data[FEATURES_13]
    return X, y


def tune_and_build(X_train, y_train, n_trials=N_TRIALS):
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
        print(f'    {name}: CV AUC = {study.best_value:.4f}')
        return study.best_params, study.best_value

    print('    Tuning LightGBM...')
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

    print('    Tuning XGBoost...')
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

    print('    Tuning CatBoost...')
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

    print('    Fitting calibrated ensemble (isotonic, 5-fold)...')
    cal = CalibratedClassifierCV(voting, method='isotonic', cv=5, n_jobs=1)
    cal.fit(X_train, y_train)
    return cal, {'lgb': lgb_p, 'xgb': xgb_p, 'cat': cb_p,
                 'cv_aucs': {'lgb': lgb_a, 'xgb': xgb_a, 'cat': cb_a}}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    t0 = time.time()

    print('=' * 70)
    print('Uric Acid Module v2 — ±14d, 13F (urine + age + sex + BMI)')
    print('=' * 70)

    print('\n[1/7] Loading data...')
    df = load_data(DATA_PATH)
    df = make_uric_labels(df)
    df = add_bmi_unified(df)
    n_14d = (df['days_difference'].abs() <= TIME_WINDOW).sum()
    print(f'  Development: {len(df):,} → ±{TIME_WINDOW}d: {n_14d:,}')
    print(f'  BMI available: {df["bmi_unified"].notna().sum():,}')

    knhanes = pd.read_csv(KNHANES_PATH, low_memory=False)
    # Build bmi_unified for KNHANES too
    bmi_demo = pd.to_numeric(knhanes.get('BMI_demo', np.nan), errors='coerce')
    bmi_calc = pd.to_numeric(knhanes.get('bmi_calc', np.nan), errors='coerce')
    knhanes['bmi_unified'] = bmi_demo.fillna(bmi_calc)
    knhanes.loc[(knhanes['bmi_unified'] < 10) | (knhanes['bmi_unified'] > 70), 'bmi_unified'] = np.nan
    print(f'  KNHANES: {len(knhanes):,} (with uric_acid: {knhanes["HE_Uacid"].notna().sum():,}, '
          f'with BMI: {knhanes["bmi_unified"].notna().sum():,})')

    print('\n[2/7] Training models per target (13F)...')
    models = {}
    train_results = {}
    for tname, cfg in TARGETS.items():
        print(f'\n  --- {cfg["label"]} ---')
        X_train, X_test, y_train, y_test = prepare_dev(df, tname)
        n_total = len(X_train) + len(X_test)
        print(f'    N={n_total:,} (train={len(X_train):,}, test={len(X_test):,}), '
              f'pos={y_train.mean():.1%}')

        cal_model, tune_info = tune_and_build(X_train, y_train)
        y_prob_test = cal_model.predict_proba(X_test)[:, 1]
        m_test = bootstrap_metrics(y_test.values, y_prob_test, n_bootstrap=N_BOOTSTRAP)
        print(f'    Internal AUROC: {m_test["AUC"]:.4f} '
              f'({m_test["AUC_CI"][0]:.4f}–{m_test["AUC_CI"][1]:.4f})')
        print(f'    Brier: {brier_score_loss(y_test, y_prob_test):.4f}')

        joblib.dump(cal_model, os.path.join(MODEL_DIR, f'{tname}_calibrated_voting.joblib'))
        models[tname] = cal_model
        train_results[tname] = {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'y_prob_test': y_prob_test, 'metrics_test': m_test, 'n_total': n_total,
            'tune_info': tune_info,
        }

    print('\n[3/7] External validation (KNHANES 2016–2024)...')
    ext_results = {}
    for tname, cfg in TARGETS.items():
        kbin = cfg['knhanes_binary']
        X_ext, y_ext = prepare_knhanes(knhanes, kbin)
        if len(X_ext) < 100 or y_ext.nunique() < 2:
            print(f'  {cfg["label"]}: insufficient (N={len(X_ext)})'); continue
        y_prob_ext = models[tname].predict_proba(X_ext)[:, 1]
        m_ext = bootstrap_metrics(y_ext.values, y_prob_ext, n_bootstrap=N_BOOTSTRAP)
        brier_ext = brier_score_loss(y_ext.values, y_prob_ext)
        print(f'  {cfg["short"]}: N={len(X_ext):,}, AUROC={m_ext["AUC"]:.4f} '
              f'({m_ext["AUC_CI"][0]:.4f}–{m_ext["AUC_CI"][1]:.4f}), Brier={brier_ext:.4f}')
        ext_results[tname] = {
            'X': X_ext, 'y_true': y_ext, 'y_pred_proba': y_prob_ext,
            'metrics': m_ext, 'brier': brier_ext, 'n': len(X_ext),
        }

    print('\n[4/7] Performance summary table...')
    rows = []
    for tname, cfg in TARGETS.items():
        tr = train_results[tname]
        r = {'Target': cfg['label'],
             'N_internal': tr['n_total'],
             'Internal_AUROC': tr['metrics_test']['AUC'],
             'Internal_AUROC_CI': f"{tr['metrics_test']['AUC_CI'][0]:.3f}-{tr['metrics_test']['AUC_CI'][1]:.3f}",
             'Internal_Brier': brier_score_loss(tr['y_test'], tr['y_prob_test'])}
        if tname in ext_results:
            ext = ext_results[tname]
            r.update({'N_external': ext['n'],
                      'External_AUROC': ext['metrics']['AUC'],
                      'External_AUROC_CI': f"{ext['metrics']['AUC_CI'][0]:.3f}-{ext['metrics']['AUC_CI'][1]:.3f}",
                      'External_Brier': ext['brier']})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'performance.csv'), index=False)

    print('\n[5/7] Subgroup analysis (KNHANES)...')
    sub_rows = []
    for tname, cfg in TARGETS.items():
        if tname not in ext_results: continue
        ext = ext_results[tname]
        X_ext = ext['X'].copy()
        y_true = ext['y_true'].values
        y_prob = ext['y_pred_proba']

        def auc_ci(yt, yp):
            if len(np.unique(yt)) < 2: return None, (None, None)
            auc = roc_auc_score(yt, yp)
            rng = np.random.RandomState(42)
            aucs = []
            for _ in range(500):
                idx = rng.randint(0, len(yt), len(yt))
                if len(np.unique(yt[idx])) < 2: continue
                aucs.append(roc_auc_score(yt[idx], yp[idx]))
            return auc, (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))

        auc, ci = auc_ci(y_true, y_prob)
        sub_rows.append({'Model': cfg['short'], 'Subgroup': 'Overall',
                          'N': len(y_true), 'Prevalence': y_true.mean()*100,
                          'AUROC': auc, 'AUROC_CI_low': ci[0], 'AUROC_CI_high': ci[1]})
        for lo, hi, lbl in [(19,39,'Age 19-39'), (40,59,'Age 40-59'),
                             (60,79,'Age 60-79'), (80,200,'Age 80+')]:
            mask = (X_ext['age'] >= lo) & (X_ext['age'] <= hi)
            if mask.sum() > 50 and len(np.unique(y_true[mask])) == 2:
                a, c = auc_ci(y_true[mask], y_prob[mask])
                sub_rows.append({'Model': cfg['short'], 'Subgroup': lbl,
                                  'N': int(mask.sum()), 'Prevalence': y_true[mask].mean()*100,
                                  'AUROC': a, 'AUROC_CI_low': c[0], 'AUROC_CI_high': c[1]})
        for sv, sl in [(1,'Male'), (0,'Female')]:
            mask = X_ext['male'] == sv
            if mask.sum() > 50 and len(np.unique(y_true[mask])) == 2:
                a, c = auc_ci(y_true[mask], y_prob[mask])
                sub_rows.append({'Model': cfg['short'], 'Subgroup': sl,
                                  'N': int(mask.sum()), 'Prevalence': y_true[mask].mean()*100,
                                  'AUROC': a, 'AUROC_CI_low': c[0], 'AUROC_CI_high': c[1]})
    pd.DataFrame(sub_rows).to_csv(os.path.join(RESULT_DIR, 'subgroup_analysis.csv'), index=False)

    print('\n[6/7] Summary table:')
    elapsed = (time.time() - t0) / 60
    print(f'\n{"Target":<42} {"N_int":>8} {"Int AUROC":>10} {"N_ext":>8} {"Ext AUROC":>10}')
    print('-' * 90)
    for tname, cfg in TARGETS.items():
        tr = train_results[tname]
        ext = ext_results.get(tname, {})
        n_e = ext.get('n', '')
        a_e = ext.get('metrics', {}).get('AUC', '')
        nes = f'{n_e:>8,}' if n_e else f'{"N/A":>8}'
        aes = f'{a_e:>10.4f}' if a_e else f'{"N/A":>10}'
        print(f'{cfg["short"]:<42} {tr["n_total"]:>8,} {tr["metrics_test"]["AUC"]:>10.4f} {nes} {aes}')

    summary = {
        'module': 'uric_v2',
        'time_window_days': TIME_WINDOW,
        'features': FEATURES_13,
        'n_features': len(FEATURES_13),
        'tuning_trials': N_TRIALS,
        'bootstrap_n': N_BOOTSTRAP,
        'targets': list(TARGETS.keys()),
        'elapsed_min': elapsed,
    }
    with open(os.path.join(RESULT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\n[7/7] Done. Total: {elapsed:.1f} min')
    print(f'Results → {RESULT_DIR}')
    print(f'Models  → {MODEL_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
