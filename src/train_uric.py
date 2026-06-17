"""
Train calibrated soft-voting ensembles for hyperuricemia detection.

Pipeline (per target):
  1. Stratified 80/20 train/test split (13 features: 10 dipstick + age + sex + BMI)
  2. Tree-structured Parzen Estimator (TPE) Bayesian hyperparameter tuning
     (Optuna, 25 trials per learner) on LightGBM, XGBoost, CatBoost
  3. Soft-voting ensemble of the three tuned learners
  4. Post-hoc isotonic probability calibration (5-fold CV)
  5. Save calibrated model artifact + bootstrap performance metrics

Targets:
    uric70:   Serum uric acid ≥ 7.0 mg/dL   (unified clinical reporting standard)
    uric90:   Serum uric acid ≥ 9.0 mg/dL   (markedly elevated, gout risk)
    uric100:  Serum uric acid ≥ 10.0 mg/dL  (institutional ULT initiation threshold)

Reference: Jang et al., JMIR Medical Informatics (under review, 2026).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

from data_utils import BASE_FEATURES, load_data
from eval_utils import bootstrap_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25
N_BOOTSTRAP = 1000
TIME_WINDOW_DAYS = 14

TARGETS = {
    "uric70":  {"col": "serum_uric_acid", "threshold": 7.0,  "label": "Serum UA ≥ 7.0 mg/dL"},
    "uric90":  {"col": "serum_uric_acid", "threshold": 9.0,  "label": "Serum UA ≥ 9.0 mg/dL"},
    "uric100": {"col": "serum_uric_acid", "threshold": 10.0, "label": "Serum UA ≥ 10.0 mg/dL"},
},
    "fpg100":  {"col": "Glucose", "threshold": 100, "label": "FPG ≥ 100 mg/dL"},
    "hba1c65": {"col": "hba1c",   "threshold": 6.5, "label": "HbA1c ≥ 6.5%"},
    "hba1c57": {"col": "hba1c",   "threshold": 5.7, "label": "HbA1c ≥ 5.7%"},
}


# ── Hyperparameter search spaces ─────────────────────────────

def _lgb_params(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }


def _xgb_params(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth":        trial.suggest_int("max_depth", 4, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
        "use_label_encoder": False, "eval_metric": "logloss",
    }


def _cb_params(trial):
    return {
        "iterations":    trial.suggest_int("iterations", 100, 300, step=50),
        "depth":         trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": 42, "verbose": False, "thread_count": -1,
    }


def _tune(model_cls, param_fn, X, y, n_trials=N_TRIALS):
    """3-fold CV Bayesian tuning, return best params + CV AUROC."""

    def objective(trial):
        params = param_fn(trial)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr, va in skf.split(X, y):
            m = model_cls(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def build_calibrated_ensemble(X_train, y_train, n_trials=N_TRIALS):
    """Tune three learners, build voting ensemble, apply isotonic calibration."""
    print("    Tuning LightGBM...")
    lgb_p, lgb_cv = _tune(lgb.LGBMClassifier, _lgb_params, X_train, y_train, n_trials)

    print("    Tuning XGBoost...")
    xgb_p, xgb_cv = _tune(xgb.XGBClassifier, _xgb_params, X_train, y_train, n_trials)

    print("    Tuning CatBoost...")
    cb_p, cb_cv = _tune(cb.CatBoostClassifier, _cb_params, X_train, y_train, n_trials)

    # Re-inject defaults
    lgb_p |= {"random_state": 42, "verbose": -1, "n_jobs": -1}
    xgb_p |= {"random_state": 42, "verbosity": 0, "n_jobs": -1,
              "use_label_encoder": False, "eval_metric": "logloss"}
    cb_p  |= {"random_seed": 42, "verbose": False, "thread_count": -1}

    voting = VotingClassifier(
        estimators=[
            ("lgb", lgb.LGBMClassifier(**lgb_p)),
            ("xgb", xgb.XGBClassifier(**xgb_p)),
            ("cat", cb.CatBoostClassifier(**cb_p)),
        ],
        voting="soft", n_jobs=1,
    )

    print("    Fitting calibrated ensemble (isotonic, 5-fold)...")
    cal = CalibratedClassifierCV(voting, method="isotonic", cv=5, n_jobs=1)
    cal.fit(X_train, y_train)

    tune_info = {
        "lgb": {"params": lgb_p, "cv_auc": lgb_cv},
        "xgb": {"params": xgb_p, "cv_auc": xgb_cv},
        "cat": {"params": cb_p,  "cv_auc": cb_cv},
    }
    return cal, tune_info


# ── Target preparation ────────────────────────────────────────

def make_target(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cfg = TARGETS[name]
    col, thr = cfg["col"], cfg["threshold"]
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[name] = np.where(df[col].notna(), (df[col] >= thr).astype(float), np.nan)
    return df


def prepare(df: pd.DataFrame, target_col: str):
    df_f = df[df["days_difference"].abs() <= TIME_WINDOW_DAYS].copy()
    cols = BASE_FEATURES + [target_col]
    data = df_f[cols].dropna().copy()
    data[target_col] = data[target_col].astype(int)
    y = data[target_col]
    X = data[BASE_FEATURES]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to processed CSV (urine + blood + demographics)")
    p.add_argument("--out", default="./models", help="Output directory for model artifacts")
    p.add_argument("--targets", nargs="+", default=list(TARGETS.keys()),
                   choices=list(TARGETS.keys()),
                   help="Subset of targets to train (default: all 3)")
    p.add_argument("--n_trials", type=int, default=N_TRIALS,
                   help=f"Optuna trials per learner (default: {N_TRIALS})")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)

    summary = {}
    for name in args.targets:
        cfg = TARGETS[name]
        print(f"\n=== {cfg['label']} ===")
        df_t = make_target(df, name)
        X_tr, X_te, y_tr, y_te = prepare(df_t, name)
        print(f"    N train={len(X_tr):,}, test={len(X_te):,}, prevalence={y_tr.mean():.1%}")

        cal, tune_info = build_calibrated_ensemble(X_tr, y_tr, args.n_trials)
        y_prob = cal.predict_proba(X_te)[:, 1]
        m = bootstrap_metrics(y_te.values, y_prob, n_bootstrap=N_BOOTSTRAP)
        brier = brier_score_loss(y_te, y_prob)

        artifact = out / f"{name}_calibrated_voting.joblib"
        joblib.dump(cal, artifact)
        print(f"    Internal AUROC: {m['AUC']:.4f} "
              f"({m['AUC_CI'][0]:.4f}–{m['AUC_CI'][1]:.4f}), Brier: {brier:.4f}")
        print(f"    Saved → {artifact}")

        summary[name] = {
            "label": cfg["label"],
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "internal_auc": m["AUC"],
            "internal_auc_ci": list(m["AUC_CI"]),
            "internal_brier": float(brier),
            "tune_info": tune_info,
        }

    with (out / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Training summary → {out / 'training_summary.json'}")


if __name__ == "__main__":
    main()
