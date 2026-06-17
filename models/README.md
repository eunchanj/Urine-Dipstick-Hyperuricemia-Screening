# Trained Models

This directory holds the calibrated soft-voting ensemble artifacts used in the manuscript.

## Files

| File                                 | Target                          | Internal AUROC | External AUROC | External Brier |
|--------------------------------------|---------------------------------|----------------|----------------|----------------|
| `uric70_calibrated_voting.joblib`    | Serum uric acid ≥7.0 mg/dL      | 0.839          | 0.850          | 0.078          |
| `uric90_calibrated_voting.joblib`    | Serum uric acid ≥9.0 mg/dL      | 0.867          | 0.866          | 0.009          |
| `uric100_calibrated_voting.joblib`   | Serum uric acid ≥10.0 mg/dL     | 0.892          | 0.865          | 0.002          |

All artifacts are `sklearn.calibration.CalibratedClassifierCV` objects wrapping a `VotingClassifier(soft)` of LightGBM, XGBoost, and CatBoost base learners with Bayesian-tuned hyperparameters and 5-fold isotonic recalibration.

## Distribution

Trained model files exceed GitHub's recommended commit size (~10–80 MB each). They are distributed via **GitHub Releases** rather than tracked in the working tree.

Download the latest release bundle:

```bash
# Replace <tag> with the latest version (e.g., v1.0.0)
gh release download <tag> --pattern '*.joblib' --dir models/
```

Or via direct browser download: https://github.com/eunchanj/Urine-Dipstick-Hyperuricemia-Screening/releases

After download, place all `*.joblib` files in this `models/` directory. They are referenced relative to the repository root by `src/predict.py` and the notebooks.

## License

Trained model weights are released under **CC BY 4.0**. See the top-level [LICENSE](../LICENSE) for code licensing terms and patent notice (Korean Intellectual Property Office). Commercial deployment may require a separate license agreement.

## Reproducibility

Models were trained with the exact dependency versions listed in `../requirements.txt`. Random seeds are fixed throughout the training pipeline (`random_state=42`, Optuna `TPESampler(seed=42)`); training is therefore reproducible up to platform-dependent floating-point variability of the underlying gradient boosting libraries.

To retrain from scratch (development data required):

```bash
python src/train_uric.py --data data/processed/development_cohort.csv --out models/
```
