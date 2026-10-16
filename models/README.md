# Model files (not distributed)

The fitted model files are **not included in this repository and are not publicly released** because of the
pending patent application (see the article's Conflicts of Interest statement). Requests for access for independent
validation may be made to the corresponding author (Kiheon Lee, keyhone2@snu.ac.kr). The README at commit 2d8b5b0
announced downloads from GitHub Releases under CC BY 4.0; no release was made and that statement is withdrawn.

## Retrained models (the models of the revised article; bestmodel_v5)

Written on 2026-10-16 by `revision_R1/train_v5.py train` into `PROJ_U_MODELS_V5` (originally
`03_models/module_uric/bestmodel_v5`):

| File | Content |
|---|---|
| `uric_hyper_70_calibrated_voting.joblib` | serum urate >=7.0 mg/dL (primary), 5.4 MB |
| `uric_severe_9_calibrated_voting.joblib` | serum urate >=9.0 mg/dL (secondary), 4.9 MB |
| `uric_severe_10_calibrated_voting.joblib` | serum urate >=10.0 mg/dL (exploratory), 5.3 MB |
| `tune_<target>.json` | tuned hyperparameters, cross-validated AUROC of each learner, events per calibration fold |
| `optuna_trials_<target>_<learner>.csv`, `train_log.txt` | the Optuna trials and the training log |
| `config.json` | the record written before any model was fitted: **included** as [`bestmodel_v5/config.json`](bestmodel_v5/config.json) |

Each model file is a scikit-learn `CalibratedClassifierCV(method='isotonic', cv=5)` around a
`VotingClassifier(voting='soft')` of `LGBMClassifier`, `XGBClassifier` and `CatBoostClassifier`, whose
hyperparameters were tuned by Optuna (TPE sampler, seed 42, 25 trials per learner, 3-fold cross-validated AUROC) on
the 103,628 training persons; the saved object averages five fold-specific calibrated ensembles (15 base learners).
The input is the 12 columns of `FMODEL` in `revision_R1/cohort.py`, in this order and on the development coding
(`docs/feature_dictionary.md`); each model stores them as `feature_names_in_`:

    ua_glucose, ua_protein, ua_ketone, ua_bilirubin, ua_blood, ua_urobilinogen, ua_nitrite, ua_pH, ua_sg, age, male, bmi_unified

The tuned hyperparameters are listed in Multimedia Appendix 1, Table S8 (`revision_R1/calc_importance_params.py`
reads them from the files).

`config.json` (written 2026-10-16 23:09:07, before the first model was fitted) records the inputs, the cohort steps of
Figure 1, the numbers of persons (129,535 in the cohort, 103,628 for training, 25,907 for testing), the split, the
recipe with its search spaces, sha256 checksums of the training inputs, of the training and test identifiers and of
the code (`train_v5.py`, `cohort.py` and `run_uric_module_v4.py` as they were at that moment; see
`revision_R1/README.md`), and the library versions (`requirements.txt`). It holds no model parameters and no
patient-level data.

Performance in the revised article (AUROC with 95% CI from 1,000 bootstrap resamples):

| Outcome | Internal test, n (events) | AUROC (95% CI) | KNHANES adults, n (events) | AUROC (95% CI) | KNHANES O:E |
|---|---|---|---|---|---|
| >=7.0 mg/dL | 25,907 (4,225) | 0.825 (0.820-0.831) | 49,987 (5,068) | 0.845 (0.840-0.849) | 0.73 |
| >=9.0 mg/dL | 25,907 (494) | 0.859 (0.845-0.873) | 49,987 (444) | 0.860 (0.845-0.873) | 0.48 |
| >=10.0 mg/dL (exploratory) | 25,907 (147) | 0.877 (0.854-0.900) | 49,987 (103) | 0.863 (0.833-0.891) | 0.35 |

O:E: observed-to-expected ratio (below 1 means over-prediction). The models over-predict in KNHANES; the article
reports an intercept update fitted on local data and states that, before any use, the models would need local
recalibration and evaluation in the intended setting.

Loading (with the model files obtained from the corresponding author):

```python
import joblib
model = joblib.load('/path/to/uric_hyper_70_calibrated_voting.joblib')
p = model.predict_proba(X[list(model.feature_names_in_)])[:, 1]   # X: DataFrame with the 12 inputs
```

## Submitted models (bestmodel_v4)

`src/train_uric.py` wrote the three submitted models (same file names and structure) into `MODEL_DIR`
(`PROJ_U_MODELS`, originally `03_models/module_uric/bestmodel_v4`). They take 13 inputs (`FEATURES_13` in
`src/train_uric.py`: the 12 above plus leukocyte esterase, `ua_wbc(s)`) and were trained on the submitted
record-level frame. The revised article does not report them. The revision uses them only in
`revision_R1/train_v5.py reproduce` (the copied recipe reproduces them exactly) and in `revision_R1/calc_main.py`,
for the comparison with the retrained models in the response letter (`original_vs_retrained.csv`). `src/predict.py`
is a scoring example for these files.
