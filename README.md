# Urine-Dipstick-Hyperuricemia-Screening

Code and resources accompanying the manuscript:

> **Routine urine dipstick analysis for opportunistic hyperuricemia screening: development and external validation of calibrated machine learning models**
> Jang EC, Song P, Kim JE, Han MH, Han H, Lee K.
> *JMIR Medical Informatics* (under review, 2026).

A calibrated soft-voting ensemble (LightGBM + XGBoost + CatBoost) using 10 routine urine dipstick parameters plus age, sex, and body mass index (BMI) for opportunistic detection of hyperuricemia at three clinically anchored thresholds:

- **≥7.0 mg/dL** — unified clinical reporting standard / monosodium urate solubility limit (~6.8 mg/dL at 37°C)
- **≥9.0 mg/dL** — markedly elevated, associated with substantially increased gout incidence
- **≥10.0 mg/dL** — institutional urate-lowering therapy (ULT) initiation threshold

External validation in the Korea National Health and Nutrition Examination Survey (KNHANES) 2016–2024 (n = 54,660): AUROC 0.850 / 0.866 / 0.865 for the three thresholds respectively.

---

## Key Features

- **Inputs**: 10 urine dipstick parameters (glucose, protein, ketone, bilirubin, occult blood, urobilinogen, nitrite, leukocyte esterase, pH, specific gravity) + age + sex + BMI
- **Targets**: 3 binary classification models — serum uric acid ≥7.0 / ≥9.0 / ≥10.0 mg/dL
- **Model**: Soft-voting ensemble (LightGBM + XGBoost + CatBoost), Bayesian-tuned hyperparameters (Optuna TPE, 25 trials per learner)
- **Calibration**: Isotonic regression with 5-fold cross-validation
- **Development**: Seoul National University Bundang Hospital, 2010–2025 (N = 145,678 within ±14-day urine–blood window)
- **External validation**: KNHANES 2016–2024 (n = 54,660 across 9 annual waves)
- **Reporting**: TRIPOD+AI, STROBE, RECORD compliant

---

## Repository Structure

```
.
├── README.md                       This file
├── LICENSE                         MIT License
├── requirements.txt                Pinned Python dependencies
├── environment.yml                 Conda environment (optional)
├── src/                            Source modules
│   ├── data_utils.py               Data loading and 13-feature preprocessing
│   ├── eval_utils.py               Performance metrics, bootstrap CIs
│   ├── train_uric.py               Model training pipeline (three thresholds)
│   └── predict.py                  Inference example with Youden thresholds
├── notebooks/                      Reproducibility notebooks
│   ├── 01_baseline_table1.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_external_validation.ipynb
├── models/                         Trained model artifacts (see models/README.md)
├── data/
│   ├── knhanes_processing.py       KNHANES preparation script (serum uric acid harmonization)
│   └── README.md                   Data access instructions
└── docs/
    ├── tripod_ai_checklist.md      TRIPOD+AI reporting checklist
    └── feature_dictionary.md       13-feature definitions and encoding
```

---

## Installation

### Option A — pip

```bash
git clone https://github.com/eunchanj/Urine-Dipstick-Hyperuricemia-Screening.git
cd Urine-Dipstick-Hyperuricemia-Screening
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — conda

```bash
git clone https://github.com/eunchanj/Urine-Dipstick-Hyperuricemia-Screening.git
cd Urine-Dipstick-Hyperuricemia-Screening
conda env create -f environment.yml
conda activate urine-dipstick-uric
```

Python 3.11 is required.

---

## Quick Start

### Loading a trained model and making predictions

```python
import joblib
import pandas as pd
from src.data_utils import BASE_FEATURES

# Load calibrated voting ensemble for the unified ≥7.0 mg/dL threshold
model = joblib.load('models/uric70_calibrated_voting.joblib')

# Example input — single record with 13 features (10 dipstick + age + sex + BMI)
sample = pd.DataFrame([{
    'ua_glucose': 0, 'ua_protein': 0, 'ua_ketone': 0, 'ua_bilirubin': 0,
    'ua_blood': 0, 'ua_urobilinogen': 0, 'ua_nitrite': 0, 'ua_wbc(s)': 0.321,
    'ua_pH': 5.5, 'ua_sg': 1.020,
    'age': 55, 'male': 1, 'BMI': 26.0,
}])

# Predict probability of hyperuricemia (serum UA ≥7.0 mg/dL)
prob = model.predict_proba(sample[BASE_FEATURES])[:, 1]
print(f'Predicted probability of hyperuricemia (≥7.0 mg/dL): {prob[0]:.3f}')
```

### Reproducing the analysis

See [notebooks/](notebooks/) for end-to-end reproducibility:

1. `01_baseline_table1.ipynb` — Cohort characteristics (Table 1)
2. `02_model_training.ipynb` — Bayesian optimization, ensemble training, isotonic calibration across three thresholds
3. `03_external_validation.ipynb` — KNHANES evaluation, subgroup analyses (age × sex), calibration plots, SHAP

---

## Data Access

- **KNHANES (external validation cohort)**: publicly available via the Korea Disease Control and Prevention Agency portal — https://knhanes.kdca.go.kr/knhanes/main.do (Korean). See [data/README.md](data/README.md) for variable mapping and access workflow.
- **Hospital development cohort**: not publicly available owing to institutional privacy regulations and the Korean Personal Information Protection Act. Reasonable requests for collaborative access may be directed to the corresponding author and are subject to Institutional Review Board approval (Seoul National University Bundang Hospital IRB No. B-2509-994-106).

---

## Trained Models

Pre-trained calibrated ensembles for all 3 hyperuricemia thresholds are provided in the `models/` directory:

| File | Target | Internal AUROC | External AUROC | External Brier |
|---|---|---|---|---|
| `uric70_calibrated_voting.joblib`  | Serum uric acid ≥7.0 mg/dL  | 0.839 | 0.850 | 0.078 |
| `uric90_calibrated_voting.joblib`  | Serum uric acid ≥9.0 mg/dL  | 0.867 | 0.866 | 0.009 |
| `uric100_calibrated_voting.joblib` | Serum uric acid ≥10.0 mg/dL | 0.892 | 0.865 | 0.002 |

See [models/README.md](models/README.md) for distribution details (large files distributed via GitHub Release).

---

## Reporting Compliance

This work follows the following reporting standards:

- **TRIPOD+AI** (Collins et al., *BMJ* 2024) — see [docs/tripod_ai_checklist.md](docs/tripod_ai_checklist.md)
- **STROBE** — observational study reporting
- **RECORD** — routinely-collected data extension

---

## Citation

If you use this code or the trained models in your research, please cite:

```bibtex
@article{jang2026uric,
  title  = {Routine urine dipstick analysis for opportunistic hyperuricemia screening: development and external validation of calibrated machine learning models},
  author = {Jang, Eun Chan and Song, Pado and Kim, Joung Eun and Han, Myung Hwa and Han, Hwigyeong and Lee, Kiheon},
  journal= {JMIR Medical Informatics},
  year   = {2026},
  note   = {Under review}
}
```

(Citation will be updated upon acceptance.)

---

## License

This code is released under the [MIT License](LICENSE). Trained model artifacts are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The methods described herein are the subject of a pending patent application (Korean Intellectual Property Office). The code is provided for academic and research use; commercial deployment may require separate licensing — contact the corresponding author.

---

## Related Modules

This work is the **hyperuricemia module** of an integrated multi-disease screening platform:

- **Diabetes module** — https://github.com/eunchanj/Urine-Dipstick-Diabetes-Screening
- **Liver module** — https://github.com/eunchanj/Urine-Dipstick-Liver-Screening
- **Hyperuricemia module** — *this repository*
- **CKD module** — Jang EC, Park YM, Han HW, et al. *J Am Med Inform Assoc* 2023;30(6):1114–1124

---

## Contact

- **Corresponding author**: Kiheon Lee, MD, PhD — keyhone2@snu.ac.kr
- **First author / code maintainer**: Eun Chan Jang, MPH — eunchan.j93@gmail.com

Department of Family Medicine, Seoul National University Bundang Hospital
82, Gumi-ro 173 Beon-gil, Bundang-gu, Seongnam-si, Gyeonggi-do 13620, Republic of Korea

---

## Acknowledgments

This research was supported by:

- Korea Health Technology R&D Project, Korea Health Industry Development Institute (KHIDI), Ministry of Health & Welfare (RS-2025-02307233)
- Doctoral Student Research Grant Program, National Research Foundation of Korea (NRF), Ministry of Education (RS-2025-2540035)
