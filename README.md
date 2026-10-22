# Urine-Dipstick-Hyperuricemia-Screening

Code accompanying the manuscript

> **Machine Learning Models Combining Routine Urine Dipstick Results With Age, Sex, and Body Mass Index to Triage Serum Urate Testing: Development and External Validation Study**
> Jang EC, Song P, Kim JE, Han MH, Han H, Lee K.
> *JMIR Medical Informatics*, manuscript #107515, revision R1 (revision under review, 2026).

**Status: revision under review.** The `revision-r1` branch holds the code of the revised article: the corrected
development cohort, the retrained models and every analysis, table, figure and appendix. It replaces the code at
commit 2d8b5b0, which did not produce the submitted results (see [History of this repository](#history-of-this-repository)).
A tagged version with an archival DOI will be created from it before publication. The fitted model files are not
included (pending patent application; available on request for independent validation).

## What the study did

Three calibrated soft-voting ensembles (LightGBM + XGBoost + CatBoost; isotonic calibration with 5-fold
cross-fitting) estimate the probability that serum urate is at or above a threshold from **12 inputs**: 9 routine urine
dipstick fields (glucose, protein, ketone, bilirubin, occult blood, urobilinogen, nitrite, pH, specific gravity), age,
sex and body mass index (BMI).

| Model | Outcome | Role in the article |
|---|---|---|
| `uric_hyper_70` | serum urate >=7.0 mg/dL | primary |
| `uric_severe_9` | serum urate >=9.0 mg/dL | secondary |
| `uric_severe_10` | serum urate >=10.0 mg/dL | exploratory (Multimedia Appendix 1, Section S12) |

- **Development data**: each person's first urine test at a Korean tertiary hospital (2010-2025), with **serum urate
  and BMI recorded on the same day** (values from other days are not used): 129,535 adults aged 19 years or older
  (21,124 with serum urate >=7.0 mg/dL, 2,470 with >=9.0 mg/dL and 736 with >=10.0 mg/dL; 81.6% examined at a
  same-day health checkup).
- **One person-level split shared by the three thresholds**: 80/20, stratified on serum urate bands, seed 42; 103,628
  training and 25,907 internal test persons, none in both sets.
- **External validation**: KNHANES 2016-2024, 49,987 adults aged 19 years or older (5,068 events at >=7.0 mg/dL, 444
  at >=9.0 mg/dL, 103 at >=10.0 mg/dL). The KNHANES dipstick results enter with the **label-matched coding X1**: each
  KNHANES grade is assigned to the development grade with the same label, 4+ is merged into 3+, and a normal or trace
  urobilinogen is coded 0.5, the development code for a normal result. Other codings are sensitivity analyses.
- **Training**: hyperparameters tuned on the training persons only (Optuna TPE sampler, seed 42, 25 trials per learner,
  3-fold cross-validated AUROC). The cohort, inputs, split, KNHANES coding and recipe were fixed before the models were
  fitted or evaluated and recorded with checksums of the training data, the training and test identifiers and the code
  ([`models/bestmodel_v5/config.json`](models/bestmodel_v5/config.json)).

Headline results of the revised article (AUROC with 95% CI from 1,000 bootstrap resamples):

| Outcome | Internal test (25,907 persons) | KNHANES adults (49,987) | KNHANES O:E |
|---|---|---|---|
| >=7.0 mg/dL | 0.825 (0.820-0.831) | 0.845 (0.840-0.849) | 0.73 |
| >=9.0 mg/dL | 0.859 (0.845-0.873) | 0.860 (0.845-0.873) | 0.48 |
| >=10.0 mg/dL (exploratory) | 0.877 (0.854-0.900) | 0.863 (0.833-0.891) | 0.35 |

O:E: observed-to-expected ratio (below 1 means over-prediction). Across four model classes the dipstick added 0.017-0.024
to the external AUROC of age-sex-BMI models at >=7.0 mg/dL. Externally the models over-predicted, which an intercept
update fitted in KNHANES corrected, and at the internally derived 80%-sensitivity threshold 97.4% of the flagged
adults were men. The article concludes that, before any use, the models would need local recalibration and evaluation
in the intended setting.

## Repository structure

```
.
├── README.md                     this file
├── LICENSE                       MIT License (code) with a patent notice
├── requirements.txt              pinned versions: the environment recorded with the retrained models, plus helpers
├── environment.yml               conda environment (Python 3.13.5 + requirements.txt)
├── revision_R1/                  the code of the revised article: corrected cohort (cohort.py, build_cohort_v2.py),
│   │                             retraining (train_v5.py), evaluation (calc_main.py, calc_*.py) and the table, figure
│   │                             and appendix generators; README with the run order, inputs and outputs
│   └── retired_v4/               scripts that concerned only the submitted models (kept for the audit history)
├── src/                          the code of the SUBMITTED models (see below)
│   ├── train_uric.py             the script that trained the submitted models and wrote the submitted evaluation
│   │                             (verbatim 05_src/run_uric_module_v4.py; paths configurable)
│   ├── data_utils.py             feature lists and data loading (verbatim; paths configurable)
│   ├── eval_utils.py             metrics and bootstrap CIs used by train_uric.py (identical to the authors' copy)
│   └── predict.py                scoring example for the submitted model files (13 inputs)
├── data/
│   ├── knhanes_processing.py     the KNHANES preparation script of the submission (verbatim
│   │                             05_src/prepare_knhanes_uric.py; coding M1; paths configurable)
│   ├── add_knhanes_labels.py     reconstructed step: outcome columns the submitted training script reads
│   └── README.md                 data access, the files the scripts expect, eligibility, KNHANES codings
├── models/
│   ├── README.md                 what the model files are and why they are not here
│   └── bestmodel_v5/config.json  record of the retrained models written before they were fitted (inputs, cohort
│                                 steps, split, recipe, checksums, library versions); no model parameters
└── docs/
    ├── tripod_ai_checklist.md    TRIPOD+AI and TRIPOD+AI for Abstracts checklists of the revised manuscript
    └── feature_dictionary.md     the 12 inputs, their coding and the KNHANES coding X1
```

## How the code in `src/` relates to the retrained models

`src/` holds the code of the submitted models (bestmodel_v4): `src/train_uric.py` built the submitted frame (records
with a blood test within 14 days, 13 inputs including leukocyte esterase, no age limit, a separate 80/20 split for
each threshold), tuned and trained the three submitted models and wrote the submitted evaluation. The models of the
revised article were produced by `revision_R1/train_v5.py`, whose `tune_and_build` is copied line for line from
`src/train_uric.py` (search spaces, Optuna sampler and seeds, cross-validation, learners, soft voting and isotonic
calibration; the only change is that the Optuna studies are returned so that their trials can be saved). What changed
is the data passed to the recipe: `revision_R1/cohort.py` builds the corrected cohort (`dev_v2`), the shared
person-level split (`split_v2`) and the KNHANES inputs under X1 (`ext_v2`), without leukocyte esterase.
`revision_R1/cohort.py` imports `src/data_utils.py` and `src/train_uric.py` (the feature lists, the BMI rule
`add_bmi_unified` and the data and model paths). Before training, `train_v5.py reproduce` confirmed that the copied
recipe, run on the submitted frame, reproduces the three submitted models exactly (identical tuned parameters and
predictions). Running `src/train_uric.py` itself retrains the submitted 13-input models and overwrites the files in
`PROJ_U_MODELS`; it is kept to document the submission, not to build the models of the revised article.

## History of this repository

1. **Commit 2d8b5b0** (cited in the submission) did not hold the code that produced the submitted results: it
   implemented the printed KNHANES mapping (M2) and an age filter that the analysis code lacked, its training script
   could not be run (syntax error; outcome and BMI columns that the data do not contain), its pinned versions
   (Python 3.11, scikit-learn 1.3.2, XGBoost 2.0.3) did not describe the environment of the submitted models, and its
   README announced model downloads under CC BY 4.0, which were never made; that statement is withdrawn.
2. **Commit caff249** (branch `revision-r1`) released the code that produced the submitted results (`src/`, `data/`)
   and the first revision analyses, which re-evaluated the submitted models on corrected cohorts.
3. **This version**: after the audit the authors corrected the development data and retrained the models.
   `revision_R1/` now holds the code of the retrained models and of every analysis of the revised manuscript; the
   scripts that concerned only the submitted models are in `revision_R1/retired_v4/`.

What the audit found in the submitted analysis, and how the retrained models handle it (details in the response
letter):

| Submitted analysis | Retrained models |
|---|---|
| No lower age limit: KNHANES participants aged 10-18 years were in the external cohort, and 7,295 of the 116,542 training records at >=7.0 mg/dL were of persons younger than 19 years | adults aged 19 years or older in both cohorts |
| Nitrite results recorded as "Positive"/"POSITIVE" were parsed as missing | nitrite read correctly from the raw text (1,035 nitrite-positive persons in the cohort) |
| Serum urate "<1.0 mg/dL" dropped; repeat records of the same person; record-level splits with persons in training and test | "<1.0 mg/dL" counted as 0.5; one record per person; one person-level split |
| Checkup BMI linked by nearest date with no time window (every non-same-day value came from a later checkup); the urine-blood interval referred to the nearest laboratory test of any kind, not to serum urate | serum urate and BMI recorded on the day of the urine test only |
| Leukocyte esterase, which KNHANES does not measure, entered externally as a constant (0.321) | not an input |
| KNHANES coding printed as one uniform map (M2) but implemented differently (M1) | label-matched coding X1, fixed before fitting; M1, M2, M3 and X2 as sensitivity analyses |

## Installation

```bash
git clone https://github.com/eunchanj/Urine-Dipstick-Hyperuricemia-Screening.git
cd Urine-Dipstick-Hyperuricemia-Screening
git checkout revision-r1
conda env create -f environment.yml && conda activate urine-dipstick-uric   # or: pip install -r requirements.txt
```

Python 3.13.5 is the recorded version. `pandoc` is needed only for the optional render check in
`revision_R1/make_tables.py`.

## Paths and environment variables

The scripts never contain the authors' data. Every data, model and output location is read from an environment
variable, with the authors' original layout as the fallback (`<root>` = the authors' project folder,
`/Users/chan/Desktop/Proj_U`):

| Variable | Meaning | Used by | Original value |
|---|---|---|---|
| `PROJ_U_ROOT` | root of the authors' project folder | `src/train_uric.py`, `data/knhanes_processing.py`, `data/add_knhanes_labels.py` | parent of `05_src` (`<root>`) |
| `PROJ_U_DATA` | processed hospital development file | `src/train_uric.py`, `src/data_utils.py`; `revision_R1/cohort.py` (through `src/train_uric.py`), `calc_missing_bias.py`, `calc_followup.py` | `<root>/00_data/processed/final_urine_lab_mr_demo_v2.csv` |
| `PROJ_U_DATA_ROOT` | hospital data folder (raw first-visit urine file, nursing and checkup records) | `revision_R1/cohort.py`, `calc_followup.py`, `retired_v4/calc_nitrite.py` | `<root>/00_data` |
| `PROJ_U_ATLAS` | validated data cache of the hospital data, version 2 (parquet) | `revision_R1/cohort.py`, `build_cohort_v2.py` | `<root>/00_data/rawdata/parquet_cache/v2` |
| `PROJ_U_KNHANES_RAW` | folder with `hn16_all.sas7bdat` ... `hn24_all.sas7bdat` | `data/knhanes_processing.py`, `revision_R1/cohort.py` | `<root>/00_data/KNHANES` |
| `PROJ_U_KNHANES` | harmonised KNHANES file | `data/*.py`, `src/train_uric.py`, `revision_R1/cohort.py` | `<root>/00_data/processed/knhanes_uric_validation.csv` |
| `PROJ_U_KNHANES_PREP` | the submission's KNHANES preparation script, parsed to verify the coding M1 | `revision_R1/calc_shift_encoding.py` | `<root>/05_src/prepare_knhanes_uric.py` (now `data/knhanes_processing.py`) |
| `PROJ_U_MODELS_V5` | folder of the retrained model files (bestmodel_v5) | `revision_R1/cohort.py`, `train_v5.py` | `<root>/03_models/module_uric/bestmodel_v5` |
| `PROJ_U_MODELS` | folder of the submitted model files (bestmodel_v4) | `src/train_uric.py`, `revision_R1/cohort.py` (`train_v5.py reproduce`, `calc_main.py`) | `<root>/03_models/module_uric/bestmodel_v4` |
| `PROJ_U_RESULTS` | outputs of `src/train_uric.py` (`performance.csv`, `subgroup_analysis.csv`, `summary.json`) | `src/train_uric.py`, `revision_R1/retired_v4/calc_original.py` | `<root>/04_results/module_uric_v4` |
| `PROJ_U_SRC` | folder of the training code imported by the revision scripts | `revision_R1/cohort.py` | `<root>/05_src` (now `src/` of this repository) |
| `PROJ_U_REVISION_DIR` | manuscript folder where tables, figures and appendices are written | `revision_R1/make_*.py`, `build_manuscript.py`, `figstyle.py` | parent folder of `revision_R1` |
| `URIC_MODEL_SET` | models returned by `cohort.load_model()` by default: `v5` (retrained) or `v4` (submitted) | `revision_R1/cohort.py` | `v5` |

## Reproducing

1. **KNHANES inputs** (public data): `python data/knhanes_processing.py` then `python data/add_knhanes_labels.py`
   (`data/README.md`). The retrained models also read the raw SAS files directly to apply the coding X1.
2. **Retrained models and every analysis of the revised article** (hospital data, the validated data cache and the
   raw KNHANES files required): `revision_R1/README.md` gives the order
   `cohort.write_cache() -> build_cohort_v2.py -> train_v5.py reproduce / train -> calc_main.py -> calc_*.py ->
   make_tables.py / make_figures.py / make_supplement.py -> build_manuscript.py -> make_checklist.py` and what each
   step writes. Aggregated results of every analysis are in Multimedia Appendix 1 of the article.
3. **Submitted models** (for the record): `python src/train_uric.py` retrains the three submitted models and writes
   the model files, `performance.csv`, `subgroup_analysis.csv` and `summary.json`.

## Data and model availability

- **KNHANES** is publicly available from the Korea Disease Control and Prevention Agency
  (https://knhanes.kdca.go.kr/knhanes/main.do, registration required).
- **Hospital data**, including the validated data cache built from them, cannot be shared publicly (institutional
  privacy regulations and the Korean Personal Information Protection Act); requests for collaborative access may be
  directed to the corresponding author and are subject to IRB approval (Seoul National University Bundang Hospital
  IRB No. B-2509-994-106).
- **Fitted model files** (the retrained models and the submitted ones) are not publicly released because of the
  pending patent application; requests for access for independent validation may be made to the corresponding
  author.

## Reporting

The article is reported according to the TRIPOD+AI statement; the completed TRIPOD+AI and TRIPOD+AI for
Abstracts checklists of the revised manuscript are in [docs/tripod_ai_checklist.md](docs/tripod_ai_checklist.md).

## Citation

```bibtex
@article{jang2026uric,
  title   = {Machine Learning Models Combining Routine Urine Dipstick Results With Age, Sex, and Body Mass Index
             to Triage Serum Urate Testing: Development and External Validation Study},
  author  = {Jang, Eun Chan and Song, Pado and Kim, Joung Eun and Han, Myung Hwa and Han, Hwigyeong and Lee, Kiheon},
  journal = {JMIR Medical Informatics},
  year    = {2026},
  note    = {Manuscript 107515, revision under review}
}
```

## Licence and patent

The code is released under the [MIT License](LICENSE). ECJ, KL, PS and JEK are inventors on a pending Korean patent
application (10-2026-0130915, filed July 15, 2026; applicant: Seoul National University Hospital) for an apparatus
and method that predict hyperuricemia risk from urinalysis information, which is related to this work (see the
article's Conflicts of Interest statement). Commercial deployment of the trained models or derived products may
require a separate licence agreement (contact the corresponding author).

## Contact

- Corresponding author: Kiheon Lee, MD, PhD, keyhone2@snu.ac.kr
- First author and code maintainer: Eun Chan Jang, MPH, eunchan.j93@gmail.com

Department of Family Medicine, Seoul National University Bundang Hospital, 82 Gumi-ro 173 Beon-gil,
Bundang-gu, Seongnam-si, Gyeonggi-do 13620, Republic of Korea

## Funding

Korea Health Technology R&D Project through the Korea Health Industry Development Institute (KHIDI), funded by
the Ministry of Health & Welfare, Republic of Korea (RS-2025-02307233); Doctoral Student Research Grant Program
through the National Research Foundation of Korea (NRF), funded by the Ministry of Education (RS-2025-2540035).
The funders had no role in the study.

## Use of generative AI

As stated in the article (Acknowledgments): the authors used Claude (Anthropic) through the Claude Code agent
(Claude Opus 4.6, Opus 4.7, Opus 4.8, Opus 5.5, and Fable 5.1; March-October 2026) to draft analysis code (data
processing, model development, statistical analysis, tables, and figures) and manuscript text. The authors directed
the work, reviewed and revised the drafts, and take full responsibility for the content.
