# Revision R1: retrained models and the analyses of the revised manuscript

The scripts in this folder built the corrected development cohort, retrained the three models and produced every
number, table, figure and appendix of the revised manuscript (JMIR Medical Informatics #107515, revision R1). They are
the authors' working scripts, released as they ran, with two kinds of change: the training code is imported from this
repository's `src/` folder (and the submission's KNHANES preparation script is read from `data/`) instead of the
authors' `05_src` folder, and every data, model and output location is read from an environment variable with the
original value as the fallback (see the table in the top-level README). Nothing else was altered. Scripts that
concerned only the submitted (original) models were retired and are kept in [`retired_v4/`](retired_v4/README.md).

Every output is written into this folder (CSV files and logs, ignored by git) or into the manuscript folder
`PROJ_U_REVISION_DIR` (tables, figures, appendices); the `make_*.py` scripts turn the CSV files into tables, figures
and the appendices, so no number in the manuscript is typed by hand.

## The retrained models

| Item | Definition (code) |
|---|---|
| Inputs | 12 (`cohort.FMODEL`): 9 dipstick fields (glucose, protein, ketone, bilirubin, occult blood, urobilinogen, nitrite, pH, specific gravity), age, sex and BMI. Leukocyte esterase is not an input because KNHANES does not measure it. |
| Development cohort | `cohort.dev_v2`: each person's first urine test, with serum urate and BMI recorded on the day of that test (values from other days are not used); adults aged 19 years or older; nitrite parsed correctly from the raw text; serum urate "<1.0 mg/dL" counted as 0.5; all 12 inputs recorded. 129,535 persons. |
| Split | `cohort.split_v2`: one person-level 80/20 split shared by the three thresholds, stratified on serum urate bands (<7.0, 7.0 to <9.0, 9.0 to <10.0, >=10.0 mg/dL), seed 42. 103,628 training and 25,907 internal test persons. |
| External cohort | `cohort.ext_v2`: KNHANES 2016-2024 adults aged 19 years or older with the 12 inputs and serum urate, 49,987 persons; dipstick fields coded by the label-matched mapping X1 (`cohort.knhanes_x1`). |
| Outcomes | serum urate >=7.0 mg/dL (`uric_hyper_70`, primary), >=9.0 mg/dL (`uric_severe_9`, secondary), >=10.0 mg/dL (`uric_severe_10`, exploratory; Multimedia Appendix 1, Section S12). |
| Recipe | `train_v5.py`: `tune_and_build` of `src/train_uric.py` copied line for line. Optuna TPE sampler (seed 42), 25 trials per learner, mean AUROC of 3-fold shuffled stratified cross-validation (seed 42); LightGBM, XGBoost and CatBoost combined by equal-weight soft voting; isotonic `CalibratedClassifierCV(cv=5)`. Hyperparameters are tuned on the training persons only. |

The cohort, inputs, split, KNHANES coding and recipe were fixed before the models were fitted or evaluated.
`train_v5.py train` writes `config.json` before it fits the first model: inputs, cohort steps, split, recipe and
search spaces, sha256 of the training inputs, of the training and test identifiers and of the code, and the library
versions. It computes no internal test or KNHANES result. The authors' record (2026-10-16 23:09:07) is
[`models/bestmodel_v5/config.json`](../models/bestmodel_v5/config.json). Its code checksums are those of the authors'
files at that moment (`train_v5.py`, `cohort.py` and `05_src/run_uric_module_v4.py`). The copies released here
differ from them by the path changes described above; `cohort.py` also differs by edits made after the freeze that
change no function used in training (module docstring, `__main__`, and the helper `persons_v2`). Before the retrained
models were fitted, `train_v5.py reproduce` confirmed that the copied recipe reproduces the three submitted models
exactly (identical tuned parameters and predictions).

## What you need

| Input | Variable | Needed by | Availability |
|---|---|---|---|
| Processed hospital development file `final_urine_lab_mr_demo_v2.csv` | `PROJ_U_DATA` | `cohort.write_cache()` (step 1), `calc_missing_bias.py`, `calc_followup.py` | **not shareable** (hospital data; collaborative access on request, IRB approval required) |
| Raw first-visit urine file `urine_wide_first_visit_raw.csv` | `PROJ_U_DATA_ROOT` | `build_cohort_v2.py` (nitrite from the raw text), `calc_followup.py` | not shareable (hospital data) |
| Nursing records `medical_record/medical_record1-7.xlsx` and health-checkup records `demographic/demographic.xlsx` | `PROJ_U_DATA_ROOT` | `calc_followup.py` | not shareable (hospital data) |
| Validated data cache of the hospital data, version 2: `derived/lab_day_wide.parquet` (serum urate and exact age on the day of the urine test) and `derived/dipstick_episode_wide.parquet` (check of the recovered nitrite) | `PROJ_U_ATLAS` | `build_cohort_v2.py` | **not shareable** (built by the authors from the hospital data) |
| Harmonised KNHANES file | `PROJ_U_KNHANES` | `cohort.write_cache()` (step 1) | built from the public KDCA files with `data/knhanes_processing.py` and `data/add_knhanes_labels.py` (`data/README.md`) |
| Raw KNHANES SAS files `hn16_all.sas7bdat` ... `hn24_all.sas7bdat` (lower-case names) | `PROJ_U_KNHANES_RAW` | `build_cohort_v2.py` (X1 inputs), `calc_shift_encoding.py`, `calc_feature_tiers.py` | **public** (KDCA, after registration) |
| Retrained model files `uric_hyper_70 / uric_severe_9 / uric_severe_10_calibrated_voting.joblib` and `tune_<target>.json` (bestmodel_v5) | `PROJ_U_MODELS_V5` | written by `train_v5.py train`; read by `calc_main.py` and every `calc_*.py` (`cohort.load_model`; `calc_importance_params.py` also reads `tune_<target>.json`) | **not released** (pending patent application); available from the corresponding author on request for independent validation |
| Submitted model files (bestmodel_v4) | `PROJ_U_MODELS` | `train_v5.py reproduce`; `calc_main.py` (`original_vs_retrained.csv`, written last, for the response letter only) | not released; available on request |
| Manuscript source text `manuscript_uric_v3.0_src.md` and the submitted checklist `MA2_v2.1_extracted.md` | `PROJ_U_REVISION_DIR` | `build_manuscript.py`, `make_checklist.py` | not included (manuscript under review) |
| `pandoc` and `python-docx` | | optional render check in `make_tables.py` | optional |

The submission's KNHANES preparation script, which `calc_shift_encoding.py` parses (without running it) to verify the
coding M1, is this repository's `data/knhanes_processing.py` (`PROJ_U_KNHANES_PREP` overrides the file).

## Run order

All commands are run from this folder with the environment of `requirements.txt` (the analysis scripts cap BLAS and
OpenMP threads at 4 themselves). Scripts within one step are independent of each other unless stated.

```bash
cd revision_R1                                    # from the repository root; every command below runs in this folder
export PROJ_U_DATA=... PROJ_U_DATA_ROOT=... PROJ_U_ATLAS=... PROJ_U_KNHANES=... PROJ_U_KNHANES_RAW=... \
       PROJ_U_MODELS=... PROJ_U_MODELS_V5=...

# 1. analysis caches and the corrected cohort
python -c "import cohort; cohort.write_cache()"   # cache/dev_window.parquet, cache/knhanes.parquet (the +/-14-day window
                                                  #   of first-visit urine tests and the harmonised KNHANES file)
python build_cohort_v2.py                         # cache/dev_window_v2.parquet, cache/knhanes_v2.parquet (nitrite from the
                                                  #   raw text, same-day serum urate and BMI, KNHANES inputs under X1),
                                                  #   cohort_info_v2.csv (exclusion steps, split and event counts, checks)

# 2. training
python train_v5.py reproduce                      # the copied recipe on the submitted frame must reproduce the submitted
                                                  #   models exactly; writes train_v5_reproduce_log.txt only
python train_v5.py train                          # config.json first (the freeze), then the three model files,
                                                  #   tune_<target>.json, optuna_trials_<target>_<learner>.csv and
                                                  #   train_log.txt in PROJ_U_MODELS_V5 (refuses to overwrite model files);
                                                  #   train_v5_log.txt here; no test or KNHANES result

# 3. first evaluation after the freeze
python calc_main.py                               # main_performance.csv, preds_internal_clean.csv,
                                                  #   preds_external_corrected.csv, original_vs_retrained.csv

# 4. analyses (read the v2 cache, the model files, main_performance.csv and the preds_*.csv files of step 3)
python calc_calibration.py                        # calibration.csv, calibration_curve.csv, recalibration.csv,
                                                  #   platt_vs_isotonic.csv, preds_calibration_methods.csv,
                                                  #   refit_sigmoid_<target>.joblib
python calc_utility.py                            # discrimination.csv, operating_points.csv, splithalf_operating.csv,
                                                  #   decision_curve.csv, decision_curve_ranges.csv, utility_12F_vs_3F.csv
python calc_feature_tiers.py                      # feature_tiers.csv, tuned_per_set.csv, ensemble_vs_simple.csv,
                                                  #   feature_tiers_coding_sensitivity.csv
python calc_sex_nested.py                         # subgroups.csv, sex_difference.csv, nested_coherence.csv, continuous.csv
python calc_temporal_design.py                    # temporal.csv, knhanes_by_year.csv, survey_weighted.csv, adult_dev.csv,
                                                  #   dipstick_by_year.csv
python calc_missing_bias.py                       # missing_table.csv, mcar.csv, mice_bmi.csv, bmi_window.csv,
                                                  #   verification_bias.csv
python calc_importance_params.py                  # hyperparameters.csv, permutation_importance.csv, shap_ensemble.csv,
                                                  #   pdp_calibrated.csv, shap_by_sg.csv, shap_age_by_sex.csv, sg_univariate.csv
python calc_shift_encoding.py                     # encoding.csv, mapping_sensitivity.csv
python calc_supplement_derived.py                 # supplement_derived.csv
python calc_followup_persons.py                   # fu_int_persons.csv
python calc_followup.py                           # fu_*.csv (16 files); run AFTER calc_calibration, calc_utility,
                                                  #   calc_feature_tiers and calc_sex_nested, whose outputs it reads

# 5. tables, figures and Multimedia Appendix 1 (read the CSV files and calc_followup_log.txt; make_tables.py also
#    builds baseline_urate.csv from the v2 cache with cohort.main_cohort() when that file is missing or does not
#    match cohort_info_v2.csv, or with --rebuild-urate)
python make_tables.py [--rebuild-urate] [--no-render-check]   # ../tables.md, ../tables_appendix_ge10.md, baseline_urate.csv
python make_figures.py                                        # ../figures/Figure_1..5.pdf/.png
python make_supplement.py                                     # ../supplement/Multimedia_Appendix_1.md,
                                                              #   ../supplement/figures/FigS*, supplement_index.csv

# 6. manuscript assembly and Multimedia Appendix 2 (need the manuscript source files, not included)
python build_manuscript.py                        # ../manuscript_uric_v3.0.md from ../manuscript_uric_v3.0_src.md,
                                                  #   ../tables.md and supplement_index.csv
python make_checklist.py                          # ../supplement/Multimedia_Appendix_2_TRIPOD_AI.md; run AFTER
                                                  #   build_manuscript.py (checks the cited headings against
                                                  #   ../manuscript_uric_v3.0.md)
```

`../` is the manuscript folder (`PROJ_U_REVISION_DIR`; by default the parent of this folder, which in this repository
is the repository root, where `.gitignore` excludes these files). `make_supplement.py` stops if any of its input CSV
files is older than `cohort_info_v2.csv`, so that no output of an analysis not rerun for the retrained models can
enter the appendix. Each script's docstring states its purpose, the reviewer comment it answers, every output file
and the bootstrap and seed conventions; each `calc_*.py` also writes a `*_log.txt` with the printed numbers.

## The scripts

| Script | Purpose |
|---|---|
| `cohort.py` | Cohort definitions and data access: the corrected cohort of the retrained models (`dev_v2`, `split_v2`, `ext_v2`), the KNHANES coding X1 (`knhanes_raw`, `knhanes_x1`), the nitrite recovery (`recover_nitrite`), the caches (`write_cache`, `write_cache_v2`) and helpers for the analyses (`main_cohort`, `train_xy`, `cal_folds`, `persons_v2`). `load_model` returns the retrained models (`URIC_MODEL_SET=v4` or `model_set="v4"` for the submitted ones). The frames of the submission (`dev_submitted`, `split_submitted`, `ext_submitted`, `ext_corrected`) are kept to reproduce the submitted models and for the response letter's comparison. |
| `common.py` | Shared statistics: bootstrap and paired-bootstrap CIs, DeLong tests, exact maximum-likelihood calibration intercept and slope, O:E, scaled Brier, recalibration on repeated half-splits, decile curves, operating points, net benefit, PSI, and the fixed LightGBM configuration and grid of the comparison models. |
| `build_cohort_v2.py` | Builds the v2 caches and checks them: inputs and same-day urate constant within person, no person younger than 19 years by exact completed years, recovered nitrite equal to the validated data cache, training and test persons disjoint, KNHANES adults identical to the adult cohort of the revision; writes `cohort_info_v2.csv` (Figure 1). |
| `train_v5.py` | Retraining, modes `reproduce` and `train` (above). |
| `calc_main.py` | First evaluation after the freeze: predictions of the retrained models for the internal test persons and KNHANES adults, headline performance, and the comparison with the submitted models on the same persons (response letter only). |
| `calc_calibration.py` | Calibration (intercept, slope, O:E, Brier, curves), recalibration in KNHANES, and Platt and log-odds logistic calibrators refitted on the models' own calibration folds (reviewer AW #16). |
| `calc_utility.py` | AUPRC, operating points chosen internally and applied externally, split-half operating points, decision curves, and the 12-input vs 3-input net-benefit comparison (AW #7). |
| `calc_feature_tiers.py` | What the dipstick adds: input sets 2F, 3F, 9F, 11F and 12F with four model classes, per-set tuning, the ensemble vs simpler 12-input models on the same training persons, and dependence on the KNHANES coding (AW #5, #18). |
| `calc_sex_nested.py` | Subgroup AUROCs with within-subgroup bootstrap, the female-male difference, nested-threshold coherence and a continuous-urate model (AW #8, #12, #19; BZ #2). |
| `calc_temporal_design.py` | Temporal holdout (2010-2021 -> 2022-2025), KNHANES by survey period, survey-weighted sensitivity analysis, internal results of adults, inputs and outcome by year (AW #11). |
| `calc_missing_bias.py` | Missingness and attrition, Little's MCAR test, multiple imputation of BMI, BMI source and agreement, and verification bias by inverse probability weighting (Editor; AW #6, #9). |
| `calc_importance_params.py` | Tuned hyperparameters read from the models, whole-ensemble permutation importance, ensemble SHAP, partial dependence (specific gravity, pH, age, BMI) overall and by sex, and the crude specific-gravity association. |
| `calc_shift_encoding.py` | Input distributions and PSI, and the KNHANES coding sensitivity analysis (X1 primary; M1 as submitted, M2 as printed, M3, and X2 with specific gravity on the development grid), rebuilt from the raw SAS files (AW #10, #13, #17; BZ #3, #4). |
| `calc_supplement_derived.py` | Small tabulations for the appendix: development flow in persons, urine-blood interval, KNHANES attrition, dipstick levels by year. |
| `calc_followup_persons.py` | Persons versus records in the internal test set (equal by construction; checked). |
| `calc_followup.py` | Follow-up analyses of the authors' internal review: care setting and urine-only encounters, BMI source, operating points and net benefit by sex with reference strategies, within-sex increment, age-and-sex-only models, out-of-fold thresholds, internal period, ketone, KNHANES by year, specific-gravity restriction, KNHANES flow, isotonic outputs of 0, continuous-urate variants. |
| `make_tables.py` | Main-text Tables 1-5 and the >=10.0 mg/dL counterparts, formatted from the CSV files. |
| `make_figures.py`, `figstyle.py` | Main-text Figures 1-5 (participant flow, ROC, calibration, decision curves, subgroups) in one visual style. |
| `make_supplement.py` | Multimedia Appendix 1 (Sections S1-S12, Tables S1-S22, Figures S1-S4) and `supplement_index.csv`. |
| `build_manuscript.py` | Assembles the manuscript from its source text, resolves table and figure tokens, renumbers citations, and fails on unresolved items. |
| `make_checklist.py` | Multimedia Appendix 2: TRIPOD+AI and TRIPOD+AI for Abstracts checklists, with every cited section heading checked against the built manuscript. |

## Not included on purpose

Analysis outputs (`*.csv`, `*_log.txt`), the parquet caches, `catboost_info/`, the model files of the retrained and
submitted models, the refitted comparison models (`refit_sigmoid_*.joblib`), figures and appendices are not
committed: they are derived from hospital data, are subject to the pending patent application or are large. Every
aggregated result is in Multimedia Appendix 1 of the article. The authors' frozen copies of `cohort.py` and
`train_v5.py` (whose checksums are in `config.json`) are kept with the model files.
