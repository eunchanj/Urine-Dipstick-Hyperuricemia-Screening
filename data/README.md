# Data

**No patient-level data are stored in this repository**, and `.gitignore` blocks CSV, SAS, xlsx and parquet
files. This folder holds the KNHANES preparation code of the submission and the instructions below.

## Development cohort (hospital)

- Source: laboratory information system of Seoul National University Bundang Hospital; each person's first urine
  dipstick test between January 2010 and August 2025, linked to blood tests within 14 days before or after it.
- Cohort of the retrained models (`revision_R1/cohort.py`, `dev_v2`): adults aged 19 years or older at the urine
  test, with serum urate and BMI recorded on the day of the urine test and all 9 dipstick fields and sex recorded:
  **129,535 persons** (21,124 with serum urate >=7.0 mg/dL; 2,470 >=9.0; 736 >=10.0), one record per person,
  split once into 103,628 training and 25,907 internal test persons. The exclusion steps are in Figure 1 of the
  article and in `cohort_info_v2.csv` (written by `revision_R1/build_cohort_v2.py`).
- The submitted models were trained on a record-level version of these data without an age limit (145,678 records
  of 145,656 persons; 116,542 training records per threshold).
- Availability: not publicly distributable (institutional privacy regulations and the Korean Personal
  Information Protection Act). Requests for collaborative access may be directed to the corresponding author
  (keyhone2@snu.ac.kr) and are subject to IRB approval (Seoul National University Bundang Hospital IRB
  No. B-2509-994-106).
- Files the scripts expect, named as in the authors' data folder:

  | File | Variable | Read by |
  |---|---|---|
  | `processed/final_urine_lab_mr_demo_v2.csv` (processed development file) | `PROJ_U_DATA` | `src/train_uric.py`; `revision_R1/cohort.py` (`write_cache`), `calc_missing_bias.py`, `calc_followup.py` |
  | `urine_wide_first_visit_raw.csv` (raw first-visit urine results) | `PROJ_U_DATA_ROOT` | `revision_R1/cohort.py` (nitrite read from the raw text), `calc_followup.py` |
  | `medical_record/medical_record1-7.xlsx`, `demographic/demographic.xlsx` (nursing and health-checkup records) | `PROJ_U_DATA_ROOT` | `revision_R1/calc_followup.py` |
  | validated data cache, version 2: `derived/lab_day_wide.parquet`, `derived/dipstick_episode_wide.parquet` | `PROJ_U_ATLAS` | `revision_R1/cohort.py`, `build_cohort_v2.py` (serum urate and exact age on the day of the urine test; check of the recovered nitrite) |

  The validated data cache was built by the authors from the same hospital data and is not shareable either.

## External validation cohort (KNHANES 2016-2024)

- Source: Korea National Health and Nutrition Examination Survey, Korea Disease Control and Prevention Agency
  (KDCA). Public after registration: https://knhanes.kdca.go.kr/knhanes/main.do. Serum urate has been part of
  the survey's laboratory tests since 2016.
- Raw files: the annual "ALL" SAS files for 2016-2024, read as `hn16_all.sas7bdat` ... `hn24_all.sas7bdat`
  (lower-case names, as in the authors' folder; on a case-sensitive file system rename or link the downloaded
  files) from `PROJ_U_KNHANES_RAW`, by `data/knhanes_processing.py` and by `revision_R1/cohort.py` (`knhanes_raw`).
- Processing, in this order:

  ```bash
  export PROJ_U_KNHANES_RAW=/path/to/knhanes/sas            # folder with hn16_all.sas7bdat ... hn24_all.sas7bdat
  export PROJ_U_KNHANES=/path/to/knhanes_uric_validation.csv  # harmonised output file
  python data/knhanes_processing.py     # the script of the submission (coding M1); 67,019 records, all ages
  python data/add_knhanes_labels.py     # appends uric_hyper_70, uric_severe_10, uric_severe_9 (reconstructed step)
  ```

  For the retrained models, `revision_R1/build_cohort_v2.py` (`cohort.write_cache_v2`) keeps the other columns of
  the harmonised file and replaces its dipstick fields with the coding X1, applied to the raw SAS files and linked
  by the KNHANES `id`.
- Eligibility: `knhanes_processing.py` applies no eligibility criteria. The external cohort of the retrained models
  (`revision_R1/cohort.py`, `ext_v2`) is the participants aged 19 years or older with serum urate, the 9 dipstick
  fields and measured BMI: **49,987 adults** (5,068 events at >=7.0 mg/dL, 444 at >=9.0, 103 at >=10.0). The
  submitted analysis had no age filter (54,660 participants, of whom 4,673 were aged 10-18); the repository at commit
  2d8b5b0 filtered age >=19 inside the preparation script, which the analysis code never did.

### KNHANES dipstick codings

KNHANES records semi-quantitative results as codes 0-5 (negative, trace, 1+, 2+, 3+, 4+); the development
laboratory reports negative, trace, 1+, 2+ and "3+ or more", coded 0, 0.5, 1, 2, 3 (bilirubin without a trace level;
a normal urobilinogen coded 0.5).

| KNHANES code | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **X1** (retrained models; `cohort.knhanes_x1`): glucose, protein, ketone, occult blood, bilirubin | 0 | 0.5 | 1 | 2 | 3 | 3 |
| **X1**: urobilinogen | 0.5 | 0.5 | 1 | 2 | 3 | 3 |
| **M1** (submission; `knhanes_processing.py`): glucose, protein, ketone, occult blood (`map_urine_standard`) | 0 | 0.5 | 1 | 2 | 3 | 3 |
| **M1**: bilirubin (`map_ua_bilirubin`) | 0 | 1 | 2 | 3 | 3 | 3 |
| **M1**: urobilinogen (`map_ua_urobilinogen`) | 0 | 0.2 | 0.5 | 1 | 2 | 3 |
| nitrite, both codings | 0 | 1 | 1 | 1 | 1 | 1 |

X1 assigns each KNHANES code to the development grade with the same label and was fixed before the retrained models
were fitted or evaluated. Under M1, KNHANES bilirubin "+" (code 2) entered as 2 (the development code for 2+) and a
normal urobilinogen (code 0) as 0; the submission printed the uniform map of the first row for all six fields (M2).
M1, M2, M3 and X2 are sensitivity analyses of the retrained models (`revision_R1/calc_shift_encoding.py`; Multimedia
Appendix 1, Table S4); they changed the external AUROC by less than 0.005 and the O:E ratio by 0.02 or less at
>=7.0 and >=9.0 mg/dL. pH and specific gravity are taken as recorded; KNHANES specific gravity is recorded in steps
of 0.001 up to 1.050, whereas the development data use steps of 0.005 up to 1.030. Leukocyte esterase is not measured
in KNHANES and is not an input of the retrained models (the submission entered the constant 0.321).

### Columns of the harmonised file

`year, id, age, he_glu, he_hba1c, he_crea, wt_itvex, wt_tot, kstrata, psu, he_bmi, he_wt, he_ht, male,
ua_glucose, ua_protein, ua_ketone, ua_bilirubin, ua_blood, ua_urobilinogen, ua_nitrite, ua_pH, ua_sg,
ua_wbc(s), HE_Uacid, uric_hyper_sex, uric_hyper_65, uric_severe, BMI_demo, bmi_calc` (written by
`knhanes_processing.py`), then `uric_hyper_70, uric_severe_10, uric_severe_9` (written by
`add_knhanes_labels.py`; each equals `HE_Uacid >= 7.0 / 10.0 / 9.0` with missing urate kept missing).
`bmi_unified = BMI_demo` else `bmi_calc` (10-70) is built in memory by `src/train_uric.py` and
`revision_R1/cohort.py`.

## Citation for KNHANES

Kweon S, Kim Y, Jang MJ, Kim Y, Kim K, Choi S, et al. Data resource profile: the Korea National Health and
Nutrition Examination Survey (KNHANES). *Int J Epidemiol*. 2014;43(1):69-77. doi:10.1093/ije/dyt228
