# Data

This directory contains data processing scripts and access instructions.
**No patient-level data are stored in this repository.**

---

## Development Cohort (Hospital EMR)

- **Source**: Seoul National University Bundang Hospital, clinical diagnostic laboratory information system, January 2010 – August 2025
- **Size**: 145,678 paired urine–blood records within ±14-day urine–serum-uric-acid matching window with complete 13-feature input
- **Availability**: **Not publicly distributed** owing to institutional privacy regulations and the Korean Personal Information Protection Act
- **Access**: Reasonable requests for collaborative access may be directed to the corresponding author (keyhone2@snu.ac.kr) and are subject to Institutional Review Board approval (Seoul National University Bundang Hospital IRB No. B-2509-994-106)

---

## External Validation Cohort (KNHANES)

- **Source**: Korea National Health and Nutrition Examination Survey, 2016–2024 (9 consecutive annual waves), Korea Disease Control and Prevention Agency (KDCA)
- **Access**: Publicly available via https://knhanes.kdca.go.kr/knhanes/main.do (Korean portal; registration required)
- **Variable manual**: https://knhanes.kdca.go.kr/knhanes/sub04/sub04_03.do
- **Eligibility used in this study**:
  - Age ≥19 years
  - Complete urine dipstick test results
  - Serum uric acid measurement available
  - Available height and weight (for BMI)
- **Eligible analytic sample**: 54,660 participants

### Processing Workflow

1. Download `HN16_ALL.sas7bdat` … `HN24_ALL.sas7bdat` from the KNHANES portal into a local directory (e.g., `data/raw/knhanes/`).
2. Run the preprocessing script:

   ```bash
   python data/knhanes_processing.py \
       --input-dir data/raw/knhanes \
       --out data/processed/knhanes_external.csv
   ```

3. This produces a harmonized CSV with the 13 model features (10 dipstick + age + sex + BMI) plus serum uric acid target column.

---

## Variable Mapping (Development ↔ KNHANES)

| Model feature       | Hospital variable        | KNHANES variable  | Scale conversion                                  |
|---------------------|--------------------------|-------------------|---------------------------------------------------|
| ua_glucose          | ua_glucose (L6103)       | HE_Uglu           | 0–5 → 0/0.5/1/2/3                                 |
| ua_protein          | ua_protein (L6104)       | HE_Upro           | 0–5 → 0/0.5/1/2/3                                 |
| ua_ketone           | ua_ketone (L6105)        | HE_Uket           | 0–5 → 0/0.5/1/2/3                                 |
| ua_bilirubin        | ua_bilirubin (L6107)     | HE_Ubil           | 0–5 → 0/0.5/1/2/3*                                |
| ua_blood            | ua_blood (L6108)         | HE_Ubld           | 0–5 → 0/0.5/1/2/3                                 |
| ua_urobilinogen     | ua_urobilinogen (L6109)  | HE_Uro            | 0–5 → 0/0.5/1/2/3                                 |
| ua_nitrite          | ua_nitrite (L6110)       | HE_Unitr          | Binary {0, 1}                                     |
| ua_wbc(s)           | ua_wbc(s) (L6111)        | *not measured*    | Imputed with development cohort mean = **0.321**  |
| ua_pH               | ua_pH (L6101)            | HE_Uph            | Continuous, no conversion                         |
| ua_sg               | ua_sg (L6102)            | HE_Usg            | Continuous, no conversion                         |
| age                 | age_at_urine_test        | age               | Continuous                                        |
| male                | gender (M=1, F=0)        | sex (1=M, 2=F)    | Binary recoding                                   |
| BMI                 | BMI (questionnaire / fallback to height & weight) | HE_wt / HE_ht | Computed: weight (kg) / height (m)² |

*For ua_bilirubin in KNHANES, only codes 0, 2, and 3 were observed (no trace values).

### KNHANES dipstick scale → Model scale

| KNHANES code | Meaning      | Model value |
|--------------|--------------|-------------|
| 0            | Negative (−) | 0           |
| 1            | Trace (±)    | 0.5         |
| 2            | 1+           | 1           |
| 3            | 2+           | 2           |
| 4            | 3+           | 3           |
| 5            | 4+           | 3 (capped)  |

### Target laboratory variable

| Target | KNHANES variable | Thresholds |
|--------|------------------|------------|
| Serum uric acid (mg/dL) | HE_Uacid | ≥7.0 (unified); ≥9.0 (markedly elevated); ≥10.0 (institutional ULT initiation) |

---

## Citations

If you use KNHANES data in derivative work, please cite:

> Kweon S, Kim Y, Jang MJ, Kim Y, Kim K, Choi S, et al. Data resource profile: the Korea National Health and Nutrition Examination Survey (KNHANES). *Int J Epidemiol*. 2014;43(1):69–77. doi:10.1093/ije/dyt228
