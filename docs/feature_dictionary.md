# Feature Dictionary — 13 Input Features

The hyperuricemia screening models use **13 routinely collected features**: 10 urine dipstick parameters + age + sex + BMI. No blood-based inputs are required.

| # | Variable name      | Clinical name              | Type / Scale                                                                  | Clinical rationale (hyperuricemia context)                                                                                                                            |
|--:|--------------------|----------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `ua_glucose`       | Urine glucose              | Ordinal 0 / 0.5 / 1 / 2 / 3 (negative → 3+)                                    | Reflects hyperglycemia; insulin resistance reduces renal urate excretion.                                                                                            |
| 2 | `ua_protein`       | Urine protein              | Ordinal 0–3                                                                    | Proteinuria reflects glomerular damage that impairs uric acid clearance — a recognized mechanism in chronic kidney disease-associated hyperuricemia.                  |
| 3 | `ua_ketone`        | Urine ketone               | Ordinal 0–3                                                                    | Ketonuria reflects fasting state and altered metabolism; affects renal urate handling.                                                                                |
| 4 | `ua_bilirubin`     | Urine bilirubin            | Ordinal 0–3                                                                    | Reflects hepatobiliary function; hyperuricemia frequently co-occurs with non-alcoholic fatty liver disease.                                                            |
| 5 | `ua_blood`         | Urine occult blood         | Ordinal 0–3                                                                    | Hematuria may indicate renal pathology; uric acid nephrolithiasis is a complication of hyperuricemia.                                                                  |
| 6 | `ua_urobilinogen`  | Urine urobilinogen         | Ordinal 0–3                                                                    | Reflects hepatic function and bilirubin metabolism.                                                                                                                    |
| 7 | `ua_nitrite`       | Urine nitrite              | Binary 0 / 1                                                                   | Indicates urinary tract infection / inflammation.                                                                                                                       |
| 8 | `ua_wbc(s)`        | Leukocyte esterase         | Ordinal 0–3 *(not measured in KNHANES; imputed with development mean 0.321)*    | Indicates urinary inflammation / infection.                                                                                                                            |
| 9 | `ua_pH`            | Urine pH                   | Continuous, 4.5–9.0                                                            | **Mechanistically critical**: acidic urine (low pH) promotes monosodium urate precipitation; alkaline urine increases urate solubility and excretion.                |
| 10 | `ua_sg`            | Urine specific gravity     | Continuous, 1.000–1.035                                                        | Reflects hydration status; concentrated urine (high specific gravity) is associated with reduced uric acid excretion.                                                |
| 11 | `age`              | Age                        | Continuous, years                                                              | Serum uric acid rises progressively with age in both sexes.                                                                                                            |
| 12 | `male`             | Biological sex             | Binary, 0 = female / 1 = male                                                  | Men have approximately three-fold higher prevalence of hyperuricemia than premenopausal women due to estrogen-mediated uricosuric effects.                              |
| 13 | `BMI`              | Body mass index            | Continuous, kg/m²                                                              | Higher BMI is associated with insulin resistance, which reduces renal urate excretion and increases hyperuricemia risk.                                                |

## Encoding Notes

**Semi-quantitative dipstick variables** are encoded on a standardized 0–3 scale across the development cohort:

| Code | Description |
|------|-------------|
| 0    | Negative    |
| 0.5  | Trace       |
| 1    | 1+          |
| 2    | 2+          |
| 3    | 3+          |

**KNHANES harmonization**: KNHANES uses a 0–5 ordinal scale (negative through 4+). Codes 4 and 5 are both mapped to the model value 3 (3+ ceiling) to match the development cohort coding scheme. See [data/README.md](../data/README.md) for the full variable mapping table.

**Leukocyte esterase**: not collected in KNHANES; imputed with the development cohort mean (0.321) for external evaluation only.

**BMI**: calculated from directly measured height and weight in KNHANES; in the development cohort BMI is obtained from a questionnaire-derived value with fall-back to measured height and weight.
