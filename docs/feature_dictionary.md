# Feature dictionary: the 12 model inputs

The retrained models of the revised article take exactly these 12 columns, in this order (`FMODEL` in
`revision_R1/cohort.py`). Everything below is read from the code in this repository (`revision_R1/cohort.py`,
`src/data_utils.py`, `src/train_uric.py`) and from the revised manuscript (Methods, Input Features; Multimedia
Appendix 1, Table S2).

| # | Column | Meaning | Development coding (hospital laboratory) | KNHANES source and coding X1 (primary) |
|--:|---|---|---|---|
| 1 | `ua_glucose` | urine glucose | 0 negative, 0.5 trace, 1 (1+), 2 (2+), 3 (3+ or more) | `HE_Uglu` codes 0-5 -> 0, 0.5, 1, 2, 3, 3 |
| 2 | `ua_protein` | urine protein | as above | `HE_Upro` codes 0-5 -> 0, 0.5, 1, 2, 3, 3 |
| 3 | `ua_ketone` | urine ketone | as above | `HE_Uket` codes 0-5 -> 0, 0.5, 1, 2, 3, 3 |
| 4 | `ua_bilirubin` | urine bilirubin | 0, 1, 2, 3 (the development laboratory reports no trace level) | `HE_Ubil` codes 0-5 -> 0, 0.5, 1, 2, 3, 3 (KNHANES "+", code 2, is coded 1, the development 1+) |
| 5 | `ua_blood` | urine occult blood | 0, 0.5, 1, 2, 3 | `HE_Ubld` codes 0-5 -> 0, 0.5, 1, 2, 3, 3 |
| 6 | `ua_urobilinogen` | urine urobilinogen | normal is coded 0.5; 1, 2, 3 above normal | `HE_Uro` codes 0-5 -> 0.5, 0.5, 1, 2, 3, 3 (a normal or trace result is coded 0.5, the development normal) |
| 7 | `ua_nitrite` | urine nitrite | 0 negative, 1 positive (read from the raw text: "Positive", "POSITIVE" and "+" are 1) | `HE_Unitr` code >= 1 -> 1, otherwise 0 |
| 8 | `ua_pH` | urine pH | numeric (5-9 in steps of 0.5) | `HE_Uph`, as recorded |
| 9 | `ua_sg` | urine specific gravity | numeric, recorded in steps of 0.005 up to 1.030 | `HE_Usg`, as recorded (steps of 0.001 up to 1.050) |
| 10 | `age` | age at the urine test, years | whole years (days since birth / 365.25, rounded down); adults aged 19 years or older | KNHANES `age` (top-coded at 80); adults aged 19 years or older |
| 11 | `male` | sex | 1 male, 0 female | KNHANES `sex` 1 -> 1, 2 -> 0 |
| 12 | `bmi_unified` | body mass index, kg/m2 | recorded on the day of the urine test: the health-checkup BMI when the checkup was on that day, otherwise weight/height2 from the nursing record of that day (height 50-220 cm, weight 10-300 kg); values outside 10-70 set to missing | `HE_BMI`, otherwise `HE_wt / (HE_ht/100)^2`; values outside 10-70 set to missing |

## Notes

- **X1 was fixed before the models were fitted or evaluated.** Each KNHANES code is assigned to the development grade
  with the same label, and 4+ is merged into 3 because the development laboratory's top category is "3+ or more".
  `cohort.knhanes_x1` applies it to the raw SAS files (`cohort.knhanes_raw`).
- **Alternative codings are sensitivity analyses** (`revision_R1/calc_shift_encoding.py`; Multimedia Appendix 1,
  Tables S2 panel B and S4): M1, the coding of the submission as implemented in `data/knhanes_processing.py`
  (bilirubin codes 0-5 -> 0, 1, 2, 3, 3, 3; urobilinogen codes 0-5 -> 0, 0.2, 0.5, 1, 2, 3); M2, as X1 but a normal
  urobilinogen coded 0 (the uniform map printed in the submission); M3, as M1 but a normal urobilinogen coded 0.5;
  X2, X1 with specific gravity rounded to the development grid (steps of 0.005, capped at 1.030). No crosswalk or
  paired-strip study validates any of these codings.
- **Tree models use only the order of the values**, so the unequal numeric intervals of the codes do not affect the
  ensembles; they do affect the logistic comparison models, whose dependence on the coding is reported
  (`revision_R1/calc_feature_tiers.py`).
- **Leukocyte esterase** (`ua_wbc(s)`) is reported by the development laboratory but is not an input, because KNHANES
  does not measure it.
- **Outcome.** Serum urate in mg/dL measured on the day of the urine test (development: the earliest result of that
  day, results reported at the same time averaged, "<1.0 mg/dL" counted as 0.5; KNHANES: `HE_Uacid`). Labels:
  >=7.0 (primary), >=9.0 (secondary), >=10.0 mg/dL (exploratory).
- **Submitted models.** The submitted models (`src/train_uric.py`, `FEATURES_13`) took 13 inputs: the 12 above plus
  leukocyte esterase, for which a constant (0.321) was entered in KNHANES. Their development BMI came from the
  health-checkup record nearest in date, without a time window, and their KNHANES inputs were coded with M1.
