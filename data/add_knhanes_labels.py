"""Append the outcome columns that src/train_uric.py reads from the harmonised KNHANES file.

RECONSTRUCTED STEP (revision R1, JMIR Medical Informatics #107515).

data/knhanes_processing.py writes the harmonised KNHANES file with the outcome columns uric_hyper_sex,
uric_hyper_65 and uric_severe (serum urate >= 9.0 mg/dL). The external evaluation in src/train_uric.py
(TARGETS[...]['knhanes_binary']) also reads uric_hyper_70 and uric_severe_10 from that file, and the authors'
copy of the file holds a third extra column, uric_severe_9. No script in the authors' code base writes these
three columns: uric_hyper_70 was present before 2026-05-28 and uric_severe_10 and uric_severe_9 were appended
on or before 2026-05-28 (commit 3b87fb3 of the analysis code base), by a step that was not found (presumably
an interactive command). In the authors' file each column equals the indicator HE_Uacid >= 7.0 / 10.0 / 9.0
mg/dL with missing urate kept missing (checked row by row against all 67,019 records on 2026-10-07); that is
what this script writes, in the same column order. The revision analyses (revision_R1/cohort.py) do not use
these columns: they derive every label from HE_Uacid.

Usage
    python data/add_knhanes_labels.py                 rewrites $PROJ_U_KNHANES in place
    python data/add_knhanes_labels.py IN.csv OUT.csv
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get('PROJ_U_ROOT', '/Users/chan/Desktop/Proj_U')
DEFAULT = os.environ.get('PROJ_U_KNHANES', f'{PROJECT_ROOT}/00_data/processed/knhanes_uric_validation.csv')
LABELS = (('uric_hyper_70', 7.0), ('uric_severe_10', 10.0), ('uric_severe_9', 9.0))


def add_labels(df):
    u = pd.to_numeric(df['HE_Uacid'], errors='coerce')
    for col, thr in LABELS:
        df[col] = np.where(u.notna(), (u >= thr).astype(float), np.nan)
    return df


def main(argv):
    src = argv[1] if len(argv) > 1 else DEFAULT
    dst = argv[2] if len(argv) > 2 else src
    df = add_labels(pd.read_csv(src, low_memory=False))
    df.to_csv(dst, index=False)
    print(f'{dst}: {len(df):,} rows; columns written: ' + ', '.join(c for c, _ in LABELS))


if __name__ == '__main__':
    main(sys.argv)
