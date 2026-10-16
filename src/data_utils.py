# ---------------------------------------------------------------------------
# PROVENANCE (revision R1): verbatim copy of 05_src/data_utils.py from the authors' analysis
# code base (the module imported by the training script). The only change is that
# load_data() takes its default path from the environment variable PROJ_U_DATA. The feature
# lists here are the 12 base features of the submitted models (10 dipstick fields, including
# leukocyte esterase, with age and sex); the training script adds 'bmi_unified'
# (FEATURES_13 in src/train_uric.py). The retrained models of the revised article use 12
# other inputs (revision_R1/cohort.py, FMODEL: no leukocyte esterase, with BMI). The file at
# commit 2d8b5b0 listed a feature named 'BMI' instead, which the submitted models do not use.
# ---------------------------------------------------------------------------
"""
Data loading and preprocessing utilities for Proj_U.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# === Feature Definitions ===

DIPSTICK_FEATURES = [
    'ua_glucose', 'ua_protein', 'ua_ketone', 'ua_bilirubin', 'ua_blood',
    'ua_urobilinogen', 'ua_nitrite', 'ua_wbc(s)', 'ua_pH', 'ua_sg'
]

DEMO_FEATURES = ['age', 'male']

BASE_FEATURES = DIPSTICK_FEATURES + DEMO_FEATURES  # 12 features

FEATURE_LABELS = {
    'ua_glucose': 'Urine Glucose',
    'ua_protein': 'Urine Protein',
    'ua_ketone': 'Urine Ketone',
    'ua_bilirubin': 'Urine Bilirubin',
    'ua_blood': 'Urine Occult Blood',
    'ua_urobilinogen': 'Urine Urobilinogen',
    'ua_nitrite': 'Urine Nitrite',
    'ua_wbc(s)': 'Urine Leukocyte Esterase',
    'ua_pH': 'Urine pH',
    'ua_sg': 'Urine Specific Gravity',
    'age': 'Age',
    'male': 'Sex (Male)',
}


def load_data(path=None):
    """Load the main dataset and encode basic columns.

    Revision R1: `path` defaults to the environment variable PROJ_U_DATA, otherwise to the original
    relative default '../00_data/processed/final_urine_lab_mr_demo_v2.csv'.
    """
    if path is None:
        path = os.environ.get('PROJ_U_DATA', '../00_data/processed/final_urine_lab_mr_demo_v2.csv')
    df = pd.read_csv(path, low_memory=False)
    df['male'] = (df['성별'] == 'M').astype(int)
    df['age'] = df['age_at_urine_test']
    return df


def prepare_classification_data(df, features, target_col, threshold, direction='>=',
                                test_size=0.2, random_state=42):
    """
    Prepare binary classification data.

    Parameters
    ----------
    df : DataFrame
    features : list of str
    target_col : str - continuous column name (e.g., 'Glucose', 'eGFR_CKD_EPI')
    threshold : float
    direction : str - '>=' or '<'
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test, data (filtered DataFrame)
    """
    cols = features + [target_col]
    data = df[cols].dropna().copy()
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    data = data.dropna()

    if direction == '>=':
        y = (data[target_col] >= threshold).astype(int)
    else:
        y = (data[target_col] < threshold).astype(int)

    X = data[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, data
