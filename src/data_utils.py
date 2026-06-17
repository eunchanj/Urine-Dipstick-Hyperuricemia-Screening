"""
Data loading and preprocessing utilities for Proj_U.

13-feature configuration: 10 urine dipstick parameters + age + sex + BMI.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# === Feature Definitions ===

DIPSTICK_FEATURES = [
    'ua_glucose', 'ua_protein', 'ua_ketone', 'ua_bilirubin', 'ua_blood',
    'ua_urobilinogen', 'ua_nitrite', 'ua_wbc(s)', 'ua_pH', 'ua_sg'
]

DEMO_FEATURES = ['age', 'male', 'BMI']

BASE_FEATURES = DIPSTICK_FEATURES + DEMO_FEATURES  # 13 features

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
    'BMI': 'Body Mass Index',
}


def load_data(path='../00_data/processed/final_urine_lab_mr_demo_v2.csv'):
    """Load the main dataset and encode basic columns."""
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
    target_col : str - continuous column name (e.g., 'ALT', 'serum_uric_acid')
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
