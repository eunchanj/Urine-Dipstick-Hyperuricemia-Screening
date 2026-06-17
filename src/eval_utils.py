"""
Evaluation and metrics utilities for Proj_U.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    brier_score_loss, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def evaluate_binary(y_true, y_pred_proba, threshold=None):
    """
    Evaluate binary classification with comprehensive metrics.
    If threshold is None, uses Youden's J to find optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    if threshold is None:
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        threshold = thresholds[optimal_idx]

    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'AUC': auc,
        'AP': average_precision_score(y_true, y_pred_proba),
        'Brier': brier_score_loss(y_true, y_pred_proba),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(y_true, y_pred),
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'threshold': threshold,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def bootstrap_metrics(y_true, y_pred_proba, n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Calculate metrics with bootstrap confidence intervals.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Point estimates
    results = evaluate_binary(y_true, y_pred_proba)

    # Bootstrap for AUC CI
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))

    alpha = (1 - ci) / 2
    results['AUC_CI'] = (np.percentile(boot_aucs, alpha * 100),
                         np.percentile(boot_aucs, (1 - alpha) * 100))
    return results


def subgroup_analysis(X, y_true, y_pred_proba, age_col='age', sex_col='male', min_n=100):
    """
    Perform subgroup analysis by age and sex.

    Returns DataFrame with subgroup performance.
    """
    df = pd.DataFrame(X).copy()
    df['y_true'] = np.asarray(y_true)
    df['y_pred_proba'] = np.asarray(y_pred_proba)

    results = []

    # Overall
    results.append(_eval_subgroup(df, 'Overall'))

    # Sex
    for val, name in [(1, 'Male'), (0, 'Female')]:
        sub = df[df[sex_col] == val]
        results.append(_eval_subgroup(sub, name, min_n))

    # Age groups
    for lo, hi, name in [(19, 39, '19-39'), (40, 59, '40-59'), (60, 79, '60-79'), (80, 120, '80+')]:
        sub = df[(df[age_col] >= lo) & (df[age_col] <= hi)]
        results.append(_eval_subgroup(sub, name, min_n))

    return pd.DataFrame([r for r in results if r is not None])


def _eval_subgroup(df, name, min_n=0):
    y = df['y_true'].values
    yp = df['y_pred_proba'].values
    if len(y) < min_n or len(np.unique(y)) < 2:
        return None
    return {
        'Subgroup': name,
        'N': len(y),
        'N_Positive': int(y.sum()),
        'Prevalence_%': y.mean() * 100,
        'AUC': roc_auc_score(y, yp),
        'AP': average_precision_score(y, yp),
    }


# ---------------------------------------------------------------------------
# Demographic / Baseline Characteristics Table
# ---------------------------------------------------------------------------

from scipy import stats


def demographic_table(data, features, target_col, target_label='Target',
                      split_col=None, split_labels=None,
                      continuous_cols=None, binary_cols=None,
                      extra_continuous=None, feature_labels=None):
    """
    Generate Table 1 (Baseline Characteristics).

    Urine dipstick variables (ordinal 0-3/0-4) are treated as continuous
    (mean ± SD, median [IQR]) since they are ordinal scales.
    Only truly binary variables (0/1 with exactly 2 unique values like
    'male', 'ua_nitrite') are shown as N (%).

    Parameters
    ----------
    data : DataFrame
    features : list of str — feature columns to describe
    target_col : str — binary target column (0/1)
    target_label : str — display name for the target
    split_col : str or None — e.g. 'split' column with 'Train'/'Test' values
    split_labels : list or None — e.g. ['Train', 'Test']
    continuous_cols : list or None — force these as continuous
    binary_cols : list or None — force these as binary N(%)
    extra_continuous : list or None — additional continuous columns not in features
        (e.g., the raw target variable like eGFR_CKD_EPI)
    feature_labels : dict or None — {col_name: display_name}
    """
    df = data.copy()
    if feature_labels is None:
        feature_labels = {}

    # Auto-detect: binary (exactly {0,1}) vs continuous (everything else)
    if continuous_cols is None or binary_cols is None:
        _cont, _bin = [], []
        for col in features:
            if col not in df.columns:
                continue
            unique = set(df[col].dropna().unique())
            if unique == {0, 1} or unique == {0} or unique == {1}:
                _bin.append(col)
            else:
                _cont.append(col)
        if continuous_cols is None:
            continuous_cols = _cont
        if binary_cols is None:
            binary_cols = _bin

    # Add extra continuous columns (e.g., raw eGFR)
    if extra_continuous:
        continuous_cols = list(extra_continuous) + continuous_cols

    # Define groups
    if split_col and split_labels:
        groups = {lbl: df[df[split_col] == lbl] for lbl in split_labels}
    else:
        groups = {
            'Overall': df,
            f'{target_label} = 0': df[df[target_col] == 0],
            f'{target_label} = 1': df[df[target_col] == 1],
        }

    rows = []

    # N
    row = {'Variable': 'N'}
    for gname, gdf in groups.items():
        row[gname] = f"{len(gdf):,}"
    rows.append(row)

    # Continuous variables: mean ± SD (median [IQR])
    for col in continuous_cols:
        if col not in df.columns:
            continue
        label = feature_labels.get(col, col)
        row = {'Variable': label}
        for gname, gdf in groups.items():
            s = gdf[col].dropna()
            mean, sd = s.mean(), s.std()
            med = s.median()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            row[gname] = f"{mean:.2f} ± {sd:.2f} ({med:.2f} [{q1:.2f}-{q3:.2f}])"
        # p-value (target groups only)
        if f'{target_label} = 0' in groups and f'{target_label} = 1' in groups:
            g0 = groups[f'{target_label} = 0'][col].dropna()
            g1 = groups[f'{target_label} = 1'][col].dropna()
            if len(g0) > 0 and len(g1) > 0:
                _, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
                row['p-value'] = _fmt_pval(p)
        rows.append(row)

    # Binary variables: N (%)
    for col in binary_cols:
        if col not in df.columns:
            continue
        label = feature_labels.get(col, col)
        row = {'Variable': label}
        for gname, gdf in groups.items():
            s = gdf[col].dropna()
            n_pos = int((s == 1).sum())
            pct = n_pos / len(s) * 100 if len(s) > 0 else 0
            row[gname] = f"{n_pos:,} ({pct:.1f}%)"
        # p-value
        if f'{target_label} = 0' in groups and f'{target_label} = 1' in groups:
            ct = pd.crosstab(df[col].dropna(), df.loc[df[col].notna(), target_col])
            if ct.shape[0] >= 2 and ct.shape[1] == 2:
                _, p, _, _ = stats.chi2_contingency(ct)
                row['p-value'] = _fmt_pval(p)
        rows.append(row)

    return pd.DataFrame(rows)


def _fmt_pval(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def demographic_table_train_test(X_train, X_test, y_train, y_test,
                                 features, target_label='Target',
                                 continuous_cols=None, binary_cols=None,
                                 extra_continuous=None, feature_labels=None):
    """
    Generate demographic table comparing Train vs Test sets,
    plus target-positive vs target-negative within each.

    Returns
    -------
    dict with keys: 'train_test' (Train vs Test comparison),
                    'by_target' (Overall positive vs negative)
    """
    df_train = pd.DataFrame(X_train, columns=features).copy()
    df_train['_target'] = y_train.values if hasattr(y_train, 'values') else y_train
    df_train['_split'] = 'Train'

    df_test = pd.DataFrame(X_test, columns=features).copy()
    df_test['_target'] = y_test.values if hasattr(y_test, 'values') else y_test
    df_test['_split'] = 'Test'

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    common = dict(continuous_cols=continuous_cols, binary_cols=binary_cols,
                  extra_continuous=extra_continuous, feature_labels=feature_labels)

    # Train vs Test
    table_split = demographic_table(
        df_all, features, '_target', target_label=target_label,
        split_col='_split', split_labels=['Train', 'Test'], **common,
    )

    # Target positive vs negative (overall)
    table_target = demographic_table(
        df_all, features, '_target', target_label=target_label, **common,
    )

    return {'train_test': table_split, 'by_target': table_target}


# ---------------------------------------------------------------------------
# Clinical-style Table 1 (paper-standard format)
# ---------------------------------------------------------------------------
# Matches format: Table1_baseline.md from module_dm submission
#   - Dipstick: negative n(%) / trace or above n(%)
#   - pH, SG, Age: mean (SD)
#   - Sex: n (%)
#   - Outcome prevalence: n/N (%)
#   - Section headers, footnotes, abbreviations

# Dipstick columns and their "negative" threshold (value == 0 means negative)
DIPSTICK_SPECS = {
    'ua_glucose':      ('Glucose',              0),
    'ua_protein':      ('Protein',              0),
    'ua_ketone':       ('Ketone',               0),
    'ua_bilirubin':    ('Bilirubin',            0),
    'ua_blood':        ('Blood',                0),
    'ua_urobilinogen': ('Urobilinogen',         None),  # special: normal=0.5
    'ua_nitrite':      ('Nitrite',              0),
    'ua_wbc(s)':       ('WBC (leukocytes)',     0),
}


def clinical_baseline_table(cohorts, outcomes=None, footnotes=None, abbreviations=None):
    """
    Generate a clinical-standard Table 1 (Baseline Characteristics).

    Parameters
    ----------
    cohorts : list of dict, each with:
        - 'name': str — column header (e.g., 'Development Cohort')
        - 'data': DataFrame — must contain BASE_FEATURES columns
        - 'N': int or None — override N (otherwise len(data))
        - 'outcome_data': DataFrame or None — for outcome prevalence
          (may have different N due to missing lab values)
    outcomes : list of dict or None, each with:
        - 'label': str — e.g., 'Fasting glucose >= 126 mg/dL'
        - 'col': str — column name in outcome_data (binary 0/1)
        - 'available': list of bool — per cohort, whether data is available
    footnotes : list of str or None
    abbreviations : str or None

    Returns
    -------
    str — Markdown-formatted table
    """
    col_headers = [c['name'] for c in cohorts]
    header = '| Variable | ' + ' | '.join(col_headers) + ' |'
    separator = '|---|' + '|'.join(['---'] * len(cohorts)) + '|'

    lines = [
        '# Table 1. Baseline Characteristics',
        '',
        header,
        separator,
    ]

    # --- Demographics ---
    lines.append('| **Demographics** | ' + ' | '.join([''] * len(cohorts)) + ' |')

    # Age
    vals = []
    for c in cohorts:
        df = c['data']
        if 'age' in df.columns:
            s = df['age'].dropna()
            vals.append(f"{s.mean():.2f} ({s.std():.2f})")
        else:
            vals.append('N/A')
    lines.append('| Age, years, mean (SD) | ' + ' | '.join(vals) + ' |')

    # Male sex
    vals = []
    for c in cohorts:
        df = c['data']
        if 'male' in df.columns:
            s = df['male'].dropna()
            n = int(s.sum())
            pct = n / len(s) * 100
            vals.append(f"{n:,} ({pct:.1f})")
        else:
            vals.append('N/A')
    lines.append('| Male sex, n (%) | ' + ' | '.join(vals) + ' |')

    # --- Urine Dipstick Parameters ---
    lines.append('| **Urine Dipstick Parameters** | ' + ' | '.join([''] * len(cohorts)) + ' |')

    for col, (display_name, neg_val) in DIPSTICK_SPECS.items():
        if col == 'ua_urobilinogen':
            # Special: normal = 0.5, elevated >= 1.0
            vals_normal = []
            vals_elevated = []
            for c in cohorts:
                df = c['data']
                if col not in df.columns:
                    vals_normal.append('N/A')
                    vals_elevated.append('N/A')
                    continue
                s = df[col].dropna()
                n_total = len(s)
                n_normal = int((s <= 0.5).sum())
                n_elevated = int((s >= 1.0).sum())
                vals_normal.append(f"{n_normal:,} ({n_normal/n_total*100:.1f})")
                vals_elevated.append(f"{n_elevated:,} ({n_elevated/n_total*100:.1f})")
            lines.append(f'| {display_name}, normal (0.5), n (%) | ' + ' | '.join(vals_normal) + ' |')
            lines.append(f'| {display_name}, elevated (>=1.0), n (%) | ' + ' | '.join(vals_elevated) + ' |')

        elif col == 'ua_nitrite':
            # Binary: negative (0) vs positive (1)
            vals_neg = []
            vals_pos = []
            for c in cohorts:
                df = c['data']
                if col not in df.columns:
                    vals_neg.append('N/A')
                    vals_pos.append('N/A')
                    continue
                s = df[col].dropna()
                n_total = len(s)
                n_neg = int((s == 0).sum())
                n_pos = int((s > 0).sum())
                vals_neg.append(f"{n_neg:,} ({n_neg/n_total*100:.1f})")
                vals_pos.append(f"{n_pos:,} ({n_pos/n_total*100:.1f})")
            lines.append(f'| {display_name}, negative, n (%) | ' + ' | '.join(vals_neg) + ' |')
            lines.append(f'| {display_name}, positive, n (%) | ' + ' | '.join(vals_pos) + ' |')

        elif col in ('ua_pH', 'ua_sg'):
            # Continuous: mean (SD)
            vals = []
            for c in cohorts:
                df = c['data']
                if col not in df.columns:
                    vals.append('N/A')
                    continue
                s = df[col].dropna()
                if col == 'ua_sg':
                    vals.append(f"{s.mean():.3f} ({s.std():.3f})")
                else:
                    vals.append(f"{s.mean():.2f} ({s.std():.2f})")
            label = 'pH, mean (SD)' if col == 'ua_pH' else 'Specific gravity, mean (SD)'
            lines.append(f'| {label} | ' + ' | '.join(vals) + ' |')

        else:
            # Standard dipstick: negative vs trace or above
            vals_neg = []
            vals_pos = []
            for c in cohorts:
                df = c['data']
                if col not in df.columns:
                    vals_neg.append('N/A')
                    vals_pos.append('N/A')
                    continue
                s = df[col].dropna()
                n_total = len(s)
                n_neg = int((s == neg_val).sum())
                n_pos = n_total - n_neg
                vals_neg.append(f"{n_neg:,} ({n_neg/n_total*100:.1f})")
                vals_pos.append(f"{n_pos:,} ({n_pos/n_total*100:.1f})")
            lines.append(f'| {display_name}, negative, n (%) | ' + ' | '.join(vals_neg) + ' |')
            lines.append(f'| {display_name}, trace or above, n (%) | ' + ' | '.join(vals_pos) + ' |')

    # --- Outcome Prevalence ---
    if outcomes:
        lines.append('| **Outcome Prevalence*** | ' + ' | '.join([''] * len(cohorts)) + ' |')
        for outcome in outcomes:
            vals = []
            for i, c in enumerate(cohorts):
                if not outcome.get('available', [True] * len(cohorts))[i]:
                    vals.append('N/A')
                    continue
                odata = c.get('outcome_data', c['data'])
                col_name = outcome['col'] if isinstance(outcome['col'], str) else outcome['col'][i]
                if col_name not in odata.columns:
                    vals.append('N/A')
                    continue
                s = odata[col_name].dropna()
                n_total = len(s)
                n_pos = int(s.sum())
                pct = n_pos / n_total * 100 if n_total > 0 else 0
                vals.append(f"{n_pos:,}/{n_total:,} ({pct:.1f})")
            lines.append(f'| {outcome["label"]}, n (%) | ' + ' | '.join(vals) + ' |')

    # Footnotes
    if footnotes:
        lines.append('')
        for fn in footnotes:
            lines.append(fn)

    # Abbreviations
    if abbreviations:
        lines.append('')
        lines.append(abbreviations)

    return '\n'.join(lines)
