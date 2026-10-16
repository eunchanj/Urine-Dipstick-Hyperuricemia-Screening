# ---------------------------------------------------------------------------
# PROVENANCE (revision R1, JMIR Medical Informatics #107515)
#
# This file is the KNHANES preparation script that produced the external validation inputs
# behind every reported result: a verbatim copy of 05_src/prepare_knhanes_uric.py from the
# authors' analysis code base (file dated 2026-05-13, committed 2026-05-19). Only the three
# path constants (PROJECT_ROOT, KNHANES_DIR, OUT_CSV) were changed, so that they can be set
# through environment variables; every mapping, column and output is unchanged.
#
# Mapping actually used (M1), as implemented by the functions below:
#   glucose, protein, ketone, occult blood   KNHANES codes 0,1,2,3,4,5 -> 0, 0.5, 1, 2, 3, 3
#   bilirubin                                KNHANES codes 0,1,2,3,4,5 -> 0, 1, 2, 3, 3, 3
#       KNHANES records "+" as code 2, which enters as 2 (the development code for 2+;
#       the development code for 1+ is 1). No KNHANES record carries code 1.
#   urobilinogen                             KNHANES codes 0,1,2,3,4,5 -> 0, 0.2, 0.5, 1, 2, 3
#       a normal result (code 0) enters as 0; the development laboratory codes a normal
#       urobilinogen as 0.5.
#   nitrite                                  code >= 1 -> 1, otherwise 0
#   pH, specific gravity                     numeric, unchanged
#   leukocyte esterase                       not measured in KNHANES; the constant 0.321 for
#       every record. The in-code comment below calls 0.321 the "development cohort mean";
#       the revision found that it is the mean over all 577,401 urine-blood window records,
#       whereas the analytic-cohort mean is 0.236 (Multimedia Appendix 1, Table S16). The
#       comment is left as written.
#   age                                      no age filter is applied; the file holds
#       participants of all ages. The age >= 19 restriction of the revised analysis is applied
#       downstream (revision_R1/cohort.py, ext_corrected); the submitted analysis did not
#       apply it, although its Methods stated it.
#
# The submission printed the intended rule (M2: the uniform map applied to bilirubin and
# urobilinogen as well), and the file this one replaces (data/knhanes_processing.py at commit
# 2d8b5b0) implemented M2 together with an age >= 19 filter; neither matched the code behind
# the reported results. The difference M1 vs M2 changes the inputs of 1,440 participants and
# the AUROC by at most 0.0001 (Table S12).
#
# Outcome columns: this script writes uric_hyper_sex, uric_hyper_65 and uric_severe
# (>= 9.0 mg/dL). The training script (src/train_uric.py) also reads uric_hyper_70 and
# uric_severe_10 from the output file; those columns were appended by a step that is not in
# the code base and is reconstructed in data/add_knhanes_labels.py.
#
# Input files: hn16_all.sas7bdat ... hn24_all.sas7bdat (lower-case names, as in the authors'
# data folder; the KDCA portal distributes HN16_ALL(SAS).zip etc.). Years whose file is
# missing are skipped with a message. The files are read with pandas.read_sas.
# ---------------------------------------------------------------------------
"""
Build KNHANES external validation dataset for the uric acid module.
Re-extracts directly from raw .sas7bdat files (sex coding + measurement integrity).

Output:
  - 00_data/processed/knhanes_uric_validation.csv
"""
import os
import pandas as pd
import numpy as np

# Paths (revision R1): set through environment variables; the fallbacks are the original values.
PROJECT_ROOT = os.environ.get('PROJ_U_ROOT', '/Users/chan/Desktop/Proj_U')
KNHANES_DIR = os.environ.get('PROJ_U_KNHANES_RAW', f'{PROJECT_ROOT}/00_data/KNHANES')
OUT_CSV = os.environ.get('PROJ_U_KNHANES', f'{PROJECT_ROOT}/00_data/processed/knhanes_uric_validation.csv')

# Column mappings (lowercase) — KNHANES naming
URINE_VARS = {
    'he_uph':    'ua_pH_raw',
    'he_usg':    'ua_sg_raw',
    'he_upro':   'ua_protein_raw',
    'he_uglu':   'ua_glucose_raw',
    'he_uket':   'ua_ketone_raw',
    'he_ubil':   'ua_bilirubin_raw',
    'he_ubld':   'ua_blood_raw',
    'he_uro':    'ua_urobilinogen_raw',
    'he_unitr':  'ua_nitrite_raw',
}

# ---- Mapping functions (per existing notebook 13 logic) ----
def map_urine_standard(x):
    """KNHANES 0–5 ordinal → development 0/0.5/1/2/3."""
    if pd.isna(x): return np.nan
    return {0: 0, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}.get(int(x), np.nan)

def map_ua_bilirubin(x):
    if pd.isna(x): return np.nan
    return {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}.get(int(x), np.nan)

def map_ua_urobilinogen(x):
    """Approximate KNHANES 0–5 → development normal/elevated."""
    if pd.isna(x): return np.nan
    return {0: 0, 1: 0.2, 2: 0.5, 3: 1, 4: 2, 5: 3}.get(int(x), np.nan)

def map_ua_nitrite(x):
    """Binary."""
    if pd.isna(x): return np.nan
    return 1 if int(x) >= 1 else 0


# ---- Load + concat all years ----
frames = []
for year in range(2016, 2025):
    yr2 = str(year)[-2:]
    path = f'{KNHANES_DIR}/hn{yr2}_all.sas7bdat'
    if not os.path.exists(path):
        print(f'{year}: file not found, skipping'); continue
    df = pd.read_sas(path, encoding='latin-1')
    df.columns = [c.lower() for c in df.columns]
    df['year'] = year
    frames.append(df)
    print(f'{year}: {len(df):,} rows loaded')

all_df = pd.concat(frames, ignore_index=True)
print(f'\nConcatenated: {len(all_df):,} rows × {all_df.shape[1]} cols')

# ---- Subset & rename ----
need = ['year', 'id', 'sex', 'age', 'he_uacid', 'he_glu', 'he_hba1c', 'he_crea',
        'wt_itvex', 'wt_tot', 'kstrata', 'psu',
        'he_bmi', 'he_wt', 'he_ht'] + list(URINE_VARS.keys())
have = [c for c in need if c in all_df.columns]
miss = [c for c in need if c not in all_df.columns]
if miss:
    print(f'\nMissing columns (will be filled NaN): {miss}')
    for c in miss:
        all_df[c] = np.nan

sub = all_df[need].copy()

# ---- Sex coding: KNHANES sex=1 (M), 2 (F) → male = 1/0 ----
sub['male'] = (sub['sex'] == 1).astype(int)

# ---- Urine dipstick mappings ----
sub['ua_glucose']      = sub['he_uglu'].apply(map_urine_standard)
sub['ua_protein']      = sub['he_upro'].apply(map_urine_standard)
sub['ua_ketone']       = sub['he_uket'].apply(map_urine_standard)
sub['ua_bilirubin']    = sub['he_ubil'].apply(map_ua_bilirubin)
sub['ua_blood']        = sub['he_ubld'].apply(map_urine_standard)
sub['ua_urobilinogen'] = sub['he_uro'].apply(map_ua_urobilinogen)
sub['ua_nitrite']      = sub['he_unitr'].apply(map_ua_nitrite)
sub['ua_pH']           = pd.to_numeric(sub['he_uph'], errors='coerce')
sub['ua_sg']           = pd.to_numeric(sub['he_usg'], errors='coerce')
# Leukocyte esterase: not measured in KNHANES → imputed later with development cohort mean (0.321)
sub['ua_wbc(s)'] = 0.321

# ---- Targets (uric-acid based) ----
sub['HE_Uacid'] = pd.to_numeric(sub['he_uacid'], errors='coerce')

sex_thr = np.where(sub['male'] == 1, 7.0, 6.0)
sub['uric_hyper_sex'] = np.where(
    sub['HE_Uacid'].notna(),
    (sub['HE_Uacid'] >= sex_thr).astype(float),
    np.nan)

sub['uric_hyper_65'] = np.where(
    sub['HE_Uacid'].notna(),
    (sub['HE_Uacid'] >= 6.5).astype(float),
    np.nan)

sub['uric_severe'] = np.where(
    sub['HE_Uacid'].notna(),
    (sub['HE_Uacid'] >= 9.0).astype(float),
    np.nan)

# BMI (from KNHANES bodily measurement)
sub['BMI_demo'] = pd.to_numeric(sub['he_bmi'], errors='coerce')
sub['bmi_calc'] = pd.to_numeric(sub['he_wt'], errors='coerce') / (pd.to_numeric(sub['he_ht'], errors='coerce') / 100) ** 2
sub.loc[(sub['BMI_demo'] < 10) | (sub['BMI_demo'] > 70), 'BMI_demo'] = np.nan
sub.loc[(sub['bmi_calc'] < 10) | (sub['bmi_calc'] > 70), 'bmi_calc'] = np.nan

# Drop raw he_* helper cols
sub = sub.drop(columns=list(URINE_VARS.keys()) + ['sex', 'he_uacid'])
# Keep he_glu/he_hba1c/he_crea/he_bmi/he_wt/he_ht raw in case useful

# Save
sub.to_csv(OUT_CSV, index=False)

# ---- Summary ----
print(f'\n━━━ Output: {OUT_CSV} ━━━')
print(f'Shape: {sub.shape}')
ua = sub.dropna(subset=['HE_Uacid'])
print(f'With HE_Uacid: {len(ua):,} ({len(ua)/len(sub)*100:.1f}%)')
print(f'  UA median (M=1): {ua[ua["male"]==1]["HE_Uacid"].median():.2f}')
print(f'  UA median (M=0): {ua[ua["male"]==0]["HE_Uacid"].median():.2f}')
print(f'Hyperuricemia sex-specific: {int(ua["uric_hyper_sex"].sum()):,} ({ua["uric_hyper_sex"].mean()*100:.1f}%)')
print(f'  Males:   {int(ua[ua["male"]==1]["uric_hyper_sex"].sum()):,} ({ua[ua["male"]==1]["uric_hyper_sex"].mean()*100:.1f}%)')
print(f'  Females: {int(ua[ua["male"]==0]["uric_hyper_sex"].sum()):,} ({ua[ua["male"]==0]["uric_hyper_sex"].mean()*100:.1f}%)')
print(f'Hyperuricemia ≥6.5: {int(ua["uric_hyper_65"].sum()):,} ({ua["uric_hyper_65"].mean()*100:.1f}%)')
print(f'Severe ≥9.0: {int(ua["uric_severe"].sum()):,} ({ua["uric_severe"].mean()*100:.1f}%)')

# Records with all 12 features + HE_Uacid
needed = ['ua_glucose','ua_protein','ua_ketone','ua_bilirubin','ua_blood',
          'ua_urobilinogen','ua_nitrite','ua_wbc(s)','ua_pH','ua_sg',
          'age','male','HE_Uacid']
complete = sub.dropna(subset=needed)
print(f'\nRecords with 12 features + HE_Uacid: {len(complete):,}')
