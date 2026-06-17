"""
KNHANES (Korea National Health and Nutrition Examination Survey) preprocessing
for external validation of the hyperuricemia screening models.

Inputs
------
Raw SAS files for each survey year (HN16_ALL.sas7bdat … HN24_ALL.sas7bdat),
downloaded from https://knhanes.kdca.go.kr/knhanes/main.do.

Outputs
-------
data/processed/knhanes_external.csv  — harmonized cohort with 13 model features
                                       (10 dipstick + age + sex + BMI) and
                                       serum uric acid target laboratory value.

Variable mapping (development cohort 0–3 scale ← KNHANES 0–5 scale)
-------------------------------------------------------------------
    KNHANES code  →  Model scale
    0 (negative)  →  0
    1 (trace)     →  0.5
    2 (1+)        →  1
    3 (2+)        →  2
    4 (3+)        →  3
    5 (4+)        →  3 (capped)

See `docs/feature_dictionary.md` for full variable definitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


# Development scale: 0, 0.5, 1, 2, 3
KNHANES_TO_MODEL = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 3.0}

DIPSTICK_KN_TO_MODEL = {
    "HE_Uglu":  "ua_glucose",
    "HE_Upro":  "ua_protein",
    "HE_Uket":  "ua_ketone",
    "HE_Ubil":  "ua_bilirubin",
    "HE_Ubld":  "ua_blood",
    "HE_Uro":   "ua_urobilinogen",
}

# Already harmonized in KNHANES (no scale conversion)
PASSTHROUGH = {
    "HE_Uph":   "ua_pH",
    "HE_Usg":   "ua_sg",
    "HE_Unitr": "ua_nitrite",   # already binary in KNHANES
}

DEMO = {"age": "age", "sex": "_sex_raw"}
TARGETS_RAW = ["HE_Uacid"]   # serum uric acid (mg/dL)


def load_year(path: Path) -> pd.DataFrame:
    """Load a single KNHANES SAS year file with column metadata."""
    df, _ = pyreadstat.read_sas7bdat(str(path))
    df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]
    return df


def harmonize(df: pd.DataFrame, le_mean: float = 0.321) -> pd.DataFrame:
    """Map raw KNHANES variables to the 13-feature model input vector."""
    out = pd.DataFrame(index=df.index)

    # Dipstick (0–5 → 0/0.5/1/2/3)
    for kn, model in DIPSTICK_KN_TO_MODEL.items():
        if kn in df.columns:
            out[model] = pd.to_numeric(df[kn], errors="coerce").map(KNHANES_TO_MODEL)
        else:
            out[model] = np.nan

    # Pass-through continuous / binary
    for kn, model in PASSTHROUGH.items():
        if kn in df.columns:
            out[model] = pd.to_numeric(df[kn], errors="coerce")
        else:
            out[model] = np.nan

    # Nitrite — coerce to {0, 1}
    if "ua_nitrite" in out.columns:
        out["ua_nitrite"] = (out["ua_nitrite"] >= 1).astype(float)

    # Leukocyte esterase — not measured in KNHANES; impute with development mean
    out["ua_wbc(s)"] = le_mean

    # Demographics — KNHANES sex coding: 1=male, 2=female
    out["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    sex_raw = pd.to_numeric(df.get("sex", np.nan), errors="coerce")
    out["male"] = (sex_raw == 1).astype(float)

    # Target labels (passed through for external evaluation)
    for col in TARGETS_RAW:
        out[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")


    # BMI — computed from directly measured height/weight in KNHANES
    height_cm = pd.to_numeric(df.get("HE_ht", np.nan), errors="coerce")
    weight_kg = pd.to_numeric(df.get("HE_wt", np.nan), errors="coerce")
    out["BMI"] = weight_kg / ((height_cm / 100.0) ** 2)

    # Survey year (for temporal subgroup analysis)
    out["year"] = pd.to_numeric(df.get("year", np.nan), errors="coerce")

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True,
                   help="Directory containing HN16_ALL.sas7bdat … HN24_ALL.sas7bdat")
    p.add_argument("--out", default="data/processed/knhanes_external.csv",
                   help="Output CSV path")
    p.add_argument("--le-mean", type=float, default=0.321,
                   help="Leukocyte esterase imputation value (development cohort mean)")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("HN[12][0-9]_ALL.sas7bdat"))
    if not files:
        raise FileNotFoundError(f"No HNYY_ALL.sas7bdat files found in {in_dir}")

    frames = []
    for f in files:
        print(f"Loading {f.name}...")
        df_raw = load_year(f)
        df_harm = harmonize(df_raw, le_mean=args.le_mean)
        frames.append(df_harm)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["age"] >= 19].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"\nWrote {len(df_all):,} records → {out_path}")
    print(f"  With serum UA label: {df_all['HE_Uacid'].notna().sum():,}")
    print(f"  With BMI:            {df_all['BMI'].notna().sum():,}")


if __name__ == "__main__":
    main()
