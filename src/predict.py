"""
Inference example for the trained calibrated voting ensemble.

Usage
-----
Single record from JSON:
    python predict.py --model models/uric70_calibrated_voting.joblib --json '{"ua_glucose":0,"ua_protein":0,"ua_ketone":0,"ua_bilirubin":0,"ua_blood":0,"ua_urobilinogen":0,"ua_nitrite":0,"ua_wbc(s)":0.321,"ua_pH":5.5,"ua_sg":1.020,"age":55,"male":1,"BMI":26.0}'

Batch from CSV:
    python predict.py --model models/uric70_calibrated_voting.joblib --csv input.csv --out predictions.csv

Decision threshold options
--------------------------
Default thresholds are the Youden-optimized thresholds determined on the
internal test set (see manuscript Table 2):

    uric70   →  0.130
    uric90   →  0.020
    uric100  →  0.005

Use --threshold to override (e.g., for high-sensitivity screening).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from data_utils import BASE_FEATURES


YOUDEN_THRESHOLDS = {
    "uric70":  0.130,
    "uric90":  0.020,
    "uric100": 0.005,
}


def _infer_target(model_path: str) -> str | None:
    stem = Path(model_path).stem.lower()
    for key in YOUDEN_THRESHOLDS:
        if key in stem:
            return key
    return None


def predict(model, X: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Return probability + binary call (if threshold provided)."""
    missing = [f for f in BASE_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    proba = model.predict_proba(X[BASE_FEATURES])[:, 1]
    out = pd.DataFrame({"predicted_probability": proba})
    if threshold is not None:
        out["screen_positive"] = (proba >= threshold).astype(int)
        out["threshold"] = threshold
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to calibrated joblib artifact")
    p.add_argument("--csv", help="Input CSV with the 13 required features")
    p.add_argument("--json", help="Single record as JSON string")
    p.add_argument("--out", help="Output CSV (default: stdout)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold (default: target-specific Youden)")
    args = p.parse_args()

    if not args.csv and not args.json:
        p.error("Provide either --csv or --json")

    if args.json:
        X = pd.DataFrame([json.loads(args.json)])
    else:
        X = pd.read_csv(args.csv)

    model = joblib.load(args.model)
    threshold = args.threshold
    if threshold is None:
        inferred = _infer_target(args.model)
        if inferred:
            threshold = YOUDEN_THRESHOLDS[inferred]
            print(f"[info] Using Youden threshold for {inferred}: {threshold}",
                  file=sys.stderr)

    pred = predict(model, X, threshold=threshold)
    if args.out:
        pred.to_csv(args.out, index=False)
        print(f"Wrote {len(pred)} predictions → {args.out}", file=sys.stderr)
    else:
        print(pred.to_csv(index=False))


if __name__ == "__main__":
    main()
