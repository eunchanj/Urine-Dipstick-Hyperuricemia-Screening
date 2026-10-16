"""Score records with one of the submitted model files (bestmodel_v4; not distributed with this repository; see
models/README.md). Kept for the record: the revised article reports the retrained models (12 inputs, no leukocyte
esterase; revision_R1/train_v5.py), which this script does not score.

Each submitted model is a scikit-learn CalibratedClassifierCV (isotonic, 5-fold) around a soft-voting ensemble of
LightGBM, XGBoost and CatBoost (src/train_uric.py). It expects exactly the 13 input columns of
train_uric.FEATURES_13, on the development coding described in docs/feature_dictionary.md:

    ua_glucose, ua_protein, ua_ketone, ua_bilirubin, ua_blood, ua_urobilinogen, ua_nitrite, ua_wbc(s),
    ua_pH, ua_sg, age, male, bmi_unified

Usage
    python src/predict.py --model <dir>/uric_hyper_70_calibrated_voting.joblib --csv input.csv [--out preds.csv]
    python src/predict.py --model <dir>/uric_severe_9_calibrated_voting.joblib --json '{"ua_glucose": 0, ...}'
    --threshold T adds a binary column (predicted probability >= T). No default threshold is built in: the
    submitted models were trained on data with the errors listed in README.md ("History of this repository"),
    and they over-predicted in KNHANES adults (response letter, correction C5).
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_uric import FEATURES_13  # noqa: E402


def predict(model, X, threshold=None):
    """Predicted probability for every row of X (and a flag if a threshold is given)."""
    missing = [f for f in FEATURES_13 if f not in X.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    out = pd.DataFrame({"predicted_probability": model.predict_proba(X[FEATURES_13])[:, 1]})
    if threshold is not None:
        out["flagged"] = (out["predicted_probability"] >= threshold).astype(int)
        out["threshold"] = threshold
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to a *_calibrated_voting.joblib file")
    p.add_argument("--csv", help="input CSV with the 13 input columns")
    p.add_argument("--json", help="one record as a JSON object")
    p.add_argument("--out", help="output CSV (default: stdout)")
    p.add_argument("--threshold", type=float, default=None, help="optional decision threshold")
    args = p.parse_args()
    if not args.csv and not args.json:
        p.error("provide --csv or --json")
    X = pd.DataFrame([json.loads(args.json)]) if args.json else pd.read_csv(args.csv)
    pred = predict(joblib.load(args.model), X, threshold=args.threshold)
    if args.out:
        pred.to_csv(args.out, index=False)
        print(f"wrote {len(pred)} predictions to {args.out}", file=sys.stderr)
    else:
        print(pred.to_csv(index=False))


if __name__ == "__main__":
    main()
