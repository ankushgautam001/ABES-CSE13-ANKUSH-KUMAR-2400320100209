"""Score new transactions with the trained model.

    python -m src.predict --input data/new_transactions.csv --output reports/scored.csv

Input needs columns Time, V1..V28, Amount (a Class column is ignored if present).
Output adds: anomaly_score, flagged, risk_score (0-100, from the ensemble) and
top_reasons (the 3 features furthest from normal behaviour) for analyst triage.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from . import config


def score_dataframe(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    bundle = bundle or joblib.load(config.MODEL_DIR / "fraud_detector.joblib")
    prep, best = bundle["preprocessor"], bundle["best"]
    X = prep.transform(df)
    s = bundle["detectors"][best].score(X)
    out = df.copy()
    out["anomaly_score"] = s
    out["flagged"] = (s >= bundle["thresholds"][best]).astype(int)
    out["risk_score"] = (100 * bundle["ensemble"].score(X)).round(1)
    top = np.argsort(-np.abs(X), axis=1)[:, :3]
    cols = np.array(prep.columns_)
    out["top_reasons"] = [", ".join(cols[i]) for i in top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(config.REPORT_DIR / "scored_transactions.csv"))
    a = ap.parse_args()
    res = score_dataframe(pd.read_csv(a.input))
    res.to_csv(a.output, index=False)
    print(f"Scored {len(res):,} transactions, flagged {int(res.flagged.sum()):,} -> {a.output}")


if __name__ == "__main__":
    main()
