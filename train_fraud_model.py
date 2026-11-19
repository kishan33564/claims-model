"""Tabular fraud-risk model on the Kaggle "Vehicle Insurance Claim Fraud Detection" data.

  kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p raw/fraud --unzip
  python train_fraud_model.py --csv raw/fraud/fraud_oracle.csv

Saves models/fraud.joblib. Probability -> risk band: <0.3 low, <0.6 medium, else high.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "FraudFound_P"
DROP = ["PolicyNumber", "RepNumber"]


def risk_band(p: float) -> str:
    return "low" if p < 0.3 else "medium" if p < 0.6 else "high"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df.pop(TARGET)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", categorical_features=list(range(len(cat_cols))))
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(y_te, prob), 3))
    print(classification_report(y_te, (prob >= 0.5).astype(int)))

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "columns": df.columns.tolist()}, "models/fraud.joblib")
    print("saved models/fraud.joblib")


if __name__ == "__main__":
    main()
