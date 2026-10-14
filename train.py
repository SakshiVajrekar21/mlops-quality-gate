from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/scikit-learn/scikit-learn/"
    "main/sklearn/datasets/data/breast_cancer.csv"
)
TARGET_NAME = "diagnosis"
RANDOM_STATE = 42
TEST_SIZE = 0.20
IMPROVEMENT_MARGIN = 0.05
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def load_dataset() -> pd.DataFrame:
    """Download the small public CSV and validate its schema."""
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            raw = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download dataset from {DATA_URL}: {exc}") from exc

    # The first line is dataset metadata: 569,30,malignant,benign.
    from io import BytesIO
    df = pd.read_csv(BytesIO(raw), header=None, skiprows=1)

    expected_feature_count = 30
    if df.shape[1] != expected_feature_count + 1:
        raise ValueError(
            f"Dataset schema invalid: expected 30 features and 1 target, "
            f"but found {df.shape[1] - 1} features and 1 target."
        )

    feature_names = [f"feature_{i:02d}" for i in range(1, expected_feature_count + 1)]
    df.columns = feature_names + [TARGET_NAME]

    missing_features = [c for c in feature_names if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_NAME}")
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # Source labels: 0 = malignant, 1 = benign. Make malignant the positive class.
    df[TARGET_NAME] = (df[TARGET_NAME].astype(int) == 0).astype(int)
    return df


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    feature_names = [c for c in df.columns if c != TARGET_NAME]
    X = df[feature_names]
    y = df[TARGET_NAME]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_valid)
    baseline_f1 = f1_score(y_valid, baseline_predictions, zero_division=0)

    # FAILURE A DEMO: temporarily replace this candidate with
    # DummyClassifier(strategy="most_frequent"). Restore this line afterward.
    
    
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    candidate.fit(X_train, y_train)
    candidate_predictions = candidate.predict(X_valid)
    candidate_f1 = f1_score(y_valid, candidate_predictions, zero_division=0)

    required_f1 = baseline_f1 + IMPROVEMENT_MARGIN
    gate_passed = candidate_f1 >= required_f1

    metrics = {
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "dataset_source": DATA_URL,
        "target": TARGET_NAME,
        "positive_class": "malignant",
        "task": "binary classification",
        "metric": "F1-score (malignant class)",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "baseline_model": "DummyClassifier(strategy='most_frequent')",
        "baseline_f1": round(float(baseline_f1), 6),
        "candidate_model": "StandardScaler + LogisticRegression",
        "candidate_f1": round(float(candidate_f1), 6),
        "improvement_margin": IMPROVEMENT_MARGIN,
        "required_f1": round(float(required_f1), 6),
        "gate_passed": bool(gate_passed),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "feature_columns": feature_names,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if not gate_passed:
        raise SystemExit(
            f"QUALITY GATE FAILED: candidate F1 {candidate_f1:.4f} must be "
            f">= baseline F1 {baseline_f1:.4f} + margin {IMPROVEMENT_MARGIN:.2f} "
            f"(required {required_f1:.4f})."
        )

    # Save the complete fitted pipeline and schema for safe prediction.
    package = {
        "model": candidate,
        "feature_names": feature_names,
        "target_name": TARGET_NAME,
        "class_labels": {0: "benign", 1: "malignant"},
    }
    joblib.dump(package, MODEL_PATH)
    print(f"Quality gate passed. Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
