from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def predict_one(features: dict[str, float], model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Predict one row; reject missing or extra feature names clearly."""
    package = joblib.load(model_path)
    expected = package["feature_names"]
    missing = [name for name in expected if name not in features]
    extra = [name for name in features if name not in expected]
    if missing:
        raise ValueError(f"Input is missing required feature(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"Input contains unexpected feature(s): {', '.join(extra)}")

    frame = pd.DataFrame([[features[name] for name in expected]], columns=expected)
    prediction = int(package["model"].predict(frame)[0])
    return {
        "prediction": prediction,
        "label": package["class_labels"][prediction],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo prediction using the trained model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--values",
        nargs=30,
        type=float,
        required=True,
        help="Provide exactly 30 numeric feature values in feature_01 ... feature_30 order.",
    )
    args = parser.parse_args()
    package = joblib.load(args.model)
    features = dict(zip(package["feature_names"], args.values))
    print(predict_one(features, args.model))


if __name__ == "__main__":
    main()
