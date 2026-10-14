
from pathlib import Path
import sys

import joblib
import pytest

# Add the repository root so Python can find predict.py.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from predict import predict_one

MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"


def test_saved_model_can_be_loaded():
    assert MODEL_PATH.exists(), (
        "Training must create artifacts/model.joblib before tests run."
    )
    package = joblib.load(MODEL_PATH)
    assert "model" in package
    assert len(package["feature_names"]) == 30


def test_valid_sample_returns_expected_shape_and_type():
    package = joblib.load(MODEL_PATH)
    sample = {name: 1.0 for name in package["feature_names"]}
    result = predict_one(sample, MODEL_PATH)

    assert isinstance(result, dict)
    assert set(result) == {"prediction", "label"}
    assert isinstance(result["prediction"], int)
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("benign", "malignant")


def test_missing_feature_is_rejected_clearly():
    package = joblib.load(MODEL_PATH)
    sample = {name: 1.0 for name in package["feature_names"]}
    sample.pop(package["feature_names"][0])

    with pytest.raises(ValueError, match="missing required feature"):
        predict_one(sample, MODEL_PATH)
