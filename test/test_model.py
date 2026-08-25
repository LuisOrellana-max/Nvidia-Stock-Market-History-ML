import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.model import train_model
from src.feature_engineering import create_market_features


@pytest.fixture
def mock_raw_csv(tmp_path) -> str:
    """Generates a synthetic stock history CSV file in Pytest's temporary directory.

    Requires at least ~60 rows to ensure rolling windows (e.g., 50-day average)
    don't drop all rows during feature engineering.
    """
    n_samples = 70
    dates = pd.date_range(start="2026-01-01", periods=n_samples, freq="D")

    # Generate realistic deterministic synthetic stock data
    np.random.seed(42)
    base_price = 100.0 + np.cumsum(np.random.randn(n_samples) * 0.5)

    raw_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": base_price + 0.1,
            "High": base_price + 1.0,
            "Low": base_price - 1.0,
            "Close": base_price,
            "Adj Close": base_price,
            "Volume": np.random.randint(100000, 500000, size=n_samples),
        }
    )

    csv_path = tmp_path / "mock_stock_history.csv"
    raw_df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_train_model_creates_artifacts(mock_raw_csv, tmp_path):
    """Verifies that train_model executes without errors and outputs valid .joblib

    artifacts.
    """
    model_path = str(tmp_path / "model.joblib")
    scaler_path = str(tmp_path / "scaler.joblib")

    # Execute training with parameter paths
    train_model(
        data_path=mock_raw_csv,
        model_output_path=model_path,
        scaler_output_path=scaler_path,
    )

    # Check existence
    assert os.path.exists(model_path), "Model file was not created."
    assert os.path.exists(scaler_path), "Scaler file was not created."

    # Check that files are not empty (size > 0 bytes)
    assert os.path.getsize(model_path) > 0, "Model joblib file is empty."
    assert os.path.getsize(scaler_path) > 0, "Scaler joblib file is empty."


def test_saved_model_inference_and_probabilities(mock_raw_csv, tmp_path):
    """Loads saved artifacts and asserts valid prediction output structure and probability

    ranges.
    """
    model_path = str(tmp_path / "model.joblib")
    scaler_path = str(tmp_path / "scaler.joblib")

    # 1. Train model to produce saved artifacts
    train_model(
        data_path=mock_raw_csv,
        model_output_path=model_path,
        scaler_output_path=scaler_path,
    )

    # 2. Load model and scaler back into memory
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Prepare single feature sample row from mock data
    raw_df = pd.read_csv(mock_raw_csv)
    features_df = create_market_features(raw_df).drop(columns=["Target"])
    sample_row = features_df.iloc[[-1]]  # Retain 2D DataFrame shape

    # 4. Scale feature row and predict
    scaled_sample = scaler.transform(sample_row)
    prediction = model.predict(scaled_sample)
    probabilities = model.predict_proba(scaled_sample)

    # 5. Assertions
    assert len(prediction) == 1, "Expected exactly 1 prediction output."
    assert prediction[0] in [0, 1], f"Unexpected target class: {prediction[0]}"
    assert probabilities.shape == (
        1,
        2,
    ), "Probabilities output shape must be (1, 2)."
    assert (
        0.0 <= probabilities[0][1] <= 1.0
    ), "Predicted probability must be bounded in [0.0, 1.0]."