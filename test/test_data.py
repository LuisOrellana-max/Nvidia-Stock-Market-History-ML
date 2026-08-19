import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import create_market_features

@pytest.fixture
def mock_stock_data():
    n_samples = 60
    dates = pd.date_range(start = "2026-01-01", periods=n_samples, freq = "D")
    np.random.seed(42)
    return pd.DataFrame({
        'Date': dates,
        'Open': np.linspace(100, 110, n_samples),
        'High': np.linspace(102, 112, n_samples),
        "Low": np.linspace(98, 108, n_samples),
        "Close": np.linspace(101, 111, n_samples),
        'Adj Close': np.linspace(101, 111, n_samples),
        "Volume": [1000] * n_samples
    })

def test_feature_creation(mock_stock_data):
    df = create_market_features(mock_stock_data)

    expected_columns = [
        'Returns', 'roll_avg_5', 'roll_avg_7', 'roll_avg_50', 
        'Volatility_5', 'Volume_Change', 'High_Low_Range', 
        'Price_Change', 'Open_Close_Ratio', 'High_Low_Ratio', 
        'lag_1', 'lag_7', 'Adj_Close_to_Lag1', 'Target'
    ]

    for col in expected_columns:
        assert col in df.columns

def test_feature_generation_no_nans(mock_stock_data):
    df = create_market_features(mock_stock_data)
    # Ensure pipeline drops rolling NaNs clean
    assert not df.isnull().values.any()

def test_target_is_binary(mock_stock_data):
    df = create_market_features(mock_stock_data)
    assert set(df['Target'].unique()).issubset({0, 1})

def test_dataframe_not_empty(mock_stock_data):
    df = create_market_features(mock_stock_data)
    assert len(df) > 0
