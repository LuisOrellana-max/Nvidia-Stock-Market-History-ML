import pandas as pd
import numpy as np

def create_market_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = data.columns.str.strip()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index()

    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()

    # Feature 1: Moving Averages
    data['roll_avg_5'] = data['Adj Close'].rolling(5).mean() # business days
    data['roll_avg_7'] = data['Adj Close'].rolling(7).mean() # whole week
    data['roll_avg_50'] = data['Adj Close'].rolling(50).mean() #quarter
    # Feature 2: Volatility
    data['Volatility_5'] = data['Returns'].rolling(window=5).std()
    # Feature 3: Volume Change
    data['Volume Change'] = data['Volume'].pct_change()

    # Basic Price Feature
    data['High-low-range'] = data['High'] - data['Low']
    data['Price-Change'] = data['Adj Close'] - data['Open']
    data['Open-Close-ratio'] = data['Adj Close'] / data['Open']
    data['High-Low-ratio'] = data['High'] / data['Low']

    # lag feature
    data['lag_1'] = data['Adj Close'].shift(1)
    data['lag_7'] = data['Adj Close'].shift(7)

    # Stationary Lag Relative Ratios (Better for Logistic Regression)
    data['Adj_Close_to_Lag1'] = data['Adj Close'] / data['lag_1'] - 1.0

    # Target: 1 if Next Day Close > Current Day Close, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    # Drop NAs created by rolling features and the final target shift
    data = data.dropna()

    return data



