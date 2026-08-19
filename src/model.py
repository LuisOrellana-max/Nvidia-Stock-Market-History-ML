import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.feature_engineering import create_market_features

def train_model(data_path: str, model_output_dir: str = "models") -> None:
    df_raw = pd.read.csv(data_path)
    df = create_market_features(df_raw)

    X = df.drop(['Target'])
    y = df['Target']