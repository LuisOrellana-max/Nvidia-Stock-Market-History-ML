import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.feature_engineering import create_market_features

def train_model(data_path: str, model_output_path: str = "models/model.joblib", scaler_output_path: str = "models/scaler.joblib"):
    df_raw = pd.read_csv(data_path)
    df = create_market_features(df_raw)

    X = df.drop(['Target'], axis = 1)
    y = df['Target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Model Training
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("--- Test Set Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    #make sure if it exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)

    
    # Save Model
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Model saved successfully to {model_output_path}")
    print(f"Scaler saved successfully to {scaler_output_path}")

    

if __name__ == "__main__":
    train_model("data/NVidia_stock_history.csv")