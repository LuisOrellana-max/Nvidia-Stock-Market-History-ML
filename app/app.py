import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import RedirectResponse 
from typing import List
from contextlib import asynccontextmanager


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

# Load artifacts on startup
model = None
scaler = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Automated startup handler: loads model and scaler into global memory upon server launch."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded successfully on startup.")

        if hasattr(scaler, "features_name_in_"):
            print("Expected Features:", list(scaler.feature_names_in_))
        
    else:
        print(f"Warning: Model ({MODEL_PATH}) or Scaler ({SCALER_PATH}) missing at startup.")
    yield
    model = None
    scaler = None
    print("Cleaning up resources...")

app = FastAPI(lifespan=lifespan)

@app.get("/", include_in_schema=False)
async def root_to_docs():
    """Redirects root URL requests straight to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")

class StockFeatures(BaseModel):
    # Raw features required by scaler
    Open: float
    High: float
    Low: float
    Close: float
    Adj_Close: float = Field(..., alias = "Adj Close")
    Volume: float
    
    # Technical & engineered features
    Returns: float
    roll_avg_5: float
    roll_avg_7: float
    roll_avg_50: float
    Volatility_5: float
    Volume_Change: float
    High_Low_Range: float
    Price_Change: float
    Open_Close_Ratio: float
    High_Low_Ratio: float
    lag_1: float
    lag_7: float
    Adj_Close_to_Lag1: float

    model_config = ConfigDict(populate_by_name=True)

   


class PredictionResponse(BaseModel):
    prediction: int
    direction: str
    probability_up: float
    confidence: float

@app.get("/health")
def get_health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_direction(features: StockFeatures):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, detail="Model or Scaler artifacts are not loaded."
        )

    data_dict = features.model_dump(by_alias=True)

    #Convert to DataFrame and checking
    input_data = pd.DataFrame([data_dict])

    # Drop 'Target' if it somehow exists for safety reason
    if "Target" in input_data.columns:
        input_data = input_data.drop(columns=["Target"])

    #  Reorder DataFrame columns to match fit-time scaler feature names
    if hasattr(scaler, "feature_names_in_"):
        input_data = input_data[scaler.feature_names_in_]

    #  Scale features & predict
    scaled_data = scaler.transform(input_data)
    pred_class = int(model.predict(scaled_data)[0])
    probabilities = model.predict_proba(scaled_data)[0]

    prob_up = float(probabilities[1])
    confidence = float(probabilities[pred_class])
    direction = "UP" if pred_class == 1 else "DOWN"

    return PredictionResponse(
        prediction=pred_class,
        direction=direction,
        probability_up=prob_up,
        confidence=confidence,
    )