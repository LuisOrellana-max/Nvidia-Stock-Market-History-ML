import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager


app = FastAPI()

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
    else:
        print(f"Warning: Model ({MODEL_PATH}) or Scaler ({SCALER_PATH}) missing at startup.")
    yield
    model = None
    scaler = None
    print("Cleaning up resources...")

app = FastAPI(lifespan=lifespan)

class StockFeatures(BaseModel):
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
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
    
    # Convert input payload to DataFrame preserving feature order
    input_data = pd.DataFrame([features.model_dump])
    
    # Scale features & run prediction
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
        confidence=confidence
    )

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "NVIDIA Stock Movement Inference API",
        "documentation": "/docs",
        "health_check": "/health"
    }