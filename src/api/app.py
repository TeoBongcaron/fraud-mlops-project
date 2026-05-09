from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# -----------------------------
# Load model + scaler at startup
# -----------------------------

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_xgb.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train the model first.")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Scaler file not found. Run preprocessing first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# FastAPI initialization
# -----------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict fraudulent transactions using a trained XGBoost model.",
    version="1.0.0"
)

# -----------------------------
# Request/Response Schemas
# -----------------------------

class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionOutput(BaseModel):
    fraud_probability: float
    fraud_label: int


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict", response_model=PredictionOutput)
def predict_transaction(payload: TransactionInput):
    """
    Predict whether a credit card transaction is fraudulent.
    """

    # Convert input to numpy array
    features = np.array([[value for value in payload.dict().values()]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prob = model.predict_proba(features_scaled)[0][1]
    label = int(prob >= 0.5)

    return PredictionOutput(
        fraud_probability=float(prob),
        fraud_label=label
    )
