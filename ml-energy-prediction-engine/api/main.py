"""
Stage 7: FastAPI prediction service.

Loads the saved model + preprocessor ONCE at startup (not on every request -
retraining or reloading per-call would be slow and pointless, since nothing
about the model changes between requests), then exposes POST /predict.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from schemas import PredictionRequest, PredictionResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(
    title="Energy Production Prediction API",
    description="Predicts next-hour renewable energy production (Wind/Solar).",
    version="1.0",
)

# Loaded once, when the API process starts - shared across all requests.
model = joblib.load(MODELS_DIR / "production_model.joblib")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
feature_config = json.load(open(MODELS_DIR / "feature_config.json"))


@app.get("/")
def root():
    return {"status": "ok", "message": "Energy Production Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_test_r2": feature_config["test_set_metrics"]["r2"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    row = pd.DataFrame([{
        "Source": request.source,
        "Day_Name": request.day_name,
        "Start_Hour": request.start_hour,
        "Day_of_Year": request.day_of_year,
        "Year": request.year,
        "Production_Lag_1": request.production_lag_1,
        "Production_Lag_24": request.production_lag_24,
        "Production_Lag_168": request.production_lag_168,
    }])[feature_config["all_features_in_order"]]

    encoded = preprocessor.transform(row)
    prediction = model.predict(encoded)[0]

    return PredictionResponse(prediction=round(float(prediction), 2))
