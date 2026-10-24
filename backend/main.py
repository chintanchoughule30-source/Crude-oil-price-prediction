import json
import math
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.pipeline.features import MODEL_FEATURES

app = FastAPI(
    title="CrudeCast AI API",
    description="WTI crude-oil price prediction and model evaluation API.",
    version="1.0.0",
)

configured_origins = os.getenv("CRUDECAST_CORS_ORIGINS")
allowed_origins = (
    [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    if configured_origins is not None
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "xgboost_model.joblib"
EVALUATION_ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "evaluation_artifacts.json"

model = None
if MODEL_PATH.exists():
    # Load the model once when the API starts, rather than for every request.
    model = joblib.load(MODEL_PATH)
    print("XGBoost Model loaded successfully.")
else:
    print("Warning: Model not found. Please run the training script first.")

class PredictionRequest(BaseModel):
    """The eight values required to predict the next price."""

    model_config = ConfigDict(extra="forbid")

    price_lag_1: float = Field(ge=0, allow_inf_nan=False)
    price_lag_2: float = Field(ge=0, allow_inf_nan=False)
    price_lag_5: float = Field(ge=0, allow_inf_nan=False)
    price_lag_20: float = Field(ge=0, allow_inf_nan=False)
    sma_5: float = Field(ge=0, allow_inf_nan=False)
    sma_20: float = Field(ge=0, allow_inf_nan=False)
    return_1d: float = Field(ge=-1, allow_inf_nan=False)
    volatility_20: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
    unit: str = "Barrel"


def load_evaluation_artifacts():
    """Read the results created by the training script."""
    try:
        with EVALUATION_ARTIFACTS_PATH.open(encoding="utf-8") as artifacts_file:
            return json.load(artifacts_file)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Evaluation artifacts are unavailable. Run backend/pipeline/train.py first.",
        ) from error
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail="Evaluation artifacts are invalid. Retrain the model to regenerate them.",
        ) from error


@app.get("/health")
def health_check():
    """Report liveness separately from prediction readiness."""
    is_ready = model is not None
    return {
        "status": "ready" if is_ready else "degraded",
        "model_loaded": is_ready,
        "ready": is_ready,
    }


@app.get("/ready")
def readiness_check():
    """Return an error when the service cannot currently make predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    return {"status": "ready"}


@app.get("/model-insights")
def get_model_insights():
    """Return the model statistics used by the dashboard."""
    artifacts = load_evaluation_artifacts()
    return {
        "metrics": artifacts["metrics"],
        "hyperparameters": artifacts["hyperparameters"],
        "feature_importances": artifacts["feature_importances"],
        "residual_distribution": artifacts["residual_distribution"],
    }


@app.get("/test-performance")
def get_test_performance():
    """Return the held-out test rows used by the chart and sample form."""
    return {"rows": load_evaluation_artifacts()["test_performance"]}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(req: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    features = [[getattr(req, feature) for feature in MODEL_FEATURES]]
    prediction = float(model.predict(features)[0])
    if not math.isfinite(prediction):
        raise HTTPException(status_code=500, detail="Model returned an invalid prediction.")

    return PredictionResponse(predicted_price=prediction)