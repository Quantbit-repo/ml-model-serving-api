"""Request/response contracts for the prediction API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One observation of the model's feature vector."""

    return_1d: float = Field(..., ge=-0.5, le=0.5, description="Previous day's return")
    volatility_20d: float = Field(..., ge=0.0, le=1.0, description="20-day rolling std of returns")
    sma_20_ratio: float = Field(..., ge=-1.0, le=5.0, description="close / SMA(20) - 1")
    sma_50_ratio: float = Field(..., ge=-1.0, le=5.0, description="close / SMA(50) - 1")
    momentum_5d: float = Field(..., ge=-1.0, le=5.0, description="5-day price change")
    momentum_20d: float = Field(..., ge=-1.0, le=5.0, description="20-day price change")
    rsi_14: float = Field(..., ge=0.0, le=100.0, description="14-day RSI")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "return_1d": 0.0042,
                "volatility_20d": 0.0135,
                "sma_20_ratio": 0.012,
                "sma_50_ratio": 0.031,
                "momentum_5d": 0.021,
                "momentum_20d": 0.048,
                "rsi_14": 56.3,
            }]
        }
    }


class BatchRequest(BaseModel):
    rows: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    prediction: str = Field(..., description="'up' or 'down'")
    probability_up: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.5, le=1.0)


class PredictResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_name: str | None = None
    trained_at: str | None = None
    source: str | None = None
    n_features: int
    features: list[str]
    metrics: dict
    honesty_note: str | None = None
