"""
FastAPI service that serves the stock-direction model.

Endpoints
---------
GET  /health         liveness + whether the model loaded
GET  /model-info     model provenance, feature contract, honest metrics
POST /predict        score a single observation
POST /predict/batch  score up to 1000 observations in one call
GET  /docs           interactive OpenAPI docs (FastAPI built-in)

Run locally:
    python -m src.train --source synthetic
    uvicorn src.app:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src import model as model_module
from src.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("ml-api")
logging.basicConfig(level=logging.INFO)

STATE: dict = {"bundle": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of per request."""
    try:
        STATE["bundle"] = model_module.get_bundle()
        log.info("model loaded: version=%s", STATE["bundle"].version)
    except Exception as exc:  # noqa: BLE001 - surfaced via /health, not a crash
        STATE["error"] = str(exc)
        log.warning("model not loaded: %s", exc)
    yield


app = FastAPI(
    title="Stock Direction Model API",
    description=(
        "Serves a next-day stock-direction classifier behind a validated REST API. "
        "Demonstrates the deployment half of the ML lifecycle: versioned artifacts, "
        "a typed request contract, batch inference, and health checks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _bundle():
    if STATE["bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"model unavailable: {STATE['error'] or 'not loaded'}",
        )
    return STATE["bundle"]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe. Reports model availability without throwing."""
    bundle = STATE["bundle"]
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Provenance and the exact feature contract callers must satisfy."""
    bundle = _bundle()
    md = bundle.metadata
    return ModelInfoResponse(
        model_version=bundle.version,
        model_name=md.get("model_name"),
        trained_at=md.get("trained_at"),
        source=md.get("source"),
        n_features=len(bundle.features),
        features=bundle.features,
        metrics=md.get("metrics", {}),
        honesty_note=md.get("honesty_note"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    """Score one observation and return the direction plus probability."""
    bundle = _bundle()
    result = bundle.predict([payload.model_dump()])[0]
    return PredictResponse(model_version=bundle.version, **result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(payload: BatchRequest) -> BatchResponse:
    """Score many observations in a single request (max 1000 rows)."""
    bundle = _bundle()
    rows = [row.model_dump() for row in payload.rows]
    results = [Prediction(**r) for r in bundle.predict(rows)]
    return BatchResponse(
        predictions=results, count=len(results), model_version=bundle.version
    )
