"""
API tests. They train a throwaway model on deterministic synthetic data into a
tmp dir, so the suite runs offline and never touches the real artifacts.
"""
from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from src import model as model_module
from src.train import train

VALID_ROW = {
    "return_1d": 0.0042,
    "volatility_20d": 0.0135,
    "sma_20_ratio": 0.012,
    "sma_50_ratio": 0.031,
    "momentum_5d": 0.021,
    "momentum_20d": 0.048,
    "rsi_14": 56.3,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Train a tiny model into a tmp dir and point the app at it."""
    model_dir = tmp_path_factory.mktemp("model")
    train(source="synthetic", tickers=[], period="", model_dir=str(model_dir))

    model_module.get_bundle.cache_clear()
    monkey_settings = str(model_dir)
    import os
    os.environ["MODEL_DIR"] = monkey_settings
    try:
        from src.app import app
        with TestClient(app) as c:
            yield c
    finally:
        os.environ.pop("MODEL_DIR", None)
        model_module.get_bundle.cache_clear()


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info_exposes_feature_contract(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == 7
    assert set(body["features"]) == set(VALID_ROW)
    assert body["metrics"], "metrics must be reported"
    for name, m in body["metrics"].items():
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0


def test_predict_returns_direction_and_probability(client):
    r = client.post("/predict", json=VALID_ROW)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"up", "down"}
    assert 0.0 <= body["probability_up"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_version"]


def test_predict_rejects_out_of_range_features(client):
    bad = dict(VALID_ROW, rsi_14=150.0)  # RSI cannot exceed 100
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_features(client):
    incomplete = {k: v for k, v in VALID_ROW.items() if k != "rsi_14"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_batch_predicts_all_rows(client):
    rows = [VALID_ROW, dict(VALID_ROW, rsi_14=22.0)]
    r = client.post("/predict/batch", json={"rows": rows})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_batch_rejects_empty_payload(client):
    r = client.post("/predict/batch", json={"rows": []})
    assert r.status_code == 422


def test_predictions_are_deterministic(client):
    first = client.post("/predict", json=VALID_ROW).json()
    second = client.post("/predict", json=VALID_ROW).json()
    assert first == second
