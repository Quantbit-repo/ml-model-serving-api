# 🚀 Stock Direction Model API

A production style **model serving** project: a trained scikit-learn classifier exposed as a validated REST API with FastAPI, typed request contracts, batch inference, health checks, a pipeline you can test offline, and a Docker image that runs standalone.

This is the **deployment half** of the ML lifecycle, the part most data science portfolios skip. Training notebooks are everywhere; a versioned, tested, monitored endpoint is what production teams actually run.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

## 🎯 What This Demonstrates (MLOps)

| Practice | How it shows up here |
|---|---|
| **Versioned model artifacts** | `models/model.joblib` + `models/metadata.json` (version, provenance, metrics) |
| **Explicit feature contract** | One `FEATURES` list shared by training, API schema, and tests, so they cannot drift |
| **Typed input validation** | Pydantic schemas with real bounds (for example RSI from 0 to 100), returning automatic `422` responses with actionable errors |
| **Load once serving** | Model loaded at startup via a lifespan handler and cached for the process, not per request |
| **Batch inference** | `/predict/batch` scores up to 1,000 rows per call |
| **Health and observability** | `/health` reports model load state; `/model-info` exposes provenance and metrics |
| **Reproducibility** | Two data sources: live `yfinance`, or deterministic synthetic data for offline use, CI, and Docker |
| **Automated tests** | 8 pytest cases covering validation, determinism, and batch behaviour |
| **Containerisation** | A layered `Dockerfile` that bakes in a working model and includes a `HEALTHCHECK` |

## 🚀 Quick Start

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train the model (real data, free, no API key)
python -m src.train --source yfinance --tickers AAPL MSFT GOOGL AMZN NVDA JPM --period 3y

#    …or fully offline and deterministic:
python -m src.train --source synthetic

# 3. Serve it
uvicorn src.app:app --reload --port 8000
```

Interactive docs: **http://127.0.0.1:8000/docs**

## 📡 API Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus whether the model loaded |
| `GET` | `/model-info` | Version, provenance, feature list, metrics |
| `POST` | `/predict` | Score one observation |
| `POST` | `/predict/batch` | Score 1 to 1000 observations |

### Example: single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "return_1d": 0.0042, "volatility_20d": 0.0135,
    "sma_20_ratio": 0.012, "sma_50_ratio": 0.031,
    "momentum_5d": 0.021, "momentum_20d": 0.048, "rsi_14": 56.3
  }'
```

```json
{
  "prediction": "down",
  "probability_up": 0.4563,
  "confidence": 0.5437,
  "model_version": "1.0.0"
}
```

### Example: rejected input

Requesting a prediction with an impossible RSI returns a precise error instead of a silent bad answer:

```json
{
  "detail": [{
    "type": "less_than_equal",
    "loc": ["body", "rsi_14"],
    "msg": "Input should be less than or equal to 100",
    "input": 150
  }]
}
```

`HTTP 422`: validation is part of the API contract, not an afterthought.

## 🧪 Tests

```bash
pytest
```

```
8 passed
```

The suite trains a throwaway model on deterministic synthetic data into a temp directory, so it runs **offline and never touches your real artifacts**. Covered: health reporting, feature contract exposure, prediction shape, rejection of out of range values, rejection of missing fields, batch scoring, rejection of empty batches, and prediction determinism.

## 🐳 Docker

```bash
docker build -t stock-model-api .
docker run -p 8000:8000 stock-model-api
```

The image trains its own deterministic model at build time, so it is **standalone**: no network access and no external artifacts needed. A `HEALTHCHECK` polls `/health` so orchestrators can restart a broken container.

> Note: the container build was authored but not executed on the development machine (no Docker runtime installed there). The same code path it runs, `python -m src.train --source synthetic` plus `uvicorn src.app:app`, is verified end to end by the test suite and the smoke test.

## 📁 Layout

```
ml-model-serving-api/
├── src/
│   ├── train.py      # data loading (yfinance | synthetic) + training + artifact persistence
│   ├── model.py      # ModelBundle: loading, feature order safety, inference
│   ├── schemas.py    # Pydantic request/response contracts with bounds
│   └── app.py        # FastAPI app: lifespan loading, 4 endpoints
├── tests/test_api.py # 8 tests against a temp model
├── scripts/smoke_test.sh  # curl based check against a live server
├── Dockerfile
├── pytest.ini
└── requirements.txt
```

## 🔬 Honest Results (no inflated metrics)

Trained on 3 years of daily data for 6 large US stocks (about 4,200 modelled rows), split by time so the model trains on the past and is tested on the future, with no lookahead leakage. Representative run:

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Logistic Regression | ~0.47 | ~0.48 |
| Histogram Gradient Boosting *(selected)* | **~0.50** | ~0.49 |

*(Live `yfinance` data shifts as new trading days arrive, so exact figures move by a few thousandths between runs. The conclusion does not.)*

**Next day direction is very close to a coin flip, and that is the honest, expected result.** Daily equity returns are nearly unpredictable; anyone reporting 95%+ accuracy on this task has leaked the future into training or is measuring the wrong thing. The model here exists to exercise a **real serving pipeline**, not to claim a trading edge. The companion project [Market Intelligence Dashboard](https://github.com/Quantbit-repo/market-intelligence-dashboard) reaches the same conclusion from the research side and shows what genuinely *does* add value (risk reduction through momentum).

Reporting an unimpressive number truthfully is itself the skill this repo is meant to demonstrate.

## ⚠️ Disclaimer

Educational portfolio project. Not investment advice. Past performance does not guarantee future results.

## 📬 Contact

Built as part of a data science portfolio: [X (@quant_bit)](https://x.com/quant_bit) · [GitHub](https://github.com/Quantbit-repo)
