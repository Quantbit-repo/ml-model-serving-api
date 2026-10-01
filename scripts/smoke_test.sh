#!/usr/bin/env bash
# End-to-end smoke test against a running API.
#   uvicorn src.app:app --port 8000 &
#   ./scripts/smoke_test.sh
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"

echo "== /health =="
curl -sS "$BASE/health" | python3 -m json.tool

echo "== /model-info =="
curl -sS "$BASE/model-info" | python3 -m json.tool | head -20

echo "== /predict =="
curl -sS -X POST "$BASE/predict" -H 'Content-Type: application/json' -d '{
  "return_1d": 0.0042, "volatility_20d": 0.0135, "sma_20_ratio": 0.012,
  "sma_50_ratio": 0.031, "momentum_5d": 0.021, "momentum_20d": 0.048, "rsi_14": 56.3
}' | python3 -m json.tool

echo "== /predict/batch =="
curl -sS -X POST "$BASE/predict/batch" -H 'Content-Type: application/json' -d '{
  "rows": [
    {"return_1d":0.0042,"volatility_20d":0.0135,"sma_20_ratio":0.012,"sma_50_ratio":0.031,"momentum_5d":0.021,"momentum_20d":0.048,"rsi_14":56.3},
    {"return_1d":-0.011,"volatility_20d":0.022,"sma_20_ratio":-0.03,"sma_50_ratio":-0.05,"momentum_5d":-0.04,"momentum_20d":-0.07,"rsi_14":28.5}
  ]
}' | python3 -m json.tool

echo
echo "Smoke test passed."
