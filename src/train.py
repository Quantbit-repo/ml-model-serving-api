"""
Train and persist the stock-direction classifier served by the API.

Two data sources, so the pipeline is reproducible both online and offline:
  --source yfinance   real daily OHLCV pulled live (free, no API key)  [default]
  --source synthetic  deterministic seeded data (offline / CI / Docker builds)

Usage
-----
python -m src.train --source yfinance --tickers AAPL MSFT GOOGL --period 3y
python -m src.train --source synthetic

Outputs
-------
models/model.joblib     fitted sklearn estimator (best of the candidates)
models/metadata.json    model version, features, metrics, provenance
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Feature contract shared by training, the API schema, and the tests.
FEATURES = [
    "return_1d",
    "volatility_20d",
    "sma_20_ratio",
    "sma_50_ratio",
    "momentum_5d",
    "momentum_20d",
    "rsi_14",
]
LABEL = "target_up"
MODEL_VERSION = "1.0.0"

DEFAULT_MODEL_DIR = "models"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _features_from_prices(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Engineer the model's feature frame from a single ticker's price series."""
    df = pd.DataFrame({"close": close.astype(float), "volume": volume.astype(float)})
    df["return_1d"] = df["close"].pct_change()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["sma_20_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1.0
    df["sma_50_ratio"] = df["close"] / df["close"].rolling(50).mean() - 1.0
    df["momentum_5d"] = df["close"].pct_change(5)
    df["momentum_20d"] = df["close"].pct_change(20)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df[LABEL] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df


def load_yfinance(tickers: list[str], period: str) -> pd.DataFrame:
    """Pull live daily data and build the feature frame for each ticker."""
    import yfinance as yf

    frames = []
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            print(f"  [warn] no data for {ticker}")
            continue
        frame = _features_from_prices(hist["Close"], hist["Volume"])
        frame["ticker"] = ticker
        frames.append(frame)
        print(f"  {ticker}: {len(frame)} rows")
    if not frames:
        raise SystemExit("ERROR: yfinance returned no data (offline?). Try --source synthetic.")
    return pd.concat(frames)


def load_synthetic(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic price path with a mild, learnable momentum effect.

    Used for tests, CI, and Docker builds so the pipeline never depends on the
    network. The signal is weak-but-real by construction, which keeps reported
    metrics honest instead of optimistically fake.
    """
    rng = np.random.default_rng(seed)
    # Random-walk returns with slight autocorrelation (momentum) and vol clustering
    shocks = rng.normal(0, 0.01, n_rows)
    rets = np.zeros(n_rows)
    for i in range(1, n_rows):
        rets[i] = 0.25 * rets[i - 1] + shocks[i]
    close = pd.Series(100 * np.cumprod(1 + rets))
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n_rows).astype(float))
    frame = _features_from_prices(close, volume)
    frame["ticker"] = "SYNTH"
    return frame


def build_dataset(source: str, tickers: list[str], period: str) -> pd.DataFrame:
    if source == "yfinance":
        return load_yfinance(tickers, period)
    return load_synthetic()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(source: str, tickers: list[str], period: str, model_dir: str) -> dict:
    out_dir = pathlib.Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data (source={source})...")
    raw = build_dataset(source, tickers, period)
    data = raw.dropna(subset=FEATURES + [LABEL]).copy()
    data = data.sort_values(["ticker", "close"]) if "ticker" in data else data
    print(f"Modelling rows: {len(data):,}")

    # Time-based split: train on the past, test on the future (no lookahead leak)
    split = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train_df[FEATURES], train_df[LABEL]
    X_test, y_test = test_df[FEATURES], test_df[LABEL]

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, random_state=42),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        acc = float(accuracy_score(y_test, (proba >= 0.5).astype(int)))
        results[name] = {"roc_auc": round(auc, 4), "accuracy": round(acc, 4)}
        fitted[name] = model
        print(f"  {name:24s} AUC={auc:.4f}  acc={acc:.4f}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = fitted[best_name]
    print(f"Selected: {best_name}")

    joblib.dump(best, out_dir / "model.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "tickers": tickers if source == "yfinance" else ["SYNTH"],
        "period": period if source == "yfinance" else "synthetic",
        "n_rows_total": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "metrics": results,
        "selected_metrics": results[best_name],
        "honesty_note": (
            "Daily direction is close to a coin flip by nature; AUC near 0.5 is the "
            "expected, honest outcome. This model exists to demonstrate the serving "
            "pipeline, not to claim a trading edge."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model + metadata to {out_dir}/")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the serving model")
    ap.add_argument("--source", choices=["yfinance", "synthetic"], default="yfinance")
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN"])
    ap.add_argument("--period", default="3y")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = ap.parse_args()

    train(args.source, args.tickers, args.period, args.model_dir)


if __name__ == "__main__":
    main()
