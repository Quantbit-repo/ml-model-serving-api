"""Model loading + inference, isolated from the web layer."""
from __future__ import annotations

import json
import os
import pathlib
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

DEFAULT_MODEL_DIR = "models"


def model_dir() -> pathlib.Path:
    """Model directory, overridable with MODEL_DIR (tests point this at tmp dirs)."""
    return pathlib.Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))


class ModelBundle:
    """A loaded estimator plus its metadata."""

    def __init__(self, estimator: Any, metadata: dict):
        self.estimator = estimator
        self.metadata = metadata
        self.features: list[str] = metadata.get("features", [])
        self.version: str = metadata.get("model_version", "unknown")

    def predict(self, rows: list[dict]) -> list[dict]:
        """
        Score feature rows.

        Each row must contain every feature in the model contract. Returns one
        dict per row: prediction label + calibrated-ish probability.
        """
        missing = [f for f in self.features if f not in rows[0]] if rows else []
        if missing:
            raise KeyError(f"missing features: {missing}")

        # Build a DataFrame (not a bare array) so sklearn sees the same feature
        # names it was fitted with - avoids silent column-order mistakes.
        frame = pd.DataFrame([[float(row[f]) for f in self.features] for row in rows],
                             columns=self.features)
        probs = self.estimator.predict_proba(frame)[:, 1]
        out = []
        for p in probs:
            p = float(p)
            out.append({
                "prediction": "up" if p >= 0.5 else "down",
                "probability_up": round(p, 4),
                "confidence": round(max(p, 1 - p), 4),
            })
        return out


def load_bundle(directory: pathlib.Path | None = None) -> ModelBundle:
    """Load model.joblib + metadata.json. Raises FileNotFoundError if untrained."""
    directory = directory or model_dir()
    model_path = directory / "model.joblib"
    meta_path = directory / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - train first: python -m src.train --source synthetic"
        )
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ModelBundle(joblib.load(model_path), metadata)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    """Process-wide cached bundle so the API loads the model once."""
    return load_bundle()
