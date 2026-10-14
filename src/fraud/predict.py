"""Load the saved champion model and score a single claim."""
import joblib
import pandas as pd

from src.fraud import config
from src.fraud.data import CATEGORICAL, NUMERIC

_CACHE: dict = {}


def load_model(path=None) -> dict:
    """Load the model bundle once per path. Raises FileNotFoundError if absent."""
    key = str(path or config.MODEL_PATH)
    if key not in _CACHE:
        _CACHE[key] = joblib.load(key)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


def risk_band(p: float, threshold: float) -> str:
    if p >= threshold * 1.5:
        return "HIGH"
    if p >= threshold:
        return "MEDIUM"
    return "LOW"


def score_claim(claim: dict) -> dict:
    bundle = load_model()
    row = pd.DataFrame([claim])[NUMERIC + CATEGORICAL]
    p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {
        "fraud_probability": round(p, 4),
        "risk_band": risk_band(p, bundle["threshold"]),
        "model": bundle["name"],
    }
