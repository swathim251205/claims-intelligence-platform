"""Load the saved champion model and score a single claim."""
from functools import lru_cache

import joblib
import pandas as pd

from src.fraud.data import CATEGORICAL, NUMERIC
from src.fraud.train import MODEL_PATH


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


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
