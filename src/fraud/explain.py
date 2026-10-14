"""SHAP explanations for the champion model.

Works for tree models (random forest). For the linear baseline the coefficients
are directly interpretable, so explanations are only generated for trees.
"""
import numpy as np
import pandas as pd
import shap

from src.fraud.data import CATEGORICAL, NUMERIC
from src.fraud.predict import load_model


def _fraud_class_values(shap_values) -> np.ndarray:
    """Return SHAP values for the fraud class across shap versions."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    return arr[..., 1] if arr.ndim == 3 else arr


def explain_claim(claim: dict, top_n: int = 5) -> list[dict]:
    bundle = load_model()
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "estimators_"):
        return []  # linear baseline: no tree explainer

    row = pd.DataFrame([claim])[NUMERIC + CATEGORICAL]
    X = pipe.named_steps["pre"].transform(row)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = pipe.named_steps["pre"].get_feature_names_out()

    values = _fraud_class_values(shap.TreeExplainer(model).shap_values(X))[0]
    order = np.argsort(-np.abs(values))[:top_n]
    return [
        {
            "feature": str(names[i]).split("__", 1)[-1],
            "impact": round(float(values[i]), 4),
            "direction": "raises risk" if values[i] > 0 else "lowers risk",
        }
        for i in order
    ]
