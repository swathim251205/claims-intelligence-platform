"""Per-claim explanations for whichever model is the champion.

Tree models: SHAP values (change in fraud probability attributed to each feature).
Linear models: coefficient times the scaled feature value, which is each feature's
contribution to the log-odds of fraud. Units differ between the two, so compare
the ranking and direction of reasons, not the raw impact across models.
"""
import numpy as np
import pandas as pd
import shap

from src.fraud import predict
from src.fraud.data import CATEGORICAL, NUMERIC


def _fraud_class_values(shap_values) -> np.ndarray:
    """Return SHAP values for the fraud class across shap versions."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    return arr[..., 1] if arr.ndim == 3 else arr


def explain_claim(claim: dict, top_n: int = 5) -> list[dict]:
    bundle = predict.load_model()
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]

    row = pd.DataFrame([claim])[NUMERIC + CATEGORICAL]
    X = pipe.named_steps["pre"].transform(row)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = pipe.named_steps["pre"].get_feature_names_out()

    if hasattr(model, "estimators_"):
        values = _fraud_class_values(shap.TreeExplainer(model).shap_values(X))[0]
    elif hasattr(model, "coef_"):
        values = model.coef_[0] * X[0]
    else:
        return []

    order = np.argsort(-np.abs(values))[:top_n]
    return [
        {
            "feature": str(names[i]).split("__", 1)[-1],
            "impact": round(float(values[i]), 4),
            "direction": "raises risk" if values[i] > 0 else "lowers risk",
        }
        for i in order
    ]
