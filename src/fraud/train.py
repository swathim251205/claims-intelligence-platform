"""Train, compare and log fraud models with MLflow.

Metrics are chosen for rare-event detection: PR-AUC and a threshold picked by
business cost, not accuracy.
"""
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.fraud.data import CATEGORICAL, NUMERIC, TARGET, generate_claims

MODEL_PATH = Path("models/fraud-model.joblib")
METRICS_PATH = Path("models/metrics.json")

# Business assumptions, stated openly so a reviewer can challenge them.
COST_MISSED_FRAUD = 4000.0   # average loss when a fraudulent claim is paid
COST_INVESTIGATION = 150.0   # cost of reviewing a claim that turns out genuine


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def best_threshold(y_true, scores) -> tuple[float, float]:
    """Return the probability threshold with the lowest expected cost."""
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.02, 0.9, 89):
        flagged = scores >= t
        fn = np.sum((~flagged) & (y_true == 1))
        fp = np.sum(flagged & (y_true == 0))
        cost = fn * COST_MISSED_FRAUD + fp * COST_INVESTIGATION
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def evaluate(pipe, X_val, y_val) -> dict:
    scores = pipe.predict_proba(X_val)[:, 1]
    threshold, cost = best_threshold(y_val.to_numpy(), scores)
    preds = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(y_val, scores)),
        "roc_auc": float(roc_auc_score(y_val, scores)),
        "threshold": threshold,
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds)),
        "expected_cost": cost,
        "base_rate": float(y_val.mean()),
    }


def run(n: int = 5000, seed: int = 42, log_to_mlflow: bool = True) -> dict:
    df = generate_claims(n=n, seed=seed)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    candidates = {
        "logistic-baseline": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random-forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        ),
    }

    if log_to_mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("claims-fraud")

    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        metrics = evaluate(pipe, X_val, y_val)
        results[name], fitted[name] = metrics, pipe
        if log_to_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("n_rows", n)
                mlflow.log_metrics(metrics)

    # Champion is chosen on expected business cost, not on a single ranking metric.
    champion = min(results, key=lambda k: results[k]["expected_cost"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": fitted[champion], "threshold": results[champion]["threshold"],
                 "name": champion}, MODEL_PATH)
    summary = {"champion": champion, "results": results}
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
