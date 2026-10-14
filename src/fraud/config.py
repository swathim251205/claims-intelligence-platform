"""Paths and business assumptions shared by training and serving."""
from pathlib import Path

MODEL_PATH = Path("models/fraud-model.joblib")
METRICS_PATH = Path("models/metrics.json")

# Business assumptions, stated openly so a reviewer can challenge them.
COST_MISSED_FRAUD = 4000.0   # average loss when a fraudulent claim is paid
COST_INVESTIGATION = 150.0   # cost of reviewing a claim that turns out genuine
