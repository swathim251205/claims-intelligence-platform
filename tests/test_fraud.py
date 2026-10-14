import numpy as np

from src.fraud.data import NUMERIC, CATEGORICAL, TARGET, generate_claims
from src.fraud import config, train
from src.fraud.predict import risk_band


def test_data_shape_and_fraud_rate():
    df = generate_claims(n=3000, seed=1)
    assert len(df) == 3000
    assert set(NUMERIC + CATEGORICAL + [TARGET]).issubset(df.columns)
    assert 0.02 < df[TARGET].mean() < 0.15


def test_data_is_reproducible():
    a = generate_claims(n=500, seed=7)
    b = generate_claims(n=500, seed=7)
    assert a.equals(b)


def test_training_beats_base_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "m.json")
    out = train.run(n=4000, log_to_mlflow=False)
    best = out["results"][out["champion"]]
    assert best["pr_auc"] > 2 * best["base_rate"]
    assert (tmp_path / "m.joblib").exists()


def test_best_threshold_returns_valid_probability():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.2, 0.15, 0.8, 0.7, 0.3, 0.9, 0.05])
    t, _ = train.best_threshold(y, s)
    assert 0 < t < 1


def test_risk_band_ordering():
    assert risk_band(0.01, 0.2) == "LOW"
    assert risk_band(0.25, 0.2) == "MEDIUM"
    assert risk_band(0.6, 0.2) == "HIGH"


def test_champion_is_lowest_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "m.json")
    out = train.run(n=4000, log_to_mlflow=False)
    costs = {k: v["expected_cost"] for k, v in out["results"].items()}
    assert out["champion"] == min(costs, key=costs.get)
