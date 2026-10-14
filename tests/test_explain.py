import joblib
import pytest

from src.fraud import train, predict
from src.fraud.data import generate_claims, NUMERIC, CATEGORICAL
from src.fraud.explain import explain_claim


@pytest.fixture
def trained(tmp_path, monkeypatch):
    path = tmp_path / "m.joblib"
    monkeypatch.setattr(train, "MODEL_PATH", path)
    monkeypatch.setattr(train, "METRICS_PATH", tmp_path / "m.json")
    train.run(n=4000, log_to_mlflow=False)
    predict.load_model.cache_clear()
    monkeypatch.setattr(predict, "load_model", lambda: joblib.load(path))
    return path


def test_scoring_and_explanation(trained, monkeypatch):
    import src.fraud.explain as explain_module
    monkeypatch.setattr(explain_module, "load_model", lambda: joblib.load(trained))
    claim = generate_claims(n=10, seed=3).iloc[0][NUMERIC + CATEGORICAL].to_dict()
    result = predict.score_claim(claim)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    reasons = explain_module.explain_claim(claim)
    assert isinstance(reasons, list)
    assert all({"feature", "impact", "direction"} <= set(r) for r in reasons)
