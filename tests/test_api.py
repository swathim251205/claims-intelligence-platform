import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.fraud import config, predict

HIGH_RISK = {
    "claim_amount": 9500, "vehicle_age": 4, "driver_age": 27, "policy_tenure_months": 5,
    "previous_claims": 2, "days_to_report": 21, "injury_claimed": True,
    "witness_present": False, "police_report": False, "late_night": True,
    "third_party_involved": True, "flagged_garage": True,
    "region": "London", "claim_type": "collision",
}
LOW_RISK = {
    "claim_amount": 350, "vehicle_age": 8, "driver_age": 52, "policy_tenure_months": 96,
    "previous_claims": 0, "days_to_report": 1, "injury_claimed": False,
    "witness_present": True, "police_report": True, "late_night": False,
    "third_party_involved": False, "flagged_garage": False,
    "region": "Wales", "claim_type": "windscreen",
}


@pytest.fixture
def client(trained_model):
    with TestClient(app) as c:
        yield c


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True}


def test_score_returns_valid_response(client):
    r = client.post("/score-claim", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_high_risk_claim_scores_above_low_risk(client):
    high = client.post("/score-claim", json=HIGH_RISK).json()["fraud_probability"]
    low = client.post("/score-claim", json=LOW_RISK).json()["fraud_probability"]
    assert high > low


def test_explain_returns_reasons(client):
    body = client.post("/explain-claim", json=HIGH_RISK).json()
    assert len(body["reasons"]) > 0
    assert {"feature", "impact", "direction"} <= set(body["reasons"][0])


@pytest.mark.parametrize("field,value", [
    ("claim_amount", -100),
    ("driver_age", 12),
    ("region", "Narnia"),
    ("claim_type", "alien abduction"),
    ("injury_claimed", "maybe"),
])
def test_invalid_input_is_rejected(client, field, value):
    bad = {**HIGH_RISK, field: value}
    assert client.post("/score-claim", json=bad).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in HIGH_RISK.items() if k != "claim_amount"}
    assert client.post("/score-claim", json=bad).status_code == 422


def test_unknown_field_is_rejected(client):
    assert client.post("/score-claim", json={**HIGH_RISK, "surprise": 1}).status_code == 422


def test_response_has_request_metadata_headers(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers
    assert "x-process-time-ms" in r.headers


def test_returns_503_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    predict.clear_cache()
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/score-claim", json=HIGH_RISK).status_code == 503
