from src.fraud import predict
from src.fraud.data import CATEGORICAL, NUMERIC, generate_claims
from src.fraud.explain import explain_claim


def test_scoring_and_explanation(trained_model):
    claim = generate_claims(n=10, seed=3).iloc[0][NUMERIC + CATEGORICAL].to_dict()
    result = predict.score_claim(claim)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    reasons = explain_claim(claim)
    assert isinstance(reasons, list)
    assert all({"feature", "impact", "direction"} <= set(r) for r in reasons)


def test_linear_model_explanations_are_not_empty():
    """The logistic baseline must also produce reasons, whichever model wins."""
    from src.fraud import train
    from src.fraud.data import TARGET
    df = generate_claims(n=3000, seed=5)
    pipe = train.build_pipeline(
        train.LogisticRegression(max_iter=1000, class_weight="balanced")
    ).fit(df[NUMERIC + CATEGORICAL], df[TARGET])
    predict._CACHE["linear-test"] = {"pipeline": pipe, "threshold": 0.5, "name": "logistic-baseline"}
    try:
        original = predict.load_model
        predict.load_model = lambda path=None: predict._CACHE["linear-test"]
        claim = df.iloc[0][NUMERIC + CATEGORICAL].to_dict()
        reasons = explain_claim(claim)
        assert len(reasons) == 5
        assert {r["direction"] for r in reasons} <= {"raises risk", "lowers risk"}
    finally:
        predict.load_model = original
        predict._CACHE.pop("linear-test", None)
