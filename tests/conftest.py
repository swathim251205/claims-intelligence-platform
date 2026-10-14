import pytest

from src.fraud import config, predict, train


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Train a small model into a temporary folder and point the code at it."""
    folder = tmp_path_factory.mktemp("model")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "MODEL_PATH", folder / "fraud-model.joblib")
    mp.setattr(config, "METRICS_PATH", folder / "metrics.json")
    predict.clear_cache()
    train.run(n=4000, log_to_mlflow=False)
    yield folder
    mp.undo()
    predict.clear_cache()
