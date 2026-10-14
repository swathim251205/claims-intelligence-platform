"""FastAPI service for claim fraud scoring and explanation.

Run locally:  uvicorn src.api.main:app --reload
Docs:         http://localhost:8000/docs
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import ClaimIn, ExplainOut, ScoreOut
from src.fraud import config, predict
from src.fraud.explain import explain_claim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("claims-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = predict.load_model()
        logger.info("Loaded model %s", bundle["name"])
    except FileNotFoundError:
        logger.warning("No model found at %s. Train one with: python -m src.fraud.train",
                       config.MODEL_PATH)
    yield


app = FastAPI(
    title="Claims Fraud Scoring API",
    description="Scores motor insurance claims for fraud risk and explains each score.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    """Attach a request id and latency to every response and log them."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
    logger.info("%s %s -> %s in %.1fms [%s]", request.method, request.url.path,
                response.status_code, elapsed_ms, request_id)
    return response


def _require_model() -> None:
    try:
        predict.load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train it with: python -m src.fraud.train",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.get("/model-info")
def model_info() -> dict:
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metrics not available.")
    return json.loads(config.METRICS_PATH.read_text())


@app.post("/score-claim", response_model=ScoreOut)
def score(claim: ClaimIn) -> dict:
    _require_model()
    return predict.score_claim(claim.to_features())


@app.post("/explain-claim", response_model=ExplainOut)
def explain(claim: ClaimIn) -> dict:
    _require_model()
    features = claim.to_features()
    result = predict.score_claim(features)
    result["reasons"] = explain_claim(features)
    return result
