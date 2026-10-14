# Claims Intelligence Platform

A motor insurance claims system that scores fraud risk with an explainable model and, in the next stage, answers questions over claim documents using retrieval-augmented generation. Built to show the full path from data to a tested, deployable service.

**Status:** fraud scoring layer and REST API complete (training, MLflow tracking, explanations, validated endpoints, Docker, tests, CI). RAG layer, monitoring and cloud deployment are in progress.

## The problem

Claims handlers cannot review every claim. Missing a fraudulent claim is expensive, but investigating genuine claims wastes time and annoys customers. The model has to support a decision about where to spend investigation effort, so it is judged on cost, not accuracy.

## Data card

The data is synthetic: 5,000 UK motor claims with a fraud rate of about 5 per cent, generated in `src/fraud/data.py`. Fraud depends on claim size, claims history, reporting delay, policy age, a flagged garage, and one interaction (new policy, large claim, slow reporting) that a linear model cannot capture. It is not real customer data and results will not transfer directly to a live portfolio.

## Approach

1. Stratified train and validation split, because fraud is rare.
2. Two models compared: a logistic regression baseline and a random forest, both with class weighting.
3. A decision threshold chosen to minimise expected cost, using stated assumptions of £4,000 for each missed fraud and £150 for each unnecessary investigation.
4. The champion is the model with the lowest expected cost. All runs are tracked in MLflow.
5. SHAP values explain each score, so a handler sees which factors raised or lowered the risk.

## Results (validation set)

| Model | PR-AUC | ROC-AUC | Precision | Recall | Expected cost |
|---|---|---|---|---|---|
| logistic-baseline | 0.36 | 0.82 | 0.15 | 0.75 | £111,650 |
| random-forest | 0.35 | 0.82 | 0.14 | 0.78 | £109,800 |

Base fraud rate in the validation set is 5.4%. Champion: **random-forest**.

Precision is low by design. At these costs it is cheaper to investigate several genuine claims than to miss one fraud, so the threshold favours recall. Changing the cost assumptions changes the threshold, and that trade-off is the main thing to discuss with the business.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.fraud.data      # generate data
python -m src.fraud.train     # train, compare, log to MLflow
python -m pytest -q           # run tests
mlflow ui --backend-store-uri sqlite:///mlflow.db

uvicorn src.api.main:app --reload   # start the API at http://localhost:8000/docs
```

### With Docker

```bash
docker build -t claims-api .
docker run -p 8000:8000 claims-api
```

The image trains the model while it builds, so it is self-contained. It runs as a non-root user and includes a health check.

## API

A FastAPI service wraps the model. Inputs are validated before they reach the model, so bad requests fail with a clear 422 instead of a silent wrong score.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and whether a model is loaded |
| `GET /model-info` | Champion model and validation metrics |
| `POST /score-claim` | Fraud probability and risk band for one claim |
| `POST /explain-claim` | The same score plus the top factors behind it |

Design choices worth knowing about:
- Pydantic schemas enforce ranges and allowed values, and reject unknown fields.
- A missing model returns 503 with instructions, not a stack trace.
- Every response carries `X-Request-ID` and `X-Process-Time-ms` headers, and each request is logged, which the monitoring stage will build on.
- Explanations work for both model types: SHAP for the forest, coefficient contributions for the logistic baseline.

Example:

```bash
curl -X POST http://localhost:8000/explain-claim \
  -H "Content-Type: application/json" \
  -d '{"claim_amount":9500,"vehicle_age":4,"driver_age":27,"policy_tenure_months":5,"previous_claims":2,"days_to_report":21,"injury_claimed":true,"witness_present":false,"police_report":false,"late_night":true,"third_party_involved":true,"flagged_garage":true,"region":"London","claim_type":"collision"}'
```

Interactive documentation is generated automatically at `/docs`.

## Limitations

- Synthetic data, so the headline numbers show the method, not real-world performance.
- Cost assumptions are illustrative.
- No fairness analysis yet. A production system would need to check outcomes across protected characteristics.

## Roadmap

- [x] Fraud model, cost-based threshold, MLflow, SHAP, tests, CI
- [x] FastAPI service with validation, request logging and Docker image
- [ ] RAG layer with citations and an evaluation set
- [ ] Cloud deployment with a public link
- [ ] Drift monitoring
- [ ] Streamlit dashboard and demo video

## Author

Swathi Mulkundkar
