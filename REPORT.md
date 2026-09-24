# Final Report — Olist Delivery Prediction (MLOps Task 3)

**Repo:** https://github.com/shoroq-tech/olist-delivery-prediction

## 1. Project Overview

This project wraps a trained delivery-delay prediction model in a FastAPI service, containerized with Docker, tested with pytest, and validated through GitHub Actions CI. It exposes prediction endpoints, a monitoring endpoint, and a JSON-lines prediction log.

## 2. Repository Structure

```
olist-delivery-prediction/
├── app/
│   ├── main.py           # FastAPI app: /health, /model-info, /predict, /predict/batch, /metrics
│   └── monitoring.py     # In-memory metrics (requests_total, errors_total, error_rate,
│                          #   avg_latency_ms, p95_latency_ms) + prediction logging
├── src/                  # Preprocessing / feature engineering / model logic
├── config/               # Configuration values (see Section 5)
├── models/               # model.pkl, state_encoder.pkl, feature_list.txt
├── logs/
│   └── predictions.jsonl # One JSON line per successful prediction (input + output)
├── docs/
│   └── RUNBOOK.md        # Where to look, alert thresholds, and response actions
├── tests/                # pytest suite (9 passing)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── .github/workflows/ci.yml   # flake8 + pytest on Python 3.12
```

## 3. Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns `ok` / `degraded` status |
| `/model-info` | GET | Model version and metadata |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |
| `/metrics` | GET | `requests_total`, `errors_total`, `error_rate`, `avg_latency_ms`, `p95_latency_ms` |

## 4. Key Technical Decisions

**Pinning `scikit-learn==1.5.1` in Docker.**
The model (`models/model.pkl`, `models/state_encoder.pkl`) was trained and serialized under scikit-learn 1.5.1. Scikit-learn's pickle format is not guaranteed to be compatible across versions — loading a pickle with a different version can silently change behavior or fail outright. Pinning the exact training version in `requirements.txt` / the Docker image guarantees that inference reproduces the training-time behavior exactly, rather than relying on chance compatibility with whatever version pip would otherwise resolve.

**Known warning: `InconsistentVersionWarning` on `state_encoder.pkl`.**
When the app loads `state_encoder.pkl`, scikit-learn prints an `InconsistentVersionWarning`. scikit-learn emits this warning when a pickled object was saved with a different library version than the one loading it. It is non-fatal here: the service loads the encoder and the Docker prediction matches the notebook reference exactly (`is_late=0`, `probability=0.2576`). The warning is documented rather than silenced so it is not mistaken for a hidden problem. A clean fix would be to re-save the encoder under the pinned version (1.5.1).

## 5. Configuration

Runtime settings are kept in `config/` and not hard-coded in the app: model and encoder paths, the decision threshold for `is_late`, and the prediction log path (`logs/predictions.jsonl`). Container port and volume settings are in `docker-compose.yml`.

## 6. Testing & CI

- **pytest:** 9 tests passing locally (`spyder-` conda environment) and in CI.
- **CI (`.github/workflows/ci.yml`):** runs on Python 3.12, executes `flake8` (lint) then `pytest`, installing both `requirements.txt` and `requirements-dev.txt`. Status: passing (green) as of the latest push.
- **Local Docker verification:** `docker compose up --build` reproduces the notebook's reference prediction exactly: `is_late=0`, `probability=0.2576`, model `version=1.0.0`.

## 7. Fault-Injection Test (Resilience Proof)

To demonstrate the service fails safely rather than crashing, a malformed request was sent to the running container:

**Command:**
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{}"
```

**Response (HTTP 422):**
```json
{"detail":[{"type":"missing","loc":["body","order"],"msg":"Field required","input":{}}]}
```

**Interpretation:** FastAPI's request validation (via Pydantic) rejected the incomplete payload before it reached the model, returning a structured `422 Unprocessable Entity` error instead of throwing an unhandled exception or returning a corrupted prediction. The API and container remained healthy and continued serving requests normally afterward — no restart was needed.

**Note on logging:** `logs/predictions.jsonl` only records *successful* predictions (input + output), not rejected/invalid requests. This is a current limitation: failed requests are currently visible only in the container's stdout/`logs/app.log`, not in the structured JSONL log. This is called out honestly here as a possible future improvement (logging validation failures to a separate error log) rather than left unstated.

## 8. Monitoring

`/metrics` exposes:
- `requests_total`, `errors_total`, `error_rate`
- `avg_latency_ms`, `p95_latency_ms`

`docs/RUNBOOK.md` documents where to look and what to do when each threshold is crossed (error rate, latency, `/health` degraded, drift suspicion).

## 9. Not Implemented (Honest Disclosure)

Due to time constraints, the following were **not completed**:

| Item | Status | Reason |
|---|---|---|
| Extra tests for preprocessing/features/data modules | Not done | Time |
| Database service in `docker-compose.yml` (API + DB + artifacts) | Not done | Time — existing `olist-db` container belongs to a separate project and was intentionally left untouched |
| Docker image build inside CI + pre-commit hooks | Not done | Time |
| Great Expectations (data validation) | Not implemented | Time — would add schema/range checks on incoming prediction requests and training data |
| DVC (data/model versioning) | Not implemented | Time — models are currently tracked manually rather than via DVC |
| MLflow (experiment tracking) | Not implemented | Time — training run metadata (params, metrics) is not currently logged to a tracking server |

## 10. Summary

The core MLOps requirements are met: a working, tested, containerized, CI-validated API with monitoring and a documented runbook, plus a demonstrated fault-tolerance case. Remaining items are data-validation/versioning/experiment-tracking tooling that would strengthen the pipeline further but were out of scope given the time available.
