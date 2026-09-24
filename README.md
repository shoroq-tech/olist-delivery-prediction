```markdown

# Olist Delivery Prediction – MLOps Task 3

Predicts whether an Olist e-commerce order will be delivered late, served as a FastAPI REST API and packaged with Docker.

## Quickstart (Clone and Run)

```bash
git clone https://github.com/shoroq-tech/olist-delivery-prediction.git
cd olist-delivery-prediction
docker compose up --build
```

The API will be available at `http://localhost:8000`. Test it:

```bash
curl http://localhost:8000/health
```

## Project Structure

| Folder/File | Purpose |
|---|---|
| `app/main.py` | FastAPI application. Defines `/health`, `/model-info`, `/predict`, `/predict/batch` endpoints. |
| `src/predict.py` | Loads the trained model and encoder, runs inference. |
| `src/features.py` | Feature engineering logic shared between training and inference. |
| `src/preprocessing.py` | Data cleaning/preprocessing steps applied before feature engineering. |
| `config/config.yaml` | Central configuration (model paths, data paths, logging, API host/port). |
| `models/model.pkl` | Trained classification model (scikit-learn). |
| `models/state_encoder.pkl` | Encoder for the Brazilian state feature. |
| `models/feature_list.txt` | Ordered list of features the model expects. |
| `tests/test_predictor.py` | Unit tests for prediction logic. |
| `tests/test_api.py` | Integration tests for the FastAPI endpoints. |
| `logs/app.log` | Runtime application log. |
| `Dockerfile` | Builds the API image (python:3.12-slim). |
| `docker-compose.yml` | Runs the API container with one command. |
| `requirements.txt` | Runtime dependencies, pinned to match the training environment. |
| `requirements-dev.txt` | Extra dependencies for testing (pytest, httpx). |

## Configuration (config/config.yaml)

| Key | Meaning |
|---|---|
| `model.path` | Path to the trained model file loaded at startup. |
| `model.encoder_path` | Path to the fitted state encoder used to transform the state feature. |
| `model.feature_list_path` | Path to the text file listing feature names/order the model expects. |
| `model.version` | Version tag returned by /model-info, used to confirm which model is deployed. |
| `data.raw_data_path` | Where raw input data is expected (local development only, not used at inference). |
| `data.processed_data_path` | Where processed/feature-engineered data is expected (local development only). |
| `logging.level` | Log verbosity (INFO). |
| `logging.log_file` | Path where application logs are written. |
| `api.host` / `api.port` | Host/port the FastAPI server binds to (0.0.0.0:8000, required for Docker). |

## Tools Used and Why

- FastAPI: serves the model as a REST API with automatic docs and request validation.
- scikit-learn: framework the model was trained with.
- Docker / Docker Compose: guarantees the API runs identically anywhere, isolated from the local Spyder/conda environment.
- pytest / httpx: automated testing of prediction logic and API endpoints.
- Git / GitHub: version control and delivery of the project.

## Testing

```bash
conda activate spyder-
cd path/to/olist-delivery-prediction
pytest tests/ -v
```

Expected: 9 passed (4 predictor tests, 5 API tests).

## Known Issue: scikit-learn Version Mismatch

requirements.txt pins scikit-learn==1.5.1 on purpose, because model.pkl and state_encoder.pkl were trained/saved with scikit-learn 1.5.1. The local Spyder environment has scikit-learn 1.9.1 installed, which is newer.

Because of this, loading state_encoder.pkl inside the Docker container raises a harmless InconsistentVersionWarning. Despite the warning, predictions were verified to be numerically identical to the original notebook output: is_late = 0, probability = 0.2576, model_version = 1.0.0.

## Not Yet Implemented

The following were not completed due to time constraints for this submission:
- Great Expectations (data validation)
- DVC (data version control)
- MLflow (experiment tracking)
```