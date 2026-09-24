# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:42:55 2026

@author: islam
"""

"""FastAPI service for olist-delivery-prediction.

Endpoints:
    GET  /health         -> service status
    GET  /model-info     -> model version and basic details
    POST /predict        -> one order
    POST /predict/batch  -> many orders
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Config paths (models/..., logs/...) are relative to the project root,
# so make sure we always run from there, whatever the launch directory is.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from src.predict import DeliveryPredictor  # noqa: E402

# ---------------------------------------------------------------- config
with open(BASE_DIR / "config" / "config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# --------------------------------------------------------------- logging
log_cfg = config.get("logging", {})
log_file = BASE_DIR / log_cfg.get("log_file", "logs/app.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------ app + model
state: Dict[str, Any] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not on every request."""
    try:
        logger.info("Loading model...")
        state["predictor"] = DeliveryPredictor(config)
        logger.info("Model loaded (version %s)", config["model"]["version"])
    except Exception:
        logger.exception("Failed to load model")
        state["predictor"] = None
    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="Olist Delivery Prediction API",
    description="Predicts whether an Olist order will be delivered late.",
    version=str(config["model"]["version"]),
    lifespan=lifespan,
)


def get_predictor() -> DeliveryPredictor:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return predictor


# --------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    order: Dict[str, Any]


class BatchRequest(BaseModel):
    orders: List[Dict[str, Any]]


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    loaded = state["predictor"] is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}


@app.get("/model-info")
def model_info():
    predictor = get_predictor()
    feature_path = BASE_DIR / config["model"]["feature_list_path"]
    with open(feature_path, encoding="utf-8") as f:
        n_features = len([line for line in f if line.strip()])
    return {
        "model_version": config["model"]["version"],
        "model_type": type(predictor.model).__name__,
        "n_features": n_features,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    predictor = get_predictor()
    try:
        return predictor.predict(req.order)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    predictor = get_predictor()
    results = []
    for i, order in enumerate(req.orders):
        try:
            results.append(predictor.predict(order))
        except Exception as exc:
            logger.exception("Prediction failed for order index %d", i)
            results.append({"index": i, "error": str(exc)})
    return {"count": len(results), "results": results}

