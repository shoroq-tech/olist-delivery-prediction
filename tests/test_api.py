"""
Integration tests for the FastAPI app (app/main.py)
Uses TestClient to simulate real HTTP requests without running uvicorn manually.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_ORDER = {
    "order_id": "e481f51cbdc54678b7cc49136f2d6af7",
    "customer_id": "9ef432eb6251297304e76186b10a928d",
    "order_status": "delivered",
    "order_purchase_timestamp": "2017-10-02 10:56:33",
    "order_approved_at": "2017-10-02 11:07:15",
    "order_delivered_carrier_date": "2017-10-04 19:55:00",
    "order_delivered_customer_date": "2017-10-10 21:25:13",
    "order_estimated_delivery_date": "2017-10-18 00:00:00",
    "n_items": 1.0,
    "total_price": 29.99,
    "total_freight": 8.72,
    "total_payment_value": 38.71,
    "n_payment_methods": 2.0,
    "customer_unique_id": "7c396fd4830fd04220f754e42b4e5bff",
    "customer_zip_code_prefix": 3149,
    "customer_city": "sao paulo",
    "customer_state": "SP",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert "model_type" in body


def test_predict_endpoint_success():
    response = client.post("/predict", json={"order": SAMPLE_ORDER})
    assert response.status_code == 200
    body = response.json()
    assert "is_late" in body
    assert "probability" in body
    assert "model_version" in body
    assert body["is_late"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_endpoint_missing_field_returns_422():
    response = client.post("/predict", json={"wrong_key": SAMPLE_ORDER})
    assert response.status_code == 422


def test_predict_batch_endpoint_success():
    response = client.post("/predict/batch", json={"orders": [SAMPLE_ORDER, SAMPLE_ORDER]})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["results"], list)
    assert len(body["results"]) == 2

import pytest


@pytest.fixture(scope="module", autouse=True)
def startup():
    with client:
        yield
