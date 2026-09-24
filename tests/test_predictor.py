"""
Unit tests for DeliveryPredictor (src/predict.py)
"""
import sys
import os
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import DeliveryPredictor


@pytest.fixture(scope="module")
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def predictor(config):
    return DeliveryPredictor(config)


def test_predictor_loads_successfully(predictor):
    """يتأكد إن الموديل انحمّل بدون أخطاء"""
    assert predictor is not None
    assert predictor.model is not None


def test_predict_returns_expected_keys(predictor):
    """يتأكد إن الناتج فيه كل المفاتيح المطلوبة"""
    order = {
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
    result = predictor.predict(order)
    assert "is_late" in result
    assert "probability" in result
    assert "model_version" in result


def test_predict_probability_in_valid_range(predictor):
    """يتأكد إن الاحتمال بين 0 و 1"""
    order = {
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
    result = predictor.predict(order)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_is_late_is_binary(predictor):
    """يتأكد إن is_late إما 0 أو 1"""
    order = {
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
    result = predictor.predict(order)
    assert result["is_late"] in (0, 1)
