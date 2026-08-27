"""
Stage 9: Minimal test suite for the prediction API.

Uses FastAPI's TestClient, which runs the app in-process (no live server
needs to be running for these to work) - the standard way to test a
FastAPI app.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "source": "Wind",
    "day_name": "Monday",
    "start_hour": 14,
    "day_of_year": 200,
    "year": 2025,
    "production_lag_1": 7200,
    "production_lag_24": 6800,
    "production_lag_168": 6500,
}


def test_valid_prediction_request():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_response_format():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert set(response.json().keys()) == {"prediction", "units"}
    assert isinstance(response.json()["prediction"], float)


def test_invalid_source_is_rejected():
    bad_payload = {**VALID_PAYLOAD, "source": "Nuclear"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_missing_required_field_is_rejected():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["production_lag_168"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_invalid_numeric_value_is_rejected():
    bad_payload = {**VALID_PAYLOAD, "start_hour": 99}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_negative_lag_is_rejected():
    bad_payload = {**VALID_PAYLOAD, "production_lag_1": -100}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
