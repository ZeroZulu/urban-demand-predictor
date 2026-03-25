"""
Integration tests for the FastAPI prediction service.
Uses TestClient to test endpoints without a running server.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_prediction_result():
    return {
        "predicted_trips":     284,
        "confidence_interval": [238, 330],
        "top_factors": [
            {"feature": "demand_lag_168h", "value": 271.0,
             "shap_value": 48.3, "direction": "increases_demand"},
            {"feature": "is_rush_hour", "value": 1.0,
             "shap_value": 31.7, "direction": "increases_demand"},
            {"feature": "precipitation_mm", "value": 0.0,
             "shap_value": -12.1, "direction": "decreases_demand"},
            {"feature": "temperature_f", "value": 78.5,
             "shap_value": 9.4, "direction": "increases_demand"},
            {"feature": "is_weekend", "value": 0.0,
             "shap_value": -7.2, "direction": "decreases_demand"},
        ],
        "model_version": "1.0.0",
    }


@pytest.fixture
def client(mock_prediction_result):
    """TestClient with predictor patched to avoid needing trained models."""
    with patch("api.routes.predict.predict_with_explanation",
               return_value=mock_prediction_result), \
         patch("api.routes.predict.predict_batch",
               return_value=[mock_prediction_result]), \
         patch("src.models.predictor.load_model", return_value=MagicMock()), \
         patch("src.models.predictor.load_explainer", return_value=MagicMock()), \
         patch("src.models.predictor.load_feature_names", return_value=["f1", "f2"]), \
         patch("src.models.predictor.load_metadata",
               return_value={"model_version": "1.0.0", "feature_count": 2}):
        from api.main import app
        yield TestClient(app)


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "model_loaded" in data
        assert "feature_count" in data

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


# ── Predict endpoint ───────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client, prediction_payload):
        resp = client.post("/predict", json=prediction_payload)
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client, prediction_payload):
        data = client.post("/predict", json=prediction_payload).json()
        assert "predicted_trips" in data
        assert "confidence_interval" in data
        assert "top_factors" in data
        assert "model_version" in data

    def test_predicted_trips_is_non_negative(self, client, prediction_payload):
        data = client.post("/predict", json=prediction_payload).json()
        assert data["predicted_trips"] >= 0

    def test_confidence_interval_has_two_values(self, client, prediction_payload):
        data = client.post("/predict", json=prediction_payload).json()
        ci = data["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    def test_top_factors_count(self, client, prediction_payload):
        data = client.post("/predict", json=prediction_payload).json()
        assert len(data["top_factors"]) == 5

    def test_shap_factors_have_direction(self, client, prediction_payload):
        data = client.post("/predict", json=prediction_payload).json()
        for factor in data["top_factors"]:
            assert factor["direction"] in ("increases_demand", "decreases_demand")

    def test_invalid_zone_id_rejected(self, client, prediction_payload):
        payload = {**prediction_payload, "zone_id": -1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_zone_id_too_high_rejected(self, client, prediction_payload):
        payload = {**prediction_payload, "zone_id": 999}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_temperature_rejected(self, client, prediction_payload):
        payload = {**prediction_payload, "temperature_f": 9999}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_precipitation_rejected(self, client, prediction_payload):
        payload = {**prediction_payload, "precipitation_mm": -5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_required_fields_rejected(self, client):
        resp = client.post("/predict", json={"zone_id": 161})
        assert resp.status_code == 422

    def test_known_zone_returns_zone_name(self, client, prediction_payload):
        payload = {**prediction_payload, "zone_id": 161}
        data = client.post("/predict", json=payload).json()
        assert data.get("zone_name") == "Midtown Center"

    def test_unknown_zone_returns_null_name(self, client, prediction_payload):
        payload = {**prediction_payload, "zone_id": 99}
        data = client.post("/predict", json=payload).json()
        assert data.get("zone_name") is None

    def test_datetime_snapped_to_hour(self, client, prediction_payload):
        """Datetime with minutes/seconds should be accepted and snapped to hour."""
        payload = {**prediction_payload, "datetime": "2024-06-15T18:34:22"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_event_fields_accepted(self, client, prediction_payload):
        payload = {**prediction_payload, "has_major_event": True, "event_attendance": 50000}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ── Batch predict endpoint ─────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_single_request(self, client, prediction_payload):
        body = {"requests": [prediction_payload]}
        resp = client.post("/predict/batch", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["predictions"]) == 1

    def test_empty_batch_rejected(self, client):
        resp = client.post("/predict/batch", json={"requests": []})
        assert resp.status_code == 422

    def test_batch_over_limit_rejected(self, client, prediction_payload):
        body = {"requests": [prediction_payload] * 101}
        resp = client.post("/predict/batch", json=body)
        assert resp.status_code == 422
