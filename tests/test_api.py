from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app, lifespan
from app.services.prediction_service import PredictionService


@pytest.fixture(autouse=True)
def mock_prediction_service():
    with patch.object(PredictionService, "_load_assets", return_value=None):
        service = PredictionService()

    service.encoder = MagicMock()
    service.encoder.classes_ = np.array(["MIT", "Stanford", "EPN"])
    service.encoder.transform = MagicMock(
        side_effect=lambda names: np.array(
            [list(service.encoder.classes_).tolist().index(n) for n in names]
        )
    )

    service.model = MagicMock()
    service.model.predict = MagicMock(return_value=np.array([55.0]))

    service.historical_df = None
    service._ranking_cache = None
    service._ranking_cache_timestamp = 0

    from app.api.v1.endpoints import analytics as analytics_module

    analytics_module.prediction_service = service


client = TestClient(app)


def test_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_affiliations_returns_503_without_model():
    response = client.get("/api/v1/affiliations")
    assert response.status_code == 503


def test_ranking_returns_503_without_model():
    response = client.get("/api/v1/ranking")
    assert response.status_code == 503


def test_model_details_returns_503_without_model():
    response = client.get("/api/v1/model-details")
    assert response.status_code == 503


def test_health_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(app.state, "recommendation_service", None, raising=False)
    monkeypatch.setattr(
        app.state, "recommendation_service_error", "boom", raising=False
    )
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "error", "service_initialized": False}


def test_health_returns_ok_with_model(monkeypatch):
    monkeypatch.setattr(app.state, "recommendation_service", MagicMock(), raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service_initialized": True}


def test_lifespan_handles_recommendation_service_failure():
    with patch("app.main.RecommendationService", side_effect=RuntimeError("init failed")):
        test_app = FastAPI(lifespan=lifespan)
        with TestClient(test_app) as client:
            response = client.get("/")
            assert response.status_code == 404
            assert test_app.state.recommendation_service is None
            assert test_app.state.recommendation_service_error == "init failed"
