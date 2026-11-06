from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app, lifespan


client = TestClient(app)


def test_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_openapi_schema_uses_gateway_prefix():
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["servers"] == [{"url": "/api/predictive", "description": "Gateway"}]
    assert "/health" not in schema["paths"]
    assert all(not path.startswith("/api/v1") for path in schema["paths"])


def test_affiliations_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(app.state, "prediction_service", None, raising=False)
    response = client.get("/api/v1/affiliations")
    assert response.status_code == 503


def test_ranking_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(app.state, "prediction_service", None, raising=False)
    response = client.get("/api/v1/ranking")
    assert response.status_code == 503


def test_model_details_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(app.state, "prediction_service", None, raising=False)
    response = client.get("/api/v1/model-details")
    assert response.status_code == 503


def test_health_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(app.state, "recommendation_service", None, raising=False)
    monkeypatch.setattr(
        app.state, "recommendation_service_error", "boom", raising=False
    )
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["global_status"] == "Offline"
    assert body["groups"][0]["services"][0] == {"name": "modelo-ml", "status": "error"}


def test_health_returns_ok_with_model(monkeypatch):
    monkeypatch.setattr(app.state, "recommendation_service", MagicMock(), raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["global_status"] == "Online"
    assert body["groups"][0]["services"][0] == {"name": "modelo-ml", "status": "ok"}


def test_lifespan_handles_recommendation_service_failure():
    with patch(
        "app.main.build_recommendation_service", side_effect=RuntimeError("init failed")
    ):
        test_app = FastAPI(lifespan=lifespan)
        with TestClient(test_app) as client:
            response = client.get("/")
            assert response.status_code == 404
            assert test_app.state.recommendation_service is None
            assert test_app.state.recommendation_service_error == "init failed"


def test_lifespan_keeps_serving_when_the_prediction_assets_fail(caplog):
    with (
        patch(
            "app.main.build_prediction_service",
            side_effect=FileNotFoundError("publication_model.pkl"),
        ),
        patch("app.main.build_recommendation_service", return_value=MagicMock()),
        patch("app.main.settings.GRS_PRECOMPUTE_ON_STARTUP", False),
    ):
        test_app = FastAPI(lifespan=lifespan)
        with caplog.at_level("ERROR"):
            with TestClient(test_app):
                assert test_app.state.prediction_service is None

    assert "publication_model.pkl" in caplog.text
