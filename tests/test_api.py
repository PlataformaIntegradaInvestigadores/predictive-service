from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_affiliations_returns_503_without_model():
    """En CI no hay modelos cargados: el servicio debe responder 503, no 500."""
    response = client.get("/api/v1/affiliations")
    assert response.status_code in (200, 503)


def test_ranking_returns_503_without_model():
    response = client.get("/api/v1/ranking")
    assert response.status_code in (200, 503)


def test_model_details_returns_503_without_model():
    response = client.get("/api/v1/model-details")
    assert response.status_code in (200, 503)
