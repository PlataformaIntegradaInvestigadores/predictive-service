import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.prediction_service import PredictionService


@pytest.fixture
def mock_prediction_service():
    """Provide a fully-mocked PredictionService for integration tests."""
    with patch.object(PredictionService, "_load_assets", return_value=None):
        service = PredictionService()

    service.encoder = MagicMock()
    service.encoder.classes_ = np.array(["MIT", "Stanford", "EPN"])
    service.encoder.transform = MagicMock(
        side_effect=lambda names: np.array(
            [list(service.encoder.classes_).index(n) for n in names]
        )
    )

    service.model = MagicMock()
    service.model.predict = MagicMock(return_value=np.array([55.0]))
    service.model.feature_importances_ = np.array([0.25, 0.35, 0.30, 0.10])
    service.model.feature_name_ = [
        "year", "affiliation_encoded", "publication_count", "distinct_authors"
    ]

    service.historical_df = pd.DataFrame({
        "affiliation_name": ["MIT", "MIT", "Stanford", "Stanford", "EPN"],
        "year": [2020, 2021, 2020, 2021, 2021],
        "publication_count": [50, 60, 30, 35, 10],
        "distinct_authors": [20, 25, 15, 18, 5],
    })

    return service


@pytest.fixture
def client(mock_prediction_service):
    """TestClient with mocked service injected via dependency override."""
    from app.api.v1.endpoints.analytics import get_prediction_service

    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.integration
class TestAffiliationsEndpoint:
    def test_get_affiliations_returns_200(self, client):
        response = client.get("/api/v1/affiliations")
        assert response.status_code == 200

    def test_get_affiliations_returns_list(self, client):
        response = client.get("/api/v1/affiliations")
        data = response.json()
        assert "affiliations" in data
        assert data["affiliations"] == ["MIT", "Stanford", "EPN"]

    def test_get_affiliations_content_type_json(self, client):
        response = client.get("/api/v1/affiliations")
        assert "application/json" in response.headers["content-type"]


@pytest.mark.integration
class TestProjectionEndpoint:
    def test_get_projection_valid_affiliation(self, client):
        response = client.get("/api/v1/projection/MIT?years=3")
        assert response.status_code == 200
        data = response.json()
        assert data["affiliation_name"] == "MIT"
        assert len(data["data"]) > 0

    def test_get_projection_unknown_affiliation_returns_error(self, client):
        response = client.get("/api/v1/projection/UnknownUniv?years=3")
        assert response.status_code in [404, 200]
        data = response.json()
        if response.status_code == 200:
            assert "error" in data

    def test_get_projection_with_hypothetical_authors(self, client):
        response = client.get("/api/v1/projection/MIT?projection_years=2&hypothetical_authors=50")
        assert response.status_code == 200
        data = response.json()
        predicted = [d for d in data["data"] if d["type"] == "predicted"]
        assert len(predicted) == 2

    def test_get_projection_response_structure(self, client):
        response = client.get("/api/v1/projection/Stanford?years=1")
        assert response.status_code == 200
        data = response.json()
        assert "affiliation_name" in data
        assert "data" in data
        for point in data["data"]:
            assert "year" in point
            assert "publications" in point
            assert "type" in point
            assert point["type"] in ["actual", "predicted"]


@pytest.mark.integration
class TestCompareEndpoint:
    def test_post_compare_valid(self, client):
        payload = {"affiliation_names": ["MIT", "Stanford"]}
        response = client.post("/api/v1/projection/compare?projection_years=3", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_post_compare_single_affiliation(self, client):
        payload = {"affiliation_names": ["EPN"]}
        response = client.post("/api/v1/projection/compare?projection_years=2", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1


@pytest.mark.integration
class TestRankingEndpoint:
    def test_get_ranking_returns_200(self, client):
        response = client.get("/api/v1/ranking")
        assert response.status_code == 200

    def test_get_ranking_returns_all_affiliations(self, client):
        response = client.get("/api/v1/ranking")
        data = response.json()
        assert "ranking" in data
        names = [item["affiliation_name"] for item in data["ranking"]]
        assert "MIT" in names
        assert "Stanford" in names
        assert "EPN" in names

    def test_get_ranking_items_have_rank_field(self, client):
        response = client.get("/api/v1/ranking")
        data = response.json()
        for item in data["ranking"]:
            assert "rank" in item
            assert "growth" in item
            assert "growth_percentage" in item


@pytest.mark.integration
class TestModelDetailsEndpoint:
    def test_get_model_details_returns_200(self, client):
        response = client.get("/api/v1/model-details")
        assert response.status_code == 200

    def test_get_model_details_response_structure(self, client):
        response = client.get("/api/v1/model-details")
        data = response.json()
        assert "model_type" in data
        assert "total_affiliations" in data
        assert "performance_metrics" in data
        assert "feature_importances" in data

    def test_get_model_details_performance_metrics(self, client):
        response = client.get("/api/v1/model-details")
        data = response.json()
        metrics = data["performance_metrics"]
        assert "mae" in metrics
        assert "rmse" in metrics


@pytest.mark.integration
class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_welcome_message(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
