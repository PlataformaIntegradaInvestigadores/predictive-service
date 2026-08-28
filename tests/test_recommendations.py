from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.recommendation_service import RecommendationService


@pytest.fixture
def mock_recommendation_service():
    with patch.object(
        RecommendationService, "_load_data"
    ), patch.object(RecommendationService, "_initialize_grs"):
        service = RecommendationService.__new__(RecommendationService)

    service.groups = {
        0: {
            "members": ["u1", "u2"],
            "n_members": 2,
            "papers_total": 5,
            "topics_p1": {"t1"},
            "topics_p2": {"t2"},
        }
    }
    service.all_topics = {"t1", "t2", "t3"}
    service.grs = MagicMock()
    service.grs.recommend_for_group.return_value = pd.DataFrame(
        {
            "topic": ["t3", "t4", "t5"],
            "novelty": [0.8, 0.6, 0.4],
            "relevance": [0.9, 0.7, 0.5],
            "collaborative_signal": [0.3, 0.2, 0.1],
            "score": [0.85, 0.65, 0.45],
        }
    )
    service.recommendations_df = pd.DataFrame(
        {
            "group_id": [0, 0, 0],
            "n_members": [2, 2, 2],
            "papers_total": [5, 5, 5],
            "rank": [1, 2, 3],
            "topic": ["t3", "t4", "t5"],
            "novelty": [0.8, 0.6, 0.4],
            "relevance": [0.9, 0.7, 0.5],
            "collaborative_signal": [0.3, 0.2, 0.1],
            "score": [0.85, 0.65, 0.45],
            "is_new_vs_recent": [1, 0, 1],
        }
    )
    service._metrics_k = None

    return service


@pytest.fixture
def recommendations_client(mock_recommendation_service):
    with TestClient(app) as client:
        app.state.recommendation_service = mock_recommendation_service
        app.state.recommendation_service_error = None
        yield client
    app.state.recommendation_service = None
    app.state.recommendation_service_error = None


class TestRecommendationsEndpoints:
    def test_health_returns_200(self, recommendations_client):
        response = recommendations_client.get("/api/v1/recommendations/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service_initialized"] is True

    def test_health_returns_unavailable_without_service(self):
        with TestClient(app) as client:
            app.state.recommendation_service = None
            app.state.recommendation_service_error = "boom"
            response = client.get("/api/v1/recommendations/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["service_initialized"] is False
        assert data["error"] == "boom"
        app.state.recommendation_service = None
        app.state.recommendation_service_error = None

    def test_get_all_groups(self, recommendations_client):
        response = recommendations_client.get("/api/v1/recommendations/groups")
        assert response.status_code == 200
        data = response.json()
        assert "groups" in data
        assert "total_groups" in data

    def test_get_group_recommendations(self, recommendations_client):
        response = recommendations_client.get("/api/v1/recommendations/group/0")
        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == 0
        assert "recommendations" in data

    def test_get_group_recommendations_not_found(self, recommendations_client):
        response = recommendations_client.get("/api/v1/recommendations/group/999")
        assert response.status_code == 404

    def test_get_recommendations_by_members_linked(self, recommendations_client):
        payload = {"scopus_ids": ["u1", "u2"]}
        response = recommendations_client.post(
            "/api/v1/recommendations/by-members", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["linked"] is True
        assert "recommendations" in data

    def test_get_recommendations_by_members_not_linked(self, recommendations_client):
        payload = {"scopus_ids": ["unknown"]}
        response = recommendations_client.post(
            "/api/v1/recommendations/by-members", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["linked"] is False

    def test_get_metrics(self, recommendations_client):
        response = recommendations_client.get("/api/v1/recommendations/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "groups_persistent" in data
        assert "coverage" in data
        assert "diversity" in data

    def test_get_all_groups_server_error(self, recommendations_client):
        original = recommendations_client.app.state.recommendation_service.get_all_recommendations
        def boom(**kwargs):
            raise Exception("boom")
        recommendations_client.app.state.recommendation_service.get_all_recommendations = boom
        response = recommendations_client.get("/api/v1/recommendations/groups")
        assert response.status_code == 500
        recommendations_client.app.state.recommendation_service.get_all_recommendations = original

    def test_get_group_recommendations_server_error(self, recommendations_client):
        original = recommendations_client.app.state.recommendation_service.get_group_recommendations
        def boom(**kwargs):
            raise Exception("boom")
        recommendations_client.app.state.recommendation_service.get_group_recommendations = boom
        response = recommendations_client.get("/api/v1/recommendations/group/0")
        assert response.status_code == 500
        recommendations_client.app.state.recommendation_service.get_group_recommendations = original

    def test_get_recommendations_by_members_server_error(self, recommendations_client):
        original = recommendations_client.app.state.recommendation_service.get_group_recommendations
        def boom(**kwargs):
            raise Exception("boom")
        recommendations_client.app.state.recommendation_service.get_group_recommendations = boom
        payload = {"scopus_ids": ["u1", "u2"]}
        response = recommendations_client.post(
            "/api/v1/recommendations/by-members", json=payload
        )
        assert response.status_code == 500
        recommendations_client.app.state.recommendation_service.get_group_recommendations = original

    def test_get_metrics_server_error(self, recommendations_client):
        original = recommendations_client.app.state.recommendation_service.get_metrics
        def boom(**kwargs):
            raise Exception("boom")
        recommendations_client.app.state.recommendation_service.get_metrics = boom
        response = recommendations_client.get("/api/v1/recommendations/metrics")
        assert response.status_code == 500
        recommendations_client.app.state.recommendation_service.get_metrics = original

    def test_by_members_invalid_payload(self, recommendations_client):
        response = recommendations_client.post(
            "/api/v1/recommendations/by-members",
            json={"scopus_ids": "not-a-list"},
        )
        assert response.status_code == 422

    def test_endpoints_return_503_when_service_unavailable(self):
        with TestClient(app) as client:
            client.app.state.recommendation_service = None
            client.app.state.recommendation_service_error = "boom"
            for path in [
                "/api/v1/recommendations/groups",
                "/api/v1/recommendations/group/0",
                "/api/v1/recommendations/metrics",
            ]:
                response = client.get(path)
                assert response.status_code == 503


class TestRecommendationService:
    def test_find_group_by_members_found(self, mock_recommendation_service):
        group_id = mock_recommendation_service.find_group_by_members(["u1", "u2"])
        assert group_id == 0

    def test_find_group_by_members_not_found(self, mock_recommendation_service):
        group_id = mock_recommendation_service.find_group_by_members(["unknown"])
        assert group_id is None

    def test_get_group_recommendations(self, mock_recommendation_service):
        result = mock_recommendation_service.get_group_recommendations(group_id=0, k=3)
        assert result["group_id"] == 0
        assert len(result["recommendations"]) == 3
        assert result["recommendations"][0]["rank"] == 1

    def test_get_group_recommendations_not_found(self, mock_recommendation_service):
        with pytest.raises(KeyError):
            mock_recommendation_service.get_group_recommendations(group_id=999)

    def test_get_all_recommendations(self, mock_recommendation_service):
        result = mock_recommendation_service.get_all_recommendations(limit=10, k=3)
        assert "total_groups" in result
        assert "groups" in result
        assert result["total_groups"] == 1

    def test_get_metrics(self, mock_recommendation_service):
        result = mock_recommendation_service.get_metrics(k=3)
        assert "groups_persistent" in result
        assert "topics_unique" in result
        assert "coverage" in result
        assert "fairness_gini" in result
