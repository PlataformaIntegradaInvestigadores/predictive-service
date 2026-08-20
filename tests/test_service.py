from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.prediction_service import PredictionService


@pytest.fixture
def mock_service():
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
        "year",
        "affiliation_encoded",
        "publication_count",
        "distinct_authors",
    ]

    service.historical_df = pd.DataFrame(
        {
            "affiliation_name": ["MIT", "MIT", "Stanford", "Stanford", "EPN"],
            "year": [2020, 2021, 2020, 2021, 2021],
            "publication_count": [50, 60, 30, 35, 10],
            "distinct_authors": [20, 25, 15, 18, 5],
        }
    )

    return service


class TestCheckAssetsLoaded:
    def test_returns_false_when_model_is_none(self, mock_service):
        mock_service.model = None
        assert mock_service.check_assets_loaded() is False

    def test_returns_false_when_encoder_is_none(self, mock_service):
        mock_service.encoder = None
        assert mock_service.check_assets_loaded() is False

    def test_returns_false_when_dataframe_is_none(self, mock_service):
        mock_service.historical_df = None
        assert mock_service.check_assets_loaded() is False

    def test_returns_true_when_all_loaded(self, mock_service):
        assert mock_service.check_assets_loaded() is True

    def test_returns_false_when_nothing_loaded(self):
        with patch.object(PredictionService, "_load_assets", return_value=None):
            service = PredictionService()
        assert service.check_assets_loaded() is False


class TestGetAllAffiliations:
    def test_returns_list_of_strings(self, mock_service):
        result = mock_service.get_all_affiliations()
        assert result == ["MIT", "Stanford", "EPN"]

    def test_return_type_is_list(self, mock_service):
        result = mock_service.get_all_affiliations()
        assert isinstance(result, list)


class TestGetProjection:
    def test_returns_error_for_unknown_affiliation(self, mock_service):
        result = mock_service.get_projection("UnknownUniversity", 3)
        assert "error" in result

    def test_returns_correct_structure(self, mock_service):
        result = mock_service.get_projection("MIT", 3)
        assert "affiliation_name" in result
        assert "data" in result
        assert result["affiliation_name"] == "MIT"

    def test_historical_data_has_actual_type(self, mock_service):
        result = mock_service.get_projection("MIT", 2)
        actual_points = [d for d in result["data"] if d["type"] == "actual"]
        assert len(actual_points) == 2

    def test_predicted_data_has_predicted_type(self, mock_service):
        result = mock_service.get_projection("MIT", 3)
        predicted_points = [d for d in result["data"] if d["type"] == "predicted"]
        assert len(predicted_points) == 3

    def test_predicted_years_are_sequential(self, mock_service):
        result = mock_service.get_projection("MIT", 3)
        predicted_points = [d for d in result["data"] if d["type"] == "predicted"]
        years = [p["year"] for p in predicted_points]
        assert years == [2022, 2023, 2024]

    def test_model_predict_is_called(self, mock_service):
        mock_service.get_projection("MIT", 2)
        assert mock_service.model.predict.called

    def test_projection_with_hypothetical_authors(self, mock_service):
        result = mock_service.get_projection("MIT", 2, hypothetical_authors=100)
        assert "error" not in result
        predicted_points = [d for d in result["data"] if d["type"] == "predicted"]
        assert len(predicted_points) == 2


class TestGetRanking:
    def test_returns_list(self, mock_service):
        result = mock_service.get_ranking()
        assert isinstance(result, list)

    def test_ranking_has_correct_fields(self, mock_service):
        result = mock_service.get_ranking()
        assert len(result) > 0
        item = result[0]
        assert "rank" in item
        assert "affiliation_name" in item
        assert "current_year_publications" in item
        assert "predicted_next_year_publications" in item
        assert "growth" in item
        assert "growth_percentage" in item

    def test_rank_numbers_are_sequential(self, mock_service):
        result = mock_service.get_ranking()
        ranks = [item["rank"] for item in result]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_all_affiliations_included(self, mock_service):
        result = mock_service.get_ranking()
        names = [item["affiliation_name"] for item in result]
        assert "MIT" in names
        assert "Stanford" in names
        assert "EPN" in names


class TestGetModelDetails:
    def test_returns_correct_model_type(self, mock_service):
        result = mock_service.get_model_details()
        assert result["model_type"] == "LightGBM Regressor"

    def test_returns_total_affiliations(self, mock_service):
        result = mock_service.get_model_details()
        assert result["total_affiliations"] == 3

    def test_returns_performance_metrics(self, mock_service):
        result = mock_service.get_model_details()
        assert "performance_metrics" in result
        metrics = result["performance_metrics"]
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_returns_feature_importances(self, mock_service):
        result = mock_service.get_model_details()
        assert "feature_importances" in result
        fi = result["feature_importances"]
        assert "year" in fi
        assert "affiliation_encoded" in fi
        assert "publication_count" in fi
        assert "distinct_authors" in fi
