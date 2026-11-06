import pandas as pd
import pytest

from app.domain.errors import AffiliationNotFound
from tests.fakes import (
    FakePublicationModel,
    history_frame,
    make_prediction_service,
)


@pytest.fixture
def model():
    return FakePublicationModel()


@pytest.fixture
def mock_service(model):
    return make_prediction_service(model=model)


class TestGetAllAffiliations:
    def test_returns_list_of_strings(self, mock_service):
        result = mock_service.get_all_affiliations()
        assert result == ["MIT", "Stanford", "EPN"]

    def test_return_type_is_list(self, mock_service):
        result = mock_service.get_all_affiliations()
        assert isinstance(result, list)


class TestGetProjection:
    def test_raises_for_unknown_affiliation(self, mock_service):
        with pytest.raises(AffiliationNotFound, match="no fue encontrada"):
            mock_service.get_projection("UnknownUniversity", 3)

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

    def test_model_predict_is_called_once_per_projected_year(
        self, mock_service, model
    ):
        mock_service.get_projection("MIT", 2)
        assert [call[:2] for call in model.calls] == [(2022, "MIT"), (2023, "MIT")]

    def test_projection_with_hypothetical_authors(self, mock_service):
        result = mock_service.get_projection("MIT", 2, hypothetical_authors=100)
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


class TestGetProjectionEdgeCases:
    def test_raises_for_unknown_affiliation(self, mock_service):
        with pytest.raises(AffiliationNotFound, match="no fue encontrada"):
            mock_service.get_projection("UnknownUniversity", 3)

    def test_raises_when_affiliation_not_known_by_the_model(self):
        service = make_prediction_service(model=FakePublicationModel(["MIT"]))
        with pytest.raises(AffiliationNotFound):
            service.get_projection("Stanford", 3)

    def test_raises_when_no_historical_data(self):
        service = make_prediction_service(
            history_frame_=pd.DataFrame(
                {
                    "affiliation_name": ["MIT"],
                    "year": [2020],
                    "publication_count": [10],
                    "distinct_authors": [5],
                }
            )
        )
        with pytest.raises(AffiliationNotFound, match="No hay datos históricos"):
            service.get_projection("Stanford", 3)

    def test_projection_with_zero_authors(self):
        service = make_prediction_service(
            history_frame_=pd.DataFrame(
                {
                    "affiliation_name": ["MIT", "MIT"],
                    "year": [2020, 2021],
                    "publication_count": [10, 12],
                    "distinct_authors": [0, 0],
                }
            )
        )
        result = service.get_projection("MIT", 2)
        assert "affiliation_name" in result


class TestGetRankingEdgeCases:
    def test_returns_empty_for_empty_historical(self):
        service = make_prediction_service(history_frame_=history_frame().iloc[0:0])
        result = service.get_ranking()
        assert result == []

    def test_returns_cached_ranking(self, mock_service):
        mock_service._ranking_cache = [{"rank": 1, "affiliation_name": "MIT"}]
        mock_service._ranking_cache_timestamp = 9999999999
        result = mock_service.get_ranking()
        assert result == [{"rank": 1, "affiliation_name": "MIT"}]


class TestGetModelDetailsEdgeCases:
    def test_returns_model_details(self, mock_service):
        result = mock_service.get_model_details()
        assert "training_data_range" in result
        assert "target_variable" in result
