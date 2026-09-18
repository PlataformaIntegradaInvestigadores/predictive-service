import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AffiliationListResponse,
    ComparisonData,
    ComparisonResponse,
    DataPoint,
    ErrorResponse,
    ModelDetailsResponse,
    ModelPerformance,
    ProjectionResponse,
    RankingItem,
    RankingResponse,
)


class TestDataPoint:
    def test_valid_actual(self):
        dp = DataPoint(year=2023, publications=42, type="actual")
        assert dp.year == 2023
        assert dp.publications == 42
        assert dp.type == "actual"

    def test_valid_predicted(self):
        dp = DataPoint(year=2025, publications=100, type="predicted")
        assert dp.type == "predicted"

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            DataPoint(year=2023, publications=10, type="forecast")

    def test_missing_year_raises(self):
        with pytest.raises(ValidationError):
            DataPoint(publications=10, type="actual")

    def test_missing_publications_raises(self):
        with pytest.raises(ValidationError):
            DataPoint(year=2023, type="actual")

    def test_wrong_year_type_raises(self):
        with pytest.raises(ValidationError):
            DataPoint(year="not_a_number", publications=10, type="actual")


class TestErrorResponse:
    def test_valid(self):
        er = ErrorResponse(error="Something went wrong")
        assert er.error == "Something went wrong"

    def test_missing_error_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse()


class TestAffiliationListResponse:
    def test_valid(self):
        resp = AffiliationListResponse(affiliations=["MIT", "Stanford", "EPN"])
        assert len(resp.affiliations) == 3

    def test_empty_list_valid(self):
        resp = AffiliationListResponse(affiliations=[])
        assert resp.affiliations == []


class TestProjectionResponse:
    def test_valid(self):
        data = [
            DataPoint(year=2020, publications=10, type="actual"),
            DataPoint(year=2021, publications=12, type="actual"),
            DataPoint(year=2022, publications=15, type="predicted"),
        ]
        resp = ProjectionResponse(affiliation_name="MIT", data=data)
        assert resp.affiliation_name == "MIT"
        assert len(resp.data) == 3

    def test_empty_data_valid(self):
        resp = ProjectionResponse(affiliation_name="EPN", data=[])
        assert resp.data == []

    def test_invalid_data_item_raises(self):
        with pytest.raises(ValidationError):
            ProjectionResponse(
                affiliation_name="MIT",
                data=[{"year": 2020, "publications": 10, "type": "invalid"}],
            )


class TestComparisonResponse:
    def test_valid(self):
        item = ComparisonData(
            affiliation_name="Stanford",
            data=[DataPoint(year=2023, publications=50, type="actual")],
        )
        resp = ComparisonResponse(results=[item])
        assert len(resp.results) == 1

    def test_empty_results_valid(self):
        resp = ComparisonResponse(results=[])
        assert resp.results == []


class TestRankingItem:
    def test_valid(self):
        item = RankingItem(
            rank=1,
            affiliation_name="MIT",
            current_year_publications=100,
            predicted_next_year_publications=120,
            growth=20,
            growth_percentage=20.0,
        )
        assert item.rank == 1
        assert item.growth_percentage == 20.0

    def test_negative_growth_valid(self):
        item = RankingItem(
            rank=5,
            affiliation_name="EPN",
            current_year_publications=50,
            predicted_next_year_publications=40,
            growth=-10,
            growth_percentage=-20.0,
        )
        assert item.growth == -10

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            RankingItem(rank=1, affiliation_name="MIT")


class TestRankingResponse:
    def test_valid(self):
        items = [
            RankingItem(
                rank=i,
                affiliation_name=f"Univ{i}",
                current_year_publications=100 - i * 10,
                predicted_next_year_publications=110 - i * 10,
                growth=10,
                growth_percentage=10.0 + i,
            )
            for i in range(1, 4)
        ]
        resp = RankingResponse(ranking=items)
        assert len(resp.ranking) == 3


class TestModelPerformance:
    def test_valid(self):
        mp = ModelPerformance(mae=2.20, rmse=4.67)
        assert mp.mae == 2.20
        assert mp.rmse == 4.67

    def test_missing_mae_raises(self):
        with pytest.raises(ValidationError):
            ModelPerformance(rmse=4.67)

    def test_missing_rmse_raises(self):
        with pytest.raises(ValidationError):
            ModelPerformance(mae=2.20)


class TestModelDetailsResponse:
    def test_valid(self):
        resp = ModelDetailsResponse(
            model_type="LightGBM Regressor",
            training_data_range="Datos historicos hasta 2021",
            target_variable="Numero de publicaciones del ano siguiente",
            total_affiliations=10,
            performance_metrics=ModelPerformance(mae=2.20, rmse=4.67),
            feature_importances={"year": 0.3, "affiliation_encoded": 0.5},
        )
        assert resp.model_type == "LightGBM Regressor"
        assert resp.total_affiliations == 10

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            ModelDetailsResponse(model_type="LightGBM Regressor")
