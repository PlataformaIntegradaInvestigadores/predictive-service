import os
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.adapters.outbound.csv_article_repository import CsvArticleRepository
from app.adapters.outbound.publication_assets import (
    TRAINING_REPORT,
    CsvPublicationHistory,
    LightGbmPublicationModel,
)
from app.bootstrap import ASSETS_DIR

MODEL = ASSETS_DIR / "publication_model.pkl"
ENCODER = ASSETS_DIR / "affiliation_encoder.pkl"


class TestCsvArticleRepository:
    def test_raises_when_no_data_path(self):
        with pytest.raises(RuntimeError, match="No se configuró CENTINELA_DATA_PATH"):
            CsvArticleRepository(None).load()

    def test_raises_when_data_path_not_exists(self):
        with pytest.raises(FileNotFoundError, match="La carpeta de datos no existe"):
            CsvArticleRepository("/nonexistent/path").load()

    def test_raises_when_required_files_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Faltan archivos requeridos"):
            CsvArticleRepository(str(tmp_path)).load()

    def test_loads_the_three_tables(self):
        dataset = CsvArticleRepository(os.environ["CENTINELA_DATA_PATH"]).load()

        assert len(dataset.articles) == 200
        assert len(dataset.author_article) == 200
        assert len(dataset.topic_article) == 2000


class TestCsvPublicationHistory:
    def test_returns_only_that_affiliation_oldest_first(self, tmp_path):
        path = tmp_path / "history.csv"
        pd.DataFrame(
            {
                "affiliation_name": ["A", "B", "A"],
                "year": [2021, 2020, 2019],
                "publication_count": [3, 9, 1],
                "distinct_authors": [2, 4, 1],
            }
        ).to_csv(path, index=False)

        rows = CsvPublicationHistory(path).for_affiliation("A")

        assert rows["year"].tolist() == [2019, 2021]

    def test_unknown_affiliation_gives_empty_frame(self, tmp_path):
        path = tmp_path / "history.csv"
        pd.DataFrame(
            {
                "affiliation_name": ["A"],
                "year": [2020],
                "publication_count": [1],
                "distinct_authors": [1],
            }
        ).to_csv(path, index=False)

        assert CsvPublicationHistory(path).for_affiliation("Z").empty


@pytest.fixture(scope="module")
def model():
    return LightGbmPublicationModel(MODEL, ENCODER)


class TestLightGbmPublicationModel:
    def test_lists_the_affiliations_of_the_encoder(self, model):
        classes = joblib.load(ENCODER).classes_.tolist()

        assert model.affiliations() == classes

    def test_prediction_matches_calling_the_pickled_model_directly(self, model):
        encoder = joblib.load(ENCODER)
        lgbm = joblib.load(MODEL)
        lgbm.set_params(n_jobs=1)  # same result, without the OpenMP start-up cost
        name = model.affiliations()[0]
        row = pd.DataFrame(
            [[2022, encoder.transform([name])[0], 12, 5]],
            columns=[
                "year",
                "affiliation_encoded",
                "publication_count",
                "distinct_authors",
            ],
        )

        assert model.predict_next(2022, name, 12, 5) == round(lgbm.predict(row)[0])

    def test_single_row_predictions_do_not_use_a_thread_pool(self, model):
        assert model._model.get_params()["n_jobs"] == 1

    def test_unknown_affiliation_is_rejected(self, model):
        with pytest.raises(KeyError):
            model.predict_next(2022, "no existe", 1, 1)

    def test_describe_combines_training_report_and_feature_importances(self, model):
        details = model.describe()

        assert details["performance_metrics"] == TRAINING_REPORT["performance_metrics"]
        assert set(details["feature_importances"]) == {
            "year",
            "affiliation_encoded",
            "publication_count",
            "distinct_authors",
        }

    def test_missing_file_fails_loudly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LightGbmPublicationModel(Path(tmp_path / "x.pkl"), ENCODER)
