"""In-memory implementations of the application ports, for tests."""

import pandas as pd

from app.application.ports import ArticleDataset
from app.application.prediction_service import PredictionService

AFFILIATIONS = ["MIT", "Stanford", "EPN"]


def history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "affiliation_name": ["MIT", "MIT", "Stanford", "Stanford", "EPN"],
            "year": [2020, 2021, 2020, 2021, 2021],
            "publication_count": [50, 60, 30, 35, 10],
            "distinct_authors": [20, 25, 15, 18, 5],
        }
    )


class FakePublicationModel:
    def __init__(self, affiliations=None, prediction=55):
        self._affiliations = list(
            AFFILIATIONS if affiliations is None else affiliations
        )
        self.prediction = prediction
        self.calls = []

    def affiliations(self):
        return self._affiliations

    def predict_next(self, year, affiliation_name, publications, authors):
        self.calls.append((year, affiliation_name, publications, authors))
        return self.prediction

    def describe(self):
        return {
            "model_type": "LightGBM Regressor",
            "training_data_range": "Datos históricos hasta el año 2021",
            "target_variable": "Número de publicaciones del año siguiente",
            "performance_metrics": {"mae": 2.20, "rmse": 4.67},
            "feature_importances": {
                "year": 0.25,
                "affiliation_encoded": 0.35,
                "publication_count": 0.30,
                "distinct_authors": 0.10,
            },
        }


class FakePublicationHistory:
    def __init__(self, frame=None):
        self.frame = history_frame() if frame is None else frame

    def for_affiliation(self, affiliation_name):
        rows = self.frame[self.frame["affiliation_name"] == affiliation_name]
        return rows.sort_values("year")


def make_prediction_service(model=None, history_frame_=None) -> PredictionService:
    return PredictionService(
        model=model or FakePublicationModel(),
        history=FakePublicationHistory(history_frame_),
    )


def article_dataset() -> ArticleDataset:
    """Smallest dataset that passes DataValidator and yields persistent groups."""
    return ArticleDataset(
        articles=pd.DataFrame(
            {
                "scopus_id": [f"a{i}" for i in range(200)],
                "publication_date": ["2020-01-01"] * 200,
            }
        ),
        author_article=pd.DataFrame(
            {
                "scopus_id": [f"a{i}" for i in range(200)],
                "authid": [f"u{i % 10 + 1}" for i in range(200)],
            }
        ),
        topic_article=pd.DataFrame(
            {
                "article_id": [f"a{i}" for i in range(2000)],
                "topic": [f"t{i % 20 + 1}" for i in range(2000)],
            }
        ),
    )


class FakeArticleRepository:
    def __init__(self, dataset=None):
        self.dataset = dataset if dataset is not None else article_dataset()

    def load(self):
        return self.dataset
