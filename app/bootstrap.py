"""Raíz de composición: único lugar que conoce las clases concretas.

Elige los adaptadores (CSV, archivos .pkl), los inyecta en los casos de uso y
lee la configuración. Todo lo demás depende solo de los puertos.
"""

from __future__ import annotations

from pathlib import Path

from app.adapters.outbound.csv_article_repository import CsvArticleRepository
from app.adapters.outbound.publication_assets import (
    CsvPublicationHistory,
    LightGbmPublicationModel,
)
from app.application.prediction_service import PredictionService
from app.application.recommendation_service import GrsPeriods, RecommendationService
from app.core.config import settings

ASSETS_DIR = Path(__file__).resolve().parent.parent


def build_recommendation_service() -> RecommendationService:
    return RecommendationService(
        repository=CsvArticleRepository(settings.CENTINELA_DATA_PATH),
        periods=GrsPeriods(
            period1_start=settings.GRS_PERIOD1_START,
            period1_end=settings.GRS_PERIOD1_END,
            period2_start=settings.GRS_PERIOD2_START,
            period2_end=settings.GRS_PERIOD2_END,
        ),
    )


def build_prediction_service() -> PredictionService:
    return PredictionService(
        model=LightGbmPublicationModel(
            ASSETS_DIR / "publication_model.pkl",
            ASSETS_DIR / "affiliation_encoder.pkl",
        ),
        history=CsvPublicationHistory(ASSETS_DIR / "publication_data.csv"),
    )
