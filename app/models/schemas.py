# backend/app/models/schemas.py

from typing import Literal

from pydantic import BaseModel, Field

# --- Modelos de Datos Base ---


class DataPoint(BaseModel):
    """
    Representa un único punto de datos para un gráfico (real o predicho).
    """

    year: int
    publications: int
    type: Literal["actual", "predicted"]


class ErrorResponse(BaseModel):
    """
    Define la estructura estándar para los mensajes de error.
    """

    error: str


# --- Endpoints Principales ---


class AffiliationListResponse(BaseModel):
    """Respuesta para la lista de todas las afiliaciones disponibles."""

    affiliations: list[str]


class ProjectionResponse(BaseModel):
    """Respuesta para la proyección de una única afiliación."""

    affiliation_name: str
    data: list[DataPoint]


# --- Endpoint de Comparación ---


class ComparisonData(BaseModel):
    """Contiene los datos de proyección para una afiliación en la comparación."""

    affiliation_name: str
    data: list[DataPoint]


class ComparisonResponse(BaseModel):
    """Respuesta para la comparación de proyecciones entre varias afiliaciones."""

    results: list[ComparisonData]


# --- Endpoint de Ranking ---


class RankingItem(BaseModel):
    """Representa un item en el ranking de crecimiento."""

    rank: int
    affiliation_name: str
    current_year_publications: int
    predicted_next_year_publications: int
    growth: int
    growth_percentage: float


class RankingResponse(BaseModel):
    """Respuesta para el ranking de afiliaciones por proyección de crecimiento."""

    ranking: list[RankingItem]


# --- Endpoint de Detalles del Modelo (ACTUALIZADO) ---


class ModelPerformance(BaseModel):
    """Define la estructura para las métricas de rendimiento del modelo."""

    mae: float = Field(..., description="Mean Absolute Error (Error Absoluto Medio)")
    rmse: float = Field(
        ..., description="Root Mean Squared Error (Raíz del Error Cuadrático Medio)"
    )


class ModelDetailsResponse(BaseModel):
    """
    Respuesta que proporciona detalles y metadatos sobre el modelo de ML,
    basado en los resultados del notebook de entrenamiento.
    """

    model_type: str
    training_data_range: str
    target_variable: str
    total_affiliations: int
    performance_metrics: ModelPerformance
    feature_importances: dict[str, float]
