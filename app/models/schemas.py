# backend/app/models/schemas.py

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# --- Modelos de Datos Base ---


class DataPoint(BaseModel):
    """
    Representa un único punto de datos para un gráfico (real o predicho).
    """

    year: int = Field(..., description="Año calendario del punto.", examples=[2024])
    publications: int = Field(
        ..., description="Número de publicaciones en ese año.", examples=[312]
    )
    type: Literal["actual", "predicted"] = Field(
        ...,
        description=(
            "`actual` si el dato es histórico; `predicted` si lo estima el modelo."
        ),
        examples=["actual"],
    )


class ErrorResponse(BaseModel):
    """
    Define la estructura estándar para los mensajes de error.
    """

    error: str = Field(
        ..., description="Motivo del error.", examples=["Afiliación no encontrada"]
    )


# --- Endpoints Principales ---


class AffiliationListResponse(BaseModel):
    """Respuesta para la lista de todas las afiliaciones disponibles."""

    affiliations: list[str] = Field(
        ...,
        description="Nombres de afiliaciones con historial de publicaciones.",
        examples=[["Escuela Politécnica Nacional", "Universidad de Cuenca"]],
    )


class ProjectionResponse(BaseModel):
    """Respuesta para la proyección de una única afiliación."""

    affiliation_name: str = Field(
        ..., description="Nombre de la afiliación.", examples=["Universidad de Cuenca"]
    )
    data: list[DataPoint] = Field(
        ..., description="Serie histórica seguida de los años proyectados."
    )


# --- Endpoint de Comparación ---


class ComparisonData(BaseModel):
    """Contiene los datos de proyección para una afiliación en la comparación."""

    affiliation_name: str = Field(..., description="Nombre de la afiliación.")
    data: list[DataPoint] = Field(
        ..., description="Serie histórica seguida de los años proyectados."
    )


class ComparisonResponse(BaseModel):
    """Respuesta para la comparación de proyecciones entre varias afiliaciones."""

    results: list[ComparisonData] = Field(
        ...,
        description=(
            "Una proyección por afiliación encontrada. "
            "Las afiliaciones desconocidas se omiten."
        ),
    )


# --- Endpoint de Ranking ---


class RankingItem(BaseModel):
    """Representa un item en el ranking de crecimiento."""

    rank: int = Field(
        ..., description="Posición en el ranking (1 = mayor crecimiento)."
    )
    affiliation_name: str = Field(..., description="Nombre de la afiliación.")
    current_year_publications: int = Field(
        ..., description="Publicaciones del último año con datos."
    )
    predicted_next_year_publications: int = Field(
        ..., description="Publicaciones estimadas para el año siguiente."
    )
    growth: int = Field(..., description="Diferencia absoluta de publicaciones.")
    growth_percentage: float = Field(
        ..., description="Crecimiento relativo, en porcentaje."
    )


class RankingResponse(BaseModel):
    """Respuesta para el ranking de afiliaciones por proyección de crecimiento."""

    ranking: list[RankingItem] = Field(
        ..., description="Afiliaciones ordenadas por crecimiento predicho."
    )


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

    model_type: str = Field(..., description="Algoritmo del modelo entrenado.")
    training_data_range: str = Field(
        ..., description="Rango de años usado en el entrenamiento."
    )
    target_variable: str = Field(..., description="Variable que el modelo predice.")
    total_affiliations: int = Field(
        ..., description="Afiliaciones presentes en los datos de entrenamiento."
    )
    performance_metrics: ModelPerformance = Field(
        ..., description="Error del modelo sobre el conjunto de validación."
    )
    feature_importances: dict[str, float] = Field(
        ..., description="Peso relativo de cada variable en la predicción."
    )


# --- Recomendación de tópicos a grupos ---


class RecommendationItem(BaseModel):
    """Un tópico recomendado a un grupo de investigación."""

    rank: int = Field(..., description="Posición dentro del grupo (1 = mejor).")
    topic: str = Field(
        ..., description="Tópico recomendado.", examples=["machine learning"]
    )
    score: float = Field(
        ...,
        description="Puntaje combinado de relevancia, señal colaborativa y novedad.",
    )
    relevance: float = Field(
        ..., description="Cercanía del tópico al perfil del grupo."
    )
    collaborative_signal: float = Field(
        ..., description="Uso del tópico en grupos similares."
    )
    novelty: float = Field(..., description="Qué tan nuevo es el tópico para el grupo.")


class GroupRecommendations(BaseModel):
    """Recomendaciones para un grupo de investigación persistente."""

    group_id: int = Field(
        ..., description="Identificador interno del grupo.", examples=[12]
    )
    members: list[str] = Field(
        ..., description="IDs de Scopus de los integrantes.", examples=[["57190000001"]]
    )
    n_members: int = Field(..., description="Número de integrantes.")
    papers_total: int = Field(..., description="Artículos del grupo en el periodo.")
    recommendations: list[RecommendationItem] = Field(
        ..., description="Tópicos recomendados, de mayor a menor puntaje."
    )
    count: int = Field(..., description="Cantidad de tópicos devueltos.")

    @field_validator("members", mode="before")
    @classmethod
    def members_as_text(cls, value):
        return [str(member) for member in value]


class GroupsResponse(BaseModel):
    """Listado de grupos con sus recomendaciones."""

    total_groups: int = Field(..., description="Grupos persistentes identificados.")
    displayed_groups: int = Field(
        ..., description="Grupos incluidos en esta respuesta."
    )
    groups: list[GroupRecommendations] = Field(..., description="Grupos devueltos.")


class ByMembersResponse(BaseModel):
    """Resultado de buscar el grupo de un conjunto de investigadores."""

    linked: bool = Field(
        ...,
        description=(
            "`true` si algún ID pertenece a un grupo persistente; "
            "`false` si no hay coincidencia."
        ),
    )
    group_id: Optional[int] = Field(None, description="Grupo encontrado.")
    members: Optional[list[str]] = Field(None, description="IDs de Scopus del grupo.")
    n_members: Optional[int] = Field(None, description="Número de integrantes.")
    papers_total: Optional[int] = Field(None, description="Artículos del grupo.")
    recommendations: Optional[list[RecommendationItem]] = Field(
        None, description="Tópicos recomendados al grupo."
    )
    count: Optional[int] = Field(None, description="Cantidad de tópicos devueltos.")

    @field_validator("members", mode="before")
    @classmethod
    def members_as_text(cls, value):
        if value is None:
            return value
        return [str(member) for member in value]


class RecommendationMetrics(BaseModel):
    """Métricas de calidad del sistema de recomendación."""

    groups_persistent: int = Field(..., description="Grupos persistentes.")
    topics_unique: int = Field(..., description="Tópicos únicos en el corpus.")
    novelty_rate: float = Field(
        ..., description="Porcentaje de recomendaciones nuevas para el grupo."
    )
    new_vs_recent: float = Field(
        ..., description="Porcentaje de tópicos nuevos frente al periodo reciente."
    )
    coverage: float = Field(
        ..., description="Porcentaje del catálogo de tópicos recomendado."
    )
    diversity: float = Field(..., description="Diversidad entre grupos, en porcentaje.")
    avg_relevance: float = Field(..., description="Relevancia promedio (0 a 1).")
    avg_score: float = Field(..., description="Puntaje promedio (0 a 1).")
    fairness_gini: float = Field(
        ..., description="Gini del puntaje entre grupos (0 = equitativo)."
    )
    fairness_score_gap_by_group_size: float = Field(
        ..., description="Brecha de puntaje según el tamaño del grupo."
    )
    n_recommendations: int = Field(..., description="Recomendaciones evaluadas.")


class RecommendationServiceStatus(BaseModel):
    """Estado de inicialización del motor de recomendación."""

    status: Literal["ok", "unavailable"] = Field(..., description="Estado del motor.")
    service_initialized: bool = Field(..., description="Si el motor cargó sus datos.")
    error: Optional[str] = Field(None, description="Error de inicialización, si hubo.")
