# backend/app/api/v1/endpoints/analytics.py

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AffiliationListResponse,
    CompareRequest,
    ComparisonResponse,
    ErrorResponse,
    ModelDetailsResponse,
    ProjectionResponse,
    RankingResponse,
)
from app.services.prediction_service import PredictionService

# --- Creación del Router y Servicio ---
api_router = APIRouter(tags=["Analítica de publicaciones"])

_MODEL_UNAVAILABLE = {
    503: {
        "description": "El modelo o los datos no están cargados.",
        "content": {
            "application/json": {
                "example": {
                    "detail": "El modelo o los datos no están disponibles. "
                    "Revisa los logs del servidor."
                }
            }
        },
    }
}
# Instancia única del servicio para toda la aplicación
prediction_service = PredictionService()


def get_prediction_service():
    """
    Función de dependencia de FastAPI.
    Verifica que el servicio y sus activos se hayan cargado correctamente.
    Si no, levanta un error 503 (Servicio No Disponible).
    """
    if not prediction_service.check_assets_loaded():
        raise HTTPException(
            status_code=503,
            detail="El modelo o los datos no están disponibles. "
            "Revisa los logs del servidor.",
        )
    return prediction_service


# --- Endpoints de Analítica ---


@api_router.get(
    "/affiliations",
    response_model=AffiliationListResponse,
    summary="Listar afiliaciones",
    responses={**_MODEL_UNAVAILABLE},
)
def get_affiliations(service: PredictionService = Depends(get_prediction_service)):
    """Devuelve los nombres de todas las afiliaciones con historial de
    publicaciones. Úsalos como valor de `affiliation_name` en las
    proyecciones."""
    affiliations = service.get_all_affiliations()
    return AffiliationListResponse(affiliations=affiliations)


@api_router.get(
    "/projection/{affiliation_name}",
    response_model=ProjectionResponse,
    summary="Proyectar publicaciones de una afiliación",
    responses={
        404: {"model": ErrorResponse, "description": "Afiliación no encontrada."},
        **_MODEL_UNAVAILABLE,
    },
)
def get_projection(
    affiliation_name: str = Path(
        ...,
        description="Nombre exacto de la afiliación (ver `/affiliations`).",
        examples=["Escuela Politécnica Nacional"],
    ),
    projection_years: int = Query(
        5, ge=1, le=20, description="Número de años a proyectar."
    ),
    hypothetical_authors: Optional[int] = Query(
        None, description="Número hipotético de autores para análisis 'What If'."
    ),
    service: PredictionService = Depends(get_prediction_service),
):
    """Devuelve la serie histórica de publicaciones de una afiliación y su
    proyección a futuro. Con `hypothetical_authors` se simula cómo cambiaría la
    proyección si la afiliación tuviera ese número de autores (escenario
    "What If")."""
    result = service.get_projection(
        affiliation_name, projection_years, hypothetical_authors
    )
    if "error" in result:
        return JSONResponse(status_code=404, content=result)
    return ProjectionResponse(**result)


@api_router.post(
    "/projection/compare",
    response_model=ComparisonResponse,
    summary="Comparar proyecciones de varias afiliaciones",
    responses={**_MODEL_UNAVAILABLE},
)
def get_comparison(
    body: CompareRequest,
    projection_years: int = Query(
        5, ge=1, le=20, description="Número de años a proyectar."
    ),
    service: PredictionService = Depends(get_prediction_service),
):
    """Calcula la proyección de cada afiliación indicada. Las que no existen se
    omiten de `results` en lugar de fallar toda la petición."""
    results = []
    for name in body.affiliation_names:
        result = service.get_projection(name, projection_years)
        if "error" in result:
            continue
        results.append(result)
    return ComparisonResponse(results=results)


@api_router.get(
    "/ranking",
    response_model=RankingResponse,
    summary="Ranking de crecimiento",
    responses={**_MODEL_UNAVAILABLE},
)
def get_ranking(service: PredictionService = Depends(get_prediction_service)):
    """Ordena las afiliaciones por crecimiento predicho de publicaciones para el
    próximo año, de mayor a menor."""
    ranking = service.get_ranking()
    return RankingResponse(ranking=ranking)


@api_router.get(
    "/model-details",
    response_model=ModelDetailsResponse,
    summary="Detalles del modelo",
    responses={**_MODEL_UNAVAILABLE},
)
def get_model_details(service: PredictionService = Depends(get_prediction_service)):
    """Devuelve metadatos del modelo de Machine Learning: tipo, rango de
    entrenamiento, error (MAE y RMSE) e importancia de cada variable."""
    details = service.get_model_details()
    return ModelDetailsResponse(**details)
