# backend/app/api/v1/endpoints/analytics.py

from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AffiliationListResponse,
    ComparisonResponse,
    ErrorResponse,
    ModelDetailsResponse,
    ProjectionResponse,
    RankingResponse,
)
from app.services.prediction_service import PredictionService

# --- Creación del Router y Servicio ---
api_router = APIRouter()
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


@api_router.get("/affiliations", response_model=AffiliationListResponse)
def get_affiliations(service: PredictionService = Depends(get_prediction_service)):
    """Devuelve una lista de todos los nombres de afiliaciones disponibles."""
    affiliations = service.get_all_affiliations()
    return AffiliationListResponse(affiliations=affiliations)


@api_router.get(
    "/projection/{affiliation_name}",
    response_model=ProjectionResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_projection(
    affiliation_name: str,
    projection_years: int = Query(
        5, ge=1, le=20, description="Número de años a proyectar."
    ),
    hypothetical_authors: Optional[int] = Query(
        None, description="Número hipotético de autores para análisis 'What If'."
    ),
    service: PredictionService = Depends(get_prediction_service),
):
    """Devuelve datos históricos y una proyección futura para una afiliación."""
    result = service.get_projection(
        affiliation_name, projection_years, hypothetical_authors
    )
    if "error" in result:
        return JSONResponse(status_code=404, content=result)
    return ProjectionResponse(**result)


@api_router.post(
    "/projection/compare",
    response_model=ComparisonResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_comparison(
    affiliation_names: list[str] = Body(
        ..., embed=True, description="Lista de nombres de afiliaciones a comparar."
    ),
    projection_years: int = Query(
        5, ge=1, le=20, description="Número de años a proyectar."
    ),
    service: PredictionService = Depends(get_prediction_service),
):
    """Compara las proyecciones de varias afiliaciones."""
    results = []
    for name in affiliation_names:
        result = service.get_projection(name, projection_years)
        if "error" in result:
            continue
        results.append(result)
    return ComparisonResponse(results=results)


@api_router.get("/ranking", response_model=RankingResponse)
def get_ranking(service: PredictionService = Depends(get_prediction_service)):
    """Devuelve un ranking de afiliaciones por crecimiento predicho para el
    próximo año."""
    ranking = service.get_ranking()
    return RankingResponse(ranking=ranking)


@api_router.get("/model-details", response_model=ModelDetailsResponse)
def get_model_details(service: PredictionService = Depends(get_prediction_service)):
    """Devuelve metadatos sobre el modelo de Machine Learning y los datos utilizados."""
    details = service.get_model_details()
    return ModelDetailsResponse(**details)
