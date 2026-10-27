from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.models.schemas import (
    ByMembersResponse,
    GroupRecommendations,
    GroupsResponse,
    RecommendationMetrics,
    RecommendationServiceStatus,
)
from app.services.recommendation_service import RecommendationService

router = APIRouter(
    prefix="/recommendations",
    tags=["Recomendación de tópicos"],
)

_UNAVAILABLE = {
    503: {
        "description": "El motor de recomendación no pudo inicializarse.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "message": "RecommendationService no está disponible",
                        "error": "No se encontró el archivo de datos",
                    }
                }
            }
        },
    }
}
_INTERNAL_ERROR = {
    500: {
        "description": "Error interno al generar las recomendaciones.",
        "content": {
            "application/json": {
                "example": {"detail": "Error generando recomendaciones: ..."}
            }
        },
    }
}


def _k_query():
    return Query(
        default=10,
        ge=1,
        le=50,
        description="Número de tópicos a recomendar por grupo.",
    )


class MembersRequest(BaseModel):
    scopus_ids: list[str] = Field(
        ...,
        description="IDs de Scopus de los investigadores. Basta que uno pertenezca "
        "a un grupo persistente para vincularlo.",
        examples=[["57190000001", "57190000002"]],
    )

    @field_validator("scopus_ids", mode="before")
    @classmethod
    def normalize_scopus_ids(cls, value):
        if not isinstance(value, list):
            raise ValueError("scopus_ids debe ser una lista")
        return [str(scopus_id).strip() for scopus_id in value]


def get_service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, "recommendation_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "RecommendationService no está disponible",
                "error": getattr(
                    request.app.state,
                    "recommendation_service_error",
                    None,
                ),
            },
        )

    return service


@router.get(
    "/health",
    response_model=RecommendationServiceStatus,
    summary="Estado del motor de recomendación",
)
def health_check(request: Request):
    """Indica si el motor de recomendación cargó sus datos. A diferencia de los
    demás endpoints, responde 200 aunque el motor no esté disponible."""
    service = getattr(request.app.state, "recommendation_service", None)

    return {
        "status": "ok" if service is not None else "unavailable",
        "service_initialized": service is not None,
        "error": getattr(
            request.app.state,
            "recommendation_service_error",
            None,
        ),
    }


@router.get(
    "/groups",
    response_model=GroupsResponse,
    summary="Listar grupos con sus recomendaciones",
    responses={**_UNAVAILABLE, **_INTERNAL_ERROR},
)
def get_all_groups(
    k: int = _k_query(),
    limit: int = Query(
        default=50,
        ge=1,
        le=1000,
        description="Máximo de grupos a devolver.",
    ),
    service: RecommendationService = Depends(get_service),
):
    """Devuelve los grupos de investigación persistentes (los que se mantienen
    entre los dos periodos analizados) junto con los `k` tópicos recomendados
    a cada uno."""
    try:
        return service.get_all_recommendations(
            k=k,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo grupos: {exc}",
        ) from exc


@router.get(
    "/group/{group_id}",
    response_model=GroupRecommendations,
    summary="Recomendaciones de un grupo",
    responses={
        404: {
            "description": "No existe un grupo con ese identificador.",
            "content": {
                "application/json": {
                    "example": {"detail": "'No existe el grupo con id 999'"}
                }
            },
        },
        **_UNAVAILABLE,
        **_INTERNAL_ERROR,
    },
)
def get_group_recommendations(
    group_id: int = Path(
        ..., description="Identificador del grupo (ver `/groups`).", examples=[12]
    ),
    k: int = _k_query(),
    service: RecommendationService = Depends(get_service),
):
    """Devuelve los `k` tópicos con mayor puntaje para un grupo, ordenados por
    `rank`."""
    try:
        return service.get_group_recommendations(
            group_id=group_id,
            k=k,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando recomendaciones: {exc}",
        ) from exc


@router.post(
    "/by-members",
    response_model=ByMembersResponse,
    response_model_exclude_none=True,
    summary="Recomendaciones a partir de investigadores",
    responses={**_UNAVAILABLE, **_INTERNAL_ERROR},
)
def get_recommendations_by_members(
    body: MembersRequest,
    k: int = _k_query(),
    service: RecommendationService = Depends(get_service),
):
    """Busca el grupo persistente al que pertenece alguno de los investigadores
    indicados. Si lo encuentra, responde `linked: true` con sus recomendaciones;
    si no, responde únicamente `linked: false`."""
    group_id = service.find_group_by_members(body.scopus_ids)

    if group_id is None:
        return {"linked": False}

    try:
        recommendations = service.get_group_recommendations(
            group_id=group_id,
            k=k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando recomendaciones: {exc}",
        ) from exc

    return {"linked": True, **recommendations}


@router.get(
    "/metrics",
    response_model=RecommendationMetrics,
    summary="Métricas del sistema de recomendación",
    responses={**_UNAVAILABLE, **_INTERNAL_ERROR},
)
def get_metrics(
    k: int = _k_query(),
    service: RecommendationService = Depends(get_service),
):
    """Evalúa la calidad de las recomendaciones de todos los grupos: novedad,
    cobertura, diversidad, relevancia y equidad (Gini). La primera llamada con
    un `k` nuevo es lenta porque genera todas las recomendaciones."""
    try:
        return service.get_metrics(k=k)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculando métricas: {exc}",
        ) from exc
