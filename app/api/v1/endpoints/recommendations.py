from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.services.recommendation_service import RecommendationService


router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


class MembersRequest(BaseModel):
    scopus_ids: list[str]


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


@router.get("/health")
def health_check(request: Request):
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


@router.get("/groups")
def get_all_groups(
    k: int = Query(default=10, ge=1, le=50),
    limit: int = Query(default=50, ge=1, le=1000),
    service: RecommendationService = Depends(get_service),
):
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


@router.get("/group/{group_id}")
def get_group_recommendations(
    group_id: int,
    k: int = Query(default=10, ge=1, le=50),
    service: RecommendationService = Depends(get_service),
):
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


@router.post("/by-members")
def get_recommendations_by_members(
    body: MembersRequest,
    k: int = Query(default=10, ge=1, le=50),
    service: RecommendationService = Depends(get_service),
):
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


@router.get("/metrics")
def get_metrics(
    k: int = Query(default=10, ge=1, le=50),
    service: RecommendationService = Depends(get_service),
):
    try:
        return service.get_metrics(k=k)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculando métricas: {exc}",
        ) from exc
