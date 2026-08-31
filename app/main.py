import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import analytics, recommendations
from app.core.config import settings
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.recommendation_service = RecommendationService()
        app.state.recommendation_service_error = None
    except Exception as exc:
        app.state.recommendation_service = None
        app.state.recommendation_service_error = str(exc)
        logger.warning("No se pudo inicializar RecommendationService: %s", exc)

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)


app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
    + ["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Router original de Centinela
app.include_router(
    analytics.api_router,
    prefix=settings.API_V1_STR,
)

# Router nuevo del GRS
app.include_router(recommendations.router, prefix=settings.API_V1_STR)


def custom_openapi():
    """Recorta el prefijo interno del spec y fija la ruta publica detras del
    gateway (nginx.conf: /api/predictive/ -> /api/v1/), para que "Try it out"
    en Swagger UI pegue a la URL real."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    prefix = settings.API_V1_STR
    schema["paths"] = {
        (path[len(prefix) :] if path.startswith(prefix) else path): item
        for path, item in schema.get("paths", {}).items()
    }
    schema["servers"] = [{"url": "/api/predictive", "description": "Gateway"}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", include_in_schema=False)
def read_root():
    return {
        "message": (
            "Bienvenido al API del Centinela Predictivo " "de Publicaciones Científicas"
        )
    }


@app.get("/health", include_in_schema=False)
def health():
    service = getattr(app.state, "recommendation_service", None)
    if service is None:
        error = getattr(app.state, "recommendation_service_error", None)
        if error:
            logger.error("Health check: recommendation service unavailable: %s", error)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "service_initialized": False},
        )
    return {"status": "ok", "service_initialized": True}
