# Centinela — predictive-service

Backend FastAPI que predice el número de publicaciones científicas futuras de afiliaciones académicas, usando un modelo LightGBM Regressor pre-entrenado, y expone un sistema de recomendación de grupos de investigación (GRS) basado en coautoría histórica.

Parte del org multi-repo `PlataformaIntegradaInvestigadores`. Servicio de solo lectura sobre datos y modelos pre-entrenados (no tiene base de datos propia); en producción se accede a través de `gateway-service`.

## Stack

- FastAPI + Uvicorn (ASGI)
- LightGBM (modelo de predicción), scikit-learn, pandas, scipy
- pydantic-settings (configuración por env vars)

## Estructura del proyecto

```
app/
  api/v1/endpoints/
    analytics.py         # Proyecciones, comparaciones, ranking, detalles del modelo
    recommendations.py   # Sistema de recomendación de grupos (GRS)
  core/config.py         # Configuración de la aplicación
  grs/grs_production.py  # Lógica del sistema de recomendación de grupos
  models/schemas.py      # Esquemas Pydantic
  services/
    prediction_service.py    # Carga el modelo/encoder/dataset y genera predicciones
    recommendation_service.py
  main.py
publication_model.pkl        # Modelo LightGBM pre-entrenado
affiliation_encoder.pkl      # Codificador de etiquetas de afiliaciones
publication_data.csv         # Datos históricos de publicaciones
preprocess_data.py           # Script de preprocesamiento de datos brutos (opcional)
tests/
```

## Requisitos previos

- Docker y Docker Compose (recomendado), o Python 3.9+ si se corre sin Docker (versión fijada en el `Dockerfile`).

## Levantar en local

### Con Docker (recomendado)
```bash
docker compose up -d --build
```
La API queda disponible en `http://localhost:8003`.

### Sin Docker (desarrollo)
```bash
python -m venv venv
venv\Scripts\activate  # En Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

## Variables de entorno

Ver `.env.example` (desarrollo) / `.env_produccion.example` (Docker prod-like). Variables clave:

| Variable | Descripción |
|---|---|
| `CENTINELA_DATA_PATH` | Ruta dentro del contenedor a los CSV que usa el GRS (`articulos_ecuador_CLEAN.csv`, `autor_articulo_CLEAN.csv`, `topic_article_CLEAN.csv`) |
| `BACKEND_CORS_ORIGINS` | Orígenes permitidos por CORS (lista de URLs) |
| `GRS_PERIOD1_START` / `GRS_PERIOD1_END` | Rango del primer período usado por el GRS para identificar grupos persistentes de autores |
| `GRS_PERIOD2_START` / `GRS_PERIOD2_END` | Rango del segundo período |

## Tests

```bash
pytest --cov=app --cov-report=term
```

Cobertura mínima exigida en CI: **90%** (`--cov-fail-under=90` en `.github/workflows/ci.yml`). Estado actual: ~90%.

## Documentación (Swagger)

Schema OpenAPI (autogenerado por FastAPI): `GET /api/v1/openapi.json`. UI local propia en `/docs`. Centralizado también en el hub del `gateway-service`: `/api/docs/v1/predictive`. `custom_openapi()` en `app/main.py` recorta el prefijo interno `/api/v1` y declara `servers: [{"url": "/api/predictive"}]` para que "Try it out" funcione a través del gateway. `/` y `/health` quedan fuera del schema (`include_in_schema=False`).

## API — Analytics (`/api/v1`)

- **`GET /`** — mensaje de bienvenida.
- **`GET /api/v1/affiliations`** — lista de nombres de afiliaciones disponibles.
- **`GET /api/v1/projection/{affiliation_name}`** — proyección histórica y futura de una afiliación.
  - Query: `projection_years` (int, default 5), `hypothetical_authors` (int, opcional, análisis "What If").
  - 404 si la afiliación no existe.
- **`POST /api/v1/projection/compare`** — compara proyecciones de varias afiliaciones (`affiliation_names: string[]`, query `projection_years`).
- **`GET /api/v1/ranking`** — ranking de afiliaciones por crecimiento de publicaciones predicho para el próximo año.
- **`GET /api/v1/model-details`** — metadatos del modelo: tipo, rango de entrenamiento, métricas (`mae`, `rmse`), importancia de features.

## API — Recomendación de grupos (`/api/v1/recommendations`)

Sistema GRS (Group Recommendation System) sobre coautoría histórica entre dos períodos configurables (`GRS_PERIOD*`). Ver `app/api/v1/endpoints/recommendations.py` para el contrato completo (listado de grupos, recomendaciones por grupo, recomendaciones por miembros, métricas de diversidad).

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`): tests unitarios → build de imagen Docker → deploy automático a staging (`develop` branch, runner self-hosted `ticcd`) con healthcheck y rollback automático.

## Convenciones

- Branches: `feature/*` → `develop`, `hotfix/*` → `main`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/), inglés, con el *por qué* en el cuerpo.
