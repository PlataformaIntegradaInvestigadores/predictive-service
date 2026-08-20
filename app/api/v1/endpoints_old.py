# backend/app/api/v1/endpoints.py

from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AffiliationListResponse,
    DataPoint,
    ErrorResponse,
    PredictionRequest,
    PredictionResponse,
    ProjectionResponse,
)

# --- Creación del Router ---
api_router = APIRouter()

# --- Carga de Activos (Modelos y Datos) ---
# Se asume que estos archivos están en la raíz del directorio 'backend'
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = BASE_DIR / "publication_model.pkl"
ENCODER_PATH = BASE_DIR / "affiliation_encoder.pkl"
# IMPORTANTE: Debes añadir tu archivo de datos históricos aquí.
# Debe contener como mínimo las columnas: 'year', 'affiliation_name',
# 'publication_count', 'distinct_authors'
HISTORICAL_DATA_PATH = BASE_DIR / "publication_data.csv"

model = None
encoder = None
historical_df = None


@api_router.on_event("startup")
def load_assets():
    """
    Carga todos los activos necesarios (modelo, encoder y datos históricos)
    al iniciar la aplicación para optimizar el rendimiento.
    """
    global model, encoder, historical_df
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Modelo y encoder cargados exitosamente.")

        # Cargar el DataFrame con los datos históricos
        historical_df = pd.read_csv(HISTORICAL_DATA_PATH)
        print("Datos históricos cargados exitosamente.")

    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo esencial: {e.filename}")
        # La aplicación no podrá funcionar correctamente sin estos archivos.
    except Exception as e:
        print(f"Ocurrió un error crítico al cargar los activos: {e}")


def get_model_and_encoder():
    """Función de dependencia para verificar que los modelos están cargados."""
    if model is None or encoder is None or historical_df is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo o los datos no están disponibles. "
            "Revisa los logs del servidor.",
        )
    return model, encoder, historical_df


# --- Nuevos Endpoints de Analítica ---


@api_router.get("/affiliations", response_model=AffiliationListResponse)
def get_affiliations():
    """
    NUEVO ENDPOINT
    Devuelve una lista de todos los nombres de afiliaciones disponibles en el modelo.
    Ideal para rellenar un combobox en el frontend.
    """
    _, enc, _ = get_model_and_encoder()
    # encoder.classes_ es un array de numpy, lo convertimos a una lista de Python
    affiliation_list = enc.classes_.tolist()
    return AffiliationListResponse(affiliations=affiliation_list)


@api_router.get(
    "/projection/{affiliation_name}",
    response_model=ProjectionResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_projection(
    affiliation_name: str,
    projection_years: int = Query(
        5, ge=1, le=20, description="Número de años a proyectar en el futuro."
    ),
):
    """
    NUEVO ENDPOINT
    Devuelve los datos históricos y una proyección futura para una afiliación
    específica.
    """
    mdl, enc, df = get_model_and_encoder()

    # Validar que la afiliación existe
    if affiliation_name not in enc.classes_:
        return JSONResponse(
            status_code=404,
            content={"error": f"La afiliación '{affiliation_name}' no fue encontrada."},
        )

    # 1. Filtrar datos históricos para la afiliación seleccionada
    affiliation_df = df[df["affiliation_name"] == affiliation_name].sort_values("year")
    if affiliation_df.empty:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"No hay datos históricos para la afiliación "
                f"'{affiliation_name}'."
            },
        )

    historical_data = [
        DataPoint(
            year=row["year"], publications=row["publication_count"], type="actual"
        )
        for index, row in affiliation_df.iterrows()
    ]

    # 2. Preparar para la proyección iterativa
    projection_data = []
    last_real_year_data = affiliation_df.iloc[-1]

    current_year = int(last_real_year_data["year"])
    current_publications = int(last_real_year_data["publication_count"])
    # Asumimos que el número de autores se mantiene constante para la proyección
    # Esta es una simplificación que se podría mejorar.
    current_authors = int(last_real_year_data["distinct_authors"])

    affiliation_encoded = enc.transform([affiliation_name])[0]

    # 3. Realizar proyección iterativa para los próximos N años
    for _i in range(projection_years):
        predict_year = current_year + 1

        input_data = pd.DataFrame(
            [
                [
                    predict_year,
                    affiliation_encoded,
                    # Usamos el dato del año anterior (que puede ser real o ya predicho)
                    current_publications,
                    current_authors,
                ]
            ],
            columns=[
                "year",
                "affiliation_encoded",
                "publication_count",
                "distinct_authors",
            ],
        )

        prediction = mdl.predict(input_data)
        predicted_pubs = round(prediction[0])

        projection_data.append(
            DataPoint(year=predict_year, publications=predicted_pubs, type="predicted")
        )

        # Actualizamos las variables para la siguiente iteración del bucle
        current_year = predict_year
        current_publications = predicted_pubs

    # 4. Combinar y devolver los datos
    full_data = historical_data + projection_data
    return ProjectionResponse(affiliation_name=affiliation_name, data=full_data)


# --- Endpoint de Predicción Simple (Existente, sin cambios) ---


@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}},
)
def predict(request: PredictionRequest):
    """
    Endpoint para realizar una predicción simple para un único año.
    """
    mdl, enc, _ = get_model_and_encoder()

    try:
        try:
            affiliation_encoded = enc.transform([request.affiliation_name])[0]
        except ValueError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"La afiliación '{request.affiliation_name}' "
                    f"no fue encontrada."
                },
            )

        input_data = pd.DataFrame(
            [
                [
                    request.year,
                    affiliation_encoded,
                    request.last_year_publications,
                    request.last_year_authors,
                ]
            ],
            columns=[
                "year",
                "affiliation_encoded",
                "publication_count",
                "distinct_authors",
            ],
        )

        prediction = model.predict(input_data)

        return PredictionResponse(predicted_publications=round(prediction[0]))

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ocurrió un error interno: {str(e)}"
        ) from e
