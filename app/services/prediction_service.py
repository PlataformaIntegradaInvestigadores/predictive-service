# backend/app/services/prediction_service.py

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd


class PredictionService:
    """
    Contiene toda la lógica de negocio para cargar modelos, datos
    y realizar las predicciones y análisis.
    """

    RANKING_CACHE_TTL_SECONDS = 3600

    def __init__(self):
        # Cargar todos los activos una sola vez al instanciar el servicio.
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.MODEL_PATH = BASE_DIR / "publication_model.pkl"
        self.ENCODER_PATH = BASE_DIR / "affiliation_encoder.pkl"
        self.HISTORICAL_DATA_PATH = BASE_DIR / "publication_data.csv"

        self.model = None
        self.encoder = None
        self.historical_df = None
        self._ranking_cache = None
        self._ranking_cache_timestamp = 0
        self._load_assets()

    def _load_assets(self):
        """Método privado para cargar los archivos .pkl y .csv."""
        try:
            self.model = joblib.load(self.MODEL_PATH)
            self.encoder = joblib.load(self.ENCODER_PATH)
            self.historical_df = pd.read_csv(self.HISTORICAL_DATA_PATH)
            print("Servicio de predicción inicializado y activos cargados.")
        except FileNotFoundError as e:
            print(
                f"CRITICAL ERROR: No se pudo inicializar PredictionService. "
                f"Archivo no encontrado: {e.filename}"
            )
        except Exception as e:
            print(
                f"CRITICAL ERROR: Ocurrió un error al cargar activos en "
                f"PredictionService: {e}"
            )

    def check_assets_loaded(self):
        """Verifica si todos los activos necesarios están cargados."""
        return not (
            self.model is None or self.encoder is None or self.historical_df is None
        )

    def get_all_affiliations(self) -> list[str]:
        """Devuelve la lista completa de nombres de afiliaciones."""
        return self.encoder.classes_.tolist()

    def _run_prediction(
        self, year: int, affiliation_encoded: int, last_pubs: int, last_authors: int
    ) -> int:
        """Función interna para ejecutar el modelo."""
        input_data = pd.DataFrame(
            [[year, affiliation_encoded, last_pubs, last_authors]],
            columns=[
                "year",
                "affiliation_encoded",
                "publication_count",
                "distinct_authors",
            ],
        )

        prediction = self.model.predict(input_data)
        return round(prediction[0])

    def get_projection(
        self,
        affiliation_name: str,
        projection_years: int,
        hypothetical_authors: int | None = None,
    ) -> dict:
        """
        Calcula la proyección histórica y futura para una afiliación.
        Incluye la lógica para el análisis "What If".
        """
        if affiliation_name not in self.encoder.classes_:
            return {"error": f"La afiliación '{affiliation_name}' no fue encontrada."}

        affiliation_df = self.historical_df[
            self.historical_df["affiliation_name"] == affiliation_name
        ].sort_values("year")
        if affiliation_df.empty:
            return {
                "error": f"No hay datos históricos para la afiliación "
                f"'{affiliation_name}'."
            }

        historical_data = [
            {
                "year": row["year"],
                "publications": row["publication_count"],
                "type": "actual",
            }
            for _, row in affiliation_df.iterrows()
        ]

        projection_data = []
        last_real_year_data = affiliation_df.iloc[-1]
        current_year = int(last_real_year_data["year"])
        current_publications = int(last_real_year_data["publication_count"])

        current_authors = (
            hypothetical_authors
            if hypothetical_authors is not None
            else int(last_real_year_data["distinct_authors"])
        )

        if current_authors > 0:
            pubs_per_author_ratio = current_publications / current_authors
        else:
            pubs_per_author_ratio = 1.0

        affiliation_encoded = self.encoder.transform([affiliation_name])[0]

        for _ in range(projection_years):
            predict_year = current_year + 1

            authors_for_prediction = max(1, round(current_authors))

            predicted_pubs = self._run_prediction(
                predict_year,
                affiliation_encoded,
                current_publications,
                authors_for_prediction,
            )

            projection_data.append(
                {
                    "year": predict_year,
                    "publications": predicted_pubs,
                    "type": "predicted",
                }
            )

            current_year = predict_year
            current_publications = predicted_pubs

            if pubs_per_author_ratio > 0.001:
                current_authors = predicted_pubs / pubs_per_author_ratio

        return {
            "affiliation_name": affiliation_name,
            "data": historical_data + projection_data,
        }

    def get_ranking(self) -> list[dict]:
        """Calcula el ranking de crecimiento para todas las afiliaciones (con caché)."""
        now = time.time()
        if (
            self._ranking_cache is not None
            and (now - self._ranking_cache_timestamp) < self.RANKING_CACHE_TTL_SECONDS
        ):
            return self._ranking_cache

        ranking_data = []
        all_affiliations = self.get_all_affiliations()

        for affiliation_name in all_affiliations:
            affiliation_df = self.historical_df[
                self.historical_df["affiliation_name"] == affiliation_name
            ].sort_values("year")
            if not affiliation_df.empty:
                last_real_year_data = affiliation_df.iloc[-1]
                current_year = int(last_real_year_data["year"])
                current_pubs = int(last_real_year_data["publication_count"])
                current_authors = int(last_real_year_data["distinct_authors"])

                affiliation_encoded = self.encoder.transform([affiliation_name])[0]

                predicted_pubs = self._run_prediction(
                    current_year + 1, affiliation_encoded, current_pubs, current_authors
                )

                growth = predicted_pubs - current_pubs
                growth_percentage = (
                    (growth / current_pubs * 100) if current_pubs > 0 else 0
                )

                ranking_data.append(
                    {
                        "affiliation_name": affiliation_name,
                        "current_year_publications": current_pubs,
                        "predicted_next_year_publications": predicted_pubs,
                        "growth": growth,
                        "growth_percentage": round(growth_percentage, 2),
                    }
                )

        sorted_ranking = sorted(ranking_data, key=lambda x: x["growth"], reverse=True)

        for i, item in enumerate(sorted_ranking):
            item["rank"] = i + 1

        self._ranking_cache = sorted_ranking
        self._ranking_cache_timestamp = now
        return sorted_ranking

    def get_model_details(self) -> dict:
        """
        Devuelve metadatos sobre el modelo y los datos,
        basado en los resultados del notebook de entrenamiento.
        """
        importances = self.model.feature_importances_
        features = self.model.feature_name_
        feature_importance_dict = dict(zip(features, importances))

        # Datos extraídos del notebook 'models_split_top_10.ipynb'
        performance = {
            "mae": 2.20,  # Mean Absolute Error (MAE): 2.20
            "rmse": 4.67,  # Root Mean Squared Error (RMSE): 4.67
        }

        return {
            "model_type": "LightGBM Regressor",
            "training_data_range": "Datos históricos hasta el año 2021",
            "target_variable": "Número de publicaciones del año siguiente",
            "total_affiliations": len(self.encoder.classes_),
            "performance_metrics": performance,
            "feature_importances": feature_importance_dict,
        }
