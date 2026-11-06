from __future__ import annotations

import time

from app.application.ports import PublicationHistory, PublicationModel
from app.domain.errors import AffiliationNotFound


class PredictionService:
    """
    Casos de uso de analítica: proyección histórica y futura, ranking de
    crecimiento y detalles del modelo. No sabe de dónde salen el modelo ni los
    datos: los recibe como puertos.
    """

    RANKING_CACHE_TTL_SECONDS = 3600

    def __init__(self, model: PublicationModel, history: PublicationHistory) -> None:
        self._model = model
        self._history = history
        self._ranking_cache = None
        self._ranking_cache_timestamp = 0

    def get_all_affiliations(self) -> list[str]:
        """Devuelve la lista completa de nombres de afiliaciones."""
        return self._model.affiliations()

    def get_projection(
        self,
        affiliation_name: str,
        projection_years: int,
        hypothetical_authors: int | None = None,
    ) -> dict:
        """
        Calcula la proyección histórica y futura para una afiliación.
        Incluye la lógica para el análisis "What If".

        Lanza AffiliationNotFound si la afiliación no existe o no tiene historial.
        """
        if affiliation_name not in self._model.affiliations():
            raise AffiliationNotFound(
                f"La afiliación '{affiliation_name}' no fue encontrada."
            )

        affiliation_df = self._history.for_affiliation(affiliation_name)
        if affiliation_df.empty:
            raise AffiliationNotFound(
                f"No hay datos históricos para la afiliación '{affiliation_name}'."
            )

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

        for _ in range(projection_years):
            predict_year = current_year + 1

            authors_for_prediction = max(1, round(current_authors))

            predicted_pubs = self._model.predict_next(
                predict_year,
                affiliation_name,
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
            affiliation_df = self._history.for_affiliation(affiliation_name)
            if not affiliation_df.empty:
                last_real_year_data = affiliation_df.iloc[-1]
                current_year = int(last_real_year_data["year"])
                current_pubs = int(last_real_year_data["publication_count"])
                current_authors = int(last_real_year_data["distinct_authors"])

                predicted_pubs = self._model.predict_next(
                    current_year + 1, affiliation_name, current_pubs, current_authors
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
        """Metadatos del modelo más el número de afiliaciones que conoce."""
        return {
            **self._model.describe(),
            "total_affiliations": len(self._model.affiliations()),
        }
