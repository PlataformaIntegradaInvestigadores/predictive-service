"""Puertos: lo que los casos de uso necesitan del exterior.

La aplicación solo conoce estas interfaces; los adaptadores de
`app/adapters/outbound` las implementan (CSV, LightGBM) y `app/bootstrap.py`
decide cuáles se usan. Así los casos de uso se prueban con datos en memoria.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import pandas as pd


class ArticleDataset(NamedTuple):
    """Datos de entrada del GRS (tablas ya cargadas)."""

    articles: pd.DataFrame
    author_article: pd.DataFrame
    topic_article: pd.DataFrame


class ArticleRepository(Protocol):
    """Origen de los artículos, autores y tópicos del GRS."""

    def load(self) -> ArticleDataset: ...


class PublicationModel(Protocol):
    """Modelo que predice publicaciones del año siguiente por afiliación."""

    def affiliations(self) -> list[str]:
        """Afiliaciones que el modelo conoce."""

    def predict_next(
        self, year: int, affiliation_name: str, publications: int, authors: int
    ) -> int:
        """Publicaciones esperadas en `year` para la afiliación."""

    def describe(self) -> dict[str, Any]:
        """Metadatos del modelo: tipo, error, rango de entrenamiento e
        importancia de variables (`feature_importances`)."""


class PublicationHistory(Protocol):
    """Histórico anual de publicaciones por afiliación."""

    def for_affiliation(self, affiliation_name: str) -> pd.DataFrame:
        """Filas de la afiliación, de la más antigua a la más reciente, con
        las columnas year, publication_count y distinct_authors."""
