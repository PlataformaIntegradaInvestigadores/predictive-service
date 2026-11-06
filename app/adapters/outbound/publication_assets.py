from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Resultado del entrenamiento (notebook 'models_split_top_10.ipynb'). Describen
# el .pkl que se despliega, así que viven junto a él: si se reentrena el
# modelo, se actualizan aquí.
TRAINING_REPORT = {
    "model_type": "LightGBM Regressor",
    "training_data_range": "Datos históricos hasta el año 2021",
    "target_variable": "Número de publicaciones del año siguiente",
    "performance_metrics": {"mae": 2.20, "rmse": 4.67},
}

FEATURES = ["year", "affiliation_encoded", "publication_count", "distinct_authors"]


class LightGbmPublicationModel:
    """Modelo LightGBM y su codificador de afiliaciones (archivos .pkl)."""

    def __init__(self, model_path: Path, encoder_path: Path) -> None:
        self._model = joblib.load(model_path)
        # Cada predicción es de una sola fila: el pool de hilos de OpenMP solo
        # añade coste (2.7k predicciones del ranking: 22 s -> 6 s, mismo resultado).
        self._model.set_params(n_jobs=1)
        encoder = joblib.load(encoder_path)
        self._affiliations = encoder.classes_.tolist()
        # LabelEncoder asigna a cada clase su posición en `classes_`.
        self._codes = {name: code for code, name in enumerate(self._affiliations)}

    def affiliations(self) -> list[str]:
        return self._affiliations

    def predict_next(
        self, year: int, affiliation_name: str, publications: int, authors: int
    ) -> int:
        row = pd.DataFrame(
            [[year, self._codes[affiliation_name], publications, authors]],
            columns=FEATURES,
        )
        return round(self._model.predict(row)[0])

    def describe(self) -> dict[str, Any]:
        importances = dict(
            zip(self._model.feature_name_, self._model.feature_importances_)
        )
        return {**TRAINING_REPORT, "feature_importances": importances}


class CsvPublicationHistory:
    """Histórico de publicaciones por afiliación (CSV)."""

    def __init__(self, csv_path: Path) -> None:
        self._data = pd.read_csv(csv_path)

    def for_affiliation(self, affiliation_name: str) -> pd.DataFrame:
        rows = self._data[self._data["affiliation_name"] == affiliation_name]
        return rows.sort_values("year")
