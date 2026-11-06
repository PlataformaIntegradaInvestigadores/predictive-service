from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.application.ports import ArticleDataset

ARTICLES_FILE = "articulos_ecuador_CLEAN.csv"
AUTHOR_ARTICLE_FILE = "autor_articulo_CLEAN.csv"
TOPIC_ARTICLE_FILE = "topic_article_CLEAN.csv"


class CsvArticleRepository:
    """Lee los tres CSV del GRS desde una carpeta."""

    def __init__(self, data_path: str | None) -> None:
        self._data_path = data_path

    def load(self) -> ArticleDataset:
        if not self._data_path:
            raise RuntimeError("No se configuró CENTINELA_DATA_PATH.")

        folder = Path(self._data_path).resolve()

        if not folder.is_dir():
            raise FileNotFoundError(f"La carpeta de datos no existe: {folder}")

        paths = [
            folder / ARTICLES_FILE,
            folder / AUTHOR_ARTICLE_FILE,
            folder / TOPIC_ARTICLE_FILE,
        ]
        missing = [str(path) for path in paths if not path.is_file()]

        if missing:
            raise FileNotFoundError(
                "Faltan archivos requeridos:\n" + "\n".join(missing)
            )

        return ArticleDataset(*(pd.read_csv(path) for path in paths))
