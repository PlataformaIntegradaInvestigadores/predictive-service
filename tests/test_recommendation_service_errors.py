import os
import tempfile

import pytest

from app.services.recommendation_service import RecommendationService


def test_raises_when_no_data_path(monkeypatch):
    monkeypatch.delenv("CENTINELA_DATA_PATH", raising=False)
    with pytest.raises(RuntimeError, match="No se configuró CENTINELA_DATA_PATH"):
        RecommendationService()


def test_raises_when_data_path_not_exists(monkeypatch):
    monkeypatch.setenv("CENTINELA_DATA_PATH", "/nonexistent/path")
    with pytest.raises(FileNotFoundError, match="La carpeta de datos no existe"):
        RecommendationService()


def test_raises_when_required_files_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CENTINELA_DATA_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Faltan archivos requeridos"):
        RecommendationService()
