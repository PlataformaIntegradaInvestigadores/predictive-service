from pydantic_settings import SettingsConfigDict

import pytest

from app.core.config import Settings


def test_default_settings():
    settings = Settings.model_construct(
        PROJECT_NAME="Centinela Predictivo de Publicaciones Científicas",
        API_V1_STR="/api/v1",
        DEBUG=False,
        ALLOWED_HOSTS=["*"],
        BACKEND_CORS_ORIGINS=["http://localhost:8082", "http://127.0.0.1:8082"],
        GRS_PERIOD1_START=2020,
        GRS_PERIOD1_END=2022,
        GRS_PERIOD2_START=2023,
        GRS_PERIOD2_END=2025,
    )
    assert settings.PROJECT_NAME == "Centinela Predictivo de Publicaciones Científicas"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.DEBUG is False
    assert settings.ALLOWED_HOSTS == ["*"]
    assert settings.BACKEND_CORS_ORIGINS == [
        "http://localhost:8082",
        "http://127.0.0.1:8082",
    ]
    assert settings.GRS_PERIOD1_START == 2020
    assert settings.GRS_PERIOD1_END == 2022
    assert settings.GRS_PERIOD2_START == 2023
    assert settings.GRS_PERIOD2_END == 2025


def test_settings_case_sensitive():
    assert Settings.model_config["case_sensitive"] is True


def test_settings_env_file():
    assert Settings.model_config["env_file"] == ".env"
    assert Settings.model_config["env_file_encoding"] == "utf-8"


def test_settings_instance_creation():
    settings = Settings.model_construct()
    assert isinstance(settings, Settings)
