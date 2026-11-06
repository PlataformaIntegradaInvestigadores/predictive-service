# backend/app/core/config.py


from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Clase para gestionar la configuración de la aplicación.
    Lee las variables de entorno desde un archivo .env.
    """

    PROJECT_NAME: str = "Centinela Predictivo de Publicaciones Científicas"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    ALLOWED_HOSTS: list[str] = ["*"]

    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:8082", "http://127.0.0.1:8082"]

    # Carpeta con los CSV del GRS (articulos_ecuador_CLEAN.csv,
    # autor_articulo_CLEAN.csv y topic_article_CLEAN.csv).
    CENTINELA_DATA_PATH: Optional[str] = None

    # Períodos usados por el GRS para identificar grupos persistentes.
    # Ajustar vía .env a medida que pase el tiempo, en vez de hardcodear en el código.
    GRS_PERIOD1_START: int = 2020
    GRS_PERIOD1_END: int = 2022
    GRS_PERIOD2_START: int = 2023
    GRS_PERIOD2_END: int = 2025

    # Calcula en segundo plano, al arrancar, las recomendaciones de todos los
    # grupos para el k por defecto, de modo que /groups y /metrics respondan al
    # instante en lugar de tardar minutos en la primera petición.
    GRS_PRECOMPUTE_ON_STARTUP: bool = True
    GRS_PRECOMPUTE_K: int = 10

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
