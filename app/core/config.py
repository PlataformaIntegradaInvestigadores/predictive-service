# backend/app/core/config.py


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

    # Períodos usados por el GRS para identificar grupos persistentes.
    # Ajustar vía .env a medida que pase el tiempo, en vez de hardcodear en el código.
    GRS_PERIOD1_START: int = 2020
    GRS_PERIOD1_END: int = 2022
    GRS_PERIOD2_START: int = 2023
    GRS_PERIOD2_END: int = 2025

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
