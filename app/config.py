# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"
    OLLAMA_TIMEOUT_SECONDS: int = 60
    CORRECTION_TEMPERATURE: float = 0.0
    NAVIGATION_TEMPERATURE: float = 0.0
    REFORMULATION_TEMPERATURE: float = 0.3
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # "development" : endpoint /api/metrics/* monté + texte brut stocké en base (debug local).
    # "production"  : endpoint /api/metrics/* non monté + aucun texte utilisateur stocké.
    ENVIRONMENT: str = "production"

    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT.strip().lower() == "development"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()