from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Backend Configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_DEBUG: bool = True
    BACKEND_RELOAD: bool = True

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ML Configuration
    ML_MODEL_PATH: str = "../ml/models/v1.0.0/model.pth"
    ML_MODEL_CONFIG: str = "../ml/models/v1.0.0/config.json"
    ML_CONFIDENCE_THRESHOLD: float = 0.7

    # Translation Configuration
    TRANSLATION_DICTIONARY_PATH: str = "../translation/data/isl_dictionary.json"
    TRANSLATION_GRAMMAR_RULES_PATH: str = "../translation/data/grammar_rules.json"

    # TTS Configuration
    TTS_ENGINE: str = "offline"
    TTS_VOICE_PROFILE: str = "default"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
