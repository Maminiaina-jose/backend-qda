from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    SKLEARN_MODEL_DIR: str = "models/sklearn"
    NEURAL_MODEL_DIR: str = "models/neural"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_parse_none_str": "null",
    }


settings = Settings()
