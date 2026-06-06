from pydantic_settings import BaseSettings
from typing import List, Union
from pydantic import field_validator
import json


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # On autorise soit une liste, soit une chaîne de caractères brute
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    SKLEARN_MODEL_DIR: str = "models/sklearn"
    NEURAL_MODEL_DIR: str = "models/neural"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: any) -> List[str]:
        # Si c'est déjà une vraie liste (comme les valeurs par défaut), on la garde
        if isinstance(v, list):
            return v
        
        # Si c'est du texte (ce qui vient de l'environnement)
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            # 1. On tente de décoder si c'est du format JSON (ex: '["http://..."]')
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # 2. Si ce n'est pas du JSON, on découpe simplement par les virgules
                return [item.strip() for item in v.split(",") if item.strip()]
        
        return []

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_parse_none_str": "null",
    }


settings = Settings()
