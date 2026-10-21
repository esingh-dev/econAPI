"""Application configuration settings for econAPI."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration settings loaded from environment or defaults."""

    APP_NAME: str = "econAPI"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", description="development or production")
    VALID_API_KEYS: list[str] = Field(default_factory=lambda: ["econ_live_demo_key", "sk_econ_institutional_2026", "econ_free_public"])
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = {
        "env_prefix": "ECONAPI_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
