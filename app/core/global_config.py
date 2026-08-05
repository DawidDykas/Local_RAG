from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    url_minio_database: str = "http://minio:9000"
    user_name_acces: str = "minio"
    user_password_acces: str = "minio123"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env", case_sensitive=False, extra="ignore"
    )


class CelerySetting(BaseSettings):
    url_broker: str = "redis://redis:6379/0"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env", case_sensitive=False, extra="ignore"
    )


class FastAPISettings(BaseSettings):
    PORT: int = 8000
    HOST: str = "127.0.0.1"  # localhost

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env", case_sensitive=False, extra="ignore"
    )


minio_database_settings = DatabaseSettings()
fastapi_settings = FastAPISettings()
celery_settings = CelerySetting()
