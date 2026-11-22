from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-6-luna"
    max_upload_mb: int = 50
    max_history_turns: int = 10
    storage_dir: str = "storage"  # 대화 DB(chat.db)와 업로드 파일이 저장되는 곳


@lru_cache
def get_settings() -> Settings:
    return Settings()
