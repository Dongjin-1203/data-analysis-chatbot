from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-6-luna"
    max_upload_mb: int = 50
    max_history_turns: int = 10
    storage_dir: str = "storage"  # 대화 DB(chat.db)와 업로드 파일이 저장되는 곳
    code_timeout_sec: int = 30  # run_python 도구의 코드 실행 시간 제한
    max_agent_steps: int = 12  # 한 질문에서 에이전트가 도구를 호출할 수 있는 최대 횟수
    embedding_model: str = "text-embedding-3-small"  # 점검 매뉴얼 검색(RAG) 임베딩 모델
    enable_shell_tool: bool = True  # 터미널 명령 실행 도구(run_shell) 사용 여부
    shell_timeout_sec: int = 180  # run_shell 시간 제한 (패키지 설치를 고려해 넉넉히)


@lru_cache
def get_settings() -> Settings:
    return Settings()
