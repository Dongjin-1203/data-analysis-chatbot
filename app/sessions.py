"""서버 메모리 기반 세션 저장소. 서버를 재시작하면 초기화된다."""

import uuid
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)
    data_summary: str | None = None
    filename: str | None = None
    dataframe: pd.DataFrame | None = None  # 2단계 도구(CSV 조회·코드 실행)에서 재사용

    def add_turn(self, user: str, assistant: str, max_turns: int) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        if len(self.history) > max_turns * 2:
            self.history = self.history[-max_turns * 2 :]


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = Session()
        return session_id

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session()
        return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


store = SessionStore()
