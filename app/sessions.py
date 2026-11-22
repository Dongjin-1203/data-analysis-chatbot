"""SQLite 기반 세션 저장소.

- 대화 기록과 업로드 데이터 정보는 storage/chat.db 에 저장되어 서버를 재시작해도 남는다.
- 업로드한 원본 파일은 storage/uploads/{session_id}/ 에 저장하고,
  DataFrame은 필요할 때 다시 읽어 메모리에 캐시한다(2단계 도구에서 재사용).
"""

import json
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.data_summary import load_dataframe

TITLE_MAX = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '새 대화',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    filename TEXT,
    data_path TEXT,
    data_summary TEXT,
    data_meta TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,          -- user | assistant | file
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Session:
    id: str
    title: str
    filename: str | None
    data_path: str | None
    data_summary: str | None
    data_meta: dict | None


class SessionStore:
    def __init__(self, storage_dir: str | Path) -> None:
        self.root = Path(storage_dir)
        self.upload_dir = self.root / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "chat.db"
        self._df_cache: dict[str, pd.DataFrame] = {}
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- 세션 ----------
    def create(self, session_id: str | None = None) -> str:
        session_id = session_id or uuid.uuid4().hex
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
        return session_id

    def get(self, session_id: str) -> Session | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            filename=row["filename"],
            data_path=row["data_path"],
            data_summary=row["data_summary"],
            data_meta=json.loads(row["data_meta"]) if row["data_meta"] else None,
        )

    def get_or_create(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            self.create(session_id)
            session = self.get(session_id)
        return session

    def list_sessions(self) -> list[dict]:
        """메시지가 하나 이상 있는 세션을 최근 순으로 돌려준다."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.updated_at, s.filename
                FROM sessions s
                WHERE EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, session_id: str) -> bool:
        with self._conn() as conn:
            deleted = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount
        self._df_cache.pop(session_id, None)
        shutil.rmtree(self.upload_dir / session_id, ignore_errors=True)
        return deleted > 0

    # ---------- 메시지 ----------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            if role == "user":
                title = content.strip().replace("\n", " ")
                title = title[:TITLE_MAX] + ("…" if len(title) > TITLE_MAX else "")
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND title = '새 대화'",
                    (title, session_id),
                )

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        self.add_message(session_id, "user", user)
        self.add_message(session_id, "assistant", assistant)

    def messages(self, session_id: str) -> list[dict]:
        """화면 복원용 전체 메시지(파일 업로드 기록 포함)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, session_id: str, max_turns: int | None = None) -> list[dict]:
        """LLM에 보낼 최근 대화(user/assistant만)."""
        max_turns = max_turns or get_settings().max_history_turns
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ? AND role IN ('user', 'assistant')
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, max_turns * 2),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ---------- 데이터 ----------
    def save_data(self, session_id: str, filename: str, content: bytes, df: pd.DataFrame, summary: str) -> dict:
        self.get_or_create(session_id)
        target_dir = self.upload_dir / session_id
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / Path(filename).name
        path.write_bytes(content)

        meta = {"filename": filename, "rows": int(df.shape[0]), "columns": [str(c) for c in df.columns]}
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET filename = ?, data_path = ?, data_summary = ?, data_meta = ?, updated_at = ? WHERE id = ?",
                (filename, str(path), summary, json.dumps(meta, ensure_ascii=False), _now(), session_id),
            )
        self._df_cache[session_id] = df
        self.add_message(session_id, "file", json.dumps(meta, ensure_ascii=False))
        return meta

    def dataframe(self, session_id: str) -> pd.DataFrame | None:
        """세션의 원본 DataFrame. 서버 재시작 후에는 저장된 파일에서 다시 읽는다."""
        if session_id in self._df_cache:
            return self._df_cache[session_id]
        session = self.get(session_id)
        if not session or not session.data_path or not Path(session.data_path).exists():
            return None
        path = Path(session.data_path)
        df = load_dataframe(path.name, path.read_bytes())
        self._df_cache[session_id] = df
        return df


store = SessionStore(get_settings().storage_dir)
