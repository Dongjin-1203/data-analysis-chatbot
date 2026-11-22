import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import app
from app.sessions import store

FIXTURE = Path(__file__).parent / "fixtures" / "sample_inspection.csv"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def captured(monkeypatch):
    """llm.stream_agent를 가짜 이벤트 스트림으로 바꾸고, 전달된 messages와 도구를 기록한다."""
    calls: list[list[dict]] = []

    async def fake_stream(messages, tools):
        calls.append(messages)
        assert [t.name for t in tools] == ["query_dataset", "run_python", "predict_defect", "search_manual", "run_shell"]
        for piece in ["분석 ", "계획", "입니다."]:
            yield {"type": "token", "text": piece}

    monkeypatch.setattr(llm, "stream_agent", fake_stream)
    return calls


def parse_sse(text: str) -> list[tuple[str, str]]:
    events = []
    for raw in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in raw.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        events.append((event, data))
    return events


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "데이터 분석 챗봇" in res.text


def test_session_lifecycle(client):
    sid = client.post("/api/session").json()["session_id"]
    assert store.get(sid) is not None
    assert client.get(f"/api/session/{sid}").json()["messages"] == []
    assert client.delete(f"/api/session/{sid}").status_code == 200
    assert client.delete(f"/api/session/{sid}").status_code == 404


def test_upload_then_chat_uses_data_context(client, captured):
    sid = client.post("/api/session").json()["session_id"]

    with FIXTURE.open("rb") as f:
        res = client.post("/api/upload", data={"session_id": sid}, files={"file": ("sample.csv", f, "text/csv")})
    assert res.status_code == 200
    body = res.json()
    assert body["rows"] == 10
    assert "defect" in body["columns"]

    res = client.post("/api/chat", json={"session_id": sid, "message": "라인별 불량률 알려줘"})
    assert res.status_code == 200
    events = parse_sse(res.text)
    deltas = [json.loads(d)["text"] for e, d in events if e == "message" and d != "[DONE]"]
    assert "".join(deltas) == "분석 계획입니다."
    assert events[-1] == ("message", "[DONE]")

    messages = captured[0]
    assert messages[0]["role"] == "system" and "분석 절차" in messages[0]["content"]
    assert "sample.csv" in messages[1]["content"] and "defect" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "라인별 불량률 알려줘"}

    # 두 번째 질문에는 이전 대화가 포함된다
    client.post("/api/chat", json={"session_id": sid, "message": "그래프도"})
    second = captured[1]
    assert {"role": "assistant", "content": "분석 계획입니다."} in second

    # 화면 복원 API: 업로드 기록과 대화가 순서대로 돌아온다
    restored = client.get(f"/api/session/{sid}").json()
    assert restored["title"] == "라인별 불량률 알려줘"
    assert [m["role"] for m in restored["messages"]] == ["file", "user", "assistant", "user", "assistant"]
    assert restored["data"]["filename"] == "sample.csv" and restored["data"]["rows"] == 10
    assert sid in [s["id"] for s in client.get("/api/sessions").json()]


def test_upload_rejects_unsupported(client):
    sid = client.post("/api/session").json()["session_id"]
    res = client.post("/api/upload", data={"session_id": sid}, files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_chat_without_api_key_returns_error_event(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "openai_api_key", "")
    sid = client.post("/api/session").json()["session_id"]
    res = client.post("/api/chat", json={"session_id": sid, "message": "안녕"})
    events = parse_sse(res.text)
    assert events[0][0] == "error"
    assert "OPENAI_API_KEY" in json.loads(events[0][1])["message"]


def test_upload_size_limit(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    sid = client.post("/api/session").json()["session_id"]
    big = b"a,b\n" + b"1,2\n" * 300_000  # 약 1.2MB
    res = client.post("/api/upload", data={"session_id": sid}, files={"file": ("big.csv", big, "text/csv")})
    assert res.status_code == 413


def test_default_upload_limit_is_50mb():
    from app.config import Settings

    assert Settings(_env_file=None).max_upload_mb == 50


def test_get_unknown_session_returns_404(client):
    assert client.get("/api/session/does-not-exist").status_code == 404


def test_system_prompt_loaded_from_markdown():
    from app.prompts import SYSTEM_PROMPT_PATH, build_messages, load_system_prompt

    assert SYSTEM_PROMPT_PATH.name == "system_prompt.md" and SYSTEM_PROMPT_PATH.exists()
    prompt = load_system_prompt()
    assert "분석 절차" in prompt
    assert "답변 범위" in prompt and "데이터 분석 관련 질문만" in prompt
    assert build_messages([], "질문")[0] == {"role": "system", "content": prompt}


def test_default_model_is_gpt_6_luna():
    from app.config import Settings

    assert Settings(_env_file=None).openai_model == "gpt-6-luna"
