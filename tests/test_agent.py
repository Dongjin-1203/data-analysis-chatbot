"""실제 OpenAI 호출 없이, 정해진 응답을 내는 가짜 모델로 에이전트 흐름을 검증한다."""

import asyncio
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import llm
from app.data_summary import load_dataframe, summarize
from app.sessions import store
from app.tools import make_tools

FIXTURE = Path(__file__).parent / "fixtures" / "sample_inspection.csv"


class ScriptedModel(BaseChatModel):
    responses: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def _session_with_data() -> str:
    sid = store.create()
    content = FIXTURE.read_bytes()
    df = load_dataframe("sample.csv", content)
    store.save_data(sid, "sample.csv", content, df, summarize(df))
    return sid


def _run(model, sid):
    async def collect():
        return [e async for e in llm.stream_agent([{"role": "user", "content": "질문"}], make_tools(sid), model=model)]

    return asyncio.run(collect())


def _call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_agent_runs_code_and_answers_from_result():
    sid = _session_with_data()
    model = ScriptedModel(
        responses=[
            _call("run_python", {"code": "print(df['defect'].sum())"}, "c1"),
            AIMessage(content="불량은 3건입니다."),
        ]
    )
    events = _run(model, sid)

    kinds = [e["type"] for e in events]
    assert kinds.index("tool_start") < kinds.index("tool_end") < kinds.index("token")
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["name"] == "run_python"
    assert end["artifact"]["ok"] is True and end["artifact"]["output"].strip() == "3"
    assert "[실행 성공]" in end["output"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "불량은 3건입니다."


def test_agent_receives_real_error_for_wrong_column():
    sid = _session_with_data()
    model = ScriptedModel(
        responses=[
            _call("run_python", {"code": "df['defect_rate'].mean()"}, "c1"),
            AIMessage(content="'defect_rate' 열이 없습니다."),
        ]
    )
    events = _run(model, sid)
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["artifact"]["ok"] is False
    assert "KeyError" in end["artifact"]["error"] and "defect_rate" in end["artifact"]["error"]
    assert "[실행 실패]" in end["output"]


def test_query_dataset_tool_via_agent():
    sid = _session_with_data()
    model = ScriptedModel(
        responses=[
            _call("query_dataset", {"columns": ["line", "defect"], "filter": "defect == 1"}, "q1"),
            AIMessage(content="불량 3건을 조회했습니다."),
        ]
    )
    events = _run(model, sid)
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["name"] == "query_dataset"
    assert "조건에 맞는 행: 3개" in end["output"]


def test_plot_image_url_is_served():
    from fastapi.testclient import TestClient

    from app.main import app

    sid = _session_with_data()
    model = ScriptedModel(
        responses=[
            _call("run_python", {"code": "df.groupby('line')['defect'].mean().plot(kind='bar')"}, "c1"),
            AIMessage(content="그래프를 그렸습니다."),
        ]
    )
    events = _run(model, sid)
    images = next(e for e in events if e["type"] == "tool_end")["artifact"]["images"]
    assert len(images) == 1 and images[0].startswith(f"/api/files/{sid}/")

    client = TestClient(app)
    res = client.get(images[0])
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"
    assert client.get(f"/api/files/{sid}/..%2Fchat.db").status_code == 404
