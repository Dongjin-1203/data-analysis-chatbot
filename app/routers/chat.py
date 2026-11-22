import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import llm
from app.prompts import build_messages
from app.sessions import store
from app.tools import make_tools

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


def _sse(data: dict | str, event: str | None = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


class TraceBuilder:
    """에이전트 이벤트를 화면 복원용 블록(text / tool)으로 모은다."""

    def __init__(self) -> None:
        self.blocks: list[dict] = []

    def add(self, event: dict) -> None:
        kind = event["type"]
        if kind == "token":
            if self.blocks and self.blocks[-1]["type"] == "text":
                self.blocks[-1]["text"] += event["text"]
            else:
                self.blocks.append({"type": "text", "text": event["text"]})
        elif kind == "tool_start":
            self.blocks.append(
                {"type": "tool", "id": event["id"], "name": event["name"], "args": event["args"], "status": "running"}
            )
        elif kind == "tool_end":
            for block in self.blocks:
                if block["type"] == "tool" and block["id"] == event["id"]:
                    block.update(
                        status="error" if event.get("status") == "error" else "done",
                        output=event["output"],
                        artifact=event.get("artifact"),
                    )

    @property
    def text(self) -> str:
        return "\n\n".join(b["text"].strip() for b in self.blocks if b["type"] == "text" and b["text"].strip())

    @property
    def has_tools(self) -> bool:
        return any(b["type"] == "tool" for b in self.blocks)


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    session = store.get_or_create(req.session_id)
    history = store.history(session.id)
    messages = build_messages(history, req.message, session.data_summary, session.filename)
    tools = make_tools(session.id)

    async def event_stream():
        trace = TraceBuilder()
        try:
            async for event in llm.stream_agent(messages, tools):
                trace.add(event)
                yield _sse(event)
        except (llm.LLMConfigError, llm.AgentStepLimitError) as e:
            yield _sse({"message": str(e)}, event="error")
            return
        except Exception as e:  # OpenAI API 오류 등
            yield _sse({"message": f"LLM 호출 중 오류가 발생했습니다: {e}"}, event="error")
            return
        store.add_turn(session.id, req.message, trace.text, trace.blocks if trace.has_tools else None)
        yield _sse("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/files/{session_id}/{name}")
def get_output_file(session_id: str, name: str) -> FileResponse:
    """run_python 이 만든 그래프 이미지."""
    base = store.output_dir(session_id).resolve()
    path = (base / name).resolve()
    if path.parent != base or path.suffix != ".png" or not path.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/png")
