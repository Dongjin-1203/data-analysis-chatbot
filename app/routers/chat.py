import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import llm
from app.config import get_settings
from app.prompts import build_messages
from app.sessions import store

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


def _sse(data: dict | str, event: str | None = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    session = store.get_or_create(req.session_id)
    messages = build_messages(session.history, req.message, session.data_summary, session.filename)

    async def event_stream():
        chunks: list[str] = []
        try:
            async for delta in llm.stream_chat(messages):
                chunks.append(delta)
                yield _sse({"delta": delta})
        except llm.LLMConfigError as e:
            yield _sse({"message": str(e)}, event="error")
            return
        except Exception as e:  # OpenAI API 오류 등
            yield _sse({"message": f"LLM 호출 중 오류가 발생했습니다: {e}"}, event="error")
            return
        session.add_turn(req.message, "".join(chunks), get_settings().max_history_turns)
        yield _sse("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
