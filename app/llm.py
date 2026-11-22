"""LLM 호출 단일 지점 (2단계: LangChain 에이전트).

stream_agent 는 에이전트 실행 과정을 아래 이벤트로 바꿔 순서대로 돌려준다.
- {"type": "token", "text": ...}                              모델이 생성한 답변 조각
- {"type": "tool_start", "id", "name", "args"}                 도구 호출 시작
- {"type": "tool_end", "id", "name", "output", "artifact", "status"}  도구 실행 결과
"""

from collections.abc import AsyncIterator, Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from app.config import get_settings


class LLMConfigError(RuntimeError):
    pass


class AgentStepLimitError(RuntimeError):
    pass


def get_model() -> BaseChatModel:
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMConfigError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, streaming=True)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


async def stream_agent(
    messages: list[dict],
    tools: Sequence[BaseTool],
    model: BaseChatModel | None = None,
) -> AsyncIterator[dict]:
    model = model or get_model()
    agent = create_agent(model, list(tools))
    config = {"recursion_limit": get_settings().max_agent_steps * 2 + 1}

    streamed_tokens = False  # 현재 모델 단계에서 토큰 스트리밍이 있었는지
    try:
        async for mode, chunk in agent.astream({"messages": messages}, config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                msg, _meta = chunk
                if isinstance(msg, AIMessageChunk):
                    text = _text(msg.content)
                    if text:
                        streamed_tokens = True
                        yield {"type": "token", "text": text}
                continue

            # mode == "updates": 노드별로 확정된 메시지
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                for m in update.get("messages", []):
                    if isinstance(m, AIMessage):
                        # 스트리밍을 지원하지 않는 모델이면 확정된 텍스트를 한 번에 보낸다
                        if not streamed_tokens and _text(m.content):
                            yield {"type": "token", "text": _text(m.content)}
                        streamed_tokens = False
                        for call in m.tool_calls:
                            yield {"type": "tool_start", "id": call["id"], "name": call["name"], "args": call["args"]}
                    elif isinstance(m, ToolMessage):
                        yield {
                            "type": "tool_end",
                            "id": m.tool_call_id,
                            "name": m.name,
                            "output": _text(m.content),
                            "artifact": m.artifact,
                            "status": m.status,
                        }
    except GraphRecursionError as e:
        raise AgentStepLimitError(
            f"도구 호출이 {get_settings().max_agent_steps}회를 넘어 중단했습니다. 질문을 더 구체적으로 나눠 주세요."
        ) from e
