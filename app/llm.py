"""LLM 호출 단일 지점. 2단계에서 LangChain/LangGraph로 교체한다."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import get_settings


class LLMConfigError(RuntimeError):
    pass


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMConfigError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    """메시지 목록을 보내고 응답 텍스트 조각을 순서대로 돌려준다."""
    client = _get_client()
    stream = await client.chat.completions.create(
        model=get_settings().openai_model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
