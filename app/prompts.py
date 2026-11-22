from pathlib import Path

# 시스템 프롬프트는 prompts/system_prompt.md 에서 관리한다.
# 요청마다 파일을 다시 읽으므로, 파일을 고치면 서버를 재시작하지 않아도 다음 질문부터 반영된다.
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"

DATA_CONTEXT_TEMPLATE = """사용자가 업로드한 데이터 요약입니다. 열 이름과 자료형은 이 정보를 기준으로 하세요.

파일명: {filename}

{summary}
"""


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_messages(
    history: list[dict],
    user_message: str,
    data_summary: str | None = None,
    filename: str | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": load_system_prompt()}]
    if data_summary:
        messages.append(
            {
                "role": "system",
                "content": DATA_CONTEXT_TEMPLATE.format(filename=filename or "(알 수 없음)", summary=data_summary),
            }
        )
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
