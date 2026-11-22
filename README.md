# data-analysis-chatbot

정형 데이터를 입력받아 사용자 지시에 따라 분석 및 예측/분류 작업을 수행해 주는 챗봇입니다.

## 개발 단계

| 단계 | 브랜치 | 내용 |
|---|---|---|
| 1단계 | `dev/stage1` | OpenAI SDK 기반 챗봇. CSV 업로드 후 분석 절차(구조 확인 → 결측치 점검 → 집계·시각화 → 원본 대조)에 따라 분석 계획과 Pandas 코드 초안을 제안 |
| 2단계 | `dev/stage2` | LangChain 전환, CSV 조회 도구와 코드 실행 도구 연결, 실행 결과에 근거한 응답 |
| 3단계 | `dev/stage3` → `main` | 불량 예측 도구(CPU 모델), 점검 매뉴얼 검색(RAG), 지원 범위 밖 입력 처리 |

## 실행 방법

[uv](https://docs.astral.sh/uv/)와 Python 3.12 이상이 필요합니다.

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY, OPENAI_MODEL 입력
uv run uvicorn app.main:app --reload
```

브라우저에서 http://localhost:8000 을 엽니다.

> **코드 실행 주의**: `run_python` 도구는 모델이 작성한 코드를 서버 PC에서 별도 프로세스로 실행합니다.
> 시간 제한(`CODE_TIMEOUT_SEC`, 기본 30초)과 프로세스 분리만 적용되며 완전한 보안 샌드박스가 아닙니다.
> 로컬·실습 환경에서만 사용하세요.

## 테스트

```bash
uv run pytest
```

## 구조

```
app/
  main.py           FastAPI 앱, 정적 파일 제공
  config.py         환경 설정(.env)
  llm.py            LLM 호출 단일 지점(LangChain 에이전트, 실행 과정 스트리밍)
  tools.py          에이전트 도구: query_dataset(데이터 조회), run_python(코드 실행)
  executor.py       run_python 코드를 별도 프로세스에서 실행(시간 제한)
  sandbox_runner.py 실제 코드 실행기(df 미리 로드, 출력·오류·그래프 수집)
  prompts.py        시스템 프롬프트 로드, 메시지 구성
  sessions.py       SQLite 세션 저장소(대화 기록, 업로드 데이터, 실행 과정) → storage/
  data_summary.py   CSV/Excel 읽기 및 요약
  routers/          /api/session(s), /api/upload, /api/chat(SSE), /api/files(그래프)
prompts/
  system_prompt.md  시스템 프롬프트 (수정하면 다음 질문부터 바로 반영)
static/             채팅 UI (HTML/CSS/JS)
data/               실습 데이터 위치
tests/              pytest
```
