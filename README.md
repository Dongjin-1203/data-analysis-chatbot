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

## 테스트

```bash
uv run pytest
```

## 구조

```
app/
  main.py           FastAPI 앱, 정적 파일 제공
  config.py         환경 설정(.env)
  llm.py            LLM 호출 단일 지점(OpenAI SDK 스트리밍)
  prompts.py        분석 절차 시스템 프롬프트
  sessions.py       메모리 세션(대화 기록, 업로드 데이터)
  data_summary.py   CSV/Excel 읽기 및 요약
  routers/          /api/session, /api/upload, /api/chat(SSE)
static/             채팅 UI (HTML/CSS/JS)
data/               실습 데이터 위치
tests/              pytest
```
