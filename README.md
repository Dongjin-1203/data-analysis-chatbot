# data-analysis-chatbot

정형 데이터를 입력받아 사용자 지시에 따라 분석 및 예측/분류 작업을 수행해 주는 챗봇입니다.

## 개발 단계

| 단계 | 브랜치 | 내용 |
|---|---|---|
| 1단계 | `dev/stage1` | OpenAI SDK 기반 챗봇. CSV 업로드 후 분석 절차(구조 확인 → 결측치 점검 → 집계·시각화 → 원본 대조)에 따라 분석 계획과 Pandas 코드 초안을 제안 |
| 2단계 | `dev/stage2` | LangChain 전환, CSV 조회 도구와 코드 실행 도구 연결, 실행 결과에 근거한 응답, 패키지 설치용 터미널 도구 |
| 3단계 | `dev/stage3` → `main` | 불량 예측 도구(CPU, PyTorch 모델), 점검 매뉴얼 검색(RAG), 지원 범위 밖 입력 처리 |

## 실행 방법

[uv](https://docs.astral.sh/uv/)와 Python 3.12 이상이 필요합니다.

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY, OPENAI_MODEL 입력
uv run uvicorn app.main:app --reload
```

브라우저에서 http://localhost:8000 을 엽니다.

> **코드 실행 주의**: `run_python`·`run_shell` 도구는 모델이 작성한 코드와 명령을 서버 PC에서 실행합니다.
> 시간 제한과 위험 명령 차단만 적용되며 완전한 보안 샌드박스가 아닙니다. 로컬·실습 환경에서만 사용하세요.
> 터미널 도구는 `.env`에 `ENABLE_SHELL_TOOL=false`로 끌 수 있습니다.

## 실습 데이터

Mević, Laber, Krivić, *Advanced Process Control and Statistical Process Control Data for Thickness Prediction of
AlCu and WTi Metal Layer in Semiconductor Manufacturing*, Zenodo, [doi:10.5281/zenodo.23442538](https://doi.org/10.5281/zenodo.23442538) (CC BY 4.0)

`apc_spc_AlCu_inputs_outputs.csv`를 받아 `data/alcu_v2.csv`로 저장합니다. (원본 CSV는 git에 올리지 않습니다.)

## 3단계: 불량 예측 모델과 점검 매뉴얼

### 불량 예측 모델 (`models/`)
- `defect_mlp.pt`: PyTorch 소형 신경망 가중치 (CPU에서 불러옴)
- `defect_mlp.json`: 입력 열 85개, 결측치 대체값(학습 중앙값), 표준화 평균·표준편차, 판정 임계값, 성능 지표
- 예측 대상: `quality_overall` (0 양품 / 1 불량). 계측 결과(`RAW_VALUE_*`, `mean_*`)와 메타데이터는 입력에서 제외
- 다시 학습하기:

  ```bash
  uv run python scripts/train_defect_model.py --data data/alcu_v2.csv
  ```

  층 수·결측 허용 비율 등은 `--hidden`, `--max-missing` 옵션으로 바꿀 수 있습니다.
  다른 가중치로 바꾸려면 같은 형식의 `.pt`와 `.json`을 `models/`에 두면 됩니다.

### 점검 매뉴얼 (`manuals/`)
- `MAN-01`~`MAN-06`: 교육·실습용으로 작성한 **가상 매뉴얼**입니다. 실제 설비·기업의 절차가 아닙니다.
- `search_manual` 도구가 `## ` 절 단위로 나눠 `text-embedding-3-small`로 임베딩·검색합니다.
  임베딩은 `storage/manual_embeddings.json`에 캐시되며, 매뉴얼을 고치면 바뀐 절만 다시 임베딩합니다.
- 원문은 `/manuals/MAN-01.md`처럼 브라우저에서 볼 수 있습니다(답변 출처와 대조용).

## 테스트

```bash
uv run pytest
```

테스트는 OpenAI를 호출하지 않습니다(가짜 모델·가짜 임베딩 사용).

## 구조

```
app/
  main.py           FastAPI 앱, 정적 파일·매뉴얼 원문 제공
  config.py         환경 설정(.env)
  llm.py            LLM 호출 단일 지점(LangChain 에이전트, 실행 과정 스트리밍)
  tools.py          에이전트 도구: query_dataset, run_python, predict_defect, search_manual, run_shell
  executor.py       run_python 코드를 별도 프로세스에서 실행(시간 제한)
  sandbox_runner.py 실제 코드 실행기(df 미리 로드, 출력·오류·그래프 수집)
  shell.py          run_shell 터미널 명령 실행(가상환경 고정, 위험 명령 차단)
  defect_model.py   불량 예측 모델 정의·전처리·저장/불러오기
  predictor.py      예측 도구 본체(입력 검증, 지원 범위 밖 입력 처리)
  rag.py            매뉴얼 청크 분할·임베딩·검색
  prompts.py        시스템 프롬프트 로드, 메시지 구성
  sessions.py       SQLite 세션 저장소(대화 기록, 업로드 데이터, 실행 과정) → storage/
  data_summary.py   CSV/Excel 읽기 및 요약
  routers/          /api/session(s), /api/upload, /api/chat(SSE), /api/files(그래프)
models/             불량 예측 모델 가중치와 메타데이터
manuals/            가상 점검 매뉴얼 (RAG 검색 대상)
scripts/            모델 학습 스크립트
prompts/
  system_prompt.md  시스템 프롬프트 (수정하면 다음 질문부터 바로 반영)
static/             채팅 UI (HTML/CSS/JS)
data/               실습 데이터 위치 (git 제외)
tests/              pytest
```
