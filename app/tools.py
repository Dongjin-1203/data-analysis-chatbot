"""에이전트가 사용하는 도구.

- query_dataset: 업로드된 CSV에서 열 정보와 데이터를 조회한다.
  (사내 데이터 레이크 조회 라이브러리가 맡는 역할을 대신한다. 실제 환경에서는 이 함수 안을
   데이터 레이크 조회 코드로 바꾸면 에이전트는 그대로 쓸 수 있다.)
- run_python: 에이전트가 작성한 Pandas 코드를 실행하고 실제 출력이나 오류를 돌려준다.

도구는 요청마다 세션에 묶어 새로 만든다(make_tools).
"""

import difflib

import pandas as pd
from langchain_core.tools import BaseTool, tool

from app.config import get_settings
from app.executor import run_code
from app.sessions import store

NO_DATA = "업로드된 데이터가 없습니다. 사용자에게 CSV/Excel 파일을 먼저 첨부해 달라고 안내하세요."
MAX_ROWS = 50
MAX_CELL_CHARS = 6000


def _unknown_columns_message(df: pd.DataFrame, unknown: list[str]) -> str:
    lines = [f"존재하지 않는 열: {', '.join(unknown)}"]
    for name in unknown:
        close = difflib.get_close_matches(name, [str(c) for c in df.columns], n=3, cutoff=0.5)
        if close:
            lines.append(f"- '{name}'와 비슷한 열: {', '.join(close)}")
    lines.append(f"전체 열 수: {df.shape[1]}개. 열 목록은 columns 없이 이 도구를 호출하면 확인할 수 있습니다.")
    return "\n".join(lines)


def query_dataset_impl(
    df: pd.DataFrame | None,
    columns: list[str] | None = None,
    filter: str | None = None,
    limit: int = 10,
) -> str:
    if df is None:
        return NO_DATA

    # 인자가 없으면 스키마(열 이름·자료형·결측치)를 돌려준다
    if not columns and not filter:
        info = pd.DataFrame(
            {"dtype": df.dtypes.astype(str), "결측치": df.isna().sum(), "고유값 수": df.nunique()}
        )
        return f"행 {df.shape[0]}개, 열 {df.shape[1]}개\n{info.to_string()}"

    if columns:
        unknown = [c for c in columns if c not in df.columns]
        if unknown:
            return "[조회 실패]\n" + _unknown_columns_message(df, unknown)

    result = df
    if filter:
        try:
            result = df.query(filter, engine="python")
        except Exception as e:
            return f"[조회 실패] filter 조건을 해석할 수 없습니다: {type(e).__name__}: {e}"

    if columns:
        result = result[columns]

    limit = max(1, min(limit, MAX_ROWS))
    text = f"조건에 맞는 행: {len(result)}개 (앞 {min(limit, len(result))}개 표시)\n{result.head(limit).to_string()}"
    if len(text) > MAX_CELL_CHARS:
        text = text[:MAX_CELL_CHARS] + "\n...(출력이 길어 생략됨)"
    return text


def make_tools(session_id: str) -> list[BaseTool]:
    @tool
    def query_dataset(columns: list[str] | None = None, filter: str | None = None, limit: int = 10) -> str:
        """업로드된 데이터(CSV/Excel)를 조회한다. 사내 데이터 레이크 조회 라이브러리에 해당한다.

        - 인자 없이 호출하면 전체 열 이름, 자료형, 결측치 수를 돌려준다.
        - columns: 조회할 열 이름 목록. 존재하지 않는 열이면 비슷한 열 이름을 알려준다.
        - filter: pandas DataFrame.query 문법의 조건식 (예: "quality_overall == 1 and slot > 10").
        - limit: 돌려받을 최대 행 수 (최대 50).
        집계나 계산이 필요하면 이 도구 대신 run_python 을 사용한다.
        """
        return query_dataset_impl(store.dataframe(session_id), columns, filter, limit)

    @tool(response_format="content_and_artifact")
    def run_python(code: str) -> tuple[str, dict]:
        """Pandas 코드를 실행하고 실제 출력이나 오류를 돌려준다.

        - 업로드된 데이터가 `df` 로 미리 로드되어 있다. `pd`, `np`, `plt` 도 import 되어 있다.
        - 결과는 print() 로 출력하거나 마지막 줄에 값을 두면 출력된다.
        - plt 로 그린 그래프는 자동으로 저장되어 사용자 화면에 표시된다(plt.show() 불필요).
        - 실행마다 새 프로세스이므로 이전 실행의 변수는 남지 않는다.
        - 파일 쓰기, 네트워크 접근, 패키지 설치는 하지 않는다.
        """
        session = store.get(session_id)
        data_path = session.data_path if session else None
        result = run_code(code, data_path, store.output_dir(session_id), timeout=get_settings().code_timeout_sec)
        images = [f"/api/files/{session_id}/{name}" for name in result.images]
        artifact = {"ok": result.ok, "output": result.output, "error": result.error, "images": images}
        return result.to_text(), artifact

    tools: list[BaseTool] = [query_dataset, run_python, _make_predict_tool(session_id), _make_manual_tool()]
    if get_settings().enable_shell_tool:
        tools.append(_make_shell_tool(session_id))
    return tools


def _make_shell_tool(session_id: str) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def run_shell(command: str) -> tuple[str, dict]:
        """터미널 명령을 실행하고 출력과 종료 코드를 돌려준다.

        주 용도: run_python 실행 중 ModuleNotFoundError 가 나면 필요한 패키지를 설치한다.
        - 패키지 설치: `uv pip install scikit-learn` (`pip install ...` 도 자동으로 uv 로 바뀐다)
        - 설치 확인: `python -c "import sklearn; print(sklearn.__version__)"`, `uv pip list`
        - `python` 은 run_python 과 같은 가상환경이므로, 설치 후 바로 run_python 에서 import 할 수 있다.
        - 세션 작업 폴더에서 실행되며 시간 제한이 있다. 대화형 입력이 필요한 명령은 쓸 수 없다.
        - 파일 삭제, 시스템 설정 변경, 패키지 제거, git 조작 등 위험한 명령은 거부된다.
        """
        from app.shell import run_shell as _run

        result = _run(command, store.workspace_dir(session_id), timeout=get_settings().shell_timeout_sec)
        artifact = {
            "ok": result.ok,
            "exit_code": result.exit_code,
            "command": result.command,
            "output": result.output,
            "blocked": result.blocked,
            "error": None if result.ok or result.blocked else f"종료 코드 {result.exit_code}",
        }
        return result.to_text(), artifact

    from app.shell import environment_note

    run_shell.description += f"\n실행 환경: {environment_note()}"
    return run_shell


# ---------- 3단계: 불량 예측 · 매뉴얼 검색 ----------
def _prediction_text(result: dict) -> str:
    rec = ", ".join(f"{k}={v}" for k, v in result["record"].items())
    m = result["model"]
    lines = [
        "[예측 완료]",
        f"검사 건: {rec}",
        f"판정: {result['prediction']} (불량 점수 {result['score']} / 임계값 {result['threshold']})",
    ]
    if result.get("actual_label") is not None:
        lines.append(f"데이터의 실제 quality_overall: {result['actual_label']} ({'불량' if result['actual_label'] else '양품'})")
    if result["imputed_features"]:
        lines.append(f"비어 있어 학습 중앙값으로 채운 입력 열: {result['imputed_features']}개")
    lines.append("학습 분포 대비 많이 벗어난 입력 열 (z = 표준편차 단위):")
    for d in result["top_deviations"]:
        lines.append(f"- {d['feature']}: 값 {d['value']}, 학습 평균 {d['train_mean']}, z {d['z']}")
    lines.append(
        f"모델: {m['name']} (입력 열 {m['n_features']}개, 테스트 ROC-AUC {m['test_roc_auc']}, F1 {m['test_f1']}). {m['note']}"
    )
    return "\n".join(lines)


def _make_predict_tool(session_id: str) -> BaseTool:
    from app.predictor import PredictionError, predict_record, predict_values, standalone_code_example

    @tool(response_format="content_and_artifact")
    def predict_defect(
        row_index: int | None = None,
        wafer_id: str | None = None,
        values: dict[str, float] | None = None,
    ) -> tuple[str, dict]:
        """제공된 PyTorch 모델 가중치(models/defect_mlp.pt)를 CPU 에서 불러와 검사 건 한 건의 불량 여부를 예측한다.

        - 업로드된 데이터의 한 행을 row_index(0부터 시작) 또는 wafer_id 로 지정한다.
        - 업로드된 데이터 없이 예측하려면 values 에 {입력 열 이름: 값} 을 넣는다.
        - 결과: 불량 점수와 임계값, 판정(불량 의심/양품 예상), 학습 분포 대비 많이 벗어난 입력 열 상위 5개.
          벗어난 열 이름은 search_manual 로 관련 점검 절차를 찾는 데 쓸 수 있다.
        - 모델은 PVD AlCu 공정 센서 통계(약 85개 열) → quality_overall(0 양품 / 1 불량) 분류만 지원한다.
          필수 열이 없거나 형식이 다르면 [예측 불가] 와 이유, 별도 환경에서 실행할 코드 예시를 돌려준다.
        """
        try:
            if values:
                result = predict_values(values)
            else:
                df = store.dataframe(session_id)
                if df is None:
                    return NO_DATA, {"ok": False, "error": "업로드된 데이터 없음"}
                result = predict_record(df, row_index, wafer_id)
        except PredictionError as e:
            if e.kind == "bad_request":
                return f"[요청 확인 필요] {e.message}", {"ok": False, "error": e.message}
            code = standalone_code_example()
            text = (
                f"[예측 불가] {e.message}\n"
                "사용자에게 처리할 수 없는 이유를 설명하고, 아래 코드를 별도 환경에서 실행하는 방법으로 안내하세요.\n"
                f"```python\n{code}```"
            )
            return text, {"ok": False, "error": e.message, "kind": e.kind, "details": e.details}
        return _prediction_text(result), {"ok": True, "result": result}

    return predict_defect


def _make_manual_tool() -> BaseTool:
    from app.rag import get_index

    @tool(response_format="content_and_artifact")
    def search_manual(query: str, k: int = 3) -> tuple[str, dict]:
        """가상 점검 매뉴얼(manuals/MAN-01~06)에서 질문과 관련된 절을 검색한다 (text-embedding-3-small).

        - query: 찾을 내용. 증상이나 센서 열 이름을 함께 넣으면 정확도가 높아진다.
          예: "PRESS_CHAMBER_DEP1_S 압력 표준편차 증가 점검 절차", "불량 의심 웨이퍼 처리 흐름"
        - k: 가져올 절 수 (기본 3, 최대 8)
        - 결과마다 [문서번호 절 제목] 출처가 붙는다. 답변에 이 출처를 그대로 인용한다.
        """
        try:
            hits = get_index().search(query, k)
        except Exception as e:
            return f"[검색 실패] {type(e).__name__}: {e}", {"ok": False, "error": str(e), "results": []}
        if not hits:
            return "[검색 결과 없음] 매뉴얼 폴더가 비어 있습니다.", {"ok": True, "results": []}
        parts, results = [], []
        for chunk, score in hits:
            parts.append(f"{chunk.citation} (유사도 {score:.3f})\n{chunk.text}")
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "score": round(score, 3),
                    "text": chunk.text,
                    "url": f"/manuals/{chunk.file}",
                }
            )
        return "\n\n---\n\n".join(parts), {"ok": True, "results": results}

    return search_manual
