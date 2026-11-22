"""업로드된 표 데이터를 읽고 LLM 컨텍스트용 요약 텍스트를 만든다.

열이 많은 데이터(예: 수백 개의 센서 통계 열)도 열 이름이 빠지지 않도록,
열 목록은 항상 전부 싣고 통계·미리보기는 열 수에 따라 줄인다.
"""

import io

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

WIDE_THRESHOLD = 30  # 이 열 수를 넘으면 '넓은 표'로 보고 요약을 압축한다
MAX_STATS_COLUMNS = 30  # 넓은 표에서 수치형 통계를 보여줄 최대 열 수
MAX_CATEGORY_UNIQUE = 20  # 이 이하의 고유값을 가진 열만 범주형 상위 값을 보여준다
PREVIEW_COLUMNS = 12
MAX_SUMMARY_CHARS = 40000


class UnsupportedFileError(ValueError):
    pass


def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        for encoding in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise UnsupportedFileError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    raise UnsupportedFileError(f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")


def _column_list(df: pd.DataFrame) -> str:
    """모든 열을 '이름 (dtype, 결측 n)' 형태로 한 줄씩 나열한다."""
    rows = len(df)
    missing = df.isna().sum()
    lines = []
    for col in df.columns:
        m = int(missing[col])
        miss = f", 결측 {m} ({m / max(rows, 1) * 100:.1f}%)" if m else ""
        lines.append(f"- {col} ({df[col].dtype}{miss})")
    return "\n".join(lines)


def _numeric_stats(df: pd.DataFrame, wide: bool) -> str | None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return None
    note = ""
    if wide and numeric.shape[1] > MAX_STATS_COLUMNS:
        note = f"(수치형 열 {numeric.shape[1]}개 중 앞 {MAX_STATS_COLUMNS}개만 표시)\n"
        numeric = numeric.iloc[:, :MAX_STATS_COLUMNS]
    stats = numeric.describe().T[["mean", "std", "min", "50%", "max"]].round(4)
    return f"## 수치형 통계\n{note}{stats.to_string()}"


def _categorical_values(df: pd.DataFrame) -> str | None:
    lines = []
    for col in df.columns:
        series = df[col]
        nunique = series.nunique()
        # 범주형 열이거나, 고유값이 적은 수치형(0/1 플래그 등) 열
        if series.dtype == object or (0 < nunique <= 5):
            if nunique > MAX_CATEGORY_UNIQUE:
                lines.append(f"- {col}: 고유값 {nunique}개 (ID 또는 자유 텍스트로 보임)")
                continue
            top = series.value_counts(dropna=False).head(MAX_CATEGORY_UNIQUE)
            values = ", ".join(f"{k}({v})" for k, v in top.items())
            lines.append(f"- {col}: {values}")
    if not lines:
        return None
    return "## 범주형 상위 값\n" + "\n".join(lines)


def summarize(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    wide = cols > WIDE_THRESHOLD

    parts = [f"## 크기\n행 {rows}개, 열 {cols}개"]
    parts.append(f"## 열 목록 (전체 {cols}개, 결측치)\n{_column_list(df)}")

    for section in (_categorical_values(df), _numeric_stats(df, wide)):
        if section:
            parts.append(section)

    preview = df.iloc[:, :PREVIEW_COLUMNS] if wide else df
    note = f" (앞 {PREVIEW_COLUMNS}개 열만 표시)" if wide else ""
    parts.append(f"## 앞 5행{note}\n{preview.head(5).to_string()}")

    text = "\n\n".join(parts)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n...(요약이 길어 생략됨)"
    return text
