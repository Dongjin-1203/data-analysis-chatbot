"""업로드된 표 데이터를 읽고 LLM 컨텍스트용 요약 텍스트를 만든다."""

import io

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")
MAX_SUMMARY_CHARS = 6000


class UnsupportedFileError(ValueError):
    pass


def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        for encoding in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnsupportedFileError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    raise UnsupportedFileError(f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")


def summarize(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    missing = df.isna().sum()
    columns_table = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "결측치": missing,
            "결측률(%)": (missing / max(rows, 1) * 100).round(1),
            "고유값 수": df.nunique(),
        }
    )

    parts = [
        f"## 크기\n행 {rows}개, 열 {cols}개",
        f"## 열 정보\n{columns_table.to_string()}",
    ]

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        parts.append(f"## 수치형 통계\n{numeric.describe().T.round(3).to_string()}")

    categorical = df.select_dtypes(exclude="number")
    if not categorical.empty:
        lines = []
        for col in categorical.columns:
            top = categorical[col].value_counts().head(5)
            values = ", ".join(f"{k}({v})" for k, v in top.items())
            lines.append(f"- {col}: {values}")
        parts.append("## 범주형 상위 값\n" + "\n".join(lines))

    parts.append(f"## 앞 5행\n{df.head(5).to_string()}")

    text = "\n\n".join(parts)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n...(요약이 길어 생략됨)"
    return text
