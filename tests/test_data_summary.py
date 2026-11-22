from pathlib import Path

import pytest

from app.data_summary import UnsupportedFileError, load_dataframe, summarize

FIXTURE = Path(__file__).parent / "fixtures" / "sample_inspection.csv"


def test_load_csv():
    df = load_dataframe("sample.csv", FIXTURE.read_bytes())
    assert df.shape == (10, 8)
    assert "defect" in df.columns


def test_load_cp949_csv():
    content = "라인,불량\nA,1\nB,0\n".encode("cp949")
    df = load_dataframe("korean.csv", content)
    assert list(df.columns) == ["라인", "불량"]


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        load_dataframe("data.json", b"{}")


def test_summary_contains_structure_and_missing():
    df = load_dataframe("sample.csv", FIXTURE.read_bytes())
    text = summarize(df)
    assert "행 10개, 열 8개" in text
    assert "결측치" in text
    assert "temperature" in text and "pressure" in text
    assert "수치형 통계" in text
    assert "범주형 상위 값" in text
