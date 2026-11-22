from pathlib import Path

from app.data_summary import load_dataframe
from app.tools import NO_DATA, query_dataset_impl

FIXTURE = Path(__file__).parent / "fixtures" / "sample_inspection.csv"
DF = load_dataframe("sample.csv", FIXTURE.read_bytes())


def test_schema_when_no_arguments():
    text = query_dataset_impl(DF)
    assert "행 10개, 열 8개" in text
    assert "temperature" in text and "결측치" in text


def test_select_columns_with_filter():
    text = query_dataset_impl(DF, columns=["inspection_id", "line"], filter="defect == 1")
    assert "조건에 맞는 행: 3개" in text
    assert "INS-0002" in text and "INS-0001" not in text
    assert "temperature" not in text


def test_unknown_column_suggests_similar():
    text = query_dataset_impl(DF, columns=["temprature"])
    assert text.startswith("[조회 실패]")
    assert "temprature" in text and "temperature" in text


def test_bad_filter():
    text = query_dataset_impl(DF, filter="no_col > 3")
    assert text.startswith("[조회 실패]")


def test_no_data():
    assert query_dataset_impl(None) == NO_DATA
