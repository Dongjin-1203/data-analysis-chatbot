from pathlib import Path

from app.executor import run_code

FIXTURE = Path(__file__).parent / "fixtures" / "sample_inspection.csv"


def test_runs_code_with_df_loaded(tmp_path):
    result = run_code("print(df.shape)\nprint(df['defect'].sum())", str(FIXTURE), tmp_path)
    assert result.ok, result.error
    assert "(10, 8)" in result.output
    assert "3" in result.output


def test_last_expression_is_printed(tmp_path):
    result = run_code("df.groupby('line')['defect'].mean().round(2)", str(FIXTURE), tmp_path)
    assert result.ok
    assert "A" in result.output and "0.25" in result.output


def test_wrong_column_returns_real_error(tmp_path):
    result = run_code("x = 1\ndf['no_such_column'].mean()", str(FIXTURE), tmp_path)
    assert not result.ok
    assert "KeyError" in result.error and "no_such_column" in result.error
    assert "2번째 줄" in result.error
    assert "[실행 실패]" in result.to_text()


def test_plot_saved_as_image(tmp_path):
    code = "df.groupby('line')['defect'].mean().plot(kind='bar')\nplt.title('라인별 불량률')"
    result = run_code(code, str(FIXTURE), tmp_path)
    assert result.ok, result.error
    assert len(result.images) == 1
    assert (tmp_path / result.images[0]).stat().st_size > 0


def test_timeout(tmp_path):
    result = run_code("while True:\n    pass", str(FIXTURE), tmp_path, timeout=2)
    assert not result.ok
    assert "2초" in result.error


def test_without_data(tmp_path):
    result = run_code("print('df' in globals())", None, tmp_path)
    assert result.ok and "False" in result.output


def test_relative_storage_dir_works(tmp_path, monkeypatch):
    """서버처럼 상대 경로(storage)로 저장소를 만들어도 코드 실행기가 데이터를 찾는다."""
    from app.data_summary import load_dataframe, summarize
    from app.sessions import SessionStore

    monkeypatch.chdir(tmp_path)
    store = SessionStore("storage")
    sid = store.create()
    content = FIXTURE.read_bytes()
    df = load_dataframe("sample.csv", content)
    store.save_data(sid, "sample.csv", content, df, summarize(df))

    data_path = store.get(sid).data_path
    assert Path(data_path).is_absolute()
    result = run_code("print(len(df))", data_path, store.output_dir(sid))
    assert result.ok, result.error
    assert result.output.strip() == "10"


def test_legacy_relative_data_path_is_resolved(tmp_path, monkeypatch):
    """1단계 DB처럼 상대 경로로 저장된 data_path 도 실행된다."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "sample.csv").write_bytes(FIXTURE.read_bytes())
    result = run_code("print(len(df))", "uploads/sample.csv", tmp_path / "out")
    assert result.ok, result.error
    assert result.output.strip() == "10"


def test_missing_data_file_is_reported(tmp_path):
    result = run_code("print(len(df))", str(tmp_path / "gone.csv"), tmp_path)
    assert not result.ok
    assert "찾을 수 없습니다" in result.error and "gone.csv" in result.error


def test_csv_path_alias(tmp_path):
    result = run_code("print(len(pd.read_csv(CSV_PATH)))", str(FIXTURE), tmp_path)
    assert result.ok, result.error
    assert result.output.strip() == "10"
