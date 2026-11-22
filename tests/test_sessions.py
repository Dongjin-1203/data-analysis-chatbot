from pathlib import Path

from app.data_summary import load_dataframe, summarize
from app.sessions import SessionStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_inspection.csv"


def test_session_survives_restart(tmp_path):
    store = SessionStore(tmp_path)
    sid = store.create()
    content = FIXTURE.read_bytes()
    df = load_dataframe("sample.csv", content)
    store.save_data(sid, "sample.csv", content, df, summarize(df))
    store.add_turn(sid, "라인별 불량률 알려줘", "분석 계획입니다.")

    # 서버 재시작을 흉내: 같은 폴더로 새 저장소 생성
    restarted = SessionStore(tmp_path)
    session = restarted.get(sid)
    assert session.filename == "sample.csv"
    assert session.title == "라인별 불량률 알려줘"
    assert "defect" in session.data_summary
    assert restarted.history(sid) == [
        {"role": "user", "content": "라인별 불량률 알려줘"},
        {"role": "assistant", "content": "분석 계획입니다."},
    ]
    assert [m["role"] for m in restarted.messages(sid)] == ["file", "user", "assistant"]
    assert restarted.dataframe(sid).shape == (10, 8)  # 저장된 파일에서 다시 읽음
    assert [s["id"] for s in restarted.list_sessions()] == [sid]


def test_history_limited_to_recent_turns(tmp_path):
    store = SessionStore(tmp_path)
    sid = store.create()
    for i in range(5):
        store.add_turn(sid, f"q{i}", f"a{i}")
    history = store.history(sid, max_turns=2)
    assert [m["content"] for m in history] == ["q3", "a3", "q4", "a4"]


def test_empty_sessions_not_listed_and_delete_removes_files(tmp_path):
    store = SessionStore(tmp_path)
    empty = store.create()
    sid = store.create()
    content = FIXTURE.read_bytes()
    df = load_dataframe("sample.csv", content)
    store.save_data(sid, "sample.csv", content, df, summarize(df))

    ids = [s["id"] for s in store.list_sessions()]
    assert sid in ids and empty not in ids

    assert store.delete(sid)
    assert store.get(sid) is None
    assert not (tmp_path / "uploads" / sid).exists()
    assert store.dataframe(sid) is None
