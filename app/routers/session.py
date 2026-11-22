import json

from fastapi import APIRouter, HTTPException

from app.sessions import store

router = APIRouter(prefix="/api", tags=["session"])


@router.post("/session")
def create_session() -> dict:
    return {"session_id": store.create()}


@router.get("/sessions")
def list_sessions() -> list[dict]:
    return store.list_sessions()


@router.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    """화면 복원용: 세션 정보, 전체 메시지, 업로드 데이터 정보."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    messages = []
    for m in store.messages(session_id):
        if m["role"] == "file":
            messages.append({"role": "file", "data": json.loads(m["content"])})
        else:
            messages.append(m)

    data = None
    if session.data_meta:
        data = {**session.data_meta, "summary": session.data_summary}

    return {"id": session.id, "title": session.title, "messages": messages, "data": data}


@router.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"deleted": True}
