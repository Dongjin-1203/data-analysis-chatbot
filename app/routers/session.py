from fastapi import APIRouter, HTTPException

from app.sessions import store

router = APIRouter(prefix="/api/session", tags=["session"])


@router.post("")
def create_session() -> dict:
    return {"session_id": store.create()}


@router.delete("/{session_id}")
def delete_session(session_id: str) -> dict:
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"deleted": True}
