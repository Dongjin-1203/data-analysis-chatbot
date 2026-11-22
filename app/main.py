from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from fastapi.staticfiles import StaticFiles

from app.rag import MANUAL_DIR
from app.routers import chat, session, upload

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Data Analysis Chatbot", version="0.3.0")

app.include_router(session.router)
app.include_router(upload.router)
app.include_router(chat.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/manuals/{name}", include_in_schema=False)
def manual(name: str) -> PlainTextResponse:
    """점검 매뉴얼 원문 (답변의 출처와 대조용)."""
    path = (MANUAL_DIR / name).resolve()
    if path.parent != MANUAL_DIR.resolve() or path.suffix != ".md" or not path.is_file():
        raise HTTPException(status_code=404, detail="매뉴얼을 찾을 수 없습니다.")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")
