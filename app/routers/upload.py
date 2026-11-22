from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.data_summary import UnsupportedFileError, load_dataframe, summarize
from app.sessions import store

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    content = await file.read()
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다. 최대 {get_settings().max_upload_mb}MB")

    filename = file.filename or "uploaded"
    try:
        df = load_dataframe(filename, content)
    except UnsupportedFileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # 깨진 파일 등 pandas 파싱 오류
        raise HTTPException(status_code=400, detail=f"파일을 읽을 수 없습니다: {e}") from e

    summary = summarize(df)
    session = store.get_or_create(session_id)
    session.dataframe = df
    session.data_summary = summary
    session.filename = filename

    return {
        "filename": filename,
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "summary": summary,
    }
