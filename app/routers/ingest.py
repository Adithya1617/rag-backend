import tempfile
from fastapi import APIRouter, UploadFile, BackgroundTasks, Form, HTTPException
from typing import Optional
from app.services.ingest_service import process_file_background

router = APIRouter()

@router.post("/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    source_name: Optional[str] = Form(None),
    max_tokens: int = Form(500),
    overlap: int = Form(100),
):
    if not file:
        raise HTTPException(status_code=400, detail="file required")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    contents = await file.read()
    tmp.write(contents)
    tmp.flush()
    tmp.close()
    # schedule background task
    background_tasks.add_task(process_file_background, tmp.name, file.filename, source_name or file.filename, max_tokens, overlap)
    return {"status": "ingestion_started", "filename": file.filename}
