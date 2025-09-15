from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.query_service import handle_query_stream

router = APIRouter()

@router.post("/chat")
async def chat(payload: dict = Body(...)):
    question = payload.get("question")
    top_k = int(payload.get("top_k", 5))
    if not question:
        return JSONResponse({"error": "question required"}, status_code=400)
    generator = handle_query_stream(question, top_k=top_k)
    return StreamingResponse(generator, media_type='text/event-stream')
