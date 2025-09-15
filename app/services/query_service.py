import json, time, asyncio
from typing import Generator
from app.embeddings import embed_texts_gemini
from app.pinecone_client import query_vectors
from app.llm_adapter import generate_answer_gemini_stream

async def _stream_generator(question: str, retrieved: list):
    meta = {"type": "retrieved", "items": retrieved}
    yield f"data: {json.dumps(meta)}\n\n"
    # stream tokens from Gemini (adapter)
    async for token in generate_answer_gemini_stream(question, retrieved):
        yield f"data: {json.dumps({'type':'token','token': token})}\n\n"
    yield f"data: {json.dumps({'type':'done'})}\n\n"

def handle_query_stream(question: str, top_k: int = 5) -> Generator:
    emb = embed_texts_gemini([question])[0]
    res = query_vectors(emb, top_k=top_k)
    retrieved = []
    matches = res.get('matches') or []
    for m in matches:
        retrieved.append({
            'id': m.get('id'),
            'score': m.get('score'),
            'text': m.get('metadata', {}).get('text_excerpt') or m.get('metadata')
        })
    return _stream_generator(question, retrieved)
