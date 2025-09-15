import os, math, time
from typing import List
from app.utils.extract import extract_text_from_path
from app.chunking import chunk_text
from app.embeddings import embed_texts_gemini
from app.pinecone_client import upsert_vectors

BATCH_SIZE = 1000  # fixed as requested

def _prepare_vectors(filename: str, chunks: List[dict]):
    vectors = []
    for c in chunks:
        vec_id = f"{filename}::chunk::{c['chunk_id']}"
        meta = {
            "source": filename,
            "start": c.get("start_char", 0),
            "end": c.get("end_char", 0),
            "text_excerpt": c['text'][:500],
        }
        vectors.append((vec_id, c['embedding'], meta))
    return vectors

def process_file_background(path: str, filename: str, source_name: str, max_tokens: int = 500, overlap: int = 100):
    try:
        text = extract_text_from_path(path, filename)
        chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
        # embed in sub-batches using Gemini
        texts = [c['text'] for c in chunks]
        embeddings = embed_texts_gemini(texts, for_query=False)
        for i, c in enumerate(chunks):
            c['embedding'] = embeddings[i]
        vectors = _prepare_vectors(source_name, chunks)
        # upsert in batches of BATCH_SIZE
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i+BATCH_SIZE]
            upsert_vectors(batch)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
