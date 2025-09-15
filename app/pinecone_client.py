import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('PINECONE_API_KEY')
ENV = os.getenv('PINECONE_ENV', 'us-west-2')  # Default to us-west-2
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME','rag-index')
# Updated dimension for sentence-transformers/all-MiniLM-L6-v2 model (384 dimensions)
DIMENSION = int(os.getenv('PINECONE_DIM', 384))

if API_KEY is None:
    raise RuntimeError('PINECONE_API_KEY required')

# Initialize Pinecone client with new API
pc = Pinecone(api_key=API_KEY)

def _get_index():
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME, 
            dimension=DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=ENV
            )
        )
    return pc.Index(INDEX_NAME)

def upsert_vectors(vectors: list):
    idx = _get_index()
    to_upsert = []
    for vid, vec, meta in vectors:
        to_upsert.append({
            "id": vid,
            "values": vec,
            "metadata": meta
        })
    idx.upsert(vectors=to_upsert)

def query_vectors(vector, top_k: int = 5, filter: dict = None):
    idx = _get_index()
    resp = idx.query(vector=vector, top_k=top_k, include_metadata=True, filter=filter)
    try:
        return resp.to_dict()
    except Exception:
        return resp
