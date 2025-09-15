# RAG Chatbot Backend (Gemini + Pinecone)

## Setup
1. Create a Python 3.10 virtualenv
2. pip install -r backend/requirements.txt
3. Set environment variables (see backend/.env.example)
4. Run locally: uvicorn app.main:app --reload --port 8000

## Notes
- Uses Google Gemini for embeddings and text generation. Configure GEMINI_API_KEY.
- Pinecone upserts are batched at 1000 vectors per call.
