from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.ingest import router as ingest_router
from app.routers.query import router as query_router
from app.routers.evaluation import router as evaluation_router

app = FastAPI(title="RAG Chatbot Backend (Gemini)")

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://rag-chatbot-orpin-phi.vercel.app/",  # Replace with your actual Vercel URL
    "https://*.vercel.app",  # Allow all Vercel preview deployments
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api/ingest")
app.include_router(query_router, prefix="/api/query")
app.include_router(evaluation_router)  # Evaluation router with its own prefix

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
