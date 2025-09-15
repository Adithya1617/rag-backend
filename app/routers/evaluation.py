from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import asyncio
from app.services.evaluation_service import (
    evaluate_rag_batch,
    generate_test_questions_from_documents
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvaluationRequest(BaseModel):
    questions: Optional[List[str]] = None  # If None, auto-generate
    top_k: int = 5
    auto_generate_count: int = 5  # Number of questions to auto-generate


class EvaluationResponse(BaseModel):
    summary: Dict[str, Any]
    detailed_results: List[Dict[str, Any]]
    recommendations: List[str]
    evaluation_time: float


@router.post("/batch", response_model=EvaluationResponse)
async def evaluate_batch(request: EvaluationRequest):
    """
    Evaluate RAG performance on a batch of questions (max 5).
    If no questions provided, auto-generate them from uploaded documents.
    """
    start_time = time.time()
    
    try:
        # Get questions (provided or auto-generated)
        if request.questions:
            questions = request.questions[:5]  # Limit to 5 questions
        else:
            questions = await generate_test_questions_from_documents(
                count=min(request.auto_generate_count, 5)
            )
            
        if not questions:
            raise HTTPException(
                status_code=400, 
                detail="No questions to evaluate. Either provide questions or ensure documents are uploaded."
            )
        
        # Run batch evaluation
        results = await evaluate_rag_batch(
            questions=questions,
            top_k=request.top_k
        )
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResponse(
            summary=results["summary"],
            detailed_results=results["detailed_results"],
            recommendations=results["recommendations"],
            evaluation_time=evaluation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/generate-questions")
async def generate_questions(count: int = 5):
    """
    Generate test questions from uploaded documents without running evaluation.
    """
    try:
        questions = await generate_test_questions_from_documents(count=min(count, 10))
        
        if not questions:
            raise HTTPException(
                status_code=400,
                detail="No documents found to generate questions from. Please upload documents first."
            )
        
        return {
            "generated_questions": questions,
            "count": len(questions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@router.get("/metrics-info")
async def get_metrics_info():
    """
    Get information about available evaluation metrics (unsupervised).
    """
    return {
        "retrieval_metrics": {
            "avg_retrieval_score": "Average similarity score of retrieved documents",
            "score_variance": "Variance in retrieval scores (higher = more diverse results)",
            "retrieval_confidence": "Confidence gap between top and lower-ranked documents",
            "coverage_diversity": "How diverse the retrieved documents are"
        },
        "generation_metrics": {
            "answer_relevance": "Semantic similarity between question and generated answer",
            "context_faithfulness": "How well the answer is grounded in retrieved documents",
            "answer_completeness": "Estimated completeness based on answer length and detail",
            "context_utilization": "Percentage of retrieved context actually used in answer"
        },
        "performance_metrics": {
            "response_time": "End-to-end response time in seconds",
            "retrieval_time": "Time taken for document retrieval",
            "generation_time": "Time taken for answer generation"
        },
        "overall_quality": {
            "composite_score": "Weighted combination of all metrics (0-1 scale)",
            "consistency_score": "How consistent the system performs across questions"
        }
    }


@router.get("/health")
async def evaluation_health():
    """
    Health check for evaluation service.
    """
    return {
        "status": "healthy",
        "evaluation_service": "ready",
        "supported_metrics": ["unsupervised", "lightweight"],
        "max_batch_size": 5
    }