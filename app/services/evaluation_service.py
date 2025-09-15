import asyncio
import time
import json
import random
import numpy as np
from typing import List, Dict, Any
from app.embeddings import embed_texts_gemini, embedding_model
from app.pinecone_client import query_vectors
from app.services.query_service import handle_query_stream
from sentence_transformers import util
import logging

logger = logging.getLogger(__name__)


async def evaluate_rag_batch(questions: List[str], top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate RAG performance on a batch of questions using unsupervised metrics.
    """
    logger.info(f"Starting batch evaluation for {len(questions)} questions")
    
    detailed_results = []
    all_metrics = []
    
    for i, question in enumerate(questions):
        logger.info(f"Evaluating question {i+1}/{len(questions)}: {question[:50]}...")
        
        question_result = await evaluate_single_question(question, top_k)
        detailed_results.append(question_result)
        all_metrics.append(question_result["metrics"])
    
    # Calculate summary statistics
    summary = calculate_summary_metrics(all_metrics)
    
    # Generate recommendations
    recommendations = generate_recommendations(all_metrics, summary)
    
    return {
        "summary": summary,
        "detailed_results": detailed_results,
        "recommendations": recommendations
    }


async def evaluate_single_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate a single question through the RAG pipeline.
    """
    start_time = time.time()
    
    # Step 1: Retrieval
    retrieval_start = time.time()
    question_embedding = embed_texts_gemini([question])[0]
    retrieved_results = query_vectors(question_embedding, top_k=top_k)
    retrieval_time = time.time() - retrieval_start
    
    # Step 2: Generation
    generation_start = time.time()
    answer_parts = []
    
    try:
        async for token in handle_query_stream(question, top_k=top_k):
            if token.startswith("data: "):
                data = json.loads(token[6:])
                if data.get("type") == "token":
                    answer_parts.append(data.get("token", ""))
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        answer_parts = ["Error generating response"]
    
    generated_answer = "".join(answer_parts)
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start_time
    
    # Step 3: Calculate metrics
    retrieved_docs = [
        {
            "id": match["id"],
            "text": match["metadata"].get("text", ""),
            "score": match["score"],
            "metadata": match["metadata"]
        }
        for match in retrieved_results.get("matches", [])
    ]
    
    metrics = calculate_unsupervised_metrics(
        question=question,
        retrieved_docs=retrieved_docs,
        generated_answer=generated_answer,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time
    )
    
    return {
        "question": question,
        "retrieved_docs": len(retrieved_docs),
        "generated_answer": generated_answer,
        "metrics": metrics,
        "timing": {
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    }


def calculate_unsupervised_metrics(
    question: str,
    retrieved_docs: List[Dict[str, Any]],
    generated_answer: str,
    retrieval_time: float,
    generation_time: float,
    total_time: float
) -> Dict[str, float]:
    """
    Calculate unsupervised quality metrics.
    """
    metrics = {}
    
    # === RETRIEVAL METRICS ===
    if retrieved_docs:
        scores = [doc["score"] for doc in retrieved_docs]
        metrics["avg_retrieval_score"] = float(np.mean(scores))
        metrics["score_variance"] = float(np.var(scores))
        metrics["max_retrieval_score"] = float(max(scores))
        metrics["min_retrieval_score"] = float(min(scores))
        
        # Retrieval confidence (gap between top and second result)
        if len(scores) >= 2:
            metrics["retrieval_confidence"] = float(scores[0] - scores[1])
        else:
            metrics["retrieval_confidence"] = float(scores[0]) if scores else 0.0
        
        # Coverage diversity (how different the retrieved docs are)
        metrics["coverage_diversity"] = calculate_coverage_diversity(retrieved_docs)
    else:
        metrics.update({
            "avg_retrieval_score": 0.0,
            "score_variance": 0.0,
            "max_retrieval_score": 0.0,
            "min_retrieval_score": 0.0,
            "retrieval_confidence": 0.0,
            "coverage_diversity": 0.0
        })
    
    # === GENERATION METRICS ===
    # Answer-Question relevance using embeddings
    metrics["answer_relevance"] = calculate_answer_relevance(question, generated_answer)
    
    # Context faithfulness (how grounded the answer is)
    metrics["context_faithfulness"] = calculate_context_faithfulness(
        generated_answer, retrieved_docs
    )
    
    # Answer completeness (based on length and detail)
    metrics["answer_completeness"] = calculate_answer_completeness(generated_answer)
    
    # Context utilization (how much context was used)
    metrics["context_utilization"] = calculate_context_utilization(
        generated_answer, retrieved_docs
    )
    
    # === PERFORMANCE METRICS ===
    metrics["retrieval_time"] = retrieval_time
    metrics["generation_time"] = generation_time
    metrics["total_time"] = total_time
    metrics["retrieval_efficiency"] = len(retrieved_docs) / retrieval_time if retrieval_time > 0 else 0
    
    # === OVERALL QUALITY ===
    metrics["composite_score"] = calculate_composite_score(metrics)
    
    return metrics


def calculate_coverage_diversity(retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Calculate how diverse the retrieved documents are using embeddings.
    """
    if len(retrieved_docs) < 2:
        return 0.0
    
    try:
        texts = [doc["text"] for doc in retrieved_docs if doc["text"]]
        if len(texts) < 2:
            return 0.0
        
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = util.cos_sim(
                    embeddings[i:i+1], 
                    embeddings[j:j+1]
                )[0][0].item()
                similarities.append(sim)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity = 1.0 - avg_similarity
        return max(0.0, diversity)
        
    except Exception as e:
        logger.warning(f"Error calculating diversity: {e}")
        return 0.0


def calculate_answer_relevance(question: str, answer: str) -> float:
    """
    Calculate semantic similarity between question and answer.
    """
    try:
        if not question.strip() or not answer.strip():
            return 0.0
        
        question_emb = embedding_model.encode([question], convert_to_tensor=False)
        answer_emb = embedding_model.encode([answer], convert_to_tensor=False)
        
        similarity = util.cos_sim(question_emb, answer_emb)[0][0].item()
        return max(0.0, similarity)
        
    except Exception as e:
        logger.warning(f"Error calculating answer relevance: {e}")
        return 0.0


def calculate_context_faithfulness(answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Calculate how well the answer is grounded in retrieved documents.
    """
    try:
        if not answer.strip() or not retrieved_docs:
            return 0.0
        
        # Combine all retrieved context
        context_text = " ".join([doc["text"] for doc in retrieved_docs if doc["text"]])
        if not context_text.strip():
            return 0.0
        
        answer_emb = embedding_model.encode([answer], convert_to_tensor=False)
        context_emb = embedding_model.encode([context_text], convert_to_tensor=False)
        
        similarity = util.cos_sim(answer_emb, context_emb)[0][0].item()
        return max(0.0, similarity)
        
    except Exception as e:
        logger.warning(f"Error calculating context faithfulness: {e}")
        return 0.0


def calculate_answer_completeness(answer: str) -> float:
    """
    Estimate answer completeness based on length and structure.
    """
    if not answer.strip():
        return 0.0
    
    word_count = len(answer.split())
    sentence_count = len([s for s in answer.split('.') if s.strip()])
    
    # Heuristic scoring based on word count and structure
    if word_count < 10:
        length_score = 0.3  # Too short
    elif word_count < 50:
        length_score = 0.7  # Reasonable
    elif word_count < 150:
        length_score = 1.0  # Good length
    else:
        length_score = 0.8  # Might be too long
    
    # Bonus for structured answers (multiple sentences)
    structure_score = min(1.0, sentence_count / 3)
    
    completeness = (length_score * 0.7) + (structure_score * 0.3)
    return min(1.0, completeness)


def calculate_context_utilization(answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Calculate how much of the retrieved context was utilized.
    """
    if not answer.strip() or not retrieved_docs:
        return 0.0
    
    answer_words = set(answer.lower().split())
    total_context_words = set()
    
    for doc in retrieved_docs:
        doc_words = set(doc["text"].lower().split())
        total_context_words.update(doc_words)
    
    if not total_context_words:
        return 0.0
    
    overlap = len(answer_words.intersection(total_context_words))
    utilization = overlap / len(total_context_words)
    
    return min(1.0, utilization)


def calculate_composite_score(metrics: Dict[str, float]) -> float:
    """
    Calculate overall quality score from individual metrics.
    """
    # Weight different metric categories
    weights = {
        "retrieval": 0.3,
        "generation": 0.5,
        "performance": 0.2
    }
    
    # Retrieval quality (higher is better)
    retrieval_score = (
        metrics.get("avg_retrieval_score", 0) * 0.4 +
        metrics.get("retrieval_confidence", 0) * 0.3 +
        metrics.get("coverage_diversity", 0) * 0.3
    )
    
    # Generation quality (higher is better)
    generation_score = (
        metrics.get("answer_relevance", 0) * 0.3 +
        metrics.get("context_faithfulness", 0) * 0.3 +
        metrics.get("answer_completeness", 0) * 0.2 +
        metrics.get("context_utilization", 0) * 0.2
    )
    
    # Performance score (lower time is better, normalize to 0-1)
    total_time = metrics.get("total_time", 5)  # Default 5 seconds
    performance_score = max(0, 1 - (total_time / 10))  # Good if under 10 seconds
    
    composite = (
        weights["retrieval"] * retrieval_score +
        weights["generation"] * generation_score +
        weights["performance"] * performance_score
    )
    
    return min(1.0, max(0.0, composite))


def calculate_summary_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Calculate summary statistics across all questions.
    """
    if not all_metrics:
        return {"error": "No metrics to summarize"}
    
    # Calculate averages for each metric
    metric_keys = all_metrics[0].keys()
    summary = {}
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        summary[f"avg_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))
        summary[f"min_{key}"] = float(np.min(values))
        summary[f"max_{key}"] = float(np.max(values))
    
    # Overall statistics
    composite_scores = [m["composite_score"] for m in all_metrics]
    summary["overall_quality"] = float(np.mean(composite_scores))
    summary["consistency"] = 1.0 - float(np.std(composite_scores))  # Lower std = more consistent
    summary["total_questions"] = len(all_metrics)
    
    return summary


def generate_recommendations(all_metrics: List[Dict[str, float]], summary: Dict[str, Any]) -> List[str]:
    """
    Generate actionable recommendations based on metrics.
    """
    recommendations = []
    
    # Check overall quality
    overall_quality = summary.get("overall_quality", 0)
    if overall_quality < 0.5:
        recommendations.append("🔴 Overall quality is low. Consider improving document quality or retrieval parameters.")
    elif overall_quality < 0.7:
        recommendations.append("🟡 Overall quality is moderate. Focus on specific weak areas identified below.")
    else:
        recommendations.append("🟢 Good overall quality! Minor optimizations may help.")
    
    # Check retrieval quality
    avg_retrieval_score = summary.get("avg_avg_retrieval_score", 0)
    if avg_retrieval_score < 0.3:
        recommendations.append("📚 Low retrieval scores suggest document chunking or embedding quality issues.")
    
    # Check diversity
    avg_diversity = summary.get("avg_coverage_diversity", 0)
    if avg_diversity < 0.3:
        recommendations.append("🔄 Low diversity in retrieved documents. Consider adjusting retrieval parameters.")
    
    # Check answer relevance
    avg_relevance = summary.get("avg_answer_relevance", 0)
    if avg_relevance < 0.5:
        recommendations.append("🎯 Answers show low relevance to questions. Review prompt engineering.")
    
    # Check faithfulness
    avg_faithfulness = summary.get("avg_context_faithfulness", 0)
    if avg_faithfulness < 0.5:
        recommendations.append("📖 Low context faithfulness. Model may be hallucinating or ignoring context.")
    
    # Check performance
    avg_time = summary.get("avg_total_time", 0)
    if avg_time > 5:
        recommendations.append("⚡ Response times are high. Consider optimizing retrieval or generation.")
    
    # Check consistency
    consistency = summary.get("consistency", 0)
    if consistency < 0.7:
        recommendations.append("📊 Inconsistent performance across questions. Review system stability.")
    
    if not recommendations:
        recommendations.append("✨ System performing well across all metrics!")
    
    return recommendations


async def generate_test_questions_from_documents(count: int = 5) -> List[str]:
    """
    Generate test questions from uploaded documents using simple extraction.
    """
    try:
        # Get a sample of documents from Pinecone
        from app.pinecone_client import _get_index
        
        index = _get_index()
        
        # First check if there are any documents in the index
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            raise Exception("No documents found in the index. Please upload documents first.")
        
        documents = []
        
        # Try multiple approaches to get documents
        # Approach 1: Use a simple semantic query with common words
        from app.embeddings import embed_texts_gemini
        
        # Generate embeddings for common query terms to find relevant docs
        query_terms = ["information", "data", "content", "document", "text"]
        for term in query_terms:
            try:
                query_embedding = embed_texts_gemini([term])
                if query_embedding:
                    results = index.query(
                        vector=query_embedding[0],
                        top_k=5,
                        include_metadata=True
                    )
                    
                    for match in results.get("matches", []):
                        text_excerpt = match["metadata"].get("text_excerpt", "")
                        if text_excerpt and len(text_excerpt) > 50:
                            documents.append(text_excerpt)
                            
                    if len(documents) >= 10:  # Got enough documents
                        break
            except Exception as e:
                logger.warning(f"Failed to query with term '{term}': {e}")
                continue
        
        # Approach 2: If still no documents, try with zero vector (fallback)
        if not documents:
            zero_vector = [0.0] * 384  # BGE dimension
            results = index.query(
                vector=zero_vector,
                top_k=10,
                include_metadata=True
            )
            
            for match in results.get("matches", []):
                text_excerpt = match["metadata"].get("text_excerpt", "")
                if text_excerpt and len(text_excerpt) > 50:
                    documents.append(text_excerpt)
        
        if not documents:
            raise Exception("No documents found to generate questions from")
        
        # Generate questions using simple heuristics
        questions = []
        question_templates = [
            "What is {}?",
            "How does {} work?",
            "Explain {} in detail.",
            "What are the key features of {}?",
            "Can you describe {}?"
        ]
        
        # Extract key terms from documents for question generation
        import re
        
        for doc in documents[:count]:
            # Simple keyword extraction
            words = re.findall(r'\b[A-Z][a-z]+\b', doc)  # Capitalized words
            if words:
                key_term = random.choice(words[:5])  # Pick from first 5 capitalized words
                template = random.choice(question_templates)
                question = template.format(key_term.lower())
                questions.append(question)
                
                if len(questions) >= count:
                    break
        
        # Fill remaining questions with generic ones if needed
        generic_questions = [
            "What is the main topic discussed in the documents?",
            "Can you provide a summary of the key concepts?",
            "What are the important details mentioned?",
            "How would you explain this topic to someone new?",
            "What are the main points covered?"
        ]
        
        while len(questions) < count and generic_questions:
            questions.append(generic_questions.pop(0))
        
        return questions[:count]
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return [
            "What is the main topic?",
            "Can you explain the key concepts?",
            "What are the important details?",
            "How does this work?",
            "What should I know about this?"
        ][:count]