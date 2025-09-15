import os
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the best free Hugging Face embedding model
# Options for different use cases:
# - 'BAAI/bge-small-en-v1.5': High-quality retrieval model (384 dim) - CURRENT
# - 'sentence-transformers/all-MiniLM-L6-v2': Fast, lightweight (384 dim) - Alternative
# - 'sentence-transformers/all-mpnet-base-v2': Better quality (768 dim) - Higher resource usage
# - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': Multilingual support (384 dim)

# Default model - BAAI BGE optimized for retrieval tasks
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
# - 'BAAI/bge-small-en-v1.5': State-of-the-art small model (384 dim) - BEST QUALITY/SPEED BALANCE
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'BAAI/bge-small-en-v1.5')

# Initialize the sentence transformer model
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Get the actual embedding dimension from the model
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded successfully. Embedding dimension: {EMBEDDING_DIM}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    logger.info("Falling back to all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    EMBEDDING_DIM = 384

def _batch(lst, n):
    """Helper function to batch lists into chunks."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def embed_texts_gemini(texts: List[str], for_query: bool = False) -> List[List[float]]:
    """
    Embed texts using Hugging Face sentence transformers with BGE formatting.
    
    Args:
        texts: List of texts to embed
        for_query: If True, format as queries; if False, format as passages
    """
    # Format texts for BAAI/bge models for optimal performance
    if for_query:
        # Prefix queries for better retrieval performance
        formatted_texts = [f"query: {text}" for text in texts]
    else:
        # Prefix passages/documents
        formatted_texts = [f"passage: {text}" for text in texts]
    
    try:
        embeddings = embedding_model.encode(formatted_texts, convert_to_tensor=False)
        return embeddings.tolist()
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise
