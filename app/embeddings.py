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
# - 'sentence-transformers/all-MiniLM-L6-v2': Fast, lightweight (384 dim) - CURRENT
# - 'sentence-transformers/all-mpnet-base-v2': Better quality (768 dim) - RECOMMENDED FOR PRODUCTION
# - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': Multilingual support (384 dim)
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
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIM = 384

def _batch(lst, n):
    """Helper function to batch lists into chunks."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def embed_texts_gemini(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Embed texts using Hugging Face sentence transformers.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process in each batch (default: 32)
    
    Returns:
        List of embedding vectors (dimensions depend on the model)
    """
    logger.info(f"Embedding {len(texts)} texts using Hugging Face model...")
    
    try:
        # Process texts in batches for better memory management
        all_embeddings = []
        
        for i, batch in enumerate(_batch(texts, batch_size)):
            logger.info(f"Processing batch {i+1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            # Get embeddings for the batch
            batch_embeddings = embedding_model.encode(
                batch,
                convert_to_tensor=False,  # Return numpy arrays
                normalize_embeddings=True,  # Normalize for better similarity search
                show_progress_bar=False,
                batch_size=batch_size  # Explicit batch size for memory control
            )
            
            # Convert to list of lists
            for embedding in batch_embeddings:
                all_embeddings.append(embedding.tolist())
        
        logger.info(f"Successfully embedded {len(texts)} texts")
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error embedding texts with Hugging Face model: {e}")
        # Return zero vectors as fallback using the actual model dimension
        return [[0.0] * EMBEDDING_DIM for _ in texts]
