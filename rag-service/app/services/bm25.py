"""Service for managing BM25 index."""

import os
import pickle
import asyncio
from typing import Optional

from app.core.vectorstore import VectorStoreManager
from app.components.bm25_retriever import TravelBM25Retriever
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Use /app/data if it exists (Docker volume), otherwise use local data directory
if os.path.exists("/app/data"):
    BASE_DATA_DIR = "/app/data"
else:
    # Fallback to project root data dir
    # .../app/services/bm25.py -> .../data
    BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

ARTIFACTS_DIR = os.path.join(BASE_DATA_DIR, "bm25")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "bm25_retriever.pkl")

async def rebuild_bm25_index():
    """Rebuild BM25 index from all documents and save to disk."""
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Initialize vector store
    vector_store_manager = VectorStoreManager()
    await vector_store_manager.initialize()
    
    try:
        # Get all documents from vector store
        logger.info("Fetching all documents from vector store...")
        documents = vector_store_manager.get_all_documents(refresh=True)
        
        if documents:
            logger.info(f"Retrieved {len(documents)} documents. Building BM25 index...")
            
            # Create retriever to build index
            # We initialize it with documents, which builds the index in memory
            retriever = TravelBM25Retriever(documents=documents)
            
            # Serialize the underlying LangChain retriever which holds the index and docs
            logger.info(f"Saving BM25 index to {INDEX_PATH}...")
            with open(INDEX_PATH, "wb") as f:
                pickle.dump(retriever.bm25_retriever, f)
                
            logger.info("BM25 index rebuilt and saved successfully.")
        else:
            logger.warning("No documents found in vector store. Skipping index build.")
            
    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index: {e}")
        raise
    finally:
        # Close vector store
        await vector_store_manager.close()
