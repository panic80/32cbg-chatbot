from typing import Any, Dict, List, Optional, TypedDict
from enum import Enum
from langchain_core.documents import Document

# Timeout constants
RETRIEVAL_TIMEOUT = 30.0  # seconds for vector search operations
LLM_TIMEOUT = 60.0  # seconds for LLM calls

class QueryType(str, Enum):
    """Types of queries for routing."""
    SIMPLE = "simple"
    TABLE = "table"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"

class RetrievalState(TypedDict):
    """State for the retrieval workflow."""
    query: str
    query_type: Optional[str]  # Store as string value, not enum
    expanded_queries: List[str]
    retrieved_documents: List[Document]
    compressed_documents: List[Document]
    reranked_documents: List[Document]
    synthesized_answer: Optional[str]
    sources: List[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]
    error: Optional[str]
    metadata: Dict[str, Any]
