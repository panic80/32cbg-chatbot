"""
Reranking components for improving retrieval results.

Provides multiple reranking strategies including cross-encoder models,
Cohere Rerank API, and LLM-based reranking.
"""

import asyncio
from typing import List, Optional, Dict, Any, Union
import logging
from functools import lru_cache

from cachetools import LRUCache
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel

from app.components.base import BaseComponent
from app.utils.retry import with_retry_async
from app.core.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.info("sentence-transformers not available for cross-encoder reranking")

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.info("cohere not available for Cohere reranking")


class RerankerType:
    """Available reranker types."""
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    LLM = "llm"


class RerankingRetriever(BaseRetriever):
    """Retriever that reranks results from a base retriever."""
    
    base_retriever: BaseRetriever = Field(description="Base retriever to get initial results")
    reranker: Any = Field(description="Reranker instance")
    top_k: int = Field(default=5, description="Number of documents to return after reranking")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents by reranking base retriever results."""
        # Get initial documents
        docs = self.base_retriever.get_relevant_documents(
            query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        if not docs:
            return []
        
        # Rerank documents
        reranked_docs = self.reranker.rerank(query, docs, self.top_k)
        
        return reranked_docs
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async get relevant documents."""
        # Get initial documents
        docs = await self.base_retriever.aget_relevant_documents(
            query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        if not docs:
            return []
        
        # Rerank documents
        reranked_docs = await self.reranker.arerank(query, docs, self.top_k)
        
        return reranked_docs


class CrossEncoderReranker(BaseComponent):
    """Reranker using cross-encoder models."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on (cpu/cuda)
            max_length: Maximum sequence length
        """
        super().__init__(component_type="reranker", component_name="cross_encoder")
        
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.model_name = model_name
        self._cache = LRUCache(maxsize=10000)
    
    def _get_cache_key(self, query: str, doc: Document) -> str:
        """Generate cache key for query-document pair."""
        return f"{query}::{doc.page_content[:100]}"
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        # Prepare pairs for scoring
        pairs = []
        for doc in documents:
            cache_key = self._get_cache_key(query, doc)
            if cache_key in self._cache:
                continue
            pairs.append([query, doc.page_content])
        
        # Score new pairs
        if pairs:
            scores = self.model.predict(pairs)
            
            # Cache scores
            idx = 0
            for doc in documents:
                cache_key = self._get_cache_key(query, doc)
                if cache_key not in self._cache:
                    self._cache[cache_key] = scores[idx]
                    idx += 1
        
        # Get all scores
        doc_scores = []
        for doc in documents:
            cache_key = self._get_cache_key(query, doc)
            score = self._cache.get(cache_key, 0.0)
            doc_scores.append((doc, score))
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        if top_k:
            doc_scores = doc_scores[:top_k]
        
        reranked_docs = [doc for doc, _ in doc_scores]
        
        self._log_event("rerank", {
            "query": query,
            "input_count": len(documents),
            "output_count": len(reranked_docs),
            "model": self.model_name
        })
        
        return reranked_docs
    
    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Async rerank - runs sync version in thread."""
        return await asyncio.to_thread(self.rerank, query, documents, top_k)


class CohereReranker(BaseComponent):
    """Reranker using Cohere Rerank API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v2.0",
        top_n: int = 5
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Rerank model to use
            top_n: Default number of results to return
        """
        super().__init__(component_type="reranker", component_name="cohere")
        
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere is required for Cohere reranking. "
                "Install with: pip install cohere"
            )
        
        self.client = cohere.Client(api_key)
        self.model = model
        self.top_n = top_n
    
    @with_retry_async
    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Rerank documents using Cohere API."""
        if not documents:
            return []
        
        top_k = top_k or self.top_n
        
        # Prepare documents for Cohere
        doc_texts = [doc.page_content for doc in documents]
        
        # Call Cohere rerank
        response = await asyncio.to_thread(
            self.client.rerank,
            query=query,
            documents=doc_texts,
            top_n=min(top_k, len(documents)),
            model=self.model
        )
        
        # Reorder documents based on Cohere results
        reranked_docs = []
        for result in response.results:
            reranked_docs.append(documents[result.index])
        
        self._log_event("rerank", {
            "query": query,
            "input_count": len(documents),
            "output_count": len(reranked_docs),
            "model": self.model
        })
        
        return reranked_docs
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Sync version of rerank."""
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.arerank(query, documents, top_k))


# Reranking prompt for LLM-based reranking
RERANK_PROMPT = PromptTemplate(
    input_variables=["query", "documents"],
    template="""You are a relevance judge for a search system. Given a query and a list of documents, 
rank them by relevance to the query. Consider both direct keyword matches and semantic relevance.

Query: {query}

Documents:
{documents}

Rank the documents from most to least relevant. Return only the document numbers in order, 
separated by commas. For example: "3,1,5,2,4"

Your ranking:"""
)


class LLMReranker(BaseComponent):
    """Reranker using LLM for relevance scoring."""
    
    def __init__(
        self,
        llm: BaseLLM,
        batch_size: int = 5
    ):
        """
        Initialize LLM reranker.
        
        Args:
            llm: Language model for reranking
            batch_size: Number of documents to rerank at once
        """
        super().__init__(component_type="reranker", component_name="llm")
        
        self.llm = llm
        self.batch_size = batch_size
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for the prompt."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            # Truncate long documents
            content = doc.page_content[:500]
            if len(doc.page_content) > 500:
                content += "..."
            formatted.append(f"{i}. {content}")
        
        return "\n\n".join(formatted)
    
    def _parse_ranking(self, ranking_str: str, num_docs: int) -> List[int]:
        """Parse LLM ranking output."""
        try:
            # Extract numbers from the response
            import re
            numbers = re.findall(r'\d+', ranking_str)
            ranking = [int(n) - 1 for n in numbers if 0 < int(n) <= num_docs]
            
            # Ensure all documents are included
            seen = set(ranking)
            for i in range(num_docs):
                if i not in seen:
                    ranking.append(i)
            
            return ranking[:num_docs]
        except Exception as e:
            logger.warning(f"Failed to parse ranking: {e}")
            # Return original order
            return list(range(num_docs))
    
    @with_retry_async
    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Rerank documents using LLM."""
        if not documents:
            return []
        
        # Process in batches
        all_rankings = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            # Format prompt
            prompt = RERANK_PROMPT.format(
                query=query,
                documents=self._format_documents(batch)
            )
            
            # Get LLM ranking
            response = await self.llm.agenerate([prompt])
            ranking_str = response.generations[0][0].text
            
            # Parse ranking
            ranking = self._parse_ranking(ranking_str, len(batch))
            
            # Add to results with global indices
            for local_idx, global_idx in enumerate(ranking):
                all_rankings.append((i + global_idx, local_idx))
        
        # Sort by ranking score
        all_rankings.sort(key=lambda x: x[1])
        
        # Get reranked documents
        reranked_docs = [documents[idx] for idx, _ in all_rankings]
        
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        self._log_event("rerank", {
            "query": query,
            "input_count": len(documents),
            "output_count": len(reranked_docs),
            "method": "llm"
        })
        
        return reranked_docs
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Sync version of rerank."""
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.arerank(query, documents, top_k))


class RerankerFactory:
    """Factory for creating rerankers."""
    
    @staticmethod
    def create_reranker(
        reranker_type: str,
        **kwargs
    ) -> Union[CrossEncoderReranker, CohereReranker, LLMReranker]:
        """
        Create a reranker of the specified type.
        
        Args:
            reranker_type: Type of reranker to create
            **kwargs: Arguments for the specific reranker
            
        Returns:
            Reranker instance
        """
        if reranker_type == RerankerType.CROSS_ENCODER:
            return CrossEncoderReranker(**kwargs)
        elif reranker_type == RerankerType.COHERE:
            return CohereReranker(**kwargs)
        elif reranker_type == RerankerType.LLM:
            return LLMReranker(**kwargs)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
    
    @staticmethod
    def create_reranking_retriever(
        base_retriever: BaseRetriever,
        reranker_type: str,
        top_k: int = 5,
        **reranker_kwargs
    ) -> RerankingRetriever:
        """
        Create a retriever with reranking.
        
        Args:
            base_retriever: Base retriever to wrap
            reranker_type: Type of reranker to use
            top_k: Number of documents to return
            **reranker_kwargs: Arguments for the reranker
            
        Returns:
            Retriever with reranking
        """
        reranker = RerankerFactory.create_reranker(reranker_type, **reranker_kwargs)
        
        return RerankingRetriever(
            base_retriever=base_retriever,
            reranker=reranker,
            top_k=top_k
        )