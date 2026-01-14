"""Retrieval strategies for the unified retrieval framework."""

from typing import List, Dict, Any, Optional, Set
import asyncio
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from app.unified_retrieval.strategies.base import BaseStrategy, RetrievalContext, StrategyType
from app.components.bm25_retriever import TravelBM25Retriever as BM25RetrieverComponent
from app.core.config import settings
from app.core.dependencies import get_vectorstore, get_embeddings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorRetrievalStrategy(BaseStrategy):
    """
    Semantic vector search strategy.
    
    This strategy performs semantic similarity search using embeddings
    and vector stores.
    """
    
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        embeddings: Optional[Embeddings] = None,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the vector retrieval strategy.
        
        Args:
            vectorstore: Vector store to search
            embeddings: Embeddings model
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')
            search_kwargs: Additional search parameters
        """
        super().__init__(
            strategy_type=StrategyType.RETRIEVAL,
            name="vector_retrieval_strategy",
            **kwargs
        )
        self.vectorstore = vectorstore
        self.embeddings = embeddings or get_embeddings()
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}
    
    async def _get_vectorstore(self) -> VectorStore:
        """Get or initialize vector store."""
        if not self.vectorstore:
            self.vectorstore = await get_vectorstore()
        return self.vectorstore
    
    async def _search_vectors(self, query: str, k: int) -> List[Document]:
        """Perform vector search."""
        vectorstore = await self._get_vectorstore()
        
        # Merge search kwargs
        search_params = {
            "k": k,
            **self.search_kwargs,
            **self.config.get("search_kwargs", {})
        }
        
        # Remove run_manager from search params to avoid duplicate parameter error
        search_params.pop("run_manager", None)
        
        # Perform search based on type
        if self.search_type == "similarity":
            docs = await vectorstore.asimilarity_search(query, **search_params)
        elif self.search_type == "mmr":
            # Maximum Marginal Relevance search
            search_params["fetch_k"] = search_params.get("fetch_k", k * 3)
            search_params["lambda_mult"] = search_params.get("lambda_mult", 0.5)
            docs = await vectorstore.amax_marginal_relevance_search(query, **search_params)
        elif self.search_type == "similarity_score_threshold":
            # Search with score threshold
            threshold = search_params.pop("score_threshold", 0.7)
            docs_with_scores = await vectorstore.asimilarity_search_with_score(query, **search_params)
            docs = [doc for doc, score in docs_with_scores if score >= threshold]
        else:
            # Default to similarity search
            docs = await vectorstore.asimilarity_search(query, **search_params)
        
        # Add similarity scores to metadata if available
        if self.search_type == "similarity_score_threshold":
            for doc, score in docs_with_scores:
                if doc in docs:
                    doc.metadata["similarity_score"] = score
        
        return docs
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the vector retrieval."""
        # Get queries to search
        queries = context.metadata.get("generated_queries", [context.get_query()])
        k = context.top_k
        
        # Perform searches for all queries
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                docs = await self._search_vectors(query, k)
                
                # Deduplicate
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])  # Hash first 200 chars
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        doc.metadata["retrieved_by"] = "vector"
                        doc.metadata["retrieval_query"] = query
                        all_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error in vector search for query '{query}': {e}")
                context.add_error(self.component_name, e)
        
        # Update context
        context.documents.extend(all_docs)
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "docs_retrieved": len(all_docs),
                "queries_used": len(queries),
                "search_type": self.search_type,
                "unique_docs": len(seen_content)
            }
        )
        
        self._log_event(
            "vector_retrieval_complete",
            {
                "total_docs": len(all_docs),
                "search_type": self.search_type
            }
        )
        
        return context


class BM25RetrievalStrategy(BaseStrategy):
    """
    Keyword-based BM25 search strategy.
    
    This strategy performs traditional keyword-based retrieval using
    BM25 scoring algorithm.
    """
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        **kwargs
    ):
        """
        Initialize the BM25 retrieval strategy.
        
        Args:
            documents: Corpus of documents to search
            k1: BM25 k1 parameter
            b: BM25 b parameter
            epsilon: BM25 epsilon parameter
        """
        super().__init__(
            strategy_type=StrategyType.RETRIEVAL,
            name="bm25_retrieval_strategy",
            **kwargs
        )
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._retriever = None
    
    async def _get_retriever(self) -> BM25RetrieverComponent:
        """Get or initialize BM25 retriever."""
        if not self._retriever:
            if not settings.enable_bm25:
                raise RuntimeError("BM25 disabled by configuration")

            if not self.documents and not settings.bm25_require_index:
                # Load documents from vector store (expensive for large corpora)
                vectorstore = await get_vectorstore()
                self.documents = await vectorstore.asimilarity_search("", k=10000)

            if not self.documents and settings.bm25_require_index:
                self._retriever = BM25RetrieverComponent(
                    k=self.config.get("default_k", 10),
                    k1=self.k1,
                    b=self.b,
                    epsilon=self.epsilon
                )
            else:
                self._retriever = BM25RetrieverComponent(
                    documents=self.documents,
                    k=self.config.get("default_k", 10),
                    k1=self.k1,
                    b=self.b,
                    epsilon=self.epsilon
                )
        
        return self._retriever
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the BM25 retrieval."""
        if not settings.enable_bm25:
            logger.info("BM25 disabled by configuration; skipping BM25 retrieval")
            return context
        # Get queries to search
        queries = context.metadata.get("generated_queries", [context.get_query()])
        k = context.top_k
        
        # Get retriever
        retriever = await self._get_retriever()
        retriever.k = k  # Update k value
        
        # Perform searches
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                # Use the retriever's search method
                docs = await retriever.aget_relevant_documents(query)
                
                # Deduplicate and mark source
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        doc.metadata["retrieved_by"] = "bm25"
                        doc.metadata["retrieval_query"] = query
                        # Add BM25 score if available
                        if hasattr(doc, "metadata") and "score" in doc.metadata:
                            doc.metadata["bm25_score"] = doc.metadata["score"]
                        all_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error in BM25 search for query '{query}': {e}")
                context.add_error(self.component_name, e)
        
        # Update context
        context.documents.extend(all_docs)
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "docs_retrieved": len(all_docs),
                "queries_used": len(queries),
                "unique_docs": len(seen_content),
                "bm25_params": {"k1": self.k1, "b": self.b}
            }
        )
        
        return context


class HybridRetrievalStrategy(BaseStrategy):
    """
    Combine vector and BM25 retrieval.
    
    This strategy combines semantic and keyword-based retrieval
    for improved recall and precision.
    """
    
    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        vectorstore: Optional[VectorStore] = None,
        documents: Optional[List[Document]] = None,
        **kwargs
    ):
        """
        Initialize the hybrid retrieval strategy.
        
        Args:
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 results
            vectorstore: Vector store for semantic search
            documents: Document corpus for BM25
        """
        super().__init__(
            strategy_type=StrategyType.RETRIEVAL,
            name="hybrid_retrieval_strategy",
            **kwargs
        )
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Initialize sub-strategies
        self.vector_strategy = VectorRetrievalStrategy(
            vectorstore=vectorstore,
            **kwargs
        )
        self.bm25_strategy = BM25RetrievalStrategy(
            documents=documents,
            **kwargs
        )
    
    def _merge_results(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document]
    ) -> List[Document]:
        """Merge and score results from both retrievers."""
        # Create document scores
        doc_scores = {}
        
        # Score vector results
        for i, doc in enumerate(vector_docs):
            content_key = doc.page_content[:200]
            # Use reverse rank scoring
            score = (len(vector_docs) - i) / len(vector_docs)
            doc_scores[content_key] = {
                "doc": doc,
                "vector_score": score * self.vector_weight,
                "bm25_score": 0
            }
        
        # Score BM25 results
        for i, doc in enumerate(bm25_docs):
            content_key = doc.page_content[:200]
            score = (len(bm25_docs) - i) / len(bm25_docs)
            
            if content_key in doc_scores:
                # Document found by both methods
                doc_scores[content_key]["bm25_score"] = score * self.bm25_weight
            else:
                # New document from BM25
                doc_scores[content_key] = {
                    "doc": doc,
                    "vector_score": 0,
                    "bm25_score": score * self.bm25_weight
                }
        
        # Calculate final scores and sort
        for content_key, info in doc_scores.items():
            total_score = info["vector_score"] + info["bm25_score"]
            info["total_score"] = total_score
            # Add retrieval info to metadata
            doc = info["doc"]
            doc.metadata["hybrid_score"] = total_score
            doc.metadata["vector_contribution"] = info["vector_score"]
            doc.metadata["bm25_contribution"] = info["bm25_score"]
        
        # Sort by total score
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )
        
        return [result["doc"] for result in sorted_results]
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the hybrid retrieval."""
        # Create sub-contexts for each strategy
        vector_context = RetrievalContext(
            original_query=context.original_query,
            enhanced_query=context.enhanced_query,
            documents=[],
            metadata=context.metadata.copy(),
            top_k=context.top_k,
            search_kwargs=context.search_kwargs.copy()
        )
        
        bm25_context = RetrievalContext(
            original_query=context.original_query,
            enhanced_query=context.enhanced_query,
            documents=[],
            metadata=context.metadata.copy(),
            top_k=context.top_k,
            search_kwargs=context.search_kwargs.copy()
        )
        
        # Execute both strategies in parallel
        vector_task = self.vector_strategy.execute(vector_context)
        bm25_task = self.bm25_strategy.execute(bm25_context)
        
        vector_result, bm25_result = await asyncio.gather(
            vector_task, bm25_task, return_exceptions=True
        )
        
        # Handle errors
        if isinstance(vector_result, Exception):
            context.add_error("vector_retrieval", vector_result)
            vector_docs = []
        else:
            vector_docs = vector_result.documents
        
        if isinstance(bm25_result, Exception):
            context.add_error("bm25_retrieval", bm25_result)
            bm25_docs = []
        else:
            bm25_docs = bm25_result.documents
        
        # Merge results
        merged_docs = self._merge_results(vector_docs, bm25_docs)
        
        # Update context
        context.documents = merged_docs[:context.top_k]
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "vector_docs": len(vector_docs),
                "bm25_docs": len(bm25_docs),
                "merged_docs": len(merged_docs),
                "final_docs": len(context.documents),
                "weights": {
                    "vector": self.vector_weight,
                    "bm25": self.bm25_weight
                }
            }
        )
        
        return context


class ParentDocumentStrategy(BaseStrategy):
    """
    Return full documents for chunk matches.
    
    This strategy retrieves parent documents when chunks are matched,
    providing more complete context.
    """
    
    def __init__(
        self,
        parent_store: Optional[Dict[str, Document]] = None,
        chunk_overlap: int = 200,
        return_parents: bool = True,
        **kwargs
    ):
        """
        Initialize the parent document strategy.
        
        Args:
            parent_store: Mapping of chunk IDs to parent documents
            chunk_overlap: Expected overlap between chunks
            return_parents: Whether to return full parents or just expand chunks
        """
        super().__init__(
            strategy_type=StrategyType.RETRIEVAL,
            name="parent_document_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.parent_store = parent_store or {}
        self.chunk_overlap = chunk_overlap
        self.return_parents = return_parents
    
    def _get_parent_id(self, doc: Document) -> Optional[str]:
        """Extract parent document ID from chunk metadata."""
        # Try different metadata fields
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            parent_id = doc.metadata.get("source")
        if not parent_id:
            # Try to extract from chunk ID
            chunk_id = doc.metadata.get("id", "")
            if "_chunk_" in chunk_id:
                parent_id = chunk_id.split("_chunk_")[0]
        
        return parent_id
    
    def _expand_chunk_context(self, doc: Document) -> Document:
        """Expand a chunk with surrounding context."""
        # This is a simplified version - in practice, you'd need access
        # to the full document text or neighboring chunks
        expanded_doc = Document(
            page_content=doc.page_content,
            metadata=doc.metadata.copy()
        )
        expanded_doc.metadata["expanded"] = True
        return expanded_doc
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the parent document retrieval."""
        documents = context.documents
        
        if not documents:
            return context
        
        # Process documents to get parents
        processed_docs = []
        parent_ids_seen = set()
        
        for doc in documents:
            parent_id = self._get_parent_id(doc)
            
            if parent_id and self.return_parents:
                # Return full parent document
                if parent_id not in parent_ids_seen:
                    parent_ids_seen.add(parent_id)
                    
                    if parent_id in self.parent_store:
                        # Use stored parent
                        parent_doc = self.parent_store[parent_id]
                    else:
                        # Create parent from chunk (simplified)
                        parent_doc = Document(
                            page_content=doc.page_content,
                            metadata={
                                **doc.metadata,
                                "is_parent": True,
                                "child_chunks": [doc.metadata.get("id", "")]
                            }
                        )
                    
                    processed_docs.append(parent_doc)
            else:
                # Return expanded chunk or original
                if self.chunk_overlap > 0:
                    expanded_doc = self._expand_chunk_context(doc)
                    processed_docs.append(expanded_doc)
                else:
                    processed_docs.append(doc)
        
        # Update context
        context.documents = processed_docs
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "original_docs": len(documents),
                "parent_docs": len(parent_ids_seen),
                "final_docs": len(processed_docs),
                "return_parents": self.return_parents
            }
        )
        
        return context
