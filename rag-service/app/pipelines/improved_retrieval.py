"""
Improved retrieval pipeline using LangChain components.

This pipeline combines multiple retrieval strategies to handle various query types,
especially queries looking for specific values in documents.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLLM

from app.components.multi_query_retriever import MultiQueryRetriever
from app.components.ensemble_retriever import ContentBoostedEnsembleRetriever
from app.components.contextual_compressor import TravelContextualCompressor
from app.components.self_query_retriever import TravelSelfQueryRetriever
from app.components.parent_document_retriever import TravelParentDocumentRetriever
from app.components.table_query_rewriter import TableQueryRewriter
from app.components.class_a_retriever import ClassARetriever
from app.components.class_a_query_enhancer import ClassAQueryEnhancer
from app.components.restriction_aware_retriever import RestrictionAwareRetriever
from app.components.bm25_retriever import TravelBM25Retriever
from app.core.config import settings
from app.core.vectorstore import VectorStoreManager
from app.core.logging import get_logger

logger = get_logger(__name__)


def _resolve_bm25_index_path() -> str:
    """Resolve the default BM25 index path used by the retriever."""
    if os.path.exists("/app/data"):
        base_data_dir = "/app/data"
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        base_data_dir = os.path.join(project_root, "data")
    return os.path.join(base_data_dir, "bm25", "bm25_retriever.pkl")


class ImprovedRetrievalPipeline:
    """
    Improved retrieval pipeline that combines multiple strategies:
    1. Multi-query generation for better coverage
    2. Self-query parsing for structured queries
    3. Hybrid search (vector + BM25) with content boosting
    4. Smart chunking with neighbor retrieval
    5. Contextual compression for relevance filtering
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm: Optional[BaseLLM] = None,
        use_multi_query: bool = False,  # Disabled for performance (saves ~6s)
        use_compression: bool = True,
        use_smart_chunking: bool = True,
        use_self_query: bool = True
    ):
        """Initialize the improved retrieval pipeline."""
        self.vector_store = vector_store_manager
        self.llm = llm
        self.use_multi_query = use_multi_query
        self.use_compression = use_compression
        self.use_smart_chunking = use_smart_chunking
        self.use_self_query = use_self_query
        
        # BM25 state
        self._bm25_initialized = False
        self._bm25_retriever = None
        
        # Create retrievers
        self.retrievers = self._create_retrievers()
    
    def _create_retrievers(self) -> Dict[str, BaseRetriever]:
        """Create all the retrievers used in the pipeline."""
        retrievers = {}
        
        # 1. Base vector retriever
        vector_retriever = self.vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )
        retrievers["vector"] = vector_retriever
        
        # 2. MMR (Maximal Marginal Relevance) retriever for diversity
        mmr_retriever = self.vector_store.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "lambda_mult": 0.5}
        )
        retrievers["mmr"] = mmr_retriever
        
        # 3. Smart chunk retriever (includes neighbors)
        if self.use_smart_chunking:
            # Create child retriever for finding chunks
            child_retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}
            )
            
            smart_retriever = TravelParentDocumentRetriever(
                child_retriever=child_retriever,
                k=5
            )
            retrievers["smart_chunk"] = smart_retriever
        
        # 5. Class A specific retriever
        class_a_retriever = ClassARetriever(
            base_retriever=vector_retriever,
            boost_factor=1.5
        )
        retrievers["class_a"] = class_a_retriever
        
        # 6. Restriction-aware retriever
        restriction_retriever = RestrictionAwareRetriever(
            base_retriever=vector_retriever,
            boost_factor=2.0
        )
        retrievers["restriction"] = restriction_retriever
        
        # 4. Self-query retriever for structured queries
        if self.use_self_query and self.llm:
            try:
                self_query_retriever = TravelSelfQueryRetriever(
                    vectorstore=self.vector_store.vector_store,
                    llm=self.llm,
                    document_contents="Canadian Forces travel instructions and policies",
                    metadata_field_info=None,  # Will use default travel attributes
                    enable_limit=True,
                    search_kwargs={"k": 20}
                )
                retrievers["self_query"] = self_query_retriever
            except Exception as e:
                logger.warning(f"Failed to create self-query retriever: {e}")
        
        # Note: BM25 will be created lazily on first use
        # Ensemble and other retrievers will be created after BM25 is initialized
        
        return retrievers
    
    async def _get_all_documents(self) -> List[Document]:
        """Get all documents from vector store for BM25 index."""
        try:
            # Wrap the synchronous operation in asyncio.to_thread
            def _get_docs_sync():
                # This is implementation-specific
                # For Chroma, we can get all documents
                collection = self.vector_store.vector_store._collection
                results = collection.get(include=["documents", "metadatas"])
                
                if results and results['documents']:
                    documents = []
                    for i, content in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if results['metadatas'] else {}
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        documents.append(doc)
                    return documents
                return []
            
            # Run the sync operation in a thread
            documents = await asyncio.to_thread(_get_docs_sync)
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
        
        return []
    
    def _create_ensemble_retriever(self) -> ContentBoostedEnsembleRetriever:
        """Create the ensemble retriever with all base retrievers."""
        base_retrievers = []
        weights = []
        
        if "vector" in self.retrievers:
            base_retrievers.append(self.retrievers["vector"])
            weights.append(0.35)
        
        if "mmr" in self.retrievers:
            base_retrievers.append(self.retrievers["mmr"])
            weights.append(0.20)
        
        if "bm25" in self.retrievers:
            base_retrievers.append(self.retrievers["bm25"])
            weights.append(0.35)
        
        additional_weights = {
            "smart_chunk": 0.1,
            "class_a": 0.1,
            "restriction": 0.1,
        }

        for key, weight in additional_weights.items():
            retriever = self.retrievers.get(key)
            if retriever:
                base_retrievers.append(retriever)
                weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if not base_retrievers or total_weight == 0:
            logger.warning("No base retrievers available for ensemble; using vector retriever only")
            fallback = self.retrievers.get("vector")
            if not fallback:
                raise RuntimeError("Vector retriever missing; cannot build ensemble")
            base_retrievers = [fallback]
            weights = [1.0]
        else:
            weights = [w / total_weight for w in weights]
        
        # Create content-boosted ensemble
        ensemble_retriever = ContentBoostedEnsembleRetriever(
            retrievers=base_retrievers,
            weights=weights,
            k=25
        )
        
        return ensemble_retriever
    
    async def _ensure_bm25_initialized(self):
        """Ensure BM25 retriever is initialized (lazy loading)."""
        if self._bm25_initialized:
            return

        try:
            if not settings.enable_bm25:
                logger.info("BM25 disabled by configuration; skipping initialization")
            else:
                bm25_retriever = None
                index_path = _resolve_bm25_index_path()
                index_exists = os.path.exists(index_path)

                if index_exists:
                    try:
                        bm25_retriever = TravelBM25Retriever(k=25, index_path=index_path)
                        self._bm25_retriever = bm25_retriever
                        self.retrievers["bm25"] = bm25_retriever
                        logger.info("Loaded BM25 retriever from disk index")
                    except Exception as e:
                        logger.warning(f"Failed to load BM25 index from disk: {e}")

                if bm25_retriever is None:
                    if settings.bm25_require_index:
                        logger.warning(
                            "BM25 index missing and request-time build disabled; skipping BM25"
                        )
                    else:
                        all_docs = await self._get_all_documents()
                        if all_docs:
                            bm25_retriever = TravelBM25Retriever(documents=all_docs, k=25)
                            self._bm25_retriever = bm25_retriever
                            self.retrievers["bm25"] = bm25_retriever
                            logger.info(f"Created BM25 retriever with {len(all_docs)} documents")
                        else:
                            logger.warning("No documents available for BM25; skipping")

            self.retrievers["ensemble"] = self._create_ensemble_retriever()
            self._create_retriever_chain()
        except Exception as e:
            logger.error(f"Failed to initialize BM25 retriever: {e}")
        finally:
            self._bm25_initialized = True  # Prevent retry on every request
    
    def _create_retriever_chain(self):
        """Create the full retriever chain with all components."""
        # Start with ensemble as base
        base_retriever = self.retrievers.get("ensemble")
        if not base_retriever:
            logger.warning("No ensemble retriever available")
            return
        
        # 1. Add table-aware processing
        table_query_rewriter = TableQueryRewriter(llm=self.llm)
        table_aware_retriever = TableAwareRetriever(
            base_retriever=base_retriever,
            query_rewriter=table_query_rewriter
        )
        self.retrievers["table_aware"] = table_aware_retriever
        current_retriever = table_aware_retriever
        
        # 2. Add multi-query generation (if LLM available)
        if self.use_multi_query and self.llm:
            try:
                multi_query_retriever = TravelMultiQueryRetriever(
                    base_retriever=current_retriever,
                    llm=self.llm,
                    include_original=True
                )
                self.retrievers["multi_query"] = multi_query_retriever
                current_retriever = multi_query_retriever
            except Exception as e:
                logger.warning(f"Failed to create multi-query retriever: {e}")
        
        # 3. Add contextual compression (if embeddings available)
        if self.use_compression:
            try:
                compressor = TravelContextualCompressor(
                    llm=self.llm,
                    embeddings=self.vector_store.embeddings,
                    compression_mode="hybrid",  # Use hybrid mode for best results
                    similarity_threshold=0.5
                )
                
                compression_retriever = compressor.compress_documents(
                    base_retriever=current_retriever
                )
                self.retrievers["compressed"] = compression_retriever
                current_retriever = compression_retriever
            except Exception as e:
                logger.warning(f"Failed to create compression retriever: {e}")
        
        # Store the final retriever
        self.retrievers["final"] = current_retriever
    
    def _expand_table_query(self, query: str) -> str:
        """Expand generic table queries to be more specific."""
        query_lower = query.lower()
        
        # Generic expansion patterns for any type of rate/allowance query
        expansion_keywords = {
            # Pattern: (trigger_words, expansion_terms)
            ("rate", "allowance", "amount", "value", "price"): [
                "table", "rates", "allowance", "per day", "daily", 
                "Canada", "USA", "CAD", "$", "dollars"
            ],
            ("meal", "breakfast", "lunch", "dinner"): [
                "breakfast", "lunch", "dinner", "meal allowance",
                "Yukon", "Alaska", "NWT", "Nunavut"
            ],
            ("incidental", "incidentals"): [
                "incidental expense", "incidental allowance", "per day",
                "17.30", "13.00", "75%", "31st day", "CIL", "change in location",
                "last day", "final day", "return travel", "extended stay",
                "30 days same location", "75 percent", "reduction"
            ],
            ("kilometric", "mileage", "km"): [
                "per kilometer", "per km", "vehicle", "PMV"
            ],
            ("accommodation", "hotel", "lodging"): [
                "overnight", "private", "commercial"
            ]
        }
        
        # Check which patterns match and build expansion
        expansions = []
        
        for trigger_words, expansion_terms in expansion_keywords.items():
            if any(word in query_lower for word in trigger_words):
                # Add relevant expansion terms that aren't already in query
                for term in expansion_terms:
                    if term.lower() not in query_lower:
                        expansions.append(term)
        
        # If we found expansions, add them to the query
        if expansions:
            # Limit expansions to avoid too long queries
            expansions = expansions[:5]
            expanded = f"{query} {' '.join(expansions)}"
            logger.info(f"Expanded query from '{query}' to '{expanded}'")
            return expanded
        
        return query
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using the improved pipeline.
        
        Args:
            query: The search query
            k: Number of documents to return
            filters: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Enhance query with Class A context if applicable
            enhanced_query = ClassAQueryEnhancer.enhance_query(query)
            if enhanced_query != query:
                logger.info(f"Query enhanced with Class A context: {enhanced_query}")
                query = enhanced_query

            # Ensure BM25 is initialized if not already
            await self._ensure_bm25_initialized()
            
            # Check if this is a PMV/restriction query
            if any(term in query_lower for term in ["drive", "pmv", "vehicle", "car", "restriction", "limit", "km"]):
                # Use larger k for restriction queries
                if k < 25:
                    k = 25
                logger.info(f"Detected PMV/restriction query, increased k to {k}")

            # Check if query might benefit from self-query parsing
            if self.use_self_query and "self_query" in self.retrievers:
                # Simple heuristic: if query mentions dates, jurisdictions, or specific document types
                query_lower = query.lower()
                use_self_query = any(keyword in query_lower for keyword in [
                    "before", "after", "since", "until", "in 20",  # dates
                    "ontario", "quebec", "alberta", "yukon",  # jurisdictions
                    "policy", "directive", "guide", "form"  # document types
                ])
                
                if use_self_query:
                    try:
                        logger.info("Using self-query retriever for structured query")
                        retriever = self.retrievers["self_query"]
                        docs = await retriever.aget_relevant_documents(query)
                        
                        # If we got good results, use them
                        if docs:
                            results = []
                            for i, doc in enumerate(docs[:k]):
                                score = 1.0 - (i * 0.1)  # Decreasing score by rank
                                results.append((doc, score))
                            return results
                    except Exception as e:
                        logger.warning(f"Self-query retrieval failed, falling back: {e}")
            
            # Use the final retriever from the chain
            retriever = self.retrievers.get("final") or self.retrievers.get("compressed") or \
                       self.retrievers.get("multi_query") or self.retrievers.get("table_aware") or \
                       self.retrievers.get("ensemble") or self.retrievers.get("vector")
            
            if not retriever:
                logger.error("No retriever available")
                return []
            
            # Expand query if it's a generic table request
            expanded_query = self._expand_table_query(query)
            
            # Retrieve documents
            logger.info(f"Retrieving with query: {expanded_query}")
            docs = await retriever.aget_relevant_documents(expanded_query)
            
            # Convert to tuples with scores
            # Since most retrievers don't return scores, we'll assign based on rank
            results = []
            for i, doc in enumerate(docs[:k]):
                score = 1.0 - (i * 0.1)  # Decreasing score by rank
                results.append((doc, score))
            
            logger.info(f"Retrieved {len(results)} documents")
            
            # Log some details about top results
            if results:
                top_doc = results[0][0]
                logger.info(f"Top result source: {top_doc.metadata.get('source', 'Unknown')}")
                logger.info(f"Top result preview: {top_doc.page_content[:100]}...")
                
                # Check if we found specific values
                if "$" in top_doc.page_content:
                    logger.info("Top result contains dollar amounts")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            # Fallback to simple vector search
            try:
                results = await self.vector_store.search(query, k, filters)
                return results
            except Exception as e2:
                logger.error(f"Fallback retrieval also failed: {e2}")
                return []
    
    def retrieve_sync(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Synchronous version of retrieve."""
        # Run async version in event loop
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.retrieve(query, k, filters))
