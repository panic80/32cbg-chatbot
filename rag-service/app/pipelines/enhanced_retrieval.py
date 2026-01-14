"""Enhanced retrieval pipeline using LangGraph for orchestrated workflows."""
import time
from typing import Any, Dict, List, Optional, Union

from app.core.logging import get_logger

from app.services.cache import CacheService
from app.components.ensemble_retriever import WeightedEnsembleRetriever
from app.components.contextual_compressor import TravelContextualCompressor
from app.components.reranker import CrossEncoderReranker, CohereReranker, LLMReranker
from app.components.result_processor import ResultProcessor
from app.components.table_query_rewriter import TableQueryRewriter
from app.services.llm_pool import LLMPool

from app.pipelines.definitions import RetrievalState, QueryType
from app.pipelines.nodes.retrieval_nodes import RetrievalNodes
from app.pipelines.builders.enhanced_graph import EnhancedGraphBuilder

class EnhancedRetrievalPipeline:
    """Advanced retrieval pipeline with LangGraph orchestration."""

    def __init__(
        self,
        retriever: WeightedEnsembleRetriever,
        compressor: TravelContextualCompressor,
        reranker: Union[CrossEncoderReranker, CohereReranker, LLMReranker],
        processor: ResultProcessor,
        table_rewriter: TableQueryRewriter,
        cache_service: Optional[CacheService] = None,
        llm_pool: Optional[LLMPool] = None,
        fallback_keywords: Optional[str] = None,
    ):
        self.logger = get_logger(__name__)
        
        # Initialize nodes with dependencies
        self.nodes = RetrievalNodes(
            retriever=retriever,
            compressor=compressor,
            reranker=reranker,
            processor=processor,
            table_rewriter=table_rewriter,
            cache_service=cache_service,
            llm_pool=llm_pool,
            fallback_keywords=fallback_keywords
        )
        
        # Build the workflow graph
        builder = EnhancedGraphBuilder(self.nodes)
        self.workflow = builder.build()

    async def retrieve(
        self, 
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Execute the enhanced retrieval workflow."""
        start_time = time.time()
        
        # Initialize state
        initial_state = RetrievalState(
            query=query,
            query_type=None,
            expanded_queries=[],
            retrieved_documents=[],
            compressed_documents=[],
            reranked_documents=[],
            synthesized_answer=None,
            sources=[],
            conversation_history=conversation_history or [],
            error=None,
            metadata={}
        )
        
        try:
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Process results
            result = {
                "answer": final_state["synthesized_answer"],
                "sources": final_state["sources"],
                "query_type": final_state["query_type"] if final_state["query_type"] else "unknown",
                "metadata": {
                    **final_state["metadata"],
                    "retrieval_time": time.time() - start_time,
                    "num_documents": len(final_state["reranked_documents"]),
                    "expanded_queries": final_state["expanded_queries"]
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced retrieval failed: {e}", exc_info=True)
            return {
                "answer": "I apologize, but I couldn't process your query. Please try again.",
                "sources": [],
                "error": str(e),
                "metadata": {
                    "retrieval_time": time.time() - start_time,
                    "error_type": type(e).__name__
                }
            }