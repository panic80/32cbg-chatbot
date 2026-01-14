import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.logging import get_logger
from app.core.config import settings
from app.core.prompts.retrieval import (
    get_query_classifier_prompt,
    get_query_expander_prompt,
    get_answer_synthesizer_prompt
)
from app.pipelines.definitions import RetrievalState, QueryType, RETRIEVAL_TIMEOUT, LLM_TIMEOUT

from app.services.cache import CacheService
from app.components.ensemble_retriever import WeightedEnsembleRetriever
from app.components.contextual_compressor import TravelContextualCompressor
from app.components.reranker import CrossEncoderReranker, CohereReranker, LLMReranker
from app.components.result_processor import ResultProcessor
from app.components.table_query_rewriter import TableQueryRewriter
from app.components.table_ranker import TableRanker
from app.services.llm_pool import LLMPool
from app.models.query import Provider

class RetrievalNodes:
    """Nodes for the enhanced retrieval workflow."""

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
        auxiliary_model: Optional[str] = None,
    ):
        self.retriever = retriever
        self.compressor = compressor
        self.reranker = reranker
        self.processor = processor
        self.table_rewriter = table_rewriter
        self.table_ranker = TableRanker()
        self.cache_service = cache_service
        self.llm_pool = llm_pool or LLMPool()
        self.fallback_keywords = fallback_keywords or ""
        self.auxiliary_model = auxiliary_model or settings.auxiliary_model
        self.logger = get_logger(__name__)
        
        # Prompts
        self.query_classifier = get_query_classifier_prompt()
        self.query_expander = get_query_expander_prompt()
        self.answer_synthesizer = get_answer_synthesizer_prompt()

    async def _invoke_json_prompt(
        self,
        prompt: ChatPromptTemplate,
        llm: Any,
        timeout: float = LLM_TIMEOUT,
        **inputs: Any
    ) -> Dict[str, Any]:
        """Execute a prompt expecting JSON output using the retryable LLM."""
        parser = JsonOutputParser()
        messages = prompt.format_messages(**inputs)
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"LLM call timed out after {timeout}s")
            raise
        content = self._extract_text_content(response)
        return parser.parse(content)

    @staticmethod
    def _extract_text_content(response: Any) -> str:
        """Normalize LangChain/LLM responses into a plain string."""
        if response is None:
            return ""

        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text_value = block.get("text") or block.get("content")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                    elif text_value is not None:
                        parts.append(json.dumps(text_value))
                else:
                    text_attr = getattr(block, "text", None)
                    if isinstance(text_attr, str):
                        parts.append(text_attr)
                    else:
                        parts.append(str(block))
            return "".join(parts)

        text_attr = getattr(content, "text", None)
        if isinstance(text_attr, str):
            return text_attr

        return str(content)

    @staticmethod
    def _serialize_document(doc: Document) -> Dict[str, Any]:
        """Convert a LangChain document into a JSON-serializable dict."""
        serialized_metadata: Dict[str, Any] = {}
        for key, value in (doc.metadata or {}).items():
            try:
                json.dumps(value)
                serialized_metadata[key] = value
            except (TypeError, ValueError):
                serialized_metadata[key] = str(value)

        return {
            "page_content": doc.page_content,
            "metadata": serialized_metadata
        }

    @classmethod
    def _serialize_documents(cls, docs: List[Document]) -> List[Dict[str, Any]]:
        """Serialize a list of documents for caching."""
        return [cls._serialize_document(doc) for doc in docs]

    @staticmethod
    def _deserialize_documents(data: List[Dict[str, Any]]) -> List[Document]:
        """Rehydrate cached document payloads back into LangChain documents."""
        documents: List[Document] = []
        for item in data or []:
            documents.append(
                Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {}) or {}
                )
            )
        return documents

    @staticmethod
    def _get_doc_id(doc: Document) -> str:
        """Generate a unique document ID for deduplication."""
        if doc.metadata.get("id"):
            return str(doc.metadata["id"])
        if doc.metadata.get("doc_id"):
            return str(doc.metadata["doc_id"])
        if doc.metadata.get("chunk_id"):
            return str(doc.metadata["chunk_id"])
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
        source = doc.metadata.get("source", "unknown")
        return f"{source}:{content_hash}"

    async def understand_query(self, state: RetrievalState) -> RetrievalState:
        """Understand and classify the query."""
        try:
            self.logger.debug(f"Starting query classification for: {state['query']}")
            
            async with self.llm_pool.acquire(Provider.OPENAI, self.auxiliary_model) as llm:
                invoke_params = {"query": state["query"]}
                try:
                    result = await self._invoke_json_prompt(
                        self.query_classifier,
                        llm,
                        **invoke_params
                    )
                except Exception as chain_error:
                    self.logger.error(f"Chain invocation error: {chain_error}", exc_info=True)
                    raise

                self.logger.debug(f"Classification result: {result}")
                
                query_type_str = result.get("type", "simple")
                if query_type_str in [qt.value for qt in QueryType]:
                    state["query_type"] = query_type_str
                else:
                    state["query_type"] = QueryType.SIMPLE.value
                    
                state["metadata"]["classification"] = result
                
                if result.get("needs_table_data", False):
                    state["metadata"]["needs_table_data"] = True
                    self.logger.info(f"Query classified as: {state['query_type']} (also needs table data)")
                else:
                    self.logger.info(f"Query classified as: {state['query_type']}")
            
        except Exception as e:
            self.logger.error(f"Query classification failed: {e}", exc_info=True)
            state["query_type"] = QueryType.SIMPLE.value
            
        return state

    async def expand_query(self, state: RetrievalState) -> RetrievalState:
        """Expand complex queries into sub-queries."""
        try:
            if state["query_type"] in [QueryType.MULTI_HOP.value, QueryType.COMPLEX.value]:
                self.logger.debug(f"Expanding query. Type: {state['query_type']}")
                
                async with self.llm_pool.acquire(Provider.OPENAI, self.auxiliary_model) as llm:
                    invoke_params = {
                        "query": state["query"],
                        "query_type": state["query_type"]
                    }

                    try:
                        result = await self._invoke_json_prompt(
                            self.query_expander,
                            llm,
                            **invoke_params
                        )
                    except Exception as chain_error:
                        self.logger.error(f"Expansion chain error: {chain_error}", exc_info=True)
                        raise

                    state["expanded_queries"] = result.get("sub_queries", [state["query"]])
                    self.logger.info(f"Expanded query into {len(state['expanded_queries'])} sub-queries")
            else:
                state["expanded_queries"] = [state["query"]]
                
        except Exception as e:
            self.logger.error(f"Query expansion failed: {e}", exc_info=True)
            state["expanded_queries"] = [state["query"]]
            
        return state

    async def retrieve_documents(self, state: RetrievalState) -> RetrievalState:
        """Retrieve documents for all queries."""
        try:
            all_docs = []
            value_patterns = []

            if state["query_type"] == QueryType.TABLE.value or state["metadata"].get("needs_table_data", False):
                rewritten_result = await self.table_rewriter.arewrite_query(state["query"])
                rewritten_query = rewritten_result.get("rewritten_query", state["query"])
                value_patterns = rewritten_result.get("value_patterns", [])

                if rewritten_query != state["query"]:
                    state["expanded_queries"].append(rewritten_query)

                if state["metadata"].get("needs_table_data", False) and state["query_type"] != QueryType.TABLE.value:
                    state["expanded_queries"].extend([
                        "meal allowances breakfast lunch dinner rates",
                        "kilometric rates per km",
                        "incidental allowances daily rates"
                    ])

                state["metadata"]["value_patterns"] = value_patterns
                state["metadata"]["table_keywords"] = rewritten_result.get("table_keywords", [])

            if not state["expanded_queries"]:
                state["expanded_queries"] = [state["query"]]

            async def fetch_docs_for_query(query: str) -> List[Document]:
                if self.cache_service:
                    cached = await self.cache_service.get(f"retrieval:{query}")
                    if cached:
                        return self._deserialize_documents(cached)

                try:
                    docs = await asyncio.wait_for(
                        self.retriever.aget_relevant_documents(query),
                        timeout=RETRIEVAL_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"Retrieval timeout for query: {query[:50]}...")
                    return []

                if self.cache_service and docs:
                    await self.cache_service.set(
                        f"retrieval:{query}",
                        self._serialize_documents(docs),
                        ttl=300
                    )
                return docs

            results = await asyncio.gather(
                *[fetch_docs_for_query(q) for q in state["expanded_queries"]],
                return_exceptions=True
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Query '{state['expanded_queries'][i][:50]}...' failed: {result}")
                elif isinstance(result, list):
                    all_docs.extend(result)
            
            seen = set()
            unique_docs = []
            for doc in all_docs:
                doc_id = self._get_doc_id(doc)
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            
            if state["query_type"] == QueryType.TABLE.value and unique_docs:
                unique_docs = self.table_ranker.filter_and_rerank(
                    unique_docs,
                    state["query"],
                    top_k=30,
                    query_type=state["query_type"],
                    value_patterns=value_patterns
                )
            
            state["retrieved_documents"] = unique_docs
            self.logger.info(f"Retrieved {len(unique_docs)} unique documents")
            
        except Exception as e:
            self.logger.error(f"Document retrieval failed: {e}")
            state["error"] = str(e)
            
        return state

    async def compress_documents(self, state: RetrievalState) -> RetrievalState:
        """Compress documents to relevant passages."""
        try:
            compressed = await self.compressor.retrieve(
                state["query"],
                k=len(state["retrieved_documents"])
            )
            state["compressed_documents"] = compressed
            self.logger.info(f"Compressed to {len(compressed)} documents")
        except Exception as e:
            self.logger.error(f"Document compression failed: {e}")
            state["compressed_documents"] = state["retrieved_documents"]
        return state

    async def rerank_documents(self, state: RetrievalState) -> RetrievalState:
        """Rerank documents for relevance."""
        try:
            if state["query_type"] == QueryType.TABLE.value or state["metadata"].get("needs_table_data", False):
                value_patterns = state["metadata"].get("value_patterns", [])
                table_ranked = self.table_ranker.filter_and_rerank(
                    state["compressed_documents"],
                    state["query"],
                    top_k=15,
                    query_type=state["query_type"],
                    value_patterns=value_patterns
                )
                reranked = await self.reranker.arerank(
                    state["query"],
                    table_ranked
                )
            else:
                reranked = await self.reranker.arerank(
                    state["query"],
                    state["compressed_documents"]
                )
            
            max_docs = {
                QueryType.SIMPLE.value: 5,
                QueryType.TABLE.value: 8,
                QueryType.COMPLEX.value: 10,
                QueryType.MULTI_HOP.value: 12,
                QueryType.COMPARISON.value: 10
            }
            
            limit = max_docs.get(state["query_type"], 5)
            state["reranked_documents"] = reranked[:limit]
            self.logger.info(f"Reranked to top {len(state['reranked_documents'])} documents")
            
        except Exception as e:
            self.logger.error(f"Document reranking failed: {e}")
            state["reranked_documents"] = state["compressed_documents"][:5]
        return state

    async def synthesize_answer(self, state: RetrievalState) -> RetrievalState:
        """Synthesize final answer from documents."""
        try:
            self.logger.debug(f"Starting answer synthesis. Query type: {state.get('query_type', 'unknown')}")
            
            context = "\n\n".join([
                doc.page_content
                for doc in state["reranked_documents"]
            ])
            
            model = self.auxiliary_model
            async with self.llm_pool.acquire(Provider.OPENAI, model) as llm:
                try:
                    messages = self.answer_synthesizer.format_messages(
                        context=context,
                        query=state["query"]
                    )
                    response = await asyncio.wait_for(
                        llm.ainvoke(messages),
                        timeout=LLM_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    self.logger.error(f"Answer synthesis LLM call timed out after {LLM_TIMEOUT}s")
                    raise
                except Exception as chain_error:
                    self.logger.error(f"Synthesis chain error: {chain_error}", exc_info=True)
                    raise

                state["synthesized_answer"] = self._extract_text_content(response)
            
            state["sources"] = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", ""),
                    "page": doc.metadata.get("page", 0),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    "page_content": doc.page_content
                }
                for doc in state["reranked_documents"]
            ]
            self.logger.info("Answer synthesized successfully")
        except Exception as e:
            self.logger.error(f"Answer synthesis failed: {e}", exc_info=True)
            state["error"] = str(e)
        return state

    async def fallback_retrieval(self, state: RetrievalState) -> RetrievalState:
        """Fallback retrieval strategy for poor results."""
        try:
            self.logger.info("Attempting fallback retrieval")
            if not self.retriever:
                self.logger.error("No retriever available for fallback")
                state["retrieved_documents"] = []
                return state

            broader_query = f"{state['query']} {self.fallback_keywords}".strip()
            try:
                docs = await asyncio.wait_for(
                    self.retriever.aget_relevant_documents(broader_query),
                    timeout=RETRIEVAL_TIMEOUT
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Fallback retrieval timed out for query: {broader_query[:50]}...")
                docs = []
            
            all_docs = state["retrieved_documents"] + docs
            seen = set()
            unique_docs = []
            for doc in all_docs:
                doc_id = self._get_doc_id(doc)
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            
            state["retrieved_documents"] = unique_docs
            self.logger.info(f"Fallback retrieval added {len(docs)} documents")
        except Exception as e:
            self.logger.error(f"Fallback retrieval failed: {e}")
        return state

    async def handle_error(self, state: RetrievalState) -> RetrievalState:
        """Handle errors gracefully."""
        self.logger.error(f"Workflow error: {state['error']}")
        
        if not state["synthesized_answer"]:
            state["synthesized_answer"] = (
                "I apologize, but I encountered an error while processing your query. "
                "Please try rephrasing your question or contact support if the issue persists."
            )
        return state
