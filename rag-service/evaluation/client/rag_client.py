"""HTTP client for RAG API endpoints."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    documents: List[Dict[str, Any]]
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ChatResult:
    """Result from a chat query."""

    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ChunkInfo:
    """Information about a document chunk."""

    id: str
    content: str
    metadata: Dict[str, Any]


class RAGClient:
    """Async HTTP client for RAG service API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        admin_token: str = "",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "RAGClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._headers(),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.admin_token:
            headers["Authorization"] = f"Bearer {self.admin_token}"
        return headers

    async def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if self._client is None:
            await self.connect()

    async def search(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """Search for documents matching a query.

        Args:
            query: The search query
            k: Number of results to return
            use_hybrid: Whether to use hybrid search (vector + BM25)

        Returns:
            RetrievalResult with documents and latency
        """
        await self._ensure_connected()

        start_time = time.time()
        try:
            response = await self._client.post(
                "/api/v1/sources/search",
                json={
                    "query": query,
                    "k": k,
                    "use_hybrid_search": use_hybrid,
                },
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response - can be list or dict with results key
            if isinstance(data, list):
                raw_docs = data
            else:
                raw_docs = data.get("results", data.get("documents", []))

            # Flatten nested document structure from DocumentSearchResult
            documents = []
            for item in raw_docs:
                if isinstance(item, dict):
                    # Handle DocumentSearchResult format: {document: {...}, score: ...}
                    if "document" in item:
                        doc = item["document"].copy()
                        doc["score"] = item.get("score")
                        documents.append(doc)
                    else:
                        documents.append(item)

            return RetrievalResult(
                documents=documents,
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                documents=[],
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                documents=[],
                latency_ms=latency_ms,
                error=str(e),
            )

    async def chat(
        self,
        message: str,
        use_rag: bool = True,
        use_hybrid: bool = True,
        short_answer: bool = False,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Send a chat message and get a response.

        Uses the streaming endpoint and collects the full response.

        Args:
            message: The user message
            use_rag: Whether to use RAG for the response
            use_hybrid: Whether to use hybrid search
            short_answer: Whether to request a short answer
            retrieval_config: Optional per-request retrieval configuration overrides

        Returns:
            ChatResult with answer, sources, and latency
        """
        # Use streaming endpoint (sync endpoint is deprecated)
        return await self.chat_stream(message, use_rag, use_hybrid, retrieval_config)

    async def chat_stream(
        self,
        message: str,
        use_rag: bool = True,
        use_hybrid: bool = True,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Send a chat message and collect streamed response.

        Args:
            message: The user message
            use_rag: Whether to use RAG for the response
            use_hybrid: Whether to use hybrid search
            retrieval_config: Optional per-request retrieval configuration overrides

        Returns:
            ChatResult with complete answer, sources, and latency
        """
        await self._ensure_connected()

        start_time = time.time()
        answer_parts = []
        sources = []

        request_body = {
            "message": message,
            "use_rag": use_rag,
            "use_hybrid_search": use_hybrid,
        }
        # Pass retrieval_config if provided (even if empty dict for baseline)
        if retrieval_config is not None:
            request_body["retrieval_config"] = retrieval_config

        try:
            async with self._client.stream(
                "POST",
                "/api/v1/chat/stream",
                json=request_body,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        import json

                        data = json.loads(data_str)
                        event_type = data.get("type", "")

                        if event_type == "token":
                            answer_parts.append(data.get("content", ""))
                        elif event_type == "content":
                            # Alternative event name for content
                            answer_parts.append(data.get("content", ""))
                        elif event_type == "sources":
                            sources = data.get("sources", [])
                        elif event_type == "error":
                            return ChatResult(
                                answer="",
                                sources=[],
                                latency_ms=(time.time() - start_time) * 1000,
                                error=data.get("error", "Unknown error"),
                            )
                    except json.JSONDecodeError:
                        continue

            latency_ms = (time.time() - start_time) * 1000

            return ChatResult(
                answer="".join(answer_parts),
                sources=sources,
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return ChatResult(
                answer="",
                sources=[],
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ChatResult(
                answer="",
                sources=[],
                latency_ms=latency_ms,
                error=str(e),
            )

    async def get_all_chunks(self) -> List[ChunkInfo]:
        """Get all document chunks from the vector store.

        Uses multiple diverse search queries to fetch a broad sample of chunks.

        Returns:
            List of ChunkInfo objects
        """
        await self._ensure_connected()

        try:
            # Diverse search queries to fetch different types of content
            search_queries = [
                "policy procedure guidelines",
                "rate amount allowance cost",
                "travel expenses reimbursement",
                "requirements eligibility criteria",
                "process steps instructions",
                "approval authorization",
                "benefits entitlements",
                "documentation records",
                "rules regulations",
                "definitions terms",
                "meal breakfast lunch dinner",
                "transportation vehicle kilometric",
                "accommodation lodging hotel",
                "duty travel temporary",
                "member employee civilian",
                "day days daily period",
                "claim claims receipt receipts",
                "family dependent relocation",
                "medical health dental",
                "leave absence vacation",
            ]

            all_chunks = {}  # Use dict to dedupe by ID

            for query in search_queries:
                try:
                    response = await self._client.post(
                        "/api/v1/sources/search",
                        json={
                            "query": query,
                            "k": 50,
                            "use_hybrid_search": True,
                        },
                    )
                    if response.status_code == 200:
                        data = response.json()
                        raw_docs = data if isinstance(data, list) else data.get("results", data.get("documents", []))

                        for item in raw_docs:
                            if isinstance(item, dict):
                                # Handle DocumentSearchResult format
                                if "document" in item:
                                    doc = item["document"]
                                else:
                                    doc = item

                                chunk_id = doc.get("id", doc.get("chunk_id", ""))
                                content = (
                                    doc.get("text") or
                                    doc.get("content") or
                                    doc.get("page_content") or
                                    ""
                                )

                                if chunk_id and content and chunk_id not in all_chunks:
                                    all_chunks[chunk_id] = ChunkInfo(
                                        id=chunk_id,
                                        content=content,
                                        metadata=doc.get("metadata", {}),
                                    )
                except Exception as e:
                    print(f"Warning: Search query '{query}' failed: {e}")
                    continue

            chunks = list(all_chunks.values())
            print(f"Retrieved {len(chunks)} unique chunks via diverse search")
            return chunks

        except Exception as e:
            print(f"Warning: Could not fetch all chunks: {e}")
            return []

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with stats like total_chunks, sources, etc.
        """
        await self._ensure_connected()

        try:
            response = await self._client.get("/api/v1/sources/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def retrieve_only(
        self,
        query: str,
        retrieval_config: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """Fast retrieval-only request - no LLM generation.

        Args:
            query: The search query
            retrieval_config: Optional retrieval configuration overrides
            use_hybrid: Whether to use hybrid search

        Returns:
            RetrievalResult with documents and latency
        """
        await self._ensure_connected()

        start_time = time.time()
        try:
            request_body = {
                "query": query,
                "use_hybrid_search": use_hybrid,
            }
            if retrieval_config is not None:
                request_body["retrieval_config"] = retrieval_config

            response = await self._client.post(
                "/api/v1/retrieval",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = data.get("latency_ms", (time.time() - start_time) * 1000)

            # Convert sources to document format
            documents = []
            for source in data.get("sources", []):
                doc = {
                    "id": source.get("id") or source.get("source_id") or source.get("chunk_id", ""),
                    "text": source.get("text", ""),
                    "score": source.get("score", 0.0),
                    "metadata": source.get("metadata", {}),
                }
                documents.append(doc)

            return RetrievalResult(
                documents=documents,
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                documents=[],
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                documents=[],
                latency_ms=latency_ms,
                error=str(e),
            )

    async def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update RAG configuration.

        Args:
            config_updates: Dictionary of config key-value pairs to update

        Returns:
            True if successful, False otherwise
        """
        await self._ensure_connected()

        try:
            response = await self._client.post(
                "/api/v1/admin/config/update",
                json={"config_updates": config_updates},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Warning: Could not update config: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if RAG service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        await self._ensure_connected()

        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
