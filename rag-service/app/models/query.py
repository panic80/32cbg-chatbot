"""Query and chat models for RAG service."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timezone
from enum import Enum


class Provider(str, Enum):
    """LLM provider enumeration."""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class Source(BaseModel):
    """Source reference model."""
    id: str = Field(..., description="Source document ID")
    source_id: Optional[str] = Field(None, description="Canonical source identifier")
    text: str = Field(..., description="Source text snippet")
    title: Optional[str] = Field(None, description="Source title")
    url: Optional[str] = Field(None, description="Source URL")
    section: Optional[str] = Field(None, description="Source section")
    page: Optional[int] = Field(None, description="Page number")
    score: Optional[float] = Field(None, description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RetrievalConfig(BaseModel):
    """Per-request retrieval configuration overrides."""
    retrieval_k: Optional[int] = Field(None, description="Number of documents to retrieve")
    rrf_k: Optional[int] = Field(None, description="RRF fusion parameter k")
    enable_hyde: Optional[bool] = Field(None, description="Enable HyDE retrieval")
    enable_reranker: Optional[bool] = Field(None, description="Enable cross-encoder reranking")
    reranker_skip_similarity_threshold: Optional[float] = Field(
        None, description="Skip reranking if top score above this threshold"
    )
    unified_retrieval_mode: Optional[str] = Field(
        None, description="Retrieval mode: simple, balanced, advanced"
    )
    auxiliary_model: Optional[str] = Field(
        None, description="Model to use for auxiliary tasks (classification, HyDE, etc.)"
    )


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    provider: Provider = Field(Provider.OPENAI, description="LLM provider")
    model: Optional[str] = Field(None, description="Specific model to use")
    use_rag: bool = Field(True, description="Use RAG for response")
    include_sources: bool = Field(True, description="Include source citations")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, description="Maximum response tokens")
    short_answer_mode: bool = Field(False, description="Provide brief, concise responses")
    # HYBRID_SEARCH_TOGGLE_START - Remove this line to disable hybrid search
    use_hybrid_search: bool = Field(False, description="Enable hybrid BM25+Vector search for improved accuracy")
    # HYBRID_SEARCH_TOGGLE_END
    additional_instructions: Optional[str] = Field(
        default=None,
        description="Extra system prompt guidance specific to the request",
        alias="additionalInstructions",
    )
    reasoning_effort: Optional[str] = Field(
        None,
        description="Hint for OpenAI reasoning effort",
        alias="reasoningEffort",
    )
    response_verbosity: Optional[str] = Field(
        None,
        description="Hint for OpenAI response verbosity",
        alias="responseVerbosity",
    )
    audience: Optional[str] = Field(
        default=None,
        description="Target audience for differences (e.g., 'classA')",
    )
    retrieval_config: Optional[RetrievalConfig] = Field(
        None, description="Per-request retrieval configuration overrides"
    )
    mode: Optional[Literal["fast", "smart"]] = Field(
        None, description="Optimization mode: 'fast' for quick responses, 'smart' for thorough retrieval"
    )

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="AI response")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    conversation_id: str = Field(..., description="Conversation ID")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    processing_time: float = Field(..., description="Response time in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    confidence_score: Optional[float] = Field(None, description="Response confidence")


class FollowUpQuestion(BaseModel):
    """Follow-up question model."""
    id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Follow-up question text")
    category: str = Field("general", description="Question category")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class FollowUpRequest(BaseModel):
    """Follow-up questions request model."""
    user_question: str = Field(..., description="Original user question")
    ai_response: str = Field(..., description="AI response")
    sources: Optional[List[Source]] = Field(default_factory=list)
    max_questions: int = Field(3, ge=1, le=5)


class FollowUpResponse(BaseModel):
    """Follow-up questions response model."""
    questions: List[FollowUpQuestion] = Field(..., description="Follow-up questions")
    
    
class QueryExpansionRequest(BaseModel):
    """Query expansion request model."""
    query: str = Field(..., description="Original query")
    context: Optional[List[ChatMessage]] = Field(default_factory=list)
    max_expansions: int = Field(3, ge=1, le=5)


class QueryExpansionResponse(BaseModel):
    """Query expansion response model."""
    original_query: str = Field(..., description="Original query")
    expanded_queries: List[str] = Field(..., description="Expanded queries")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
