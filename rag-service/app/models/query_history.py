"""Query history models for tracking and analyzing system usage."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class QueryStatus(str, Enum):
    """Query execution status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class QueryHistoryEntry(BaseModel):
    """Single query history entry."""
    id: str = Field(..., description="Unique query ID")
    timestamp: datetime = Field(..., description="Query timestamp")
    user_query: str = Field(..., description="Original user query")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model name")
    use_rag: bool = Field(..., description="Whether RAG was enabled")
    response_preview: Optional[str] = Field(None, description="First 500 chars of response")
    sources_count: int = Field(0, description="Number of sources retrieved")
    processing_time: float = Field(..., description="Total processing time in seconds")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    status: QueryStatus = Field(..., description="Query execution status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryHistoryFilter(BaseModel):
    """Filters for querying history."""
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    provider: Optional[str] = Field(None, description="Filter by provider")
    model: Optional[str] = Field(None, description="Filter by model")
    status: Optional[QueryStatus] = Field(None, description="Filter by status")
    use_rag: Optional[bool] = Field(None, description="Filter by RAG usage")
    conversation_id: Optional[str] = Field(None, description="Filter by conversation")
    search_query: Optional[str] = Field(None, description="Search in user queries")
    limit: int = Field(100, description="Maximum results to return")
    offset: int = Field(0, description="Offset for pagination")
    order_by: str = Field("timestamp", description="Field to order by")
    order_desc: bool = Field(True, description="Order descending")


class QueryStatistics(BaseModel):
    """Aggregated query statistics."""
    total_queries: int = Field(..., description="Total number of queries")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    average_processing_time: float = Field(..., description="Average processing time in seconds")
    total_tokens_used: int = Field(..., description="Total tokens consumed")
    queries_per_provider: Dict[str, int] = Field(..., description="Query count by provider")
    queries_per_model: Dict[str, int] = Field(..., description="Query count by model")
    rag_enabled_queries: int = Field(..., description="Queries with RAG enabled")
    average_sources_per_query: float = Field(..., description="Average sources retrieved per query")
    queries_by_hour: Dict[int, int] = Field(..., description="Query distribution by hour of day")
    queries_by_day: Dict[str, int] = Field(..., description="Query distribution by day of week")
    top_errors: List[Dict[str, Any]] = Field(..., description="Most common errors")
    query_trends: List[Dict[str, Any]] = Field(..., description="Query trends over time")


class QueryExportRequest(BaseModel):
    """Request for exporting query history."""
    format: str = Field("csv", description="Export format: csv or json")
    filters: QueryHistoryFilter = Field(default_factory=QueryHistoryFilter, description="Filters to apply")
    include_responses: bool = Field(False, description="Include full response text")
    include_metadata: bool = Field(True, description="Include metadata")
    anonymize: bool = Field(False, description="Anonymize sensitive data")