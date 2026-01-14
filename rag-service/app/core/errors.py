"""Custom error types and error handling utilities."""

from typing import Optional, Dict, Any
from enum import Enum
import traceback
from datetime import datetime, timezone


class ErrorCategory(Enum):
    """Categories of errors for better handling."""
    NETWORK = "network"
    PARSING = "parsing"
    VALIDATION = "validation"
    STORAGE = "storage"
    PROCESSING = "processing"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class IngestionError(Exception):
    """Base exception for ingestion errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None
    ):
        """Initialize ingestion error."""
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.now(timezone.utc)
        self.traceback = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/response."""
        return {
            "error": self.message,
            "category": self.category.value,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback if self.details.get("include_traceback") else None
        }


class NetworkError(IngestionError):
    """Network-related errors."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        details = {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
            
        # Determine if error is recoverable based on status code
        recoverable = status_code not in [400, 401, 403, 404] if status_code else True
        
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            details=details,
            recoverable=recoverable
        )


class ParsingError(IngestionError):
    """Document parsing errors."""
    
    def __init__(self, message: str, document_type: Optional[str] = None, source: Optional[str] = None):
        details = {}
        if document_type:
            details["document_type"] = document_type
        if source:
            details["source"] = source
            
        super().__init__(
            message=message,
            category=ErrorCategory.PARSING,
            details=details,
            recoverable=False  # Parsing errors usually require fixing the source
        )


class ValidationError(IngestionError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
            
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            details=details,
            recoverable=False  # Validation errors require fixing input
        )


class StorageError(IngestionError):
    """Vector store or database errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            category=ErrorCategory.STORAGE,
            details=details,
            recoverable=True  # Storage errors might be temporary
        )


class RateLimitError(IngestionError):
    """Rate limiting errors."""
    
    def __init__(self, message: str, retry_after: int):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            details={"retry_after": retry_after},
            recoverable=True,
            retry_after=retry_after
        )


import re

# Exception type to (recoverable, retry_after_seconds) mapping
# More specific exception types get priority over keyword matching
RECOVERABLE_EXCEPTION_PATTERNS = {
    # OpenAI errors
    "openai.RateLimitError": (True, 60),
    "openai.APIStatusError": (True, 5),
    "openai.APIConnectionError": (True, 3),
    "openai.APITimeoutError": (True, 5),
    # Chroma errors
    "chromadb.errors.ChromaError": (True, 2),
    "chromadb.errors.InvalidCollectionException": (False, 0),
    # Redis errors
    "redis.ConnectionError": (True, 5),
    "redis.TimeoutError": (True, 3),
    "redis.BusyLoadingError": (True, 10),
    # HTTP errors
    "aiohttp.ClientConnectorError": (True, 3),
    "aiohttp.ServerDisconnectedError": (True, 2),
    "httpx.ConnectError": (True, 3),
    "httpx.TimeoutException": (True, 5),
    # General network
    "ConnectionResetError": (True, 2),
    "ConnectionRefusedError": (True, 5),
    "TimeoutError": (True, 5),
    "OSError": (True, 2),  # Often network-related
}

# Patterns that indicate non-recoverable errors (regex)
NON_RECOVERABLE_PATTERNS = [
    r"invalid api key",
    r"invalid.*key",
    r"authentication failed",
    r"unauthorized",
    r"model not found",
    r"model.*does not exist",
    r"context length exceeded",
    r"maximum context length",
    r"file not found",
    r"no such file",
    r"permission denied",
    r"access denied",
    r"invalid.*format",
    r"unsupported.*type",
    r"corrupt.*file",
    r"malformed.*document",
]


def categorize_error(error: Exception) -> IngestionError:
    """Categorize generic exceptions into specific ingestion errors.

    Uses a multi-stage classification:
    1. Check exception type against known recoverable patterns
    2. Check error message against non-recoverable patterns
    3. Fall back to keyword-based classification
    """
    error_type = f"{type(error).__module__}.{type(error).__name__}"
    error_str = str(error).lower()

    # Stage 1: Check exception type against known patterns
    for pattern, (recoverable, retry_after) in RECOVERABLE_EXCEPTION_PATTERNS.items():
        if pattern in error_type or error_type.endswith(pattern.split('.')[-1]):
            if recoverable:
                if "rate" in error_str or "limit" in error_str:
                    return RateLimitError(f"Rate limit: {error}", retry_after=retry_after)
                elif "timeout" in error_str:
                    return NetworkError(f"Timeout: {error}")
                else:
                    return NetworkError(f"Connection error: {error}")
            else:
                return IngestionError(
                    message=str(error),
                    category=ErrorCategory.UNKNOWN,
                    details={"original_error": error_type},
                    recoverable=False
                )

    # Stage 2: Check for non-recoverable error patterns
    for pattern in NON_RECOVERABLE_PATTERNS:
        if re.search(pattern, error_str, re.IGNORECASE):
            # Determine category based on pattern
            if any(word in pattern for word in ["api key", "auth", "unauthorized"]):
                return IngestionError(
                    message=f"Authentication error: {error}",
                    category=ErrorCategory.AUTHENTICATION,
                    details={"original_error": error_type},
                    recoverable=False
                )
            elif any(word in pattern for word in ["file", "permission", "access"]):
                return ParsingError(f"File access error: {error}")
            elif any(word in pattern for word in ["format", "type", "corrupt", "malformed"]):
                return ParsingError(f"Invalid document: {error}")
            elif any(word in pattern for word in ["context", "length", "model"]):
                return ValidationError(f"Model constraint error: {error}")

    # Stage 3: Keyword-based classification (fallback)
    # Network-related errors
    if any(keyword in error_str for keyword in ["connection", "timeout", "network", "refused", "unreachable"]):
        return NetworkError(f"Network error: {error}")

    # Parsing errors
    elif any(keyword in error_str for keyword in ["parse", "decode", "invalid format", "malformed", "encoding"]):
        return ParsingError(f"Parsing error: {error}")

    # Validation errors - be careful not to catch "invalid" too broadly
    elif any(keyword in error_str for keyword in ["validation failed", "required field", "missing required"]):
        return ValidationError(f"Validation error: {error}")

    # Storage errors
    elif any(keyword in error_str for keyword in ["database", "vector", "storage", "persist", "collection"]):
        return StorageError(f"Storage error: {error}")

    # Rate limit errors
    elif any(keyword in error_str for keyword in ["rate limit", "too many requests", "429", "quota"]):
        return RateLimitError("Rate limit exceeded", retry_after=60)

    # Default to unknown but check if it looks transient
    else:
        # Check if error looks transient (network-like)
        transient_indicators = ["temporary", "retry", "503", "502", "500", "unavailable", "overloaded"]
        is_likely_transient = any(ind in error_str for ind in transient_indicators)

        return IngestionError(
            message=str(error),
            category=ErrorCategory.UNKNOWN,
            details={"original_error": error_type},
            recoverable=is_likely_transient,
            retry_after=5 if is_likely_transient else None
        )
