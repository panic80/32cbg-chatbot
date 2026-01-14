"""LangChain utilities including retry logic and error handling."""

from typing import TypeVar, Callable, Any, Union, Optional, Type, Tuple
from functools import wraps
import asyncio
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from langchain_core.exceptions import (
    LangChainException,
    OutputParserException
)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from openai import RateLimitError, APIError, APITimeoutError
from httpx import TimeoutException, ConnectError

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

# Define retryable exceptions
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIError,
    APITimeoutError,
    TimeoutException,
    ConnectError,
    ConnectionError,
    OSError
)

# Provider-specific exceptions
PROVIDER_EXCEPTIONS = {
    "openai": (RateLimitError, APIError, APITimeoutError),
    "google": (Exception,),  # Google client exceptions are less specific
    "anthropic": (Exception,)  # Anthropic client exceptions are less specific
}


def get_retry_decorator(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Create a retry decorator with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=initial_wait,
            max=max_wait,
            exp_base=exponential_base
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logger.level),
        after=after_log(logger, logger.level),
        reraise=True
    )


def langchain_retry(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying LangChain operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to wait times
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = min(
                            initial_wait * (exponential_base ** attempt),
                            max_wait
                        )
                        if jitter:
                            wait_time *= (0.5 + random.random())
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
                except Exception as e:
                    logger.error(f"Non-retryable error: {e}")
                    raise
            
            if last_exception:
                raise last_exception
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = min(
                            initial_wait * (exponential_base ** attempt),
                            max_wait
                        )
                        if jitter:
                            wait_time *= (0.5 + random.random())
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
                except Exception as e:
                    logger.error(f"Non-retryable error: {e}")
                    raise
            
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RetryableLLM:
    """Wrapper for LLMs with built-in retry logic."""
    
    def __init__(
        self,
        llm: Union[ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic],
        max_attempts: int = 3,
        initial_wait: float = 1.0
    ):
        self.llm = llm
        self.max_attempts = max_attempts
        self.initial_wait = initial_wait
        self._retry_decorator = langchain_retry(
            max_attempts=max_attempts,
            initial_wait=initial_wait
        )
    
    @property
    def provider(self) -> str:
        """Get the provider name."""
        if isinstance(self.llm, ChatOpenAI):
            return "openai"
        elif isinstance(self.llm, ChatGoogleGenerativeAI):
            return "google"
        elif isinstance(self.llm, ChatAnthropic):
            return "anthropic"
        else:
            return "unknown"
    
    async def ainvoke(self, *args, **kwargs):
        """Async invoke with retry."""
        @self._retry_decorator
        async def _invoke():
            return await self.llm.ainvoke(*args, **kwargs)
        
        return await _invoke()
    
    def invoke(self, *args, **kwargs):
        """Sync invoke with retry."""
        @self._retry_decorator
        def _invoke():
            return self.llm.invoke(*args, **kwargs)
        
        return _invoke()
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped LLM."""
        return getattr(self.llm, name)


def parse_with_retry(parser, text: str, max_attempts: int = 3) -> Any:
    """
    Parse output with retry logic for OutputParserException.
    
    Args:
        parser: The LangChain parser to use
        text: The text to parse
        max_attempts: Maximum number of attempts
        
    Returns:
        Parsed output
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return parser.parse(text)
        except OutputParserException as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Parse attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    "Retrying with cleaned text..."
                )
                # Try to clean the text
                text = text.strip()
                # Remove common formatting issues
                if text.startswith("```") and text.endswith("```"):
                    text = text[3:-3].strip()
                if text.startswith("```json") and text.endswith("```"):
                    text = text[7:-3].strip()
            else:
                logger.error(f"All parse attempts failed: {e}")
    
    if last_exception:
        raise last_exception


def handle_llm_error(
    error: Exception,
    provider: str,
    context: Optional[str] = None
) -> Tuple[str, int]:
    """
    Handle LLM errors and return user-friendly message and status code.
    
    Args:
        error: The exception that occurred
        provider: The LLM provider name
        context: Optional context about what was being done
        
    Returns:
        Tuple of (error message, HTTP status code)
    """
    error_type = type(error).__name__
    context_str = f" while {context}" if context else ""
    
    # Rate limit errors
    if isinstance(error, RateLimitError):
        return (
            f"Rate limit exceeded for {provider}{context_str}. Please try again later.",
            429
        )
    
    # Timeout errors
    if isinstance(error, (APITimeoutError, TimeoutException)):
        return (
            f"Request to {provider} timed out{context_str}. Please try again.",
            504
        )
    
    # Connection errors
    if isinstance(error, (ConnectError, ConnectionError)):
        return (
            f"Unable to connect to {provider}{context_str}. Please check your network connection.",
            503
        )
    
    # API errors
    if isinstance(error, APIError):
        return (
            f"{provider} API error{context_str}: {str(error)}",
            502
        )
    
    # Generic LangChain errors
    if isinstance(error, LangChainException):
        return (
            f"LangChain error{context_str}: {str(error)}",
            500
        )
    
    # Fallback
    return (
        f"Unexpected error with {provider}{context_str}: {error_type} - {str(error)}",
        500
    )


# Utility functions for common patterns
async def safe_llm_call(
    llm_func: Callable,
    *args,
    fallback_result: Optional[Any] = None,
    error_message: str = "LLM call failed",
    **kwargs
) -> Any:
    """
    Safely call an LLM function with error handling.
    
    Args:
        llm_func: The LLM function to call
        *args: Positional arguments for the function
        fallback_result: Result to return on failure
        error_message: Error message to log
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result from LLM or fallback_result
    """
    try:
        return await llm_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return fallback_result


def batch_with_concurrency_limit(
    items: list,
    batch_size: int,
    max_concurrent: int = 5
) -> list:
    """
    Create batches with concurrency limit for parallel processing.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent batches
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    # Limit concurrent batches
    if len(batches) > max_concurrent:
        # Process in chunks of max_concurrent
        chunked_batches = []
        for i in range(0, len(batches), max_concurrent):
            chunked_batches.append(batches[i:i + max_concurrent])
        return chunked_batches
    
    return [batches]
