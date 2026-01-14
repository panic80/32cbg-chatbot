"""
Retry utilities for handling transient failures.

This module provides decorators and utilities for retrying
operations that may fail due to transient issues.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Union, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            exceptions: Tuple of exceptions to retry on
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions


def calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float
) -> float:
    """
    Calculate delay for exponential backoff.
    
    Args:
        attempt: Current attempt number (0-based)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        
    Returns:
        Delay in seconds
    """
    delay = initial_delay * (exponential_base ** attempt)
    return min(delay, max_delay)


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying synchronous functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt,
                            initial_delay,
                            max_delay,
                            exponential_base
                        )
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


def with_retry_async(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying asynchronous functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt,
                            initial_delay,
                            max_delay,
                            exponential_base
                        )
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


class RetryManager:
    """Manager for handling retries with configurable policies."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a synchronous function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.config.exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = calculate_delay(
                        attempt,
                        self.config.initial_delay,
                        self.config.max_delay,
                        self.config.exponential_base
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed: {e}"
                    )
        
        raise last_exception
    
    async def execute_with_retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an asynchronous function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.config.exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = calculate_delay(
                        attempt,
                        self.config.initial_delay,
                        self.config.max_delay,
                        self.config.exponential_base
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed: {e}"
                    )
        
        raise last_exception


# Common retry configurations
QUICK_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=5.0
)

STANDARD_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=30.0
)

PERSISTENT_RETRY = RetryConfig(
    max_attempts=10,
    initial_delay=2.0,
    max_delay=120.0
)


# Specific exception retry configurations
NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError, OSError)
)

LLM_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=10.0,
    exceptions=(Exception,)  # Catch all LLM-related exceptions
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=10,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    exceptions=(Exception,)  # Catch all exceptions for aggressive retry
)

# Alias for compatibility
retry_async = with_retry_async