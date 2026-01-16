"""
Error handling decorators and utilities for FastAPI endpoints.

Provides standardized error handling to reduce boilerplate across API endpoints.
"""

from __future__ import annotations

import functools
import traceback
from typing import Any, Callable, Optional, TypeVar, Union

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class APIError(Exception):
    """Base exception for API errors with status code and details."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}


def handle_errors(
    *,
    fallback: Optional[Any] = None,
    log_level: str = "error",
    error_message: str = "An error occurred",
    include_traceback: bool = False,
    reraise_http_exceptions: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for standardized error handling in FastAPI endpoints.

    Args:
        fallback: Value to return on error instead of raising (if provided)
        log_level: Logging level for errors ('error', 'warning', 'info')
        error_message: Default error message for responses
        include_traceback: Include traceback in logs
        reraise_http_exceptions: Re-raise HTTPException without wrapping

    Usage:
        @router.get("/items/{item_id}")
        @handle_errors(error_message="Failed to fetch item")
        async def get_item(item_id: int):
            return await fetch_item(item_id)

        # With fallback value (returns fallback instead of raising)
        @handle_errors(fallback=[], error_message="Failed to list items")
        async def list_items():
            return await fetch_items()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                if reraise_http_exceptions:
                    raise
                return _handle_exception(func, fallback, log_level, error_message, include_traceback)
            except APIError as e:
                _log_error(func, e, log_level, include_traceback)
                if fallback is not None:
                    return fallback
                raise HTTPException(
                    status_code=e.status_code,
                    detail={
                        "error": e.error_code,
                        "message": e.message,
                        **e.details,
                    },
                )
            except Exception as e:
                return _handle_exception(func, fallback, log_level, error_message, include_traceback, e)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                if reraise_http_exceptions:
                    raise
                return _handle_exception(func, fallback, log_level, error_message, include_traceback)
            except APIError as e:
                _log_error(func, e, log_level, include_traceback)
                if fallback is not None:
                    return fallback
                raise HTTPException(
                    status_code=e.status_code,
                    detail={
                        "error": e.error_code,
                        "message": e.message,
                        **e.details,
                    },
                )
            except Exception as e:
                return _handle_exception(func, fallback, log_level, error_message, include_traceback, e)

        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def _log_error(
    func: Callable[..., Any],
    error: Exception,
    log_level: str,
    include_traceback: bool,
) -> None:
    """Log an error with the appropriate level and context."""
    log_func = getattr(logger, log_level, logger.error)

    log_data = {
        "function": func.__name__,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if include_traceback:
        log_data["traceback"] = traceback.format_exc()

    log_func("Error in %s: %s", func.__name__, str(error), extra=log_data)


def _handle_exception(
    func: Callable[..., Any],
    fallback: Optional[Any],
    log_level: str,
    error_message: str,
    include_traceback: bool,
    error: Optional[Exception] = None,
) -> Any:
    """Handle exception by logging and either returning fallback or raising HTTPException."""
    if error:
        _log_error(func, error, log_level, include_traceback)

    if fallback is not None:
        return fallback

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "INTERNAL_ERROR",
            "message": error_message,
        },
    )


def create_error_response(
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    error_code: str = "INTERNAL_ERROR",
    details: Optional[dict] = None,
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        message: Human-readable error message
        status_code: HTTP status code
        error_code: Machine-readable error code
        details: Additional error details

    Returns:
        JSONResponse with standardized error format
    """
    content = {
        "error": error_code,
        "message": message,
    }
    if details:
        content.update(details)

    return JSONResponse(
        status_code=status_code,
        content=content,
    )
