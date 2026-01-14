"""Utility validators for strict JSON parsing and basic sanity checks.

These are scaffolds for later enforcement of structured outputs.
"""

from __future__ import annotations

from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_json_model(model: Type[T], payload: Any) -> T:
    """Parse a dict-like payload into a Pydantic model, raising on failure.

    This should be used after LLM calls to enforce strict schemas.
    """
    if isinstance(payload, model):
        return payload
    if isinstance(payload, dict):
        return model(**payload)
    # Allow pydantic-like objects
    dump = getattr(payload, "model_dump", None)
    if callable(dump):
        return model(**dump())
    asdict = getattr(payload, "dict", None)
    if callable(asdict):
        return model(**asdict())
    raise ValidationError([{"loc": ("payload",), "msg": "Invalid payload type", "type": "type_error"}], model)

