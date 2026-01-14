"""Utility helpers for securing admin-only API endpoints."""

import os
from fastapi import Header, HTTPException, status

ADMIN_TOKEN_ENV = "ADMIN_API_TOKEN"
EXPECTED_ADMIN_TOKEN = os.getenv(ADMIN_TOKEN_ENV)

if not EXPECTED_ADMIN_TOKEN:
    raise RuntimeError("ADMIN_API_TOKEN must be set for admin endpoints")


def verify_admin_bearer_token(authorization: str = Header(None, convert_underscores=False)) -> bool:
    """Ensure that the caller supplied the correct admin bearer token."""

    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Admin bearer token required.'
        )

    provided_token = authorization.split(' ', 1)[1].strip()
    if provided_token != EXPECTED_ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Invalid admin bearer token.'
        )

    return True
