"""
JWT Authentication — token generation and validation.

This module provides:
  - Token creation (login)
  - Token verification (dependency injection)
  - Credential validation (in-memory for v1.0, easily swappable for MongoDB)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import structlog
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials

from app.config import get_settings

log = structlog.get_logger()
settings = get_settings()

security = HTTPBearer()

# ─── In-memory user store (v1.0) ─────────────────────────────────────────────
# In production, query MongoDB or a real user database
_VALID_USERS = {
    "demo": "demo123",  # username: password
    "user1": "password1",
    "user2": "password2",
}


def validate_credentials(username: str, password: str) -> bool:
    """
    Verify username + password.
    In production: query MongoDB user collection and verify hashed password.
    """
    return _VALID_USERS.get(username) == password


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate JWT token with expiry.
    Payload includes user_id and issued-at timestamp.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


async def verify_token(credentials: HTTPAuthCredentials = Depends(security)) -> str:
    """
    FastAPI dependency: validate Bearer token.
    Returns username on success, raises 401 on failure.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no subject claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError as e:
        log.warning("token_verification_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
