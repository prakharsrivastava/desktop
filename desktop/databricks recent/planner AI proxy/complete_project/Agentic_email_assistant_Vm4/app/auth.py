"""OAuth 2.0 Bearer Token Authentication Module."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import hashlib
from pydantic import BaseModel

# Security configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # TODO: Move to environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# HTTP Bearer token scheme
security = HTTPBearer()

# Helper function for password hashing (SHA256 with salt)
def _hash_password(password: str, salt: str = "email-intelligence-salt") -> str:
    """Hash password using SHA256 with salt."""
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

# Pre-hashed passwords (SHA256 hashes of "admin123" and "user123")
MOCK_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": _hash_password("admin123"),
        "email": "admin@example.com",
        "full_name": "Administrator",
    },
    "user": {
        "username": "user",
        "hashed_password": _hash_password("user123"),
        "email": "user@example.com",
        "full_name": "Regular User",
    },
}


class Token(BaseModel):
    """OAuth 2.0 token response."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return _hash_password(plain_password) == hashed_password


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return _hash_password(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in MOCK_USERS_DB:
        user_dict = MOCK_USERS_DB[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Validate token and return current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception

    return User(username=user.username, email=user.email, full_name=user.full_name)
