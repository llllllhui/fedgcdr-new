"""密码哈希与 JWT 工具函数 — 使用 hashlib 避免 passlib + bcrypt 兼容问题"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

from backend.core.config import settings

ALGORITHM = "sha256"
SALT_LENGTH = 16


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码: hashed = salt$hash"""
    parts = hashed_password.split("$", 1)
    if len(parts) != 2:
        return False
    salt, stored_hash = parts
    computed = hashlib.pbkdf2_hmac(
        ALGORITHM,
        plain_password.encode(),
        salt.encode(),
        100000,
    )
    return secrets.compare_digest(computed.hex(), stored_hash)


def hash_password(password: str) -> str:
    """哈希密码: salt$pbkdf2_sha256(salt + password)"""
    salt = secrets.token_hex(SALT_LENGTH)
    h = hashlib.pbkdf2_hmac(
        ALGORITHM,
        password.encode(),
        salt.encode(),
        100000,
    )
    return f"{salt}${h.hex()}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None
