"""JWT auth + RBAC dependencies (Phase 2 client onboarding)."""
import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt  # PyJWT
from dotenv import load_dotenv
from fastapi import Cookie, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import User, UserRole

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "")
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24 * 7
COOKIE_NAME = "access_token"


def create_access_token(user: User) -> str:
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY not set — cannot issue tokens.")
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "client_code": user.client_code,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    access_token: str = Cookie(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_role(*roles: str):
    async def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return _dep


# super_admin only
require_super_admin = require_role(UserRole.SUPER_ADMIN.value)
# super_admin or client_admin (min level for user-management)
require_client_admin = require_role(
    UserRole.SUPER_ADMIN.value, UserRole.CLIENT_ADMIN.value
)
