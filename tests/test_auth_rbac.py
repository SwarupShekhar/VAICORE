"""RBAC self-check: password hashing, JWT roundtrip, role gate. No DB needed.

Run: python tests/test_auth_rbac.py
"""
import asyncio
import os
import uuid

os.environ.setdefault("SECRET_KEY", "test-secret-key")
# database.py requires DATABASE_URL at import; a dummy is fine (no connection is opened here).
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")

import jwt
import fastapi

from models import User, UserRole, hash_password, verify_password
import auth


def test_password_roundtrip():
    h = hash_password("s3cret!")
    assert h != "s3cret!"
    assert verify_password("s3cret!", h)
    assert not verify_password("wrong", h)


def test_jwt_roundtrip():
    u = User(id=uuid.uuid4(), email="a@b.c", role=UserRole.CLIENT_ADMIN.value, client_code="CLIENT001")
    payload = jwt.decode(auth.create_access_token(u), os.environ["SECRET_KEY"], algorithms=["HS256"])
    assert payload["sub"] == str(u.id)
    assert payload["role"] == "client_admin"
    assert payload["client_code"] == "CLIENT001"


def test_require_role_gate():
    dep = auth.require_role(UserRole.SUPER_ADMIN.value)
    admin = User(id=uuid.uuid4(), email="s@x", role=UserRole.SUPER_ADMIN.value)
    biz = User(id=uuid.uuid4(), email="b@x", role=UserRole.BUSINESS_USER.value)
    assert asyncio.run(dep(admin)) is admin
    try:
        asyncio.run(dep(biz))
        assert False, "business_user should be rejected"
    except fastapi.HTTPException as e:
        assert e.status_code == 403


if __name__ == "__main__":
    test_password_roundtrip()
    test_jwt_roundtrip()
    test_require_role_gate()
    print("OK: all RBAC self-checks passed")
