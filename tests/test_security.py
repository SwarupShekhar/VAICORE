"""Security-hardening self-checks: password policy, CSRF compare, JWT jti.

No DB / celery needed. Run: PYTHONPATH=. python tests/test_security.py
"""
import os
import uuid

os.environ.setdefault("SECRET_KEY", "test-secret-key")
# database.py requires DATABASE_URL at import; a dummy is fine (no connection is opened here).
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")

import jwt

from models import User, UserRole, validate_password
import auth


def test_validate_password_accepts_good():
    assert validate_password("abcdef1234") is True   # 10 chars, letter+digit
    assert validate_password("Sup3rSecret!") is True
    assert validate_password("passw0rd12") is True


def test_validate_password_rejects_bad():
    bad_cases = [
        "",              # empty
        "short1",        # too short
        "abc123",        # too short (6)
        "abcdefghij",    # no digit
        "1234567890",    # no letter
        "         1",    # too short + no letter
    ]
    for pw in bad_cases:
        try:
            validate_password(pw)
            assert False, f"expected ValueError for {pw!r}"
        except ValueError as e:
            assert str(e), "error message should be non-empty"


def test_csrf_compare():
    # both present and equal -> True
    assert auth.csrf_tokens_match("tok-abc", "tok-abc") is True
    # mismatch -> False
    assert auth.csrf_tokens_match("tok-abc", "tok-xyz") is False
    # missing header or cookie -> False
    assert auth.csrf_tokens_match(None, "tok-abc") is False
    assert auth.csrf_tokens_match("tok-abc", None) is False
    assert auth.csrf_tokens_match(None, None) is False
    assert auth.csrf_tokens_match("", "") is False


def test_jwt_contains_jti():
    u = User(id=uuid.uuid4(), email="a@b.c", role=UserRole.CLIENT_ADMIN.value, client_code="C1")
    token = auth.create_access_token(u)
    payload = jwt.decode(token, os.environ["SECRET_KEY"], algorithms=["HS256"])
    assert "jti" in payload, "JWT must carry a jti claim for revocation"
    # jti must be a valid UUID string
    uuid.UUID(payload["jti"])
    # two tokens for the same user must have distinct jtis
    payload2 = jwt.decode(auth.create_access_token(u), os.environ["SECRET_KEY"], algorithms=["HS256"])
    assert payload["jti"] != payload2["jti"]


if __name__ == "__main__":
    test_validate_password_accepts_good()
    test_validate_password_rejects_bad()
    test_csrf_compare()
    test_jwt_contains_jti()
    print("OK: all security self-checks passed")
