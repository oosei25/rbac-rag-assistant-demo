from __future__ import annotations

import json
import os
import secrets
from typing import Callable, Dict

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Default demo users (plaintext for local/dev)
DEFAULT_USERS: Dict[str, Dict[str, str]] = {
    "Tony":   {"password": "password123",   "role": "engineering"},
    "Bruce":  {"password": "securepass",    "role": "marketing"},
    "Sam":    {"password": "financepass",   "role": "finance"},
    "Peter":  {"password": "pete123",       "role": "engineering"},
    "Mariam": {"password": "mariampass123", "role": "marketing"},
    "Natasha":{"password": "hrpass123",     "role": "hr"},
    "Cathy":  {"password": "cathyceo",      "role": "clevel"},
    "Emma":   {"password": "password",      "role": "employee"},
}


class AuthService:
    """Encapsulates user lookup + RBAC helpers for FastAPI dependencies."""

    security = HTTPBasic()

    def __init__(self):
        self._users_db = self._load_users()

    @staticmethod
    def _load_users() -> Dict[str, Dict[str, str]]:
        """
        Optional override via BASIC_USERS_JSON env:
        {
          "Alice": {"password": "alicepwd", "role": "engineering"},
          "Bob":   {"password": "bobpwd",   "role": "finance"}
        }
        """
        raw = os.getenv("BASIC_USERS_JSON")
        if not raw:
            return DEFAULT_USERS
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("BASIC_USERS_JSON must be an object")
            for u, rec in data.items():
                if (
                    not isinstance(u, str)
                    or not isinstance(rec, dict)
                    or not isinstance(rec.get("password"), str)
                    or not isinstance(rec.get("role"), str)
                ):
                    raise ValueError(f"Invalid user record for {u}")
            return data
        except Exception as e:
            print(f"[auth] WARNING: failed to parse BASIC_USERS_JSON: {e}")
            # An explicitly supplied but invalid database must fail closed;
            # silently restoring well-known demo credentials is unsafe.
            return {}

    def authenticate(
        self, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Dict[str, str]:
        matched_user: tuple[str, Dict[str, str]] | None = None
        supplied_username = credentials.username.encode("utf-8")
        supplied_password = credentials.password.encode("utf-8")

        # Check every demo record so unknown usernames do not take a visibly
        # shorter path than known usernames. compare_digest avoids early-exit
        # string comparisons for both credentials.
        for username, record in self._users_db.items():
            username_ok = secrets.compare_digest(
                supplied_username, username.encode("utf-8")
            )
            password_ok = secrets.compare_digest(
                supplied_password, record["password"].encode("utf-8")
            )
            if username_ok and password_ok:
                matched_user = (username, record)

        if matched_user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        username, record = matched_user
        return {"username": username, "role": record["role"]}

    def require_roles(self, *allowed_roles: str) -> Callable:
        auth_dep = self.authenticate

        def _dep(user=Depends(auth_dep)):
            if user["role"] not in allowed_roles:
                raise HTTPException(status_code=403, detail="Forbidden")
            return user

        return _dep


auth_service = AuthService()
authenticate = auth_service.authenticate
require_roles = auth_service.require_roles

__all__ = ["AuthService", "auth_service", "authenticate", "require_roles"]
