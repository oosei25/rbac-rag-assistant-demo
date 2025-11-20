from __future__ import annotations

import json
import os
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
            for u, rec in data.items():
                if not isinstance(rec, dict) or "password" not in rec or "role" not in rec:
                    raise ValueError(f"Invalid user record for {u}")
            return data
        except Exception as e:
            print(f"[auth] WARNING: failed to parse BASIC_USERS_JSON: {e}")
            return DEFAULT_USERS

    def authenticate(
        self, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Dict[str, str]:
        username = credentials.username
        password = credentials.password
        rec = self._users_db.get(username)
        if not rec or rec.get("password") != password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"username": username, "role": rec["role"]}

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
