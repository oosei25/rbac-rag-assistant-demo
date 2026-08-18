from __future__ import annotations

import hashlib


def scoped_thread_id(username: str, requested_thread_id: str | None) -> str:
    """Return a stable, user-scoped checkpoint key without exposing raw identifiers."""
    client_key = requested_thread_id or "default"
    material = f"{len(username)}:{username}\0{client_key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


__all__ = ["scoped_thread_id"]
