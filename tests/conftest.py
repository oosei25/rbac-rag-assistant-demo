"""
Test configuration helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)

if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _has_chroma_index(index_dir: Path) -> bool:
    return (index_dir / "chroma.sqlite3").exists()


@pytest.fixture(scope="session", autouse=True)
def ensure_vector_index():
    """
    Build the vector index once per test session so `generate()` has data.
    Skips the rebuild when SKIP_TEST_REINDEX=1.
    """
    if os.getenv("SKIP_TEST_REINDEX") == "1":
        return

    from app.services.indexer import indexer_service

    if indexer_service.vector_db == "chroma":
        if _has_chroma_index(Path(indexer_service.index_path)):
            return
    try:
        indexer_service.reindex()
    except Exception as exc:
        pytest.skip(f"Reindex failed: {exc}")
