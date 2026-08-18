from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator

from app.policy import allowed_departments
from app.schemas import DocumentDetail, DocumentSummary
from app.utils.io import read_file


SUPPORTED_SUFFIXES = {".csv", ".md", ".txt"}


def safe_document_path(path: Path, data_dir: Path) -> str:
    """Return a stable path relative to DATA_DIR, never a host path."""
    root = data_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Document is outside DATA_DIR")
    return resolved.relative_to(root).as_posix()


def canonical_document_id(path: Path, data_dir: Path) -> str:
    safe_path = safe_document_path(path, data_dir)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rbac-rag:{safe_path}"))


def department_from_path(path: Path, data_dir: Path) -> str:
    parts = Path(safe_document_path(path, data_dir)).parts
    return parts[0] if len(parts) > 1 else "general"


def document_title(path: Path, content: str) -> str:
    if path.suffix.lower() == ".md":
        for line in content.splitlines():
            if line.lstrip().startswith("#"):
                title = line.lstrip("# ").strip()
                if title:
                    return title
    return path.stem.replace("_", " ").strip().title()


def document_preview(content: str, limit: int = 400) -> str:
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    preview = " ".join(lines[:3])
    return preview[:limit] + ("…" if len(preview) > limit else "")


class DocumentService:
    """Reads only documents authorized for the authenticated role."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def _paths(self) -> Iterator[Path]:
        if not self.data_dir.exists():
            return
        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                try:
                    safe_document_path(path, self.data_dir)
                except ValueError:
                    continue
                yield path

    def list_for_role(self, role: str) -> list[DocumentSummary]:
        allowed = set(allowed_departments(role))
        if not allowed:
            return []

        documents: list[DocumentSummary] = []
        for path in self._paths():
            department = department_from_path(path, self.data_dir)
            if department not in allowed:
                continue
            content = read_file(path)
            if not content:
                continue
            documents.append(
                DocumentSummary(
                    document_id=canonical_document_id(path, self.data_dir),
                    path=safe_document_path(path, self.data_dir),
                    title=document_title(path, content),
                    department=department,
                    preview=document_preview(content),
                )
            )
        return documents

    def get_for_role(self, document_id: str, role: str) -> DocumentDetail | None:
        allowed = set(allowed_departments(role))
        if not allowed:
            return None

        for path in self._paths():
            department = department_from_path(path, self.data_dir)
            if department not in allowed:
                continue
            if canonical_document_id(path, self.data_dir) != document_id:
                continue
            content = read_file(path)
            if not content:
                return None
            return DocumentDetail(
                document_id=document_id,
                path=safe_document_path(path, self.data_dir),
                title=document_title(path, content),
                department=department,
                preview=document_preview(content),
                content=content,
            )
        return None
