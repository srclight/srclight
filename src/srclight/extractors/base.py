"""Shared helpers for document extractors."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..db import Database, SymbolRecord


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Interface that all document extractors must implement."""

    language: str
    extensions: tuple[str, ...]

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        """Extract symbols from a document. Returns count of symbols inserted."""
        ...


def make_section(
    file_id: int,
    rel_path: str,
    name: str,
    qualified_name: str,
    content: str,
    *,
    start_line: int = 1,
    end_line: int = 1,
    signature: str | None = None,
    doc_comment: str | None = None,
    parent_symbol_id: int | None = None,
    kind: str = "section",
) -> SymbolRecord:
    """Build a SymbolRecord for a document section."""
    from ..db import SymbolRecord

    body_h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return SymbolRecord(
        file_id=file_id,
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        signature=signature,
        start_line=start_line,
        end_line=end_line,
        content=content,
        doc_comment=doc_comment,
        body_hash=body_h,
        line_count=max(1, end_line - start_line + 1),
        parent_symbol_id=parent_symbol_id,
    )


def make_document(
    file_id: int,
    rel_path: str,
    content: str,
    *,
    name: str | None = None,
    signature: str | None = None,
    doc_comment: str | None = None,
) -> SymbolRecord:
    """Build a SymbolRecord for a whole-document symbol."""
    from pathlib import Path

    from ..db import SymbolRecord

    stem = Path(rel_path).stem
    doc_name = name or stem
    body_h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return SymbolRecord(
        file_id=file_id,
        kind="document",
        name=doc_name,
        qualified_name=doc_name,
        signature=signature,
        start_line=1,
        end_line=max(1, line_count),
        content=content,
        doc_comment=doc_comment,
        body_hash=body_h,
        line_count=max(1, line_count),
        parent_symbol_id=None,
    )
