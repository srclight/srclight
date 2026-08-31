# src/srclight/freshness.py
"""Does the index still describe what is on disk?

WHY THIS EXISTS. An index that cannot say it is fresh gets grepped around: on a
repo under active edit, an agent falls back to grep because a silently stale
answer is worse than a slow one (canes-fideles grain-0393, 2026-08-30 — the
same stale-server lesson mcpkit closed for MCP daemons, applied to the index).

READ-ONLY BY DESIGN. Checking freshness never mutates the index — no hash
refresh, no mtime touch-up. A checker that writes is an indexer.

CHEAP BY DESIGN. stat() mtime equality is the fast path (no file read). Only a
moved mtime pays for a read + sha256, and a touch that changed no bytes still
reports FRESH — mtime is a hint, the content hash is the fact.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .db import content_hash

if TYPE_CHECKING:
    from .db import Database

FRESH = "fresh"
STALE = "stale"
MISSING = "missing_on_disk"
NOT_INDEXED = "not_indexed"

__all__ = ["FRESH", "STALE", "MISSING", "NOT_INDEXED", "file_freshness"]


def file_freshness(db: "Database", repo_root: Path, rel_paths: Iterable[str]) -> dict[str, str]:
    """Status per relative path: FRESH / STALE / MISSING / NOT_INDEXED."""
    out: dict[str, str] = {}
    for rel in rel_paths:
        rec = db.get_file(rel)
        if rec is None:
            out[rel] = NOT_INDEXED
            continue
        disk = repo_root / rel
        try:
            st = disk.stat()
        except OSError:
            out[rel] = MISSING
            continue
        # Fast path needs mtime AND size to match. mtime alone misses an edit
        # landing in the same mtime tick as the indexing run — precisely the
        # repo-under-active-edit case this module exists for (caught by
        # test_edited_file_is_stale, 2026-08-30). A same-tick same-length edit
        # can still slip the fast path; the hash fallback catches it whenever
        # either stat field moves.
        if st.st_mtime == rec.mtime and st.st_size == rec.size:
            out[rel] = FRESH          # fast path: no read
            continue
        try:
            raw = disk.read_bytes()
        except OSError:
            out[rel] = MISSING
            continue
        out[rel] = FRESH if content_hash(raw) == rec.content_hash else STALE
    return out
