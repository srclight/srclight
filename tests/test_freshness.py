# tests/test_freshness.py
"""Freshness = does the index still describe what is on disk.

The statuses are facts, not judgements: FRESH (index matches disk), STALE
(disk changed since indexing), MISSING (indexed file no longer on disk),
NOT_INDEXED (path was never indexed). The mtime fast path must not produce
false STALE: a touch that does not change bytes is still FRESH.
"""
import os
from pathlib import Path

import pytest

from srclight.db import Database, FileRecord, content_hash
from srclight.freshness import FRESH, MISSING, NOT_INDEXED, STALE, file_freshness


@pytest.fixture
def repo(tmp_path):
    """A tiny on-disk repo plus an index that matches it."""
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    src = tmp_path / "src"
    src.mkdir()
    f = src / "a.py"
    f.write_text("def a(): pass\n")
    raw = f.read_bytes()
    db.upsert_file(FileRecord(
        path="src/a.py", content_hash=content_hash(raw),
        mtime=f.stat().st_mtime, language="python",
        size=len(raw), line_count=2,
    ))
    db.commit()
    yield db, tmp_path
    db.close()


def test_unchanged_file_is_fresh(repo):
    db, root = repo
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": FRESH}


def test_touched_but_identical_file_is_fresh_via_hash_fallback(repo):
    """mtime moved, bytes did not: the fast path misses, the hash decides."""
    db, root = repo
    f = root / "src" / "a.py"
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 100))
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": FRESH}


def test_edited_file_is_stale(repo):
    db, root = repo
    (root / "src" / "a.py").write_text("def a(): pass  # edited\n")
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": STALE}


def test_deleted_file_is_missing(repo):
    db, root = repo
    (root / "src" / "a.py").unlink()
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": MISSING}


def test_unknown_path_is_not_indexed(repo):
    db, root = repo
    assert file_freshness(db, root, ["src/ghost.py"]) == {"src/ghost.py": NOT_INDEXED}


def test_mtime_fast_path_does_not_read_the_file(repo, monkeypatch):
    """When mtimes match, the file body must not be read (cheapness is the point)."""
    db, root = repo
    def boom(self):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("read_bytes called on the fast path")
    monkeypatch.setattr(Path, "read_bytes", boom)
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": FRESH}
