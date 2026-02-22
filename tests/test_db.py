"""Tests for the database layer."""

import tempfile
from pathlib import Path

import pytest

from srclight.db import Database, FileRecord, SymbolRecord, EdgeRecord, content_hash


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def test_initialize(db):
    """Database initializes with all tables."""
    stats = db.stats()
    assert stats["files"] == 0
    assert stats["symbols"] == 0
    assert stats["edges"] == 0


def test_upsert_file(db):
    """Can insert and update file records."""
    rec = FileRecord(
        path="src/main.py",
        content_hash="abc123",
        mtime=1000.0,
        language="python",
        size=500,
        line_count=25,
    )
    file_id = db.upsert_file(rec)
    assert file_id > 0

    # Retrieve it
    got = db.get_file("src/main.py")
    assert got is not None
    assert got.content_hash == "abc123"
    assert got.language == "python"

    # Update it
    rec.content_hash = "def456"
    db.upsert_file(rec)
    got = db.get_file("src/main.py")
    assert got.content_hash == "def456"


def test_file_needs_reindex(db):
    """Change detection works via content hash."""
    rec = FileRecord(
        path="src/main.py", content_hash="abc123",
        mtime=1000.0, language="python", size=100, line_count=10,
    )
    db.upsert_file(rec)

    assert not db.file_needs_reindex("src/main.py", "abc123")
    assert db.file_needs_reindex("src/main.py", "different_hash")
    assert db.file_needs_reindex("nonexistent.py", "abc123")


def test_insert_symbol_and_search(db):
    """Can insert symbols and find them via FTS5."""
    # Insert a file first
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc",
        mtime=1000.0, language="python", size=100, line_count=10,
    ))

    # Insert a symbol
    sym = SymbolRecord(
        file_id=file_id,
        kind="function",
        name="calculate_total",
        qualified_name="src/main.py::calculate_total",
        signature="def calculate_total(items: list) -> float",
        start_line=10,
        end_line=20,
        content="def calculate_total(items: list) -> float:\n    return sum(i.price for i in items)",
        doc_comment="Calculate the total price of all items.",
        line_count=11,
    )
    sym_id = db.insert_symbol(sym, "src/main.py")
    assert sym_id > 0

    # Search by name
    results = db.search_symbols("calculate_total")
    assert len(results) > 0
    assert results[0]["name"] == "calculate_total"

    # Search by content (trigram)
    results = db.search_symbols("price")
    assert len(results) > 0

    # Search by doc (porter stemmed)
    results = db.search_symbols("calculating prices")
    assert len(results) > 0


def test_symbols_in_file(db):
    """Can list all symbols in a file."""
    file_id = db.upsert_file(FileRecord(
        path="src/lib.py", content_hash="abc",
        mtime=1000.0, language="python", size=200, line_count=30,
    ))

    for i, name in enumerate(["foo", "bar", "baz"]):
        db.insert_symbol(SymbolRecord(
            file_id=file_id, kind="function", name=name,
            start_line=i * 10 + 1, end_line=i * 10 + 8,
            content=f"def {name}(): pass", line_count=8,
        ), "src/lib.py")

    db.commit()
    syms = db.symbols_in_file("src/lib.py")
    assert len(syms) == 3
    assert [s.name for s in syms] == ["foo", "bar", "baz"]


def test_edges(db):
    """Can insert and query symbol relationships."""
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc",
        mtime=1000.0, language="python", size=100, line_count=10,
    ))

    caller_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="main",
        start_line=1, end_line=5, content="def main(): calc()", line_count=5,
    ), "src/main.py")

    callee_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="calc",
        start_line=10, end_line=15, content="def calc(): pass", line_count=6,
    ), "src/main.py")

    db.insert_edge(EdgeRecord(
        source_id=caller_id, target_id=callee_id, edge_type="calls",
    ))
    db.commit()

    callers = db.get_callers(callee_id)
    assert len(callers) == 1
    assert callers[0]["symbol"].name == "main"
    assert callers[0]["edge_type"] == "calls"

    callees = db.get_callees(caller_id)
    assert len(callees) == 1
    assert callees[0]["symbol"].name == "calc"


def test_content_hash():
    """SHA256 content hashing works."""
    h1 = content_hash(b"hello world")
    h2 = content_hash(b"hello world")
    h3 = content_hash(b"different content")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA256 hex


def test_stats(db):
    """Stats reflect database contents."""
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc",
        mtime=1000.0, language="python", size=100, line_count=10,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="foo",
        start_line=1, end_line=5, content="def foo(): pass", line_count=5,
    ), "src/main.py")

    db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="class", name="Bar",
        start_line=10, end_line=20, content="class Bar: pass", line_count=11,
    ), "src/main.py")

    db.commit()
    stats = db.stats()
    assert stats["files"] == 1
    assert stats["symbols"] == 2
    assert stats["languages"]["python"] == 1
    assert stats["symbol_kinds"]["function"] == 1
    assert stats["symbol_kinds"]["class"] == 1
