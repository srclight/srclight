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


def _symbols_from_main_file_only(src: Path, dest_dir: Path) -> int:
    """Copy ONLY index.db (no -wal/-shm) and count symbols readable from it.

    This is what a backup tool, an rsync of '*.db', or a user tidying up the
    odd-looking sidecars actually captures.
    """
    import shutil
    import sqlite3
    dest = dest_dir / "index.db"
    shutil.copyfile(src, dest)
    conn = sqlite3.connect(dest)
    try:
        return conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    finally:
        conn.close()


def test_checkpoint_makes_the_db_self_contained(tmp_path):
    """After checkpoint, index.db alone must carry the data.

    In WAL mode the main file holds only a 4096-byte header while a connection
    is open — every row lives in index.db-wal. srclight never checkpointed, so
    an index.db separated from its sidecar is an empty database with 0 tables
    (issue #16: "I keep finding 0-length index.db files").
    """
    db_path = tmp_path / "index.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    fid = db.upsert_file(FileRecord(path="a.py", content_hash="h", mtime=1.0,
                                    language="python", size=10, line_count=2))
    db.insert_symbol(SymbolRecord(file_id=fid, kind="function", name="f",
                                  start_line=1, end_line=2, content="def f(): pass",
                                  body_hash="b"), "a.py")
    db.commit()

    db.checkpoint()

    copied = tmp_path / "copied"
    copied.mkdir()
    assert _symbols_from_main_file_only(db_path, copied) == 1
    db.close()


def test_checkpoint_works_while_another_connection_holds_the_db(tmp_path):
    """The issue-#16 scenario: indexing while the MCP server holds the index open.

    A clean close already checkpoints, but the indexer's close is NOT the last
    connection when the server is running — so nothing moved the WAL into the
    main file, and the user saw a 4096-byte index.db beside a large -wal and
    read it as corruption. The checkpoint must succeed with a reader attached.
    """
    db_path = tmp_path / "index.db"
    server = Database(db_path)          # the long-running MCP server
    server.open()
    server.initialize()
    server.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()

    indexer = Database(db_path)         # the CLI indexer, second connection
    indexer.open()
    fid = indexer.upsert_file(FileRecord(path="a.py", content_hash="h", mtime=1.0,
                                         language="python", size=10, line_count=2))
    indexer.insert_symbol(SymbolRecord(file_id=fid, kind="function", name="f",
                                       start_line=1, end_line=2, content="x",
                                       body_hash="b"), "a.py")
    indexer.commit()
    indexer.checkpoint()

    copied = tmp_path / "copied"
    copied.mkdir()
    assert _symbols_from_main_file_only(db_path, copied) == 1
    indexer.close()
    server.close()


def test_checkpoint_failure_is_not_fatal(tmp_path):
    """A checkpoint that cannot run must not take the caller down with it.

    checkpoint() runs inside close(), so an exception here would turn an
    ordinary shutdown into a crash — and its own handler must not be the thing
    that raises.
    """
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    db.conn.close()          # pull the connection out from under it

    assert db.checkpoint() is None


def test_a_read_only_session_does_not_checkpoint_someone_elses_wal(tmp_path):
    """Opening and closing a Database to READ must not rewrite the index.

    v0.22.2 put `PRAGMA wal_checkpoint(TRUNCATE)` in Database.close(), and
    server.py opens/closes a Database in 17 places — get_callers, get_callees,
    find_dead_code and friends. Measured: one `SELECT count(*)` through that
    path changed the main file's md5 and truncated a 4,152-byte WAL to zero.
    The same session named that behaviour a defect two hours later and fixed it
    in one place only. Checkpointing belongs where writes happen: the indexer
    and the deliberate shutdown path.
    """
    import hashlib
    import subprocess
    import sys

    db_path = tmp_path / "index.db"
    Database(db_path).__enter__().initialize()

    # leave a hot WAL: a writer that exits without closing
    subprocess.run(
        [sys.executable, "-c",
         "import sqlite3,os,sys\n"
         "c=sqlite3.connect(sys.argv[1])\n"
         "c.execute('PRAGMA journal_mode=WAL')\n"
         "c.execute('CREATE TABLE IF NOT EXISTS scratch(x)')\n"
         "c.execute('INSERT INTO scratch VALUES (1)')\n"
         "c.commit()\n"
         "os._exit(0)\n", str(db_path)],
        check=True,
    )
    wal = db_path.with_name("index.db-wal")
    assert wal.exists() and wal.stat().st_size > 0, "fixture left no hot WAL"
    before = (hashlib.md5(db_path.read_bytes()).hexdigest(), wal.stat().st_size)

    reader = Database(db_path)
    reader.open()
    reader.conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    reader.close()

    after = (hashlib.md5(db_path.read_bytes()).hexdigest(),
             wal.stat().st_size if wal.exists() else 0)
    assert after == before, f"a read-only session rewrote the index: {before} -> {after}"
