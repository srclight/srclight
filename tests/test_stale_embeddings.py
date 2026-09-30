"""Which symbols need embedding again, found without reading the blobs."""
import pytest

from srclight.db import Database, FileRecord, SymbolRecord
from srclight.embeddings import vector_to_bytes

MODEL = "mock:test"


def _reference(db, model, limit):
    """The single query this replaced, kept as the oracle."""
    rows = db.conn.execute(
        """SELECT s.id, s.name, s.qualified_name, s.signature, s.doc_comment,
                  s.content, s.body_hash, s.kind, f.path as file_path
           FROM symbols s
           JOIN files f ON s.file_id = f.id
           LEFT JOIN symbol_embeddings e ON s.id = e.symbol_id AND e.model = ?
           WHERE e.symbol_id IS NULL OR e.body_hash != s.body_hash
           LIMIT ?""",
        (model, limit),
    ).fetchall()
    return [{k: row[k] for k in row.keys()} for row in rows]


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    file_id = db.upsert_file(FileRecord(path="a.py", content_hash="x", mtime=1.0,
                                        language="python", size=10, line_count=10))
    vec = vector_to_bytes([0.1, 0.2, 0.3])
    cases = [
        # (symbol body_hash, embedding: None for none, else (model, body_hash))
        ("h0", None),                      # never embedded
        ("h1", (MODEL, "h1")),             # current
        ("h2", (MODEL, "old")),            # body changed since
        ("h3", ("mock:other", "h3")),      # embedded by another model only
        (None, None),                      # no hash, never embedded
        (None, (MODEL, "h5")),             # symbol without a hash
        ("h6", (MODEL, None)),             # embedding without a hash
        (None, (MODEL, None)),             # neither has one
        ("h8", (MODEL, "h8")),             # current
        ("h9", (MODEL, "stale")),          # body changed since
    ]
    for i, (body_hash, embedding) in enumerate(cases):
        sid = db.insert_symbol(SymbolRecord(
            file_id=file_id, kind="function", name=f"fn{i}", start_line=i * 5 + 1,
            end_line=i * 5 + 3, content=f"def fn{i}(): pass", body_hash=body_hash), "a.py")
        if embedding is not None:
            db.upsert_embedding(sid, embedding[0], 3, vec, embedding[1])
    db.commit()
    yield db
    db.close()


@pytest.mark.parametrize("limit", [100000, 3, 1, 0])
def test_the_same_symbols_as_the_join_it_replaced(db, limit):
    got = db.get_symbols_needing_embeddings(MODEL, limit=limit)
    assert got == _reference(db, MODEL, limit)
    if limit >= 3:
        assert [s["name"] for s in got][:3] == ["fn0", "fn2", "fn3"]


def test_another_model_needs_everything_it_has_not_embedded(db):
    assert db.get_symbols_needing_embeddings("mock:other") == _reference(db, "mock:other", 100000)


def _large_index_statistics(db):
    """Planner statistics of an index of ~200k symbols, all embedded."""
    db.conn.execute("ANALYZE")
    db.conn.execute(
        "DELETE FROM sqlite_stat1 WHERE tbl IN ('symbols', 'symbol_embeddings', 'files')")
    db.conn.executemany("INSERT INTO sqlite_stat1 (tbl, idx, stat) VALUES (?, ?, ?)", [
        ("files", None, "20000"),
        ("files", "idx_files_language", "20000 2000"),
        ("files", "idx_files_hash", "20000 1"),
        ("symbols", None, "200000"),
        ("symbols", "idx_symbols_body_hash", "200000 1 1"),
        ("symbol_embeddings", None, "200000"),
        ("symbol_embeddings", "idx_symbol_embeddings_hash", "200000 1 1 1"),
        ("symbol_embeddings", "idx_symbol_embeddings_stamp", "200000 1 1 1"),
    ])
    db.commit()
    db.conn.execute("ANALYZE sqlite_schema")  # reload them


@pytest.mark.parametrize("statistics", [False, True])
def test_the_hashes_are_read_from_indexes_not_from_the_tables(db, statistics):
    """body_hash sits after the embedding blob and after the symbol's
    content: read from either table, it walks all of them on every run.
    Whatever the planner knows about the tables, the symbols' index is the
    one loop, and each embedding is one lookup in its own index."""
    if statistics:
        _large_index_statistics(db)
    statements = []
    db.conn.set_trace_callback(statements.append)
    db.get_symbols_needing_embeddings(MODEL)
    db.conn.set_trace_callback(None)
    listing = next(s for s in statements if "FROM symbol_embeddings e" in s)
    plan = [r[3] for r in db.conn.execute(
        "EXPLAIN QUERY PLAN " + listing.replace(f"'{MODEL}'", "?"), (MODEL, MODEL))]
    scans = [p for p in plan if p.startswith("SCAN")]
    assert scans == ["SCAN s USING COVERING INDEX idx_symbols_body_hash"], plan
    lookups = [p for p in plan if "symbol_embeddings" in p or p.startswith("SEARCH e")]
    assert lookups and all("COVERING INDEX idx_symbol_embeddings_hash" in p for p in lookups), plan
    assert not any("BLOOM" in p for p in plan), plan


def test_a_database_without_the_indexes_is_still_answered(db):
    """The MCP server's reindex opens an existing database without running
    initialize(): the indexes may not be there, and naming them would fail."""
    expected = _reference(db, MODEL, 100000)
    db.conn.execute("DROP INDEX idx_symbol_embeddings_hash")
    assert db.get_symbols_needing_embeddings(MODEL) == expected
