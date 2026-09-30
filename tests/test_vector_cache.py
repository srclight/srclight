"""Tests for VectorCache — GPU/CPU-resident embedding matrix with .npy sidecar."""

import json
import os
import struct
import time

import numpy as np
import pytest

from srclight.db import Database, FileRecord, SymbolRecord
from srclight.embeddings import vector_to_bytes
from srclight.vector_cache import VectorCache


# --- Helpers ---


def _make_vec(dims: int, seed: float) -> list[float]:
    """Create a deterministic unit vector."""
    vec = [(seed + i) * 0.1 for i in range(dims)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec


def _setup_db(tmp_path, n_symbols=5, dims=8):
    """Create a test database with symbols and embeddings."""
    db_path = tmp_path / ".srclight" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    db.open()
    db.initialize()

    file_id = db.upsert_file(FileRecord(
        path="test.py", content_hash="abc123", mtime=1.0,
        language="python", size=100, line_count=50,
    ))

    kinds = ["function", "class", "method", "function", "class"]
    for i in range(n_symbols):
        sym_id = db.insert_symbol(SymbolRecord(
            file_id=file_id, kind=kinds[i % len(kinds)],
            name=f"symbol_{i}", start_line=i * 10 + 1,
            end_line=i * 10 + 10, content=f"def symbol_{i}(): pass",
            body_hash=f"h{i}",
        ), "test.py")

        vec = _make_vec(dims, seed=float(i))
        db.upsert_embedding(sym_id, "mock:test", dims, vector_to_bytes(vec), f"h{i}")

    db.commit()
    return db, db_path


# --- Tests ---


def test_build_from_db(tmp_path):
    """Build sidecar from DB and verify .npy files are created."""
    db, db_path = _setup_db(tmp_path)

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    assert cache.npy_path.exists()
    assert cache.norms_path.exists()
    assert cache.meta_path.exists()

    # Verify meta content
    meta = json.loads(cache.meta_path.read_text())
    assert meta["row_count"] == 5
    assert meta["dimensions"] == 8
    assert meta["model"] == "mock:test"
    assert len(meta["symbol_ids"]) == 5
    assert len(meta["symbol_kinds"]) == 5
    assert len(meta["file_paths"]) == 5

    # Verify numpy files
    matrix = np.load(cache.npy_path)
    assert matrix.shape == (5, 8)
    norms = np.load(cache.norms_path)
    assert norms.shape == (5,)

    assert cache.is_loaded()
    db.close()


def test_load_sidecar(tmp_path):
    """Build sidecar, clear cache, reload from disk."""
    db, db_path = _setup_db(tmp_path)

    # Build
    cache1 = VectorCache(db_path.parent)
    cache1.build_from_db(db.conn)
    assert cache1.is_loaded()

    # Load fresh from disk
    cache2 = VectorCache(db_path.parent)
    assert not cache2.is_loaded()
    cache2.load_sidecar()
    assert cache2.is_loaded()
    assert cache2._dimensions == 8
    assert len(cache2._symbol_ids) == 5

    db.close()


def test_search_basic(tmp_path):
    """Search with a known query vector and verify ordering."""
    db, db_path = _setup_db(tmp_path, n_symbols=5, dims=8)

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    # Query with vector identical to symbol_0 — should be top result
    query_vec = _make_vec(8, seed=0.0)
    query_bytes = vector_to_bytes(query_vec)

    results = cache.search(query_bytes, 8, limit=3)
    assert len(results) == 3

    # First result should be symbol_0 (exact match, similarity ~1.0)
    row_idx, sim, sym_id = results[0]
    assert sim == pytest.approx(1.0, abs=0.01)
    # symbol_0 is the first inserted, so sym_id=1
    assert sym_id == 1

    # Results should be in descending similarity order
    sims = [r[1] for r in results]
    assert sims == sorted(sims, reverse=True)

    db.close()


def test_search_with_kind_filter(tmp_path):
    """Filter by kind and verify only matching symbols returned."""
    db, db_path = _setup_db(tmp_path, n_symbols=5, dims=8)

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    query_vec = _make_vec(8, seed=0.0)
    query_bytes = vector_to_bytes(query_vec)

    # Search only classes (symbols 1 and 4 have kind="class")
    results = cache.search(query_bytes, 8, limit=10, kind="class")
    assert len(results) == 2

    # All results should be classes
    for row_idx, sim, sym_id in results:
        assert cache._symbol_kinds[row_idx] == "class"

    db.close()


def test_is_valid_detects_stale(tmp_path):
    """Bump version in DB, verify is_valid() returns False."""
    db, db_path = _setup_db(tmp_path)

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)
    assert cache.is_valid(db.conn)

    # Bump the version manually (simulates a new embedding being inserted)
    db.conn.execute(
        "UPDATE schema_info SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
        "WHERE key = 'embedding_cache_version'"
    )
    db.conn.commit()

    assert not cache.is_valid(db.conn)

    db.close()


def test_invalidate_clears_cache(tmp_path):
    """Call invalidate and verify is_loaded() returns False."""
    db, db_path = _setup_db(tmp_path)

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)
    assert cache.is_loaded()

    cache.invalidate()
    assert not cache.is_loaded()

    db.close()


def test_fallback_when_no_sidecar(tmp_path):
    """No .npy files — verify graceful behavior."""
    srclight_dir = tmp_path / ".srclight"
    srclight_dir.mkdir(parents=True, exist_ok=True)

    cache = VectorCache(srclight_dir)
    assert not cache.sidecar_exists()
    assert not cache.is_loaded()

    # Search on unloaded cache should return empty
    query_bytes = vector_to_bytes([1.0, 0.0, 0.0, 0.0])
    results = cache.search(query_bytes, 4, limit=5)
    assert results == []


def test_db_vector_search_with_cache(tmp_path):
    """Test the fast path in db.vector_search() using VectorCache."""
    db, db_path = _setup_db(tmp_path, n_symbols=5, dims=8)

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    query_vec = _make_vec(8, seed=0.0)
    query_bytes = vector_to_bytes(query_vec)

    # Fast path (with cache)
    results_fast = db.vector_search(query_bytes, 8, limit=3, cache=cache)
    assert len(results_fast) == 3
    assert results_fast[0]["name"] == "symbol_0"
    assert results_fast[0]["similarity"] == pytest.approx(1.0, abs=0.01)

    # Slow path (without cache) — same results
    results_slow = db.vector_search(query_bytes, 8, limit=3, cache=None)
    assert len(results_slow) == 3
    assert results_slow[0]["name"] == "symbol_0"

    # Both should return the same top result
    assert results_fast[0]["symbol_id"] == results_slow[0]["symbol_id"]

    db.close()


def test_upsert_embedding_bumps_version(tmp_path):
    """Verify that upsert_embedding increments embedding_cache_version."""
    db, db_path = _setup_db(tmp_path, n_symbols=1, dims=4)

    # Check current version
    row = db.conn.execute(
        "SELECT value FROM schema_info WHERE key='embedding_cache_version'"
    ).fetchone()
    v1 = int(row["value"])

    # Upsert another embedding — version should bump
    vec = _make_vec(4, seed=99.0)
    db.upsert_embedding(1, "mock:test", 4, vector_to_bytes(vec), "hx")
    db.commit()

    row = db.conn.execute(
        "SELECT value FROM schema_info WHERE key='embedding_cache_version'"
    ).fetchone()
    v2 = int(row["value"])
    assert v2 > v1

    db.close()


def test_sidecar_exists(tmp_path):
    """Test sidecar_exists() with and without files."""
    db, db_path = _setup_db(tmp_path)

    cache = VectorCache(db_path.parent)
    assert not cache.sidecar_exists()

    cache.build_from_db(db.conn)
    assert cache.sidecar_exists()

    db.close()


def test_empty_db_build(tmp_path):
    """Building sidecar from an empty DB should be a no-op."""
    db_path = tmp_path / ".srclight" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    db.open()
    db.initialize()
    db.commit()

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    # No files should be created, cache not loaded
    assert not cache.npy_path.exists()
    assert not cache.is_loaded()

    db.close()


def test_rebuild_replaces_the_file_instead_of_truncating_it(tmp_path):
    """A rebuild must swap the inode, not rewrite bytes under a live mmap.

    The server keeps every sidecar mmap'd for its whole life (load_sidecar uses
    mmap_mode="r"). np.save() opens its target "wb" — O_TRUNC on the same inode
    — so `srclight index --embed` running in a second process rewrites pages
    beneath the running server's mapping. Writing a temp file and os.replace()ing
    it leaves the old inode intact for anyone still holding it.
    """
    db, db_path = _setup_db(tmp_path, n_symbols=5, dims=8)
    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    live = np.load(cache.npy_path, mmap_mode="r")
    before = np.array(live)

    # Re-embed every symbol with different vectors — same row count, new bytes.
    sym_ids = [r["symbol_id"] for r in db.conn.execute(
        "SELECT symbol_id FROM symbol_embeddings ORDER BY symbol_id")]
    for i, sid in enumerate(sym_ids):
        db.upsert_embedding(
            sid, "mock:test", 8, vector_to_bytes(_make_vec(8, seed=float(i + 100))), f"r{i}"
        )
    db.commit()
    VectorCache(db_path.parent).build_from_db(db.conn)

    # Guard against a vacuous assertion: the rebuild must really differ on disk.
    on_disk = np.array(np.load(cache.npy_path, mmap_mode="r"))
    assert not np.array_equal(on_disk, before), "rebuild produced identical bytes"

    assert np.array_equal(np.array(live), before), (
        "the rebuild rewrote bytes under a live mmap — sidecar was truncated in "
        "place rather than replaced atomically"
    )


def test_load_refuses_a_sidecar_whose_meta_disagrees_with_the_matrix(tmp_path):
    """A torn sidecar must fail loudly, never serve the wrong symbol.

    The three sidecar files are replaced one at a time, so a kill between the
    renames can leave a new matrix beside an older symbol_ids list. search()
    then evaluates self._symbol_ids[orig_idx]: when the stale list is longer the
    index is in range and the caller gets a real-but-wrong symbol carrying a
    plausible similarity score. Wrong answers must not be reachable.
    """
    db, db_path = _setup_db(tmp_path, n_symbols=5, dims=8)
    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)

    meta = json.loads(cache.meta_path.read_text())
    meta["symbol_ids"] = meta["symbol_ids"] + [999999]  # stale, one row too long
    meta["row_count"] = len(meta["symbol_ids"])
    cache.meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="sidecar"):
        VectorCache(db_path.parent).load_sidecar()


# --- Reusing the vectors of the sidecar being replaced ---


def _sidecar(cache):
    meta = json.loads(cache.meta_path.read_text())
    meta.pop("version")
    meta.pop("files", None)  # differs between any two builds, by design
    return np.load(cache.npy_path), np.load(cache.norms_path), meta


def _build_traced(db, directory):
    """Build a sidecar in `directory`; return the ids whose blobs were read."""
    statements = []
    real = VectorCache.__dict__["_snapshot_connection"]

    def traced(conn):
        own = real.__func__(conn)
        own.set_trace_callback(statements.append)
        return own

    VectorCache._snapshot_connection = staticmethod(traced)
    try:
        VectorCache(directory).build_from_db(db.conn)
    finally:
        VectorCache._snapshot_connection = real
    blob_reads = [s for s in statements if "SELECT symbol_id, embedding" in s]
    read = set()
    for s in blob_reads:
        if "WHERE symbol_id IN" not in s:
            return blob_reads, "all"
        read |= {int(x) for x in s.split("IN (", 1)[1].rstrip(")").split(",")}
    return blob_reads, read


def _change_some(db, dims=8):
    """Re-embed one symbol, delete another, add a third."""
    time.sleep(0.01)  # embedded_at has millisecond resolution
    ids = [r["symbol_id"] for r in db.conn.execute(
        "SELECT symbol_id FROM symbol_embeddings ORDER BY symbol_id")]
    db.upsert_embedding(ids[1], "mock:test", dims,
                        vector_to_bytes(_make_vec(dims, seed=42.0)), "changed")
    db.conn.execute("DELETE FROM symbols WHERE id = ?", (ids[2],))
    file_id = db.conn.execute("SELECT id FROM files").fetchone()[0]
    new_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="added", start_line=90, end_line=95,
        content="def added(): pass", body_hash="added"), "test.py")
    db.upsert_embedding(new_id, "mock:test", dims,
                        vector_to_bytes(_make_vec(dims, seed=7.0)), "added")
    db.commit()
    return {ids[1], new_id}


def test_a_rebuild_reads_only_the_vectors_that_changed(tmp_path):
    """The blobs are most of a large index and nearly all already in the
    sidecar: only new and re-embedded vectors are read again, and the result
    is the sidecar a full read builds."""
    db, db_path = _setup_db(tmp_path, n_symbols=6)
    VectorCache(db_path.parent).build_from_db(db.conn)
    changed = _change_some(db)

    _, read = _build_traced(db, db_path.parent)
    assert read == changed

    fresh = tmp_path / "fresh"
    _, read_all = _build_traced(db, fresh)
    assert read_all == "all"
    incremental, full = _sidecar(VectorCache(db_path.parent)), _sidecar(VectorCache(fresh))
    assert np.array_equal(incremental[0], full[0])
    assert np.array_equal(incremental[1], full[1])
    assert incremental[2] == full[2]
    db.close()


def test_a_symbol_id_reused_after_a_deletion_is_read_again(tmp_path):
    """SQLite reuses the rowid of a deleted symbol: the id alone would hand
    the new symbol the old one's vector. Its embedding row is new, and so is
    its stamp."""
    db, db_path = _setup_db(tmp_path, n_symbols=3)
    VectorCache(db_path.parent).build_from_db(db.conn)
    last = db.conn.execute("SELECT max(id) FROM symbols").fetchone()[0]
    db._delete_symbol_fts(last)
    db.conn.execute("DELETE FROM symbols WHERE id = ?", (last,))
    time.sleep(0.01)
    file_id = db.conn.execute("SELECT id FROM files").fetchone()[0]
    reused = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="other", start_line=70, end_line=75,
        content="def other(): pass", body_hash="other"), "test.py")
    assert reused == last, "the test needs SQLite to reuse the rowid"
    db.upsert_embedding(reused, "mock:test", 8,
                        vector_to_bytes(_make_vec(8, seed=99.0)), "other")
    db.commit()

    _, read = _build_traced(db, db_path.parent)
    assert read == {reused}
    matrix, _, meta = _sidecar(VectorCache(db_path.parent))
    row = meta["symbol_ids"].index(reused)
    assert np.allclose(matrix[row], _make_vec(8, seed=99.0))
    db.close()


@pytest.mark.parametrize("spoil", ["no_stamps", "other_model", "torn_matrix", "garbage_meta",
                                   "matrix_replaced"])
def test_a_sidecar_that_cannot_be_trusted_is_not_reused(tmp_path, spoil):
    db, db_path = _setup_db(tmp_path, n_symbols=5)
    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)
    meta = json.loads(cache.meta_path.read_text())
    if spoil == "no_stamps":  # written before the stamps were recorded
        del meta["embedded_at"]
        cache.meta_path.write_text(json.dumps(meta))
    elif spoil == "other_model":
        meta["model"] = "mock:other"
        cache.meta_path.write_text(json.dumps(meta))
    elif spoil == "torn_matrix":
        np.save(cache.npy_path, np.zeros((3, 8), dtype=np.float32))
    elif spoil == "matrix_replaced":  # same shape, another build's rows
        time.sleep(0.01)
        np.save(cache.npy_path, np.load(cache.npy_path)[::-1].copy())
    else:
        cache.meta_path.write_text("{not json")
    _change_some(db)

    _, read = _build_traced(db, db_path.parent)
    assert read == "all"
    fresh = tmp_path / "fresh"
    VectorCache(fresh).build_from_db(db.conn)
    rebuilt, full = _sidecar(VectorCache(db_path.parent)), _sidecar(VectorCache(fresh))
    assert np.array_equal(rebuilt[0], full[0]) and rebuilt[2] == full[2]
    db.close()


def test_the_stamps_are_read_without_the_blobs(tmp_path):
    """embedded_at is stored after the embedding: read from the table, it
    walks every blob. The index lets the rebuild list what it holds from
    the index alone."""
    db, _ = _setup_db(tmp_path)
    plan = " ".join(r[3] for r in db.conn.execute("""EXPLAIN QUERY PLAN
        SELECT e.symbol_id, e.model, e.dimensions, e.embedded_at, s.kind, f.path as file_path
        FROM symbol_embeddings e JOIN symbols s ON e.symbol_id = s.id
        JOIN files f ON s.file_id = f.id"""))
    assert "COVERING INDEX idx_symbol_embeddings_stamp" in plan
    db.close()


def test_a_build_killed_half_way_is_not_reused(tmp_path, monkeypatch):
    """The files are replaced one at a time. A new matrix beside the previous
    build's meta — same row count, rows moved — would hand symbols each
    other's vectors on every later build. The meta names the exact files it
    describes, and this matrix is not one of them."""
    db, db_path = _setup_db(tmp_path, n_symbols=5)
    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)
    _change_some(db)

    real = VectorCache._atomic_write

    def killed_after_the_matrix(path, write):
        if path.name == "embeddings_norms.npy":
            raise KeyboardInterrupt("killed")
        return real(path, write)

    monkeypatch.setattr(VectorCache, "_atomic_write", staticmethod(killed_after_the_matrix))
    with pytest.raises(KeyboardInterrupt):
        VectorCache(db_path.parent).build_from_db(db.conn)
    monkeypatch.setattr(VectorCache, "_atomic_write", staticmethod(real))

    _, read = _build_traced(db, db_path.parent)
    assert read == "all"
    fresh = tmp_path / "fresh"
    VectorCache(fresh).build_from_db(db.conn)
    rebuilt, full = _sidecar(VectorCache(db_path.parent)), _sidecar(VectorCache(fresh))
    assert np.array_equal(rebuilt[0], full[0]) and rebuilt[2] == full[2]
    db.close()


def test_a_deletion_committed_during_a_build_does_not_break_it(tmp_path, monkeypatch):
    """The vectors are listed, then read, in separate statements. A reindex
    in another process deleting symbols between the two made the build fail;
    one read transaction reads both from the same snapshot."""
    import sqlite3

    db, db_path = _setup_db(tmp_path, n_symbols=5)
    ids = [r[0] for r in db.conn.execute("SELECT symbol_id FROM symbol_embeddings")]
    real = VectorCache._reuse_previous

    def meanwhile_elsewhere(self, *args, **kwargs):
        other = sqlite3.connect(db_path)
        other.execute("DELETE FROM symbol_embeddings WHERE symbol_id = ?", (ids[0],))
        other.commit()
        other.close()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(VectorCache, "_reuse_previous", meanwhile_elsewhere)
    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)
    assert json.loads(cache.meta_path.read_text())["symbol_ids"] == sorted(ids)
    db.close()


def test_a_replace_that_fails_leaves_the_sidecar_reusable(tmp_path, monkeypatch):
    """On Windows a reader holding the matrix mapped makes os.replace fail.
    The build stops there, the three files are still the previous build's,
    and the next build reuses them."""
    db, db_path = _setup_db(tmp_path, n_symbols=5)
    VectorCache(db_path.parent).build_from_db(db.conn)
    changed = _change_some(db)

    real = VectorCache._atomic_write

    def mapped_elsewhere(path, write):
        if path.name == "embeddings.npy":
            raise PermissionError("the file is mapped by another process")
        return real(path, write)

    monkeypatch.setattr(VectorCache, "_atomic_write", staticmethod(mapped_elsewhere))
    with pytest.raises(PermissionError):
        VectorCache(db_path.parent).build_from_db(db.conn)
    monkeypatch.setattr(VectorCache, "_atomic_write", staticmethod(real))

    assert VectorCache(db_path.parent).sidecar_exists()
    _, read = _build_traced(db, db_path.parent)
    assert read == changed
    db.close()


def test_a_build_leaves_the_callers_connection_alone(tmp_path, monkeypatch):
    """The MCP server shares one connection between its threads, and its
    reindex writes through it. A transaction the build opened there took in
    those writes and committed them half done; the reindex could no longer
    roll them back."""
    db, db_path = _setup_db(tmp_path, n_symbols=5)
    assert not db.conn.in_transaction
    real = VectorCache._reuse_previous

    def a_reindex_writes_meanwhile(self, *args, **kwargs):
        db.conn.execute("INSERT INTO schema_info (key, value) VALUES ('half_done', '1')")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(VectorCache, "_reuse_previous", a_reindex_writes_meanwhile)
    VectorCache(db_path.parent).build_from_db(db.conn)
    assert db.conn.in_transaction, "the write's own transaction was ended by the build"
    db.conn.rollback()
    left = db.conn.execute("SELECT count(*) FROM schema_info WHERE key = 'half_done'").fetchone()[0]
    assert left == 0
    db.close()


def test_a_caller_with_uncommitted_writes_gets_a_sidecar_it_sees_as_current(tmp_path):
    """The MCP server's reindex bumps the cache version in a transaction it
    has not committed yet, on the connection the server checks the cache
    against. A sidecar read from the committed database carried the older
    version: stale again at once, rebuilt on every search until the commit."""
    db, db_path = _setup_db(tmp_path, n_symbols=5)
    VectorCache(db_path.parent).build_from_db(db.conn)
    time.sleep(0.01)
    sid = db.conn.execute("SELECT min(symbol_id) FROM symbol_embeddings").fetchone()[0]
    db.upsert_embedding(sid, "mock:test", 8, vector_to_bytes(_make_vec(8, seed=5.0)), "x")
    assert db.conn.in_transaction  # not committed

    cache = VectorCache(db_path.parent)
    cache.build_from_db(db.conn)
    assert cache.is_valid(db.conn)
    matrix, _, meta = _sidecar(cache)
    assert np.allclose(matrix[meta["symbol_ids"].index(sid)], _make_vec(8, seed=5.0))
    db.conn.rollback()
    db.close()


def test_the_meta_names_the_files_this_build_wrote(tmp_path, monkeypatch):
    """Two builds at once (a git hook's index and the server's rebuild)
    replace the same paths. Fingerprinted by path after the fact, this
    build's meta could name the other build's files and pass the check on a
    mismatched set; taken from the written file itself, it cannot.

    The other build's matrix has the same size and, as on a filesystem with
    a coarse clock (ext4 under WSL2), the same mtime: only its inode tells
    it apart."""
    db, db_path = _setup_db(tmp_path, n_symbols=5)
    real = VectorCache._atomic_write
    other = tmp_path / "other.npy"
    np.save(other, np.ones((5, 8), dtype=np.float32))

    def another_build_lands_in_between(path, write):
        written = real(path, write)
        if path.name == "embeddings_norms.npy":
            mine = (db_path.parent / "embeddings.npy").stat()
            assert other.stat().st_size == mine.st_size
            os.utime(other, ns=(mine.st_atime_ns, mine.st_mtime_ns))
            os.replace(other, db_path.parent / "embeddings.npy")
        return written

    monkeypatch.setattr(VectorCache, "_atomic_write", staticmethod(another_build_lands_in_between))
    VectorCache(db_path.parent).build_from_db(db.conn)
    monkeypatch.setattr(VectorCache, "_atomic_write", staticmethod(real))

    _change_some(db)
    _, read = _build_traced(db, db_path.parent)
    assert read == "all"
    db.close()
