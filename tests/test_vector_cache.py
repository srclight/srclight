"""Tests for VectorCache — GPU/CPU-resident embedding matrix with .npy sidecar."""

import json
import struct

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
