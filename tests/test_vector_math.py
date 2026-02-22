"""Tests for srclight.vector_math — tiered vector backend."""

import math
import struct

import pytest

from srclight.vector_math import cosine_top_k, decode_matrix, get_backend


def _pack_floats(floats: list[float]) -> bytes:
    """Pack a list of floats into a bytes blob."""
    return struct.pack(f"{len(floats)}f", *floats)


def test_backend_detected():
    """Backend should be numpy (or cupy if GPU available), not pure Python."""
    backend = get_backend()
    assert backend in ("numpy", "cupy", "python")
    # numpy is a required dependency, so it should always be available
    assert backend != "python", "numpy should be installed as a required dependency"


def test_decode_matrix_round_trip():
    """Encode floats → bytes → decode_matrix → compare."""
    vecs = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    blobs = [_pack_floats(v) for v in vecs]
    matrix = decode_matrix(blobs, 3)

    # Should recover original values
    for i, expected in enumerate(vecs):
        for j, val in enumerate(expected):
            assert abs(float(matrix[i][j]) - val) < 1e-6


def test_cosine_top_k_basic():
    """Known vectors — verify correct ordering."""
    # query is [1, 0, 0] — should match [1, 0, 0] perfectly, then [1, 1, 0], then [0, 1, 0]
    query = [1.0, 0.0, 0.0]
    vecs = [
        [0.0, 1.0, 0.0],  # orthogonal → sim=0
        [1.0, 1.0, 0.0],  # 45 degrees → sim≈0.707
        [1.0, 0.0, 0.0],  # identical → sim=1.0
    ]
    blobs = [_pack_floats(v) for v in vecs]
    matrix = decode_matrix(blobs, 3)
    results = cosine_top_k(query, matrix, k=3)

    assert len(results) == 3
    # First result should be index 2 (identical vector)
    assert results[0][0] == 2
    assert abs(results[0][1] - 1.0) < 1e-4
    # Second should be index 1 (cos 45°)
    assert results[1][0] == 1
    assert abs(results[1][1] - math.sqrt(2) / 2) < 1e-4
    # Third should be index 0 (orthogonal)
    assert results[2][0] == 0
    assert abs(results[2][1] - 0.0) < 1e-4


def test_cosine_top_k_zero_vector():
    """Zero query vector returns empty."""
    query = [0.0, 0.0, 0.0]
    blobs = [_pack_floats([1.0, 2.0, 3.0])]
    matrix = decode_matrix(blobs, 3)
    results = cosine_top_k(query, matrix, k=5)
    assert results == []


def test_cosine_top_k_k_larger_than_n():
    """k > number of rows should return all rows."""
    query = [1.0, 0.0]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    blobs = [_pack_floats(v) for v in vecs]
    matrix = decode_matrix(blobs, 2)
    results = cosine_top_k(query, matrix, k=100)
    assert len(results) == 2


def test_cosine_top_k_matches_pure_python():
    """Compare numpy/cupy results against pure Python for correctness."""
    import random

    random.seed(42)
    n_dims = 64
    n_vecs = 50
    k = 5

    query = [random.gauss(0, 1) for _ in range(n_dims)]
    vecs = [[random.gauss(0, 1) for _ in range(n_dims)] for _ in range(n_vecs)]

    # Pure Python reference
    q_norm = math.sqrt(sum(x * x for x in query))
    py_scored = []
    for i, row in enumerate(vecs):
        dot = sum(a * b for a, b in zip(query, row))
        e_norm = math.sqrt(sum(x * x for x in row))
        if e_norm == 0:
            continue
        py_scored.append((i, dot / (q_norm * e_norm)))
    py_scored.sort(key=lambda x: x[1], reverse=True)
    py_top_k = py_scored[:k]

    # vector_math implementation
    blobs = [_pack_floats(v) for v in vecs]
    matrix = decode_matrix(blobs, n_dims)
    vm_top_k = cosine_top_k(query, matrix, k)

    # Same indices in same order
    assert [idx for idx, _ in vm_top_k] == [idx for idx, _ in py_top_k]
    # Similarities match within float32 tolerance
    for (_, vm_sim), (_, py_sim) in zip(vm_top_k, py_top_k):
        assert abs(vm_sim - py_sim) < 1e-4, f"Similarity mismatch: {vm_sim} vs {py_sim}"


def test_cosine_top_k_zero_embedding_row():
    """Rows with zero embeddings should be excluded from results."""
    query = [1.0, 0.0, 0.0]
    vecs = [
        [0.0, 0.0, 0.0],  # zero vector — should have sim=0 or be excluded
        [1.0, 0.0, 0.0],  # perfect match
    ]
    blobs = [_pack_floats(v) for v in vecs]
    matrix = decode_matrix(blobs, 3)
    results = cosine_top_k(query, matrix, k=5)

    # The perfect match must be first
    assert results[0][0] == 1
    assert abs(results[0][1] - 1.0) < 1e-4
