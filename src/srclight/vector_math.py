"""Vectorized cosine similarity with tiered backend: cupy → numpy → pure Python."""

import logging
import struct

logger = logging.getLogger(__name__)

# --- Backend detection (once at import time) ---
_backend = "python"
_np = None

try:
    import cupy as _np

    # Test that GPU is actually available
    _np.zeros(1)
    _backend = "cupy"
    logger.info("vector_math: using cupy (GPU)")
except Exception:
    try:
        import numpy as _np

        _backend = "numpy"
        logger.info("vector_math: using numpy (CPU)")
    except ImportError:
        _np = None
        logger.info("vector_math: using pure Python fallback")


def get_backend() -> str:
    """Return current backend name: 'cupy', 'numpy', or 'python'."""
    return _backend


def decode_matrix(blob_rows: list[bytes], n_floats: int):
    """Decode list of embedding BLOBs into a 2D matrix.

    Returns numpy/cupy ndarray (N, n_floats) or list-of-lists for pure Python.
    """
    if _np is not None:
        buf = b"".join(blob_rows)
        flat = _np.frombuffer(buf, dtype=_np.float32)
        return flat.reshape(len(blob_rows), n_floats)
    else:
        return [list(struct.unpack(f"{n_floats}f", b)) for b in blob_rows]


def cosine_top_k_with_norms(query_vec, matrix, row_norms, k: int) -> list[tuple[int, float]]:
    """Return top-k (index, similarity) pairs, using pre-computed row norms.

    Same as cosine_top_k but skips row norm computation — useful when norms
    are cached (e.g., from VectorCache).
    """
    if _np is not None:
        q = _np.asarray(query_vec, dtype=_np.float32)
        m = _np.asarray(matrix, dtype=_np.float32)
        q_norm = _np.linalg.norm(q)
        if q_norm == 0:
            return []
        m_norms = _np.asarray(row_norms, dtype=_np.float32)
        mask = m_norms > 0
        sims = _np.zeros(len(m), dtype=_np.float32)
        sims[mask] = (m[mask] @ q) / (m_norms[mask] * q_norm)
        if len(sims) <= k:
            top_idx = _np.argsort(-sims)
        else:
            top_idx = _np.argpartition(-sims, k)[:k]
            top_idx = top_idx[_np.argsort(-sims[top_idx])]
        if _backend == "cupy":
            return [(int(i), float(sims[i].get())) for i in top_idx]
        return [(int(i), float(sims[i])) for i in top_idx]
    else:
        import math
        q_norm = math.sqrt(sum(x * x for x in query_vec))
        if q_norm == 0:
            return []
        scored = []
        for i, (row, e_norm) in enumerate(zip(matrix, row_norms)):
            if e_norm == 0:
                continue
            dot = sum(a * b for a, b in zip(query_vec, row))
            scored.append((i, dot / (q_norm * e_norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def cosine_top_k(query_vec, matrix, k: int) -> list[tuple[int, float]]:
    """Return top-k (index, similarity) pairs by cosine similarity.

    query_vec: 1D array/list of floats
    matrix: 2D matrix from decode_matrix()
    k: number of results to return
    Returns: list of (row_index, similarity_score) sorted descending
    """
    if _np is not None:
        q = _np.asarray(query_vec, dtype=_np.float32)
        m = _np.asarray(matrix, dtype=_np.float32)
        q_norm = _np.linalg.norm(q)
        if q_norm == 0:
            return []
        m_norms = _np.linalg.norm(m, axis=1)
        # Avoid division by zero
        mask = m_norms > 0
        sims = _np.zeros(len(m), dtype=_np.float32)
        sims[mask] = (m[mask] @ q) / (m_norms[mask] * q_norm)
        # Top-k via argpartition (faster than full sort for large N)
        if len(sims) <= k:
            top_idx = _np.argsort(-sims)
        else:
            top_idx = _np.argpartition(-sims, k)[:k]
            top_idx = top_idx[_np.argsort(-sims[top_idx])]
        # Convert to Python types (cupy → numpy → Python)
        if _backend == "cupy":
            return [(int(i), float(sims[i].get())) for i in top_idx]
        return [(int(i), float(sims[i])) for i in top_idx]
    else:
        # Pure Python fallback
        import math

        q_norm = math.sqrt(sum(x * x for x in query_vec))
        if q_norm == 0:
            return []
        scored = []
        for i, row in enumerate(matrix):
            dot = sum(a * b for a, b in zip(query_vec, row))
            e_norm = math.sqrt(sum(x * x for x in row))
            if e_norm == 0:
                continue
            scored.append((i, dot / (q_norm * e_norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
