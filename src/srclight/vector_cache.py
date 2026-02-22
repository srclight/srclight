"""GPU/CPU-resident embedding matrix with .npy sidecar for fast vector search.

Architecture:
- SQLite `symbol_embeddings` is the write-path (per-row CRUD during indexing)
- `.npy` sidecar is a read-optimized snapshot built after indexing
- VectorCache loads the sidecar to GPU once and serves all queries from VRAM

Performance (27K vectors x 4096 dims, RTX 3090):
- Without cache: ~585ms/query (SQLite fetch + decode every time)
- With cache: ~3ms/query (matrix resident on GPU)
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

logger = logging.getLogger("srclight.vector_cache")


class VectorCache:
    """Per-database GPU/CPU-resident embedding matrix with .npy sidecar."""

    def __init__(self, srclight_dir: Path):
        self._dir = srclight_dir
        self._matrix = None        # cupy or numpy ndarray (N, dims), on GPU if available
        self._norms = None          # cupy or numpy (N,), pre-computed row norms
        self._symbol_ids: list[int] | None = None
        self._symbol_kinds: list[str] | None = None
        self._file_paths: list[str] | None = None
        self._model: str | None = None
        self._dimensions: int | None = None
        self._loaded_version: int = -1

    # --- File paths ---

    @property
    def npy_path(self) -> Path:
        return self._dir / "embeddings.npy"

    @property
    def norms_path(self) -> Path:
        return self._dir / "embeddings_norms.npy"

    @property
    def meta_path(self) -> Path:
        return self._dir / "embeddings_meta.json"

    def sidecar_exists(self) -> bool:
        return self.npy_path.exists() and self.meta_path.exists()

    # --- Build sidecar from SQLite ---

    def build_from_db(self, conn) -> None:
        """Read all embeddings from SQLite, write .npy + meta, load to GPU."""
        import numpy as np

        rows = conn.execute("""
            SELECT e.symbol_id, e.embedding, e.model, e.dimensions,
                   s.kind, f.path as file_path
            FROM symbol_embeddings e
            JOIN symbols s ON e.symbol_id = s.id
            JOIN files f ON s.file_id = f.id
            ORDER BY e.symbol_id
        """).fetchall()

        if not rows:
            return

        dims = rows[0]["dimensions"]
        n = len(rows)

        # Decode to contiguous matrix
        buf = b"".join(row["embedding"] for row in rows)
        matrix = np.frombuffer(buf, dtype=np.float32).reshape(n, dims).copy()
        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)

        # Ensure directory exists
        self._dir.mkdir(parents=True, exist_ok=True)

        # Write sidecar files
        np.save(self.npy_path, matrix)
        np.save(self.norms_path, norms)

        version = self._get_db_version(conn)
        meta = {
            "version": version,
            "model": rows[0]["model"],
            "dimensions": dims,
            "row_count": n,
            "symbol_ids": [row["symbol_id"] for row in rows],
            "symbol_kinds": [row["kind"] for row in rows],
            "file_paths": [row["file_path"] for row in rows],
        }
        self.meta_path.write_text(json.dumps(meta))

        # Load into memory / GPU
        self._load_matrix(matrix, norms, meta)
        logger.info("Built sidecar: %d vectors x %d dims (version %d)", n, dims, version)

    # --- Load sidecar (fast path, server start) ---

    def load_sidecar(self) -> None:
        """Load .npy sidecar into GPU/CPU memory."""
        import numpy as np

        meta = json.loads(self.meta_path.read_text())
        matrix = np.load(self.npy_path, mmap_mode="r")
        norms_path = self.norms_path
        if norms_path.exists():
            norms = np.load(norms_path, mmap_mode="r")
        else:
            # Recompute if norms file missing (backwards compat)
            norms = np.linalg.norm(matrix, axis=1).astype(np.float32)

        self._load_matrix(matrix, norms, meta)
        logger.info(
            "Loaded sidecar: %d vectors x %d dims (version %d)",
            meta["row_count"], meta["dimensions"], meta["version"],
        )

    def _load_matrix(self, matrix, norms, meta: dict) -> None:
        """Transfer matrix to GPU if available, store metadata."""
        from .vector_math import _backend, _np

        if _np is not None and _backend == "cupy":
            self._matrix = _np.asarray(matrix)
            self._norms = _np.asarray(norms)
        else:
            self._matrix = matrix
            self._norms = norms

        self._symbol_ids = meta["symbol_ids"]
        self._symbol_kinds = meta["symbol_kinds"]
        self._file_paths = meta["file_paths"]
        self._model = meta["model"]
        self._dimensions = meta["dimensions"]
        self._loaded_version = meta["version"]

    # --- Validity check ---

    def is_loaded(self) -> bool:
        return self._matrix is not None

    def is_valid(self, conn) -> bool:
        if not self.is_loaded():
            return False
        return self._loaded_version == self._get_db_version(conn)

    @staticmethod
    def _get_db_version(conn) -> int:
        try:
            row = conn.execute(
                "SELECT value FROM schema_info WHERE key='embedding_cache_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        except Exception:
            return 0

    # --- Search (the hot path: ~3ms) ---

    def search(
        self,
        query_bytes: bytes,
        dimensions: int,
        limit: int,
        kind: str | None = None,
    ) -> list[tuple[int, float, int]]:
        """Return top-k (row_index, similarity, symbol_id) tuples."""
        from .vector_math import _backend, _np

        if _np is None or self._matrix is None:
            return []

        n_floats = len(query_bytes) // 4
        query_vec = struct.unpack(f"{n_floats}f", query_bytes)
        q = _np.asarray(query_vec, dtype=_np.float32)
        q_norm = float(_np.linalg.norm(q))
        if q_norm == 0:
            return []

        m = self._matrix
        norms = self._norms

        # Optional kind mask
        if kind and self._symbol_kinds:
            kind_mask = _np.array([k == kind for k in self._symbol_kinds])
            m = m[kind_mask]
            norms = norms[kind_mask]
            index_map = _np.where(kind_mask)[0]
        else:
            index_map = None

        if len(m) == 0:
            return []

        # Cosine similarity
        mask = norms > 0
        sims = _np.zeros(len(m), dtype=_np.float32)
        sims[mask] = (m[mask] @ q) / (norms[mask] * q_norm)

        # Top-k
        k = min(limit, len(sims))
        if k == 0:
            return []

        if len(sims) <= k:
            top_idx = _np.argsort(-sims)
        else:
            top_idx = _np.argpartition(-sims, k)[:k]
            top_idx = top_idx[_np.argsort(-sims[top_idx])]

        # Map back to original indices and extract results
        results = []
        for i in top_idx:
            orig_idx = int(index_map[i]) if index_map is not None else int(i)
            sim = float(sims[i].get()) if _backend == "cupy" else float(sims[i])
            results.append((orig_idx, sim, self._symbol_ids[orig_idx]))

        return results

    def invalidate(self) -> None:
        """Clear in-memory cache. Sidecar files are left on disk."""
        self._matrix = None
        self._norms = None
        self._symbol_ids = None
        self._symbol_kinds = None
        self._file_paths = None
        self._loaded_version = -1
