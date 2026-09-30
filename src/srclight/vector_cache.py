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
import os
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
        """Read all embeddings from SQLite, write .npy + meta, load to GPU.

        Reading every blob is most of the cost on a large index — gigabytes,
        cold on disk right after the indexing pass — and nearly all of them
        are already in the sidecar being replaced. A vector is taken from
        there when the database still holds the same one: same symbol,
        written at the same moment (`embedded_at` changes on every write, and
        a symbol id reused after a deletion gets a row of its own). Only the
        others are read from SQLite. The result is the sidecar a full read
        builds. A caller with uncommitted writes gets them in the sidecar, read
        in full as before.
        """
        import numpy as np

        if conn.in_transaction:
            # The caller has writes of its own in flight — the MCP server's
            # reindex, on the connection all its threads share. The sidecar
            # must describe what that connection sees, version included, or
            # the server finds it stale on every search and rebuilds it again.
            # One statement reads it all consistently, as builds always did.
            built = self._read_in_one_statement(conn)
        else:
            built = self._read_reusing_previous(conn)
        if built is None:
            return
        rows, matrix, norms, version = built
        n, dims = matrix.shape
        ids = [row[0] for row in rows]
        stamps = [row[3] for row in rows]

        # Ensure directory exists
        self._dir.mkdir(parents=True, exist_ok=True)

        files = {
            self.npy_path.name: self._atomic_write(self.npy_path, lambda fh: np.save(fh, matrix)),
            self.norms_path.name: self._atomic_write(self.norms_path, lambda fh: np.save(fh, norms)),
        }

        meta = {
            "version": version,
            "model": rows[0][1],
            "dimensions": dims,
            "row_count": n,
            "symbol_ids": ids,
            "symbol_kinds": [row[4] for row in rows],
            "file_paths": [row[5] for row in rows],
            # What the next build compares to reuse these vectors.
            "embedded_at": stamps,
            # The exact files this meta describes. They are replaced one at a
            # time: a build killed after the matrix leaves it beside the
            # previous meta, which the next build would otherwise reuse row
            # by row — with the same row count, handing symbols each other's
            # vectors from then on. A replace that fails (a reader holding
            # the file mapped on Windows) leaves all three as they were.
            "files": files,
        }
        self._atomic_write(
            self.meta_path, lambda fh: fh.write(json.dumps(meta).encode())
        )

        # Load into memory / GPU
        self._load_matrix(matrix, norms, meta)
        logger.info("Built sidecar: %d vectors x %d dims (version %d)", n, dims, version)

    def _read_reusing_previous(self, conn):
        """(rows, matrix, norms, version) of the committed database, taking
        from the sidecar on disk every vector it still holds; None when there
        are no embeddings."""
        import numpy as np

        # The embeddings are listed, then read, in several statements: one
        # read transaction makes them one snapshot, so a reindex committing
        # meanwhile cannot delete a vector between the two. It is taken on a
        # connection of its own: the caller's may be shared — the MCP server
        # hands every thread the same one, and its reindex writes through it
        # — and a transaction opened there would swallow those writes and
        # commit them half done.
        own = self._snapshot_connection(conn)
        if own is not None:
            conn = own
            conn.execute("BEGIN")
        try:
            # No ORDER BY: sorted by symbol_id, SQLite walks the table itself,
            # whose rows hold the blobs; unsorted, it reads a covering index.
            rows = sorted(conn.execute("""
                SELECT e.symbol_id, e.model, e.dimensions, e.embedded_at,
                       s.kind, f.path as file_path
                FROM symbol_embeddings e
                JOIN symbols s ON e.symbol_id = s.id
                JOIN files f ON s.file_id = f.id
            """).fetchall(), key=lambda row: row[0])
            if not rows:
                return None
            dims = rows[0][2]
            n = len(rows)
            ids = [row[0] for row in rows]
            matrix = np.empty((n, dims), dtype=np.float32)
            norms = np.empty(n, dtype=np.float32)
            unread = self._reuse_previous(ids, [row[3] for row in rows], rows[0][1], dims,
                                          matrix, norms)
            self._read_vectors(conn, ids, unread, matrix, norms)
            version = self._get_db_version(conn)
        finally:
            if own is not None:
                own.close()  # nothing written: ends the snapshot
        logger.debug("Sidecar vectors: %d reused, %d read from the database",
                     n - len(unread), len(unread))
        return rows, matrix, norms, version

    def _read_in_one_statement(self, conn):
        """(rows, matrix, norms, version) as `conn` sees them, uncommitted
        writes included, every vector read; None when there are none."""
        import numpy as np

        rows, blobs = [], []
        for row in conn.execute("""
            SELECT e.symbol_id, e.model, e.dimensions, e.embedded_at,
                   s.kind, f.path as file_path, e.embedding
            FROM symbol_embeddings e
            JOIN symbols s ON e.symbol_id = s.id
            JOIN files f ON s.file_id = f.id
            ORDER BY e.symbol_id
        """):
            rows.append(tuple(row)[:6])
            blobs.append(row[6])
        if not rows:
            return None
        matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(
            len(rows), rows[0][2]).copy()
        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
        return rows, matrix, norms, self._get_db_version(conn)

    @staticmethod
    def _snapshot_connection(conn):
        """A new connection to the database `conn` is open on, or None when
        it has no file to open again (an in-memory database), in which case
        the build reads through `conn` itself, statement by statement."""
        import sqlite3

        path = next((row[2] for row in conn.execute("PRAGMA database_list")
                     if row[1] == "main"), "")
        if not path:
            return None
        own = sqlite3.connect(path, timeout=30)
        own.row_factory = sqlite3.Row
        return own

    @staticmethod
    def _identity(st) -> list[int]:
        """What tells one written file from another: its size, its mtime and
        its inode. The mtime alone is not enough — on a filesystem with a
        coarse clock (ext4 under WSL2, network mounts, FAT) two builds a few
        milliseconds apart get the same one — but os.replace() always puts a
        file with another inode in place, and a rename keeps a file's inode."""
        return [st.st_size, st.st_mtime_ns, st.st_ino]

    def _fingerprint(self) -> dict:
        """The identity of the matrix and norms files at their paths: each
        build writes them anew, so a pair left by another build differs."""
        return {path.name: self._identity(path.stat())
                for path in (self.npy_path, self.norms_path)}

    def _reuse_previous(self, ids: list[int], stamps: list, model: str, dims: int,
                        matrix, norms) -> list[int]:
        """Fill `matrix` and `norms` from the sidecar on disk where it holds
        the same vector; return the positions it could not fill.

        Anything that does not add up — no sidecar, one from before the
        stamps were recorded, another model, a torn one — reuses nothing,
        and the build reads everything as it always did.
        """
        import numpy as np

        everything = list(range(len(ids)))
        if not self.sidecar_exists():
            return everything
        old_matrix = old_norms = None
        try:
            meta = json.loads(self.meta_path.read_text())
            old_ids, old_stamps = meta.get("symbol_ids"), meta.get("embedded_at")
            if (old_stamps is None or old_ids is None or len(old_stamps) != len(old_ids)
                    or meta.get("model") != model or meta.get("dimensions") != dims
                    or meta.get("files") != self._fingerprint()):
                return everything
            old_matrix = np.load(self.npy_path, mmap_mode="r")
            if old_matrix.shape != (len(old_ids), dims) or old_matrix.dtype != np.float32:
                return everything
            if self.norms_path.exists():
                old_norms = np.load(self.norms_path, mmap_mode="r")
                if old_norms.shape != (len(old_ids),):
                    old_norms = None
            # Still the same files once mapped: another build replacing them
            # between the check and the load would pair its matrix with this
            # meta. The same identities across both loads mean what was
            # mapped is what the meta names.
            if meta["files"] != self._fingerprint():
                return everything
            where = {key: row for row, key in enumerate(zip(old_ids, old_stamps))
                     if key[1] is not None}
            kept_at, kept_from, unread = [], [], []
            for position, key in enumerate(zip(ids, stamps)):
                row = where.get(key)
                if row is None:
                    unread.append(position)
                else:
                    kept_at.append(position)
                    kept_from.append(row)
            if kept_at:
                # Copies, not views: nothing may keep the old files mapped,
                # or replacing them fails on Windows.
                matrix[kept_at] = old_matrix[kept_from]
                norms[kept_at] = (old_norms[kept_from] if old_norms is not None
                                  else np.linalg.norm(matrix[kept_at], axis=1))
            return unread
        except (OSError, ValueError, KeyError, TypeError):
            logger.debug("Could not reuse the sidecar at %s; reading every vector",
                         self.npy_path, exc_info=True)
            return everything
        finally:
            del old_matrix, old_norms

    @staticmethod
    def _read_vectors(conn, ids: list[int], positions: list[int], matrix, norms) -> None:
        """Read the vectors at `positions` from SQLite into `matrix`, one blob
        at a time — never all of them in one list and then one joined buffer,
        which held the matrix three times over — and compute their norms."""
        import numpy as np

        if not positions:
            return
        position_of = {ids[p]: p for p in positions}
        if len(positions) * 2 > len(ids):
            # Most of the table: one pass in rowid order reads it sequentially.
            cursor = conn.execute("SELECT symbol_id, embedding FROM symbol_embeddings")
            pairs = ((sid, blob) for sid, blob in cursor if sid in position_of)
        else:
            def chunked():
                wanted = [ids[p] for p in positions]
                for start in range(0, len(wanted), 500):
                    chunk = wanted[start:start + 500]
                    yield from conn.execute(
                        "SELECT symbol_id, embedding FROM symbol_embeddings "
                        f"WHERE symbol_id IN ({','.join('?' * len(chunk))})", chunk)
            pairs = chunked()
        read = 0
        for sid, blob in pairs:
            matrix[position_of[sid]] = np.frombuffer(blob, dtype=np.float32)
            read += 1
        if read != len(positions):
            # The matrix is np.empty: a row never read holds garbage.
            raise ValueError(f"read {read} of the {len(positions)} embeddings listed")
        norms[positions] = np.linalg.norm(matrix[positions], axis=1)

    @staticmethod
    def _atomic_write(path: Path, write) -> list[int]:
        """Write through a temp file + os.replace(); return the written file's
        identity (see _identity), taken from the file itself before it is
        renamed — the rename keeps it — so it is this build's file whatever
        another build puts at `path` afterwards.

        A running server holds every sidecar mmap'd for its whole life, and
        np.save() opens its target "wb" — O_TRUNC on the same inode. Rebuilding
        in place therefore rewrites pages under the live mapping: the reader
        either SIGBUSes past the new EOF or silently reads new vectors against
        an old symbol_ids list. os.replace() swaps the inode instead, so anyone
        still holding the old file keeps a coherent view of it.
        """
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                write(fh)
                fh.flush()
                os.fsync(fh.fileno())
                st = os.fstat(fh.fileno())
            os.replace(tmp, path)
            return VectorCache._identity(st)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

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

        # The three sidecar files are replaced one at a time, so a kill between
        # renames can pair a new matrix with a stale symbol_ids list. search()
        # indexes that list directly, and a longer stale list yields a real but
        # WRONG symbol with a plausible score. Refuse to load instead.
        n = int(matrix.shape[0])
        for field in ("symbol_ids", "symbol_kinds", "file_paths"):
            got = len(meta.get(field) or [])
            if got != n:
                raise ValueError(
                    f"torn sidecar at {self.npy_path}: matrix has {n} rows but "
                    f"meta lists {got} {field} — rebuild with `srclight index --embed`"
                )
        if len(norms) != n:
            raise ValueError(
                f"torn sidecar at {self.npy_path}: matrix has {n} rows but "
                f"{len(norms)} norms — rebuild with `srclight index --embed`"
            )

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
