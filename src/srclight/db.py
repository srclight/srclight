"""SQLite database layer for Srclight.

Schema: files, symbols, 3x FTS5 indexes, symbol_edges, index_state.
External content FTS5 with trigger sync.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Paths that indicate vendored/third-party code
VENDORED_PREFIXES = ("third_party/", "third-party/", "vendor/", "ext/", "depends/")


def is_vendored_path(path: str) -> bool:
    """Check if a file path is in a vendored/third-party directory."""
    return any(path.startswith(p) or f"/{p}" in path for p in VENDORED_PREFIXES)


def split_identifier(name: str) -> str:
    """Split a code identifier into searchable tokens.

    Handles CamelCase, snake_case, :: qualifiers, and mixed styles.
    Examples:
        "SQLiteDictionary" -> "SQLite Dictionary sqlite dictionary"
        "get_callers" -> "get callers"
        "OCRManager" -> "OCR Manager ocr manager"
        "myapp::util::ConfigManager" -> "myapp util Config Manager config manager"
    """
    if not name:
        return ""

    # Split on :: and -> first (C++ qualifiers)
    parts = re.split(r"::|->|\.", name)

    tokens = []
    for part in parts:
        # Split on underscores
        sub_parts = part.split("_")
        for sp in sub_parts:
            if not sp:
                continue
            # Split CamelCase with proper handling of acronyms:
            # "SQLiteDict" -> ["SQLite", "Dict"]
            # "OCRManager" -> ["OCR", "Manager"]
            # "getHTTPSUrl" -> ["get", "HTTPS", "Url"]
            # Step 1: insert boundary between lowercase/digit and uppercase
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", sp)
            # Step 2: insert boundary between acronym and next word
            # "SQLite" -> keep as-is, "OCRManager" -> "OCR Manager"
            s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
            camel_parts = s.split()
            tokens.extend(p for p in camel_parts if p)

    # Return both original tokens and lowercased for case-insensitive matching
    result_parts = []
    for t in tokens:
        result_parts.append(t)
    # Add lowercased versions
    lower_parts = [t.lower() for t in tokens if t.lower() != t]
    result_parts.extend(lower_parts)

    return " ".join(result_parts)

SCHEMA_VERSION = 4

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- File registry (incremental indexing via hash)
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    language TEXT,
    size INTEGER,
    line_count INTEGER,
    indexed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Symbols (functions, classes, methods, structs — AST-level chunks)
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT,
    qualified_name TEXT,
    signature TEXT,
    return_type TEXT,
    parameters TEXT,         -- JSON array
    visibility TEXT,
    is_async INTEGER DEFAULT 0,
    is_static INTEGER DEFAULT 0,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    doc_comment TEXT,
    body_hash TEXT,
    line_count INTEGER,
    parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    metadata TEXT,           -- JSON
    UNIQUE(file_id, kind, name, start_line)
);

-- FTS5 Index 1: Symbol names (code-aware tokenization)
-- name_tokens stores split identifiers (CamelCase/snake_case -> words)
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_names_fts USING fts5(
    qualified_name,
    name,
    signature,
    name_tokens,
    file_path UNINDEXED,
    kind UNINDEXED,
    symbol_id UNINDEXED,
    tokenize='unicode61'
);

-- FTS5 Index 2: Source code content (trigram for substring matching)
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_content_fts USING fts5(
    content,
    file_path UNINDEXED,
    name UNINDEXED,
    kind UNINDEXED,
    symbol_id UNINDEXED,
    tokenize='trigram'
);

-- FTS5 Index 3: Documentation (natural language with stemming)
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_docs_fts USING fts5(
    doc_comment,
    name UNINDEXED,
    file_path UNINDEXED,
    kind UNINDEXED,
    symbol_id UNINDEXED,
    tokenize='porter unicode61'
);

-- Symbol relationships (graph layer)
CREATE TABLE IF NOT EXISTS symbol_edges (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    metadata TEXT,           -- JSON
    UNIQUE(source_id, target_id, edge_type)
);

-- Indexing state (resumable)
CREATE TABLE IF NOT EXISTS index_state (
    id INTEGER PRIMARY KEY,
    repo_root TEXT UNIQUE NOT NULL,
    last_commit TEXT,
    config_hash TEXT,
    files_indexed INTEGER DEFAULT 0,
    symbols_indexed INTEGER DEFAULT 0,
    indexed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    indexer_version TEXT
);

-- Embeddings (optional, populated when embed model is configured)
CREATE TABLE IF NOT EXISTS symbol_embeddings (
    symbol_id INTEGER PRIMARY KEY REFERENCES symbols(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    body_hash TEXT,
    embedded_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Regular indexes
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_kind_name ON symbols(kind, name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON symbol_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON symbol_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON symbol_edges(edge_type);
"""


@dataclass
class FileRecord:
    id: int | None = None
    path: str = ""
    content_hash: str = ""
    mtime: float = 0.0
    language: str | None = None
    size: int = 0
    line_count: int = 0
    indexed_at: str | None = None


@dataclass
class SymbolRecord:
    id: int | None = None
    file_id: int = 0
    kind: str = ""
    name: str | None = None
    qualified_name: str | None = None
    signature: str | None = None
    return_type: str | None = None
    parameters: list[dict] | None = None
    visibility: str | None = None
    is_async: bool = False
    is_static: bool = False
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    doc_comment: str | None = None
    body_hash: str | None = None
    line_count: int = 0
    parent_symbol_id: int | None = None
    metadata: dict | None = None
    # Joined fields (not in symbols table directly)
    file_path: str | None = None


@dataclass
class EdgeRecord:
    id: int | None = None
    source_id: int = 0
    target_id: int = 0
    edge_type: str = ""
    confidence: float = 1.0
    metadata: dict | None = None


class Database:
    """SQLite database for Srclight index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def initialize(self) -> None:
        """Create all tables and indexes."""
        assert self.conn is not None
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_info (key, value) VALUES (?, ?)",
            ("embedding_cache_version", "0"),
        )
        # Migrate: add indexer_version column if missing (pre-0.10.1 DBs)
        try:
            self.conn.execute("ALTER TABLE index_state ADD COLUMN indexer_version TEXT")
        except Exception:
            pass  # column already exists
        self.conn.commit()

    # --- Files ---

    def upsert_file(self, rec: FileRecord) -> int:
        """Insert or update a file record. Returns file ID."""
        assert self.conn is not None
        self.conn.execute(
            """INSERT INTO files (path, content_hash, mtime, language, size, line_count)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   content_hash=excluded.content_hash,
                   mtime=excluded.mtime,
                   language=excluded.language,
                   size=excluded.size,
                   line_count=excluded.line_count,
                   indexed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (rec.path, rec.content_hash, rec.mtime, rec.language, rec.size, rec.line_count),
        )
        # lastrowid is unreliable for ON CONFLICT DO UPDATE — fetch the actual ID
        row = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (rec.path,)
        ).fetchone()
        return row["id"]

    def get_file(self, path: str) -> FileRecord | None:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        return FileRecord(**{k: row[k] for k in row.keys()})

    def get_file_by_id(self, file_id: int) -> FileRecord | None:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if row is None:
            return None
        return FileRecord(**{k: row[k] for k in row.keys()})

    def file_needs_reindex(self, path: str, content_hash: str) -> bool:
        """Check if file needs re-indexing by comparing content hash."""
        existing = self.get_file(path)
        if existing is None:
            return True
        return existing.content_hash != content_hash

    def delete_file(self, file_id: int) -> None:
        """Delete a file and all its symbols (CASCADE)."""
        assert self.conn is not None
        # First remove FTS entries for symbols in this file
        symbols = self.conn.execute(
            "SELECT id FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        for sym in symbols:
            self._delete_symbol_fts(sym["id"])
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def all_file_paths(self) -> set[str]:
        assert self.conn is not None
        rows = self.conn.execute("SELECT path FROM files").fetchall()
        return {row["path"] for row in rows}

    # --- Symbols ---

    def insert_symbol(self, rec: SymbolRecord, file_path: str) -> int:
        """Insert a symbol and its FTS entries. Returns symbol ID."""
        assert self.conn is not None
        params_json = json.dumps(rec.parameters) if rec.parameters else None
        meta_json = json.dumps(rec.metadata) if rec.metadata else None

        cur = self.conn.execute(
            """INSERT OR REPLACE INTO symbols
               (file_id, kind, name, qualified_name, signature, return_type,
                parameters, visibility, is_async, is_static,
                start_line, end_line, content, doc_comment, body_hash,
                line_count, parent_symbol_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.file_id, rec.kind, rec.name, rec.qualified_name,
                rec.signature, rec.return_type, params_json, rec.visibility,
                int(rec.is_async), int(rec.is_static),
                rec.start_line, rec.end_line, rec.content, rec.doc_comment,
                rec.body_hash, rec.line_count, rec.parent_symbol_id, meta_json,
            ),
        )
        symbol_id = cur.lastrowid

        # Insert into all 3 FTS tables
        name_tokens = split_identifier(rec.name) if rec.name else ""
        self.conn.execute(
            """INSERT INTO symbol_names_fts
               (rowid, qualified_name, name, signature, name_tokens, file_path, kind, symbol_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol_id, rec.qualified_name or "", rec.name or "", rec.signature or "",
             name_tokens, file_path, rec.kind, str(symbol_id)),
        )

        self.conn.execute(
            """INSERT INTO symbol_content_fts
               (rowid, content, file_path, name, kind, symbol_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (symbol_id, rec.content, file_path, rec.name or "", rec.kind, str(symbol_id)),
        )

        if rec.doc_comment:
            self.conn.execute(
                """INSERT INTO symbol_docs_fts
                   (rowid, doc_comment, name, file_path, kind, symbol_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (symbol_id, rec.doc_comment, rec.name or "", file_path, rec.kind,
                 str(symbol_id)),
            )

        return symbol_id

    def _delete_symbol_fts(self, symbol_id: int) -> None:
        """Remove a symbol's FTS entries."""
        assert self.conn is not None
        for table in ("symbol_names_fts", "symbol_content_fts", "symbol_docs_fts"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE rowid = ?", (symbol_id,)
            )

    def delete_symbols_for_file(self, file_id: int) -> None:
        """Delete all symbols for a file (edges, FTS entries, then symbols)."""
        assert self.conn is not None
        symbols = self.conn.execute(
            "SELECT id FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        if not symbols:
            return
        sym_ids = [s["id"] for s in symbols]
        # Explicitly delete edges first (don't rely solely on CASCADE)
        self.delete_edges_for_symbols(sym_ids)
        for sid in sym_ids:
            self._delete_symbol_fts(sid)
        self.conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))

    def get_symbol_by_name(self, name: str) -> SymbolRecord | None:
        """Get first symbol matching exact name. Use get_symbols_by_name for all matches."""
        assert self.conn is not None
        row = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name = ? LIMIT 1""",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_symbol(row)

    def get_symbols_by_name(self, name: str, limit: int = 20) -> list[SymbolRecord]:
        """Get all symbols matching exact name, with LIKE fallback."""
        assert self.conn is not None

        # Try exact match first
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name = ?
               ORDER BY f.path, s.start_line
               LIMIT ?""",
            (name, limit),
        ).fetchall()

        if rows:
            return [self._row_to_symbol(r) for r in rows]

        # Fallback: case-insensitive LIKE match
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name LIKE ? COLLATE NOCASE
               ORDER BY
                   CASE WHEN s.name = ? THEN 0
                        WHEN s.name LIKE ? THEN 1
                        ELSE 2 END,
                   f.path, s.start_line
               LIMIT ?""",
            (f"%{name}%", name, f"{name}%", limit),
        ).fetchall()

        return [self._row_to_symbol(r) for r in rows]

    def get_symbol_by_id(self, symbol_id: int) -> SymbolRecord | None:
        assert self.conn is not None
        row = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.id = ?""",
            (symbol_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_symbol(row)

    def symbols_in_file(self, path: str) -> list[SymbolRecord]:
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE f.path = ?
               ORDER BY s.start_line""",
            (path,),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def _row_to_symbol(self, row: sqlite3.Row) -> SymbolRecord:
        d = {k: row[k] for k in row.keys()}
        d["is_async"] = bool(d.get("is_async", 0))
        d["is_static"] = bool(d.get("is_static", 0))
        if isinstance(d.get("parameters"), str):
            d["parameters"] = json.loads(d["parameters"])
        if isinstance(d.get("metadata"), str):
            d["metadata"] = json.loads(d["metadata"])
        # Filter to only SymbolRecord fields (joined queries may add extras)
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(SymbolRecord)}
        d = {k: v for k, v in d.items() if k in valid_fields}
        return SymbolRecord(**d)

    # --- Search ---

    def search_symbols(
        self, query: str, kind: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search symbols using FTS5 + LIKE fallback.

        Search tiers:
        1. FTS5 on symbol names (exact + tokenized CamelCase/snake_case)
        2. LIKE fallback on symbol names (substring match)
        3. FTS5 on source code content (trigram)
        4. FTS5 on documentation (porter stemmed)

        Kind filtering is applied inside each tier's query to avoid
        consuming the limit with non-matching results.
        """
        assert self.conn is not None
        results = []
        seen_ids: set[int] = set()

        # High-value kinds that agents typically search for
        _PRIMARY_KINDS = {"class", "struct", "interface", "enum", "function", "method"}
        query_lower = query.lower()

        def _add_row(row_dict: dict) -> None:
            sid = row_dict["symbol_id"]
            if sid not in seen_ids:
                if kind and row_dict["kind"] != kind:
                    return
                rank = row_dict.get("rank", 0)
                name = row_dict.get("name", "")
                sym_kind = row_dict.get("kind", "")
                # Boost exact name matches
                if name == query:
                    rank -= 50.0
                elif name and query_lower in name.lower():
                    rank -= 10.0
                # Boost primary symbol kinds (class/struct > prototype/namespace)
                if sym_kind in _PRIMARY_KINDS:
                    rank -= 5.0
                # Name length normalization: shorter names closer to query length
                # are more relevant (ICaptureService > CaptureServiceImpl::OnHover)
                query_len = len(query)
                name_len = len(name)
                if name_len > query_len:
                    rank += min((name_len - query_len) * 0.3, 5.0)
                # Path-based ranking: core > bindings > vendored
                file_path = row_dict.get("file", "")
                if is_vendored_path(file_path):
                    rank += 20.0
                    row_dict["vendored"] = True
                elif file_path.startswith("bindings/"):
                    rank += 3.0  # Slight penalty vs core src/
                row_dict["rank"] = rank
                results.append(row_dict)
                seen_ids.add(sid)

        # Overfetch so we can sort across tiers, not first-come-first-served.
        OVERFETCH = 3
        fetch_limit = limit * 5 if kind else limit * OVERFETCH

        # Tier 1: FTS5 on symbol names (includes tokenized name_tokens column)
        try:
            query_tokens = split_identifier(query)
            for fts_query in [query, query_tokens]:
                if not fts_query or len(results) >= fetch_limit:
                    break
                rows = self.conn.execute(
                    """SELECT symbol_id, name, file_path, kind,
                              rank, snippet(symbol_names_fts, 1, '>>>', '<<<', '...', 20) as snippet
                       FROM symbol_names_fts
                       WHERE symbol_names_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, fetch_limit),
                ).fetchall()
                for row in rows:
                    if len(results) >= fetch_limit:
                        break
                    _add_row({
                        "symbol_id": int(row["symbol_id"]),
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["snippet"],
                        "source": "name",
                        "rank": row["rank"],
                    })
        except sqlite3.OperationalError:
            pass  # Invalid FTS query syntax

        # Tier 2: LIKE on symbol names — ALWAYS runs (even if Tier 1 filled up).
        # This is critical: FTS5 tokenization can bury exact substring matches
        # (e.g., "CaptureService" finds capture_service_* tokens before ICaptureService).
        # LIKE guarantees substring matches surface.
        # ORDER BY: exact → definition kinds (class/struct/interface) → prefix → length.
        # This ensures ICaptureService (class, non-prefix) beats
        # CaptureServiceImpl::OnHover (method, prefix).
        like_limit = limit * 4  # Generous budget — dedup handles overlap
        _DEFN_KINDS_SQL = "('class','struct','interface','enum')"
        if kind:
            rows = self.conn.execute(
                f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.name LIKE ? COLLATE NOCASE AND s.kind = ?
                   ORDER BY
                       CASE WHEN s.name = ? THEN 0 ELSE 1 END,
                       CASE WHEN s.kind IN {_DEFN_KINDS_SQL} THEN 0
                            WHEN s.kind IN ('function','prototype') THEN 1
                            ELSE 2 END,
                       CASE WHEN s.name LIKE ? THEN 0 ELSE 1 END,
                       length(s.name),
                       s.name
                   LIMIT ?""",
                (f"%{query}%", kind, query, f"{query}%", like_limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.name LIKE ? COLLATE NOCASE
                   ORDER BY
                       CASE WHEN s.name = ? THEN 0 ELSE 1 END,
                       CASE WHEN s.kind IN {_DEFN_KINDS_SQL} THEN 0
                            WHEN s.kind IN ('function','prototype') THEN 1
                            ELSE 2 END,
                       CASE WHEN s.name LIKE ? THEN 0 ELSE 1 END,
                       length(s.name),
                       s.name
                   LIMIT ?""",
                (f"%{query}%", query, f"{query}%", like_limit),
            ).fetchall()
        for row in rows:
            # LIKE results are name matches — give them a competitive base rank.
            # FTS5 BM25 ranks are typically -10 to -25 for name matches.
            # Using -15 here lets _add_row's substring/kind boosts push good
            # matches above FTS token-decomposition noise.
            _add_row({
                "symbol_id": int(row["symbol_id"]),
                "name": row["name"],
                "file": row["file_path"],
                "kind": row["kind"],
                "snippet": row["name"],
                "source": "name_like",
                "rank": -15.0,
            })

        # Tier 3: FTS5 on source code content (trigram)
        if len(results) < fetch_limit:
            try:
                rows = self.conn.execute(
                    """SELECT symbol_id, name, file_path, kind,
                              rank, snippet(symbol_content_fts, 0, '>>>', '<<<', '...', 30) as snippet
                       FROM symbol_content_fts
                       WHERE symbol_content_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (query, fetch_limit),
                ).fetchall()
                for row in rows:
                    if len(results) >= fetch_limit:
                        break
                    _add_row({
                        "symbol_id": int(row["symbol_id"]),
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["snippet"],
                        "source": "content",
                        "rank": row["rank"],
                    })
            except sqlite3.OperationalError:
                pass

        # Tier 4: FTS5 on documentation
        if len(results) < fetch_limit:
            try:
                rows = self.conn.execute(
                    """SELECT symbol_id, name, file_path, kind,
                              rank, snippet(symbol_docs_fts, 0, '>>>', '<<<', '...', 30) as snippet
                       FROM symbol_docs_fts
                       WHERE symbol_docs_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (query, fetch_limit),
                ).fetchall()
                for row in rows:
                    if len(results) >= fetch_limit:
                        break
                    _add_row({
                        "symbol_id": int(row["symbol_id"]),
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["snippet"],
                        "source": "docs",
                        "rank": row["rank"],
                    })
            except sqlite3.OperationalError:
                pass

        # Sort: project code first, then by rank within each group
        results.sort(key=lambda r: (r.get("vendored", False), r.get("rank", 0)))
        return results[:limit]

    # --- Edges ---

    def delete_edges_for_symbols(self, symbol_ids: list[int]) -> None:
        """Delete all edges where source or target is in the given symbol IDs."""
        assert self.conn is not None
        if not symbol_ids:
            return
        placeholders = ",".join("?" * len(symbol_ids))
        self.conn.execute(
            f"DELETE FROM symbol_edges WHERE source_id IN ({placeholders})"
            f" OR target_id IN ({placeholders})",
            symbol_ids + symbol_ids,
        )

    def all_symbol_names(self) -> dict[str, list[int]]:
        """Return a mapping of symbol name -> list of symbol IDs.

        Used for building the call graph: scan symbol content for references
        to known symbol names.
        """
        assert self.conn is not None
        result: dict[str, list[int]] = {}
        for row in self.conn.execute("SELECT id, name FROM symbols WHERE name IS NOT NULL"):
            name = row["name"]
            if name not in result:
                result[name] = []
            result[name].append(row["id"])
        return result

    def insert_edge(self, rec: EdgeRecord) -> int:
        assert self.conn is not None
        meta_json = json.dumps(rec.metadata) if rec.metadata else None
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO symbol_edges
               (source_id, target_id, edge_type, confidence, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (rec.source_id, rec.target_id, rec.edge_type, rec.confidence, meta_json),
        )
        return cur.lastrowid

    def get_callers(self, symbol_id: int) -> list[dict]:
        """Get all symbols that call/reference the given symbol."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.source_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.target_id = ?
               ORDER BY e.edge_type, s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_callees(self, symbol_id: int) -> list[dict]:
        """Get all symbols that the given symbol calls/references."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.target_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.source_id = ?
               ORDER BY e.edge_type, s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_subclasses(self, symbol_id: int) -> list[dict]:
        """Get all classes/structs that inherit from the given symbol."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.source_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.target_id = ? AND e.edge_type = 'inherits'
               ORDER BY s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_base_classes(self, symbol_id: int) -> list[dict]:
        """Get all classes/structs that the given symbol inherits from."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.target_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.source_id = ? AND e.edge_type = 'inherits'
               ORDER BY s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_dependents(self, symbol_id: int, transitive: bool = False, max_depth: int = 5) -> list[dict]:
        """Get all symbols that depend on (call/reference) the given symbol.

        If transitive=True, walks the caller graph recursively up to max_depth.
        """
        assert self.conn is not None
        if not transitive:
            return self.get_callers(symbol_id)

        visited: set[int] = set()
        result = []

        def _walk(sid: int, depth: int):
            if depth > max_depth or sid in visited:
                return
            visited.add(sid)
            callers = self.get_callers(sid)
            for c in callers:
                caller_id = c["symbol"].id
                if caller_id not in visited:
                    c["depth"] = depth
                    result.append(c)
                    _walk(caller_id, depth + 1)

        _walk(symbol_id, 1)
        return result

    def get_implementors(self, symbol_id: int) -> list[dict]:
        """Get all classes/structs that inherit from or implement the given interface/class.

        Same as get_subclasses but named for the interface pattern.
        """
        return self.get_subclasses(symbol_id)

    def get_tests_for(self, symbol_name: str) -> list[dict]:
        """Find test functions that likely test the given symbol.

        Heuristic: looks for symbols in test files whose names contain the target name.
        Matches patterns like test_X, X_test, Test_X, test_X_something.
        """
        assert self.conn is not None
        # Find test files (common patterns)
        test_rows = self.conn.execute(
            """SELECT s.*, f.path as file_path
               FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE (f.path LIKE '%test%' OR f.path LIKE '%spec%')
                 AND s.kind IN ('function', 'method')
                 AND (s.name LIKE ? OR s.name LIKE ? OR s.name LIKE ?)
               ORDER BY f.path, s.start_line""",
            (f"%test%{symbol_name}%", f"%{symbol_name}%test%", f"%{symbol_name}%spec%"),
        ).fetchall()

        results = []
        for r in test_rows:
            sym = self._row_to_symbol(r)
            results.append({
                "symbol": sym,
                "edge_type": "tests",
                "confidence": 0.7,
            })

        # Also check for explicit 'tests' edges if any exist
        sym = self.get_symbol_by_name(symbol_name)
        if sym and sym.id:
            edge_rows = self.conn.execute(
                """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
                   FROM symbol_edges e
                   JOIN symbols s ON e.source_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.target_id = ? AND e.edge_type = 'tests'
                   ORDER BY s.name""",
                (sym.id,),
            ).fetchall()
            seen = {r["symbol"].id for r in results if r["symbol"].id}
            for r in edge_rows:
                s = self._row_to_symbol(r)
                if s.id not in seen:
                    results.append({
                        "symbol": s,
                        "edge_type": r["edge_type"],
                        "confidence": r["confidence"],
                    })

        return results

    # --- Embeddings ---

    def upsert_embedding(self, symbol_id: int, model: str, dimensions: int,
                         embedding_bytes: bytes, body_hash: str | None = None) -> None:
        """Insert or update an embedding for a symbol."""
        assert self.conn is not None
        self.conn.execute(
            """INSERT INTO symbol_embeddings (symbol_id, model, dimensions, embedding, body_hash)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(symbol_id) DO UPDATE SET
                   model=excluded.model,
                   dimensions=excluded.dimensions,
                   embedding=excluded.embedding,
                   body_hash=excluded.body_hash,
                   embedded_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (symbol_id, model, dimensions, embedding_bytes, body_hash),
        )
        # Bump embedding cache version so VectorCache knows to reload
        self.conn.execute(
            """INSERT INTO schema_info (key, value) VALUES ('embedding_cache_version', '1')
               ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)""",
        )

    def get_symbols_needing_embeddings(self, model: str, limit: int = 100000) -> list[dict]:
        """Get symbols that need embeddings (no embedding or body_hash changed)."""
        assert self.conn is not None
        rows = self.conn.execute(
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

    def vector_search(self, query_embedding: bytes, dimensions: int,
                      limit: int = 10, kind: str | None = None,
                      cache=None) -> list[dict]:
        """Search symbols by cosine similarity to the query embedding.

        If a valid VectorCache is provided, uses the GPU-resident matrix (~3ms).
        Otherwise falls back to fetching all embeddings from SQLite (~585ms).
        """
        assert self.conn is not None

        # Fast path: use GPU-resident VectorCache
        if cache is not None and cache.is_valid(self.conn):
            top_k = cache.search(query_embedding, dimensions, limit, kind)
            return self._enrich_results(top_k)

        # Slow path: fetch all embeddings from SQLite
        import struct
        from .vector_math import cosine_top_k, decode_matrix

        n_floats = len(query_embedding) // 4
        query_vec = struct.unpack(f'{n_floats}f', query_embedding)

        if kind:
            rows = self.conn.execute(
                """SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                          s.kind, s.signature, f.path as file_path,
                          s.start_line, s.end_line, s.line_count, s.doc_comment
                   FROM symbol_embeddings e
                   JOIN symbols s ON e.symbol_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.dimensions = ? AND s.kind = ?""",
                (dimensions, kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                          s.kind, s.signature, f.path as file_path,
                          s.start_line, s.end_line, s.line_count, s.doc_comment
                   FROM symbol_embeddings e
                   JOIN symbols s ON e.symbol_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.dimensions = ?""",
                (dimensions,),
            ).fetchall()

        if not rows:
            return []

        blobs = [row["embedding"] for row in rows]
        matrix = decode_matrix(blobs, n_floats)
        top_k = cosine_top_k(query_vec, matrix, limit)

        results = []
        for idx, sim in top_k:
            row = rows[idx]
            results.append({
                "symbol_id": row["symbol_id"],
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "kind": row["kind"],
                "signature": row["signature"],
                "file": row["file_path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "line_count": row["line_count"],
                "doc_comment": row["doc_comment"],
                "similarity": round(sim, 4),
            })
        return results

    def _enrich_results(self, top_k: list[tuple[int, float, int]]) -> list[dict]:
        """Fetch full metadata for top-k results from VectorCache.

        Args:
            top_k: list of (row_index, similarity, symbol_id) from VectorCache.search()
        Returns:
            list of result dicts with full symbol metadata.
        """
        assert self.conn is not None
        if not top_k:
            return []

        results = []
        for _row_idx, sim, symbol_id in top_k:
            row = self.conn.execute(
                """SELECT s.name, s.qualified_name, s.kind, s.signature,
                          f.path as file_path, s.start_line, s.end_line,
                          s.line_count, s.doc_comment
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.id = ?""",
                (symbol_id,),
            ).fetchone()
            if row is None:
                continue
            results.append({
                "symbol_id": symbol_id,
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "kind": row["kind"],
                "signature": row["signature"],
                "file": row["file_path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "line_count": row["line_count"],
                "doc_comment": row["doc_comment"],
                "similarity": round(sim, 4),
            })
        return results

    def embedding_stats(self) -> dict:
        """Get embedding statistics."""
        assert self.conn is not None
        total_symbols = self.conn.execute("SELECT COUNT(*) as n FROM symbols").fetchone()["n"]
        embedded = self.conn.execute(
            "SELECT COUNT(*) as n FROM symbol_embeddings"
        ).fetchone()["n"]
        model_row = self.conn.execute(
            "SELECT model, dimensions FROM symbol_embeddings LIMIT 1"
        ).fetchone()
        return {
            "total_symbols": total_symbols,
            "embedded_symbols": embedded,
            "coverage_pct": round(embedded / total_symbols * 100, 1) if total_symbols else 0,
            "model": model_row["model"] if model_row else None,
            "dimensions": model_row["dimensions"] if model_row else None,
        }

    # --- Index State ---

    def get_index_state(self, repo_root: str) -> dict | None:
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT * FROM index_state WHERE repo_root = ?", (repo_root,)
        ).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def update_index_state(
        self, repo_root: str, last_commit: str | None = None,
        files_indexed: int = 0, symbols_indexed: int = 0,
        indexer_version: str | None = None,
    ) -> None:
        assert self.conn is not None
        self.conn.execute(
            """INSERT INTO index_state (repo_root, last_commit, files_indexed, symbols_indexed, indexer_version)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(repo_root) DO UPDATE SET
                   last_commit=excluded.last_commit,
                   files_indexed=excluded.files_indexed,
                   symbols_indexed=excluded.symbols_indexed,
                   indexer_version=excluded.indexer_version,
                   indexed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (repo_root, last_commit, files_indexed, symbols_indexed, indexer_version),
        )
        self.conn.commit()

    # --- Stats ---

    def directory_summary(self, max_depth: int = 3) -> list[dict]:
        """Get symbol counts per directory, up to max_depth levels deep."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT f.path, COUNT(s.id) as symbol_count, f.language
               FROM files f
               LEFT JOIN symbols s ON f.id = s.file_id
               GROUP BY f.id
               ORDER BY symbol_count DESC"""
        ).fetchall()

        dir_counts: dict[str, dict] = {}
        for row in rows:
            parts = row["path"].split("/")
            # Build directory paths at each depth level
            for depth in range(1, min(len(parts), max_depth + 1)):
                dir_path = "/".join(parts[:depth])
                if dir_path not in dir_counts:
                    dir_counts[dir_path] = {"files": 0, "symbols": 0, "languages": set()}
                dir_counts[dir_path]["files"] += 1
                dir_counts[dir_path]["symbols"] += row["symbol_count"]
                if row["language"]:
                    dir_counts[dir_path]["languages"].add(row["language"])

        result = []
        for path in sorted(dir_counts.keys()):
            info = dir_counts[path]
            result.append({
                "path": path,
                "files": info["files"],
                "symbols": info["symbols"],
                "languages": sorted(info["languages"]),
            })
        return result

    def hotspot_files(self, limit: int = 10) -> list[dict]:
        """Get files with the most symbols (complexity hotspots)."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT f.path, f.language, f.line_count, COUNT(s.id) as symbol_count
               FROM files f
               JOIN symbols s ON f.id = s.file_id
               GROUP BY f.id
               ORDER BY symbol_count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "path": row["path"],
                "language": row["language"],
                "lines": row["line_count"],
                "symbols": row["symbol_count"],
            }
            for row in rows
        ]

    def stats(self) -> dict:
        assert self.conn is not None
        files = self.conn.execute("SELECT COUNT(*) as n FROM files").fetchone()["n"]
        symbols = self.conn.execute("SELECT COUNT(*) as n FROM symbols").fetchone()["n"]
        edges = self.conn.execute("SELECT COUNT(*) as n FROM symbol_edges").fetchone()["n"]

        lang_counts = {}
        for row in self.conn.execute(
            "SELECT language, COUNT(*) as n FROM files GROUP BY language ORDER BY n DESC"
        ):
            lang_counts[row["language"] or "unknown"] = row["n"]

        kind_counts = {}
        for row in self.conn.execute(
            "SELECT kind, COUNT(*) as n FROM symbols GROUP BY kind ORDER BY n DESC"
        ):
            kind_counts[row["kind"]] = row["n"]

        db_size = self.path.stat().st_size if self.path.exists() else 0

        return {
            "files": files,
            "symbols": symbols,
            "edges": edges,
            "languages": lang_counts,
            "symbol_kinds": kind_counts,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
        }

    def commit(self) -> None:
        assert self.conn is not None
        self.conn.commit()


def content_hash(data: bytes) -> str:
    """SHA256 hash of file content."""
    return hashlib.sha256(data).hexdigest()
