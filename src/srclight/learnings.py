"""Workspace-level learnings database for Srclight.

Stores decisions, corrections, discoveries, patterns, and conventions
extracted from AI conversations, labbooks, and decision logs. Searchable
via FTS5 keyword search and embedding-based semantic search.

Schema: conversations, learnings, learning_symbols, learning_sources,
learnings_fts (FTS5), learning_embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("srclight.learnings")

SCHEMA_VERSION = 1

VALID_KINDS = ("decision", "correction", "discovery", "pattern", "blocker", "convention")
VALID_SCOPES = ("workspace", "project", "file", "symbol")
VALID_SOURCE_TYPES = ("conversation", "labbook", "decisions_md", "claude_md", "agent_log")

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Session-level summaries
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    project TEXT,
    task_summary TEXT NOT NULL,
    model TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    started_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ended_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Individual learnings (the core unit of knowledge)
CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('decision','correction','discovery','pattern','blocker','convention')),
    content TEXT NOT NULL,
    reasoning TEXT,
    scope TEXT DEFAULT 'workspace' CHECK(scope IN ('workspace','project','file','symbol')),
    project TEXT,
    confidence REAL DEFAULT 1.0,
    ttl_days INTEGER,
    superseded_by INTEGER REFERENCES learnings(id) ON DELETE SET NULL,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at TEXT
);

-- Link learnings to srclight symbol names (many-to-many)
CREATE TABLE IF NOT EXISTS learning_symbols (
    learning_id INTEGER NOT NULL REFERENCES learnings(id) ON DELETE CASCADE,
    symbol_name TEXT NOT NULL,
    project TEXT,
    PRIMARY KEY (learning_id, symbol_name, project)
);

-- Link learnings to source documents
CREATE TABLE IF NOT EXISTS learning_sources (
    id INTEGER PRIMARY KEY,
    learning_id INTEGER NOT NULL REFERENCES learnings(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK(source_type IN ('conversation','labbook','decisions_md','claude_md','agent_log')),
    source_ref TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- FTS5 for keyword search (porter stemmer for natural language)
CREATE VIRTUAL TABLE IF NOT EXISTS learnings_fts USING fts5(
    content,
    reasoning,
    kind UNINDEXED,
    project UNINDEXED,
    learning_id UNINDEXED,
    tokenize='porter unicode61'
);

-- Embeddings for semantic search
CREATE TABLE IF NOT EXISTS learning_embeddings (
    learning_id INTEGER PRIMARY KEY REFERENCES learnings(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    body_hash TEXT,
    embedded_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_learnings_kind ON learnings(kind);
CREATE INDEX IF NOT EXISTS idx_learnings_project ON learnings(project);
CREATE INDEX IF NOT EXISTS idx_learnings_created ON learnings(created_at);
CREATE INDEX IF NOT EXISTS idx_learnings_superseded ON learnings(superseded_by);
CREATE INDEX IF NOT EXISTS idx_learning_symbols_name ON learning_symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_learning_sources_type ON learning_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project);
"""


@dataclass
class ConversationRecord:
    id: int | None = None
    session_id: str = ""
    project: str | None = None
    task_summary: str = ""
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    started_at: str | None = None
    ended_at: str | None = None


@dataclass
class LearningRecord:
    id: int | None = None
    kind: str = ""
    content: str = ""
    reasoning: str | None = None
    scope: str = "workspace"
    project: str | None = None
    confidence: float = 1.0
    ttl_days: int | None = None
    superseded_by: int | None = None
    conversation_id: int | None = None
    created_at: str | None = None
    expires_at: str | None = None
    # Joined fields (populated by get_learning)
    symbols: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


def prepare_learning_text(rec: LearningRecord) -> str:
    """Build text to embed for a learning."""
    parts = [f"{rec.kind}: {rec.content}"]
    if rec.reasoning:
        parts.append(rec.reasoning)
    if rec.project:
        parts.append(f"project: {rec.project}")
    return "\n".join(parts)


def _body_hash(content: str, reasoning: str | None) -> str:
    """Compute hash for embedding cache invalidation."""
    text = f"{content}\n{reasoning or ''}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class LearningsDB:
    """SQLite database for workspace-level learnings."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.open()
        self.initialize()
        return self

    def __exit__(self, *args):
        self.close()

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        assert self.conn is not None
        self.conn.executescript(SCHEMA_SQL)
        # Track schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    # --- Conversations ---

    def record_conversation(self, rec: ConversationRecord) -> int:
        """Upsert a conversation by session_id. Returns the conversation ID."""
        assert self.conn is not None
        now = _now_iso()
        self.conn.execute(
            """INSERT INTO conversations (session_id, project, task_summary, model,
               tokens_in, tokens_out, cost_usd, started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 task_summary=excluded.task_summary,
                 model=excluded.model,
                 tokens_in=excluded.tokens_in,
                 tokens_out=excluded.tokens_out,
                 cost_usd=excluded.cost_usd,
                 ended_at=excluded.ended_at""",
            (
                rec.session_id,
                rec.project,
                rec.task_summary,
                rec.model,
                rec.tokens_in,
                rec.tokens_out,
                rec.cost_usd,
                rec.started_at or now,
                rec.ended_at or now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM conversations WHERE session_id = ?", (rec.session_id,)
        ).fetchone()
        return row["id"]

    # --- Learnings CRUD ---

    def record_learning(
        self,
        rec: LearningRecord,
        symbols: list[str] | None = None,
        sources: list[dict] | None = None,
    ) -> int:
        """Insert a learning with optional symbol and source links. Returns learning ID."""
        assert self.conn is not None

        if rec.kind not in VALID_KINDS:
            raise ValueError(f"Invalid kind '{rec.kind}', must be one of {VALID_KINDS}")
        if rec.scope not in VALID_SCOPES:
            raise ValueError(f"Invalid scope '{rec.scope}', must be one of {VALID_SCOPES}")

        # Compute expires_at from ttl_days
        expires_at = rec.expires_at
        if rec.ttl_days and not expires_at:
            dt = datetime.now(timezone.utc) + timedelta(days=rec.ttl_days)
            expires_at = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        cursor = self.conn.execute(
            """INSERT INTO learnings (kind, content, reasoning, scope, project,
               confidence, ttl_days, conversation_id, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.kind,
                rec.content,
                rec.reasoning,
                rec.scope,
                rec.project,
                rec.confidence,
                rec.ttl_days,
                rec.conversation_id,
                expires_at,
            ),
        )
        learning_id = cursor.lastrowid

        # Sync FTS5
        self.conn.execute(
            "INSERT INTO learnings_fts (content, reasoning, kind, project, learning_id) VALUES (?, ?, ?, ?, ?)",
            (rec.content, rec.reasoning, rec.kind, rec.project, learning_id),
        )

        # Link symbols
        if symbols:
            for sym in symbols:
                self.conn.execute(
                    "INSERT OR IGNORE INTO learning_symbols (learning_id, symbol_name, project) VALUES (?, ?, ?)",
                    (learning_id, sym, rec.project or ""),
                )

        # Link sources
        if sources:
            for src in sources:
                self.conn.execute(
                    "INSERT INTO learning_sources (learning_id, source_type, source_ref) VALUES (?, ?, ?)",
                    (learning_id, src.get("type", "conversation"), src.get("ref", "")),
                )

        self.conn.commit()
        return learning_id

    def get_learning(self, learning_id: int) -> LearningRecord | None:
        """Fetch a learning by ID with joined symbols and sources."""
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT * FROM learnings WHERE id = ?", (learning_id,)
        ).fetchone()
        if not row:
            return None

        rec = LearningRecord(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            reasoning=row["reasoning"],
            scope=row["scope"],
            project=row["project"],
            confidence=row["confidence"],
            ttl_days=row["ttl_days"],
            superseded_by=row["superseded_by"],
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

        # Join symbols
        sym_rows = self.conn.execute(
            "SELECT symbol_name, project FROM learning_symbols WHERE learning_id = ?",
            (learning_id,),
        ).fetchall()
        rec.symbols = [r["symbol_name"] for r in sym_rows]

        # Join sources
        src_rows = self.conn.execute(
            "SELECT source_type, source_ref FROM learning_sources WHERE learning_id = ?",
            (learning_id,),
        ).fetchall()
        rec.sources = [{"type": r["source_type"], "ref": r["source_ref"]} for r in src_rows]

        return rec

    def supersede(self, old_id: int, new_id: int) -> None:
        """Mark an old learning as superseded by a new one."""
        assert self.conn is not None
        self.conn.execute(
            "UPDATE learnings SET superseded_by = ? WHERE id = ?",
            (new_id, old_id),
        )
        self.conn.commit()

    def expire_stale(self) -> int:
        """Delete learnings past their expires_at. Returns count deleted."""
        assert self.conn is not None
        now = _now_iso()
        # Get IDs to delete (for FTS cleanup)
        rows = self.conn.execute(
            "SELECT id FROM learnings WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        ).fetchall()
        if not rows:
            return 0
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        # Clean FTS
        self.conn.execute(
            f"DELETE FROM learnings_fts WHERE learning_id IN ({placeholders})", ids
        )
        # Clean embeddings, sources, symbols (CASCADE handles most, but FTS is manual)
        count = self.conn.execute(
            f"DELETE FROM learnings WHERE id IN ({placeholders})", ids
        ).rowcount
        self.conn.commit()
        return count

    # --- Search ---

    def search_fts(
        self,
        query: str,
        kind: str | None = None,
        project: str | None = None,
        limit: int = 20,
        include_superseded: bool = False,
    ) -> list[dict]:
        """FTS5 keyword search over learnings."""
        assert self.conn is not None

        # Build FTS match query
        conditions = ["learnings_fts MATCH ?"]
        params: list[Any] = [query]

        sql = f"""
            SELECT l.*, learnings_fts.rank AS fts_rank
            FROM learnings_fts
            JOIN learnings l ON l.id = learnings_fts.learning_id
            WHERE {" AND ".join(conditions)}
        """

        if kind:
            sql += " AND l.kind = ?"
            params.append(kind)
        if project:
            sql += " AND (l.project = ? OR l.project IS NULL)"
            params.append(project)
        if not include_superseded:
            sql += " AND l.superseded_by IS NULL"

        sql += " ORDER BY fts_rank LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to LIKE
            return self._search_like(query, kind, project, limit, include_superseded)

        return [
            {
                "learning_id": r["id"],
                "kind": r["kind"],
                "content": r["content"],
                "reasoning": r["reasoning"],
                "project": r["project"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "fts_rank": r["fts_rank"],
            }
            for r in rows
        ]

    def _search_like(
        self,
        query: str,
        kind: str | None,
        project: str | None,
        limit: int,
        include_superseded: bool,
    ) -> list[dict]:
        """Fallback LIKE search when FTS query fails."""
        assert self.conn is not None
        conditions = ["(content LIKE ? OR reasoning LIKE ?)"]
        like_q = f"%{query}%"
        params: list[Any] = [like_q, like_q]

        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if project:
            conditions.append("(project = ? OR project IS NULL)")
            params.append(project)
        if not include_superseded:
            conditions.append("superseded_by IS NULL")

        sql = f"SELECT * FROM learnings WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "learning_id": r["id"],
                "kind": r["kind"],
                "content": r["content"],
                "reasoning": r["reasoning"],
                "project": r["project"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "fts_rank": 0.0,
            }
            for r in rows
        ]

    def search_embeddings(
        self,
        query_embedding: list[float],
        kind: str | None = None,
        project: str | None = None,
        limit: int = 20,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Cosine similarity search over learning embeddings."""
        assert self.conn is not None
        from .embeddings import cosine_similarity

        # Load all embeddings (small table — hundreds, not thousands)
        conditions = ["1=1"]
        params: list[Any] = []
        if kind:
            conditions.append("l.kind = ?")
            params.append(kind)
        if project:
            conditions.append("(l.project = ? OR l.project IS NULL)")
            params.append(project)
        if not include_superseded:
            conditions.append("l.superseded_by IS NULL")

        sql = f"""
            SELECT le.learning_id, le.embedding, le.dimensions,
                   l.kind, l.content, l.reasoning, l.project, l.confidence, l.created_at
            FROM learning_embeddings le
            JOIN learnings l ON l.id = le.learning_id
            WHERE {' AND '.join(conditions)}
        """
        rows = self.conn.execute(sql, params).fetchall()

        # Score each row
        results = []
        for r in rows:
            emb_bytes = r["embedding"]
            dims = r["dimensions"]
            stored_emb = list(struct.unpack(f"{dims}f", emb_bytes))
            sim = cosine_similarity(query_embedding, stored_emb)
            results.append({
                "learning_id": r["learning_id"],
                "kind": r["kind"],
                "content": r["content"],
                "reasoning": r["reasoning"],
                "project": r["project"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "similarity": sim,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def hybrid_search(
        self,
        fts_results: list[dict],
        embedding_results: list[dict],
        limit: int = 20,
        k: int = 60,
    ) -> list[dict]:
        """RRF merge of FTS and embedding results (keyed on learning_id)."""
        scores: dict[int, float] = {}
        data: dict[int, dict] = {}

        for rank, result in enumerate(fts_results):
            lid = result["learning_id"]
            scores[lid] = scores.get(lid, 0.0) + 1.0 / (k + rank + 1)
            if lid not in data:
                data[lid] = dict(result)
                data[lid]["sources"] = []
            data[lid]["sources"].append("fts")

        for rank, result in enumerate(embedding_results):
            lid = result["learning_id"]
            scores[lid] = scores.get(lid, 0.0) + 1.0 / (k + rank + 1)
            if lid not in data:
                data[lid] = dict(result)
                data[lid]["sources"] = []
            data[lid]["sources"].append("embedding")
            if "similarity" in result:
                data[lid]["similarity"] = result["similarity"]

        merged = []
        for lid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            entry = data[lid]
            entry["rrf_score"] = round(score, 6)
            merged.append(entry)

        return merged[:limit]

    # --- Embeddings ---

    def upsert_embedding(
        self, learning_id: int, model: str, dims: int, embedding: list[float], body_hash: str
    ) -> None:
        """Store or update a learning embedding."""
        assert self.conn is not None
        emb_blob = struct.pack(f"{dims}f", *embedding)
        self.conn.execute(
            """INSERT INTO learning_embeddings (learning_id, model, dimensions, embedding, body_hash)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(learning_id) DO UPDATE SET
                 model=excluded.model, dimensions=excluded.dimensions,
                 embedding=excluded.embedding, body_hash=excluded.body_hash,
                 embedded_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (learning_id, model, dims, emb_blob, body_hash),
        )
        self.conn.commit()

    # --- Stats ---

    def stats(self, project: str | None = None, days: int | None = None) -> dict:
        """Get learning statistics."""
        assert self.conn is not None

        conditions = ["superseded_by IS NULL"]
        params: list[Any] = []
        if project:
            conditions.append("(project = ? OR project IS NULL)")
            params.append(project)
        if days:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{days} days")

        where = " AND ".join(conditions)

        # Counts by kind
        rows = self.conn.execute(
            f"SELECT kind, COUNT(*) as cnt FROM learnings WHERE {where} GROUP BY kind",
            params,
        ).fetchall()
        by_kind = {r["kind"]: r["cnt"] for r in rows}

        # Total
        total = sum(by_kind.values())

        # Conversation count
        conv_conditions = ["1=1"]
        conv_params: list[Any] = []
        if project:
            conv_conditions.append("(project = ? OR project IS NULL)")
            conv_params.append(project)
        if days:
            conv_conditions.append("started_at >= datetime('now', ?)")
            conv_params.append(f"-{days} days")

        conv_row = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM conversations WHERE {' AND '.join(conv_conditions)}",
            conv_params,
        ).fetchone()

        # Embedding coverage
        emb_row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM learning_embeddings"
        ).fetchone()

        return {
            "total_learnings": total,
            "by_kind": by_kind,
            "conversations": conv_row["cnt"],
            "embedded": emb_row["cnt"],
        }
