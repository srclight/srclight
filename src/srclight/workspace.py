"""Workspace management for multi-repo indexing.

A workspace groups multiple repos under one name. Each repo has its own
.srclight/index.db. At query time, we ATTACH all per-repo databases to a
:memory: connection and UNION across them — same pattern as MultiDict in
intuition/nomad-builder.

Config lives at ~/.srclight/workspaces/{name}.json
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger("srclight.workspace")


class _WarningRing(logging.Handler):
    """Keeps the last few hundred WARNING+ records so /healthz can bark.

    STUBBY (pack review 2026-09-01): 410 "Failed to attach" warnings went to a
    write-only journal while the dashboard stayed green. A ring buffer lets the
    health payload report `warnings_last_hour` without a log watcher.
    """

    def __init__(self, maxlen: int = 500):
        super().__init__(level=logging.WARNING)
        self.records: collections.deque[tuple[float, str]] = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.created, record.getMessage()))
        except Exception:  # noqa: BLE001 -- a logging handler must never raise
            pass

    def since(self, seconds: float) -> list[str]:
        import time as _time
        cutoff = _time.time() - seconds
        return [msg for t, msg in self.records if t >= cutoff]


warning_ring = _WarningRing()
logger.addHandler(warning_ring)

_LEGACY_DIR = Path.home() / ".codelight"
_NEW_DIR = Path.home() / ".srclight"

# Auto-migrate ~/.codelight/ → ~/.srclight/ on first access
if _LEGACY_DIR.exists() and not _NEW_DIR.exists():
    try:
        _LEGACY_DIR.rename(_NEW_DIR)
        logger.info("Migrated %s -> %s", _LEGACY_DIR, _NEW_DIR)
    except OSError as e:
        logger.warning("Could not migrate %s -> %s: %s", _LEGACY_DIR, _NEW_DIR, e)

WORKSPACES_DIR = _NEW_DIR / "workspaces"


@dataclass
class ProjectEntry:
    """A project within a workspace."""
    name: str
    path: str  # Absolute path to repo root

    @property
    def index_db(self) -> Path:
        return Path(self.path) / ".srclight" / "index.db"

    @property
    def has_index(self) -> bool:
        return self.index_db.exists()


@dataclass
class WorkspaceConfig:
    """Configuration for a workspace (group of repos)."""
    name: str
    projects: dict[str, str] = field(default_factory=dict)  # name -> path

    @property
    def config_path(self) -> Path:
        return WORKSPACES_DIR / f"{self.name}.json"

    @property
    def learnings_db_path(self) -> Path:
        """Path to the workspace-level learnings database."""
        return WORKSPACES_DIR / f"{self.name}_learnings.db"

    def save(self) -> None:
        if not self.name or not re.match(r"^[a-zA-Z0-9_-]+$", self.name):
            raise ValueError(
                f"Invalid workspace name '{self.name}': must be non-empty, "
                "alphanumeric with hyphens/underscores only"
            )
        WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "projects": self.projects}
        self.config_path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, name: str) -> WorkspaceConfig:
        path = WORKSPACES_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workspace '{name}' not found at {path}")
        data = json.loads(path.read_text())
        return cls(name=data["name"], projects=data.get("projects", {}))

    @classmethod
    def list_all(cls) -> list[str]:
        if not WORKSPACES_DIR.exists():
            return []
        return sorted(
            p.stem for p in WORKSPACES_DIR.glob("*.json")
        )

    def add_project(self, name: str, path: str) -> None:
        self.projects[name] = str(Path(path).resolve())
        self.save()

    def remove_project(self, name: str) -> None:
        self.projects.pop(name, None)
        self.save()

    def get_entries(self) -> list[ProjectEntry]:
        return [ProjectEntry(name=n, path=p) for n, p in sorted(self.projects.items())]


_RESERVED_SCHEMA_NAMES = {"main", "temp", "memory"}


def _sanitize_schema_name(name: str) -> str:
    """Convert a project name to a valid SQLite schema identifier.

    SQLite ATTACH AS names must be valid identifiers.
    Replace hyphens/dots with underscores, strip non-alphanumeric.
    Guards against SQLite reserved schema names (main, temp).
    """
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    if not s:
        s = "_unnamed"
    if s.lower() in _RESERVED_SCHEMA_NAMES:
        s = f"p_{s}"
    return s


# Re-warn about a dead search leg on this interval. It MUST stay below the
# window /healthz reads (web.py: warning_ring.since(3600)) — warning once and
# never again meant a permanently dead leg showed for an hour and the dashboard
# went green afterwards while the leg was still dead.
_FTS_WARN_INTERVAL_SECONDS = 900


def _read_only_uri(path) -> str:
    """A `file:` URI that ATTACH and connect() honour as read-only.

    NEVER build this with an f-string. In a raw path, '#' truncates as a URI
    fragment and '?' steals the query string, so `mode=ro` is silently dropped
    and SQLite opens — or CREATES — a *different* file, read-write, with no
    exception raised: ATTACH succeeds against an empty schema and nothing goes
    red. `Path.as_uri()` percent-encodes both, plus spaces and non-ASCII.
    """
    uri = Path(path).resolve().as_uri()
    if uri.startswith("file://") and not uri.startswith("file:///"):
        # A Windows UNC path (\\server\share, \\wsl$\...) renders with a
        # non-empty authority, which SQLite refuses outright: measured on Windows,
        # "invalid uri authority: wsl%24". Fall back to the plain path so this
        # cannot be the thing that breaks such a project. It is a narrow rescue:
        # a plain ATTACH of a \\wsl.localhost path returns "database is locked"
        # anyway, so indexes on that share were already unusable. A real SMB
        # share is untested.
        logger.warning(
            "Cannot build a read-only URI for %s (UNC path); attaching read-write", path
        )
        return str(path)
    return uri + "?mode=ro"


MAX_ATTACH = 10  # SQLite default SQLITE_MAX_ATTACHED


def _synchronized(method):
    """Run a WorkspaceDB method while holding the instance's re-entrant lock.

    The single :memory: connection carries mutable ATTACH state that a batch
    walk rewrites as it goes. Two walks interleaving from different threads
    (the web dashboard runs every /api/* handler on its own worker thread)
    drift ``_attached`` away from what SQLite really has attached, after
    which every ATTACH fails with "too many attached databases" and every
    project reports 0 files -- for MCP callers too. See
    test_workspace_db_concurrent_batch_walks_do_not_poison_connection.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class WorkspaceDB:
    """Cross-repo search via ATTACH + UNION.

    Opens :memory: as the primary connection, ATTACHes each project's
    .srclight/index.db, and queries across all with UNION ALL.

    SQLite limits ATTACH to 10 databases. When there are more projects,
    we query in batches: attach up to 10, run queries, detach, attach next batch.
    """

    def __init__(self, workspace: WorkspaceConfig):
        self.workspace = workspace
        self.conn: sqlite3.Connection | None = None
        self._attached: dict[str, str] = {}  # schema_name -> project_name
        self._all_indexable: list[ProjectEntry] = []  # all entries with an index
        self._caches: dict[str, Any] = {}  # project_name -> VectorCache
        self._stale_sidecars: set[str] = set()  # project_name, sidecar older than index.db
        self._fts_warned: dict[tuple[str, str], float] = {}  # (schema, leg) -> last warned at
        # project_name -> (index-file version key, per-project stats). See _collect_stats.
        self._stats_cache: dict[str, tuple[tuple, dict[str, Any]]] = {}
        self._attach_errors: dict[str, str] = {}  # project_name -> why ATTACH failed
        # Re-entrant so a locked method may walk _iter_batches (also locked).
        self._lock = threading.RLock()

    def open(self) -> None:
        # uri=True matters on WINDOWS ONLY, and for availability rather than
        # integrity. Measured with sqlite3_compileoption_used against the shipped
        # binaries: engine-linux libsqlite3.so.0 (3.45.1) has SQLITE_USE_URI=True,
        # engine-windows sqlite3.dll (3.49.1) has it FALSE.
        #   Linux/macOS: the compile default already interprets URIs, so mode=ro is
        #     honoured with or without this flag (verified: uri=False still refuses
        #     the write and creates no stray file). The flag is a no-op here.
        #   Windows: without it ATTACH raises "unable to open database" and EVERY
        #     project lands in _attach_errors, so the desktop app indexes nothing.
        # It does NOT prevent a silent downgrade to read-write; no such downgrade
        # was reproducible on either platform. The genuinely silent failure is a
        # metacharacter truncating the URI, which _read_only_uri() handles.
        self.conn = sqlite3.connect(
            ":memory:", check_same_thread=False, uri=True
        )
        self.conn.row_factory = sqlite3.Row
        # Discover all indexable projects
        self._all_indexable = [
            e for e in self.workspace.get_entries() if e.has_index
        ]
        # Attach first batch
        self._attach_batch(self._all_indexable[:MAX_ATTACH])

    @_synchronized
    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        self._attached.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _detach_all(self) -> None:
        """Detach every attached database.

        Trusts ``PRAGMA database_list`` rather than ``_attached`` so the
        connection can heal itself if bookkeeping and reality ever diverge.
        """
        assert self.conn is not None
        for row in self.conn.execute("PRAGMA database_list").fetchall():
            name = row["name"]
            if name in ("main", "temp"):
                continue
            try:
                self.conn.execute(f"DETACH DATABASE [{name}]")
            except sqlite3.OperationalError as e:
                logger.warning("Failed to detach %s: %s", name, e)
        self._attached.clear()

    def _verify_attachment(self, schema: str, entry: ProjectEntry) -> None:
        """Confirm the attachment landed on the file we named.

        A guard nobody tests is furniture. Two ways `mode=ro` can be dropped
        without raising: a metacharacter in the path truncating the URI, and a
        SQLite built without SQLITE_USE_URI taking the whole string as a literal
        filename. Both attach a DIFFERENT file, read-write, and report success —
        the project then serves zero symbols behind a green dashboard.
        `PRAGMA database_list`'s file column is the honest tell and writes
        nothing. Detach before raising so a bad attachment cannot linger.
        """
        want = str(Path(entry.index_db).resolve())
        got = None
        for row in self.conn.execute("PRAGMA database_list"):
            if row[1] == schema:
                got = row[2]
                break
        # realpath() costs ~21us on ext4 but ~1ms on drvfs, and several projects
        # live under /mnt/c — so only pay for it when the cheap comparison fails.
        if got != want and (
            got is None or os.path.realpath(got) != os.path.realpath(want)
        ):
            try:
                self.conn.execute(f"DETACH DATABASE [{schema}]")
            except sqlite3.DatabaseError:
                pass
            raise sqlite3.DatabaseError(
                f"attachment for [{schema}] resolved to {got!r}, expected {want!r}"
            )

    def _attach_batch(self, entries: list[ProjectEntry]) -> None:
        """ATTACH a batch of project databases."""
        assert self.conn is not None
        for entry in entries:
            schema = _sanitize_schema_name(entry.name)
            try:
                self.conn.execute(
                    f"ATTACH DATABASE ? AS [{schema}]",
                    (_read_only_uri(entry.index_db),),
                )
                self._verify_attachment(schema, entry)
                self._attached[schema] = entry.name
                self._attach_errors.pop(entry.name, None)
                logger.debug("Attached %s as [%s]", entry.index_db, schema)
            except sqlite3.DatabaseError as e:
                # DatabaseError covers OperationalError AND "file is not a
                # database": one corrupt index must cost one row, never the
                # whole workspace (BARRY, pack review 2026-09-01).
                self._attach_errors[entry.name] = str(e)
                logger.warning("Failed to attach %s: %s", entry.name, e)

    def _iter_batches(self, project_filter: str | None = None, entries: list[ProjectEntry] | None = None):
        """Yield batches of (schema, project_name) tuples.

        If all indexable projects fit in one batch (<= MAX_ATTACH), yields once
        with the already-attached schemas. Otherwise, detaches and re-attaches
        in batches of MAX_ATTACH. ``entries`` restricts the walk to a subset
        (used by the stats cache to visit only projects whose index changed).
        """
        if entries is None:
            entries = self._all_indexable
        if project_filter:
            entries = [e for e in entries if e.name == project_filter]

        # Hold the lock across every yield: the consumer's queries run
        # against the schemas attached here, and another walk rewriting
        # them mid-loop is exactly the race this guards against. The lock
        # is released when the generator is exhausted or closed.
        with self._lock:
            if len(entries) <= MAX_ATTACH:
                # Ensure these specific entries are attached
                needed = {_sanitize_schema_name(e.name) for e in entries}
                if not needed.issubset(set(self._attached.keys())):
                    self._detach_all()
                    self._attach_batch(entries)
                wanted = {e.name for e in entries}
                yield list(
                    (s, p) for s, p in self._attached.items()
                    if p in wanted
                )
            else:
                # Need to batch
                for i in range(0, len(entries), MAX_ATTACH):
                    batch = entries[i:i + MAX_ATTACH]
                    self._detach_all()
                    self._attach_batch(batch)
                    yield list(self._attached.items())

    @property
    def project_count(self) -> int:
        return len(self._all_indexable)

    @property
    @_synchronized
    def attached_projects(self) -> dict[str, str]:
        """schema_name -> project_name mapping for currently attached projects."""
        return dict(self._attached)

    # ---- per-project stats cache -------------------------------------------

    @staticmethod
    def _signal_path(entry: ProjectEntry) -> Path:
        """The indexer's last-indexed signal file (one JSON line per index run)."""
        return entry.index_db.parent / "last-indexed"

    @classmethod
    def _stats_key(cls, entry: ProjectEntry) -> tuple:
        """Version key for a project's index: (mtime_ns, size) of index.db, its
        WAL file if present, and the last-indexed signal. Any reindex changes
        at least one of them."""
        key: list = []
        for path in (
            entry.index_db,
            entry.index_db.with_name(entry.index_db.name + "-wal"),
            cls._signal_path(entry),
        ):
            try:
                st = os.stat(path)
                key.append((st.st_mtime_ns, st.st_size))
            except OSError:
                key.append(None)
        return tuple(key)

    def _read_project_stats(self, schema: str) -> dict[str, Any]:
        """Read every number the stat views need from one attached schema."""
        assert self.conn is not None
        q = self.conn.execute
        files = q(f"SELECT COUNT(*) as n FROM [{schema}].files").fetchone()["n"]
        symbols = q(f"SELECT COUNT(*) as n FROM [{schema}].symbols").fetchone()["n"]
        edges = q(f"SELECT COUNT(*) as n FROM [{schema}].symbol_edges").fetchone()["n"]
        languages = {
            (r["language"] or "unknown"): r["n"]
            for r in q(f"SELECT language, COUNT(*) as n FROM [{schema}].files "
                       f"GROUP BY language ORDER BY n DESC")
        }
        kinds = {
            r["kind"]: r["n"]
            for r in q(f"SELECT kind, COUNT(*) as n FROM [{schema}].symbols GROUP BY kind")
        }
        last_indexed = q(f"SELECT MAX(indexed_at) as t FROM [{schema}].files").fetchone()["t"]
        embedded, model, dimensions = 0, None, None
        if q(f"SELECT name FROM [{schema}].sqlite_master "
              f"WHERE type='table' AND name='symbol_embeddings'").fetchone():
            embedded = q(f"SELECT COUNT(*) as n FROM [{schema}].symbol_embeddings").fetchone()["n"]
            if embedded:
                row = q(f"SELECT model, dimensions FROM [{schema}].symbol_embeddings LIMIT 1").fetchone()
                if row:
                    model, dimensions = row["model"], row["dimensions"]
        return {
            "files": files, "symbols": symbols, "edges": edges,
            "languages": languages, "kinds": kinds, "last_indexed": last_indexed,
            "embedded": embedded, "model": model, "dimensions": dimensions,
        }

    def _collect_stats(self, project_filter: str | None = None) -> dict[str, dict[str, Any]]:
        """project_name -> stats for every indexable project (or one).

        Served from the cache when the index file has not changed, so a poll
        costs no ATTACH and no COUNT. Only projects whose key moved are
        re-read, in batches. A read failure is returned as {"error": ...}
        and not cached, so the next call retries it.
        """
        # A project indexed after open() must show up without a restart (the
        # desktop app's add-then-index flow). Cheap: one stat per entry.
        known = {e.name for e in self._all_indexable}
        newly = [e for e in self.workspace.get_entries() if e.name not in known and e.has_index]
        if newly:
            with self._lock:
                known = {e.name for e in self._all_indexable}
                self._all_indexable.extend(e for e in newly if e.name not in known)

        entries = self._all_indexable
        if project_filter:
            entries = [e for e in entries if e.name == project_filter]
        result: dict[str, dict[str, Any]] = {}
        stale: list[ProjectEntry] = []
        keys: dict[str, tuple] = {}
        for e in entries:
            key = self._stats_key(e)
            keys[e.name] = key
            hit = self._stats_cache.get(e.name)
            if hit and hit[0] == key:
                result[e.name] = hit[1]
            else:
                stale.append(e)
        if stale:
            # Only a miss touches the connection, so only a miss takes the
            # lock: a warm /healthz never queues behind a search walk (K9).
            with self._lock:
                self._walk_stale(stale, keys, result)
        return result

    def _walk_stale(self, stale: list[ProjectEntry], keys: dict[str, tuple],
                    result: dict[str, dict[str, Any]]) -> None:
        by_name = {e.name: e for e in stale}
        for batch in self._iter_batches(entries=stale):
            for schema, project_name in batch:
                try:
                    stats = self._read_project_stats(schema)
                except sqlite3.DatabaseError as e:
                    logger.warning("Error reading stats for %s: %s", project_name, e)
                    result[project_name] = {"error": str(e)}
                    continue
                # "Indexed N ago" means the last index RUN when the signal
                # file exists; MAX(files.indexed_at) only moves when a file
                # is re-parsed and would call a project checked this
                # morning "160d ago" (TOTO, pack review 2026-09-01).
                run_ts = self._read_signal_timestamp(by_name[project_name])
                if run_ts:
                    stats = {**stats, "last_indexed": run_ts, "last_file_change": stats["last_indexed"]}
                self._stats_cache[project_name] = (keys[project_name], stats)
                result[project_name] = stats
        for e in stale:
            if e.name not in result:  # ATTACH itself failed
                result[e.name] = {"error": self._attach_errors.get(e.name, "could not attach index")}

    @classmethod
    def _read_signal_timestamp(cls, entry: ProjectEntry) -> str | None:
        try:
            with open(cls._signal_path(entry), encoding="utf-8") as fh:
                ts = json.load(fh).get("timestamp")
            return str(ts) if ts else None
        except (OSError, ValueError, AttributeError):
            return None

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects in the workspace with their stats."""
        assert self.conn is not None
        stats = self._collect_stats()
        entries_by_name = {e.name: e for e in self.workspace.get_entries()}
        results = []
        for project_name in sorted(stats):
            st = stats[project_name]
            if "error" in st:
                results.append({"project": project_name, "error": st["error"]})
                continue
            entry = entries_by_name[project_name]
            db_size = entry.index_db.stat().st_size if entry.index_db.exists() else 0
            results.append({
                "project": project_name,
                "path": entry.path,
                "files": st["files"],
                "symbols": st["symbols"],
                "edges": st["edges"],
                "languages": st["languages"],
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "indexed": True,
                "last_indexed": st["last_indexed"],
                "last_file_change": st.get("last_file_change", st["last_indexed"]),
                "embedded_symbols": st["embedded"],
                "embedding_coverage": round(st["embedded"] / st["symbols"], 4) if st["symbols"] else 0.0,
            })

        # Also list unindexed projects
        for entry in self.workspace.get_entries():
            if entry.name not in stats:
                results.append({
                    "project": entry.name,
                    "path": entry.path,
                    "files": 0,
                    "symbols": 0,
                    "indexed": False,
                })

        return results

    @_synchronized
    def search_symbols(
        self, query: str, kind: str | None = None,
        project: str | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search symbols across all projects using UNION ALL.

        FTS5 virtual tables are queried per-schema, results merged and ranked.
        Uses batched ATTACH when projects exceed SQLite's 10-database limit.
        """
        assert self.conn is not None
        from .db import (
            _IDENT_RE, is_vendored_path, match_rung, split_identifier,
        )

        results: list[dict[str, Any]] = []
        seen_ids: set[tuple[str, int]] = set()  # (project, symbol_id)

        _PRIMARY_KINDS = {"class", "struct", "interface", "enum", "function", "method"}
        query_lower = query.lower()
        query_tokens = split_identifier(query)

        def _rank_result(row_dict: dict) -> float:
            """Score by MATCH QUALITY first; statistics only break ties inside it.

            A rung is worth 1000 and every within-rung adjustment together
            cannot reach that, so a better match can never be outranked by a
            worse one with a stronger bm25 -- which is exactly what happened
            when a hardcoded LIKE constant competed with raw bm25 across tiers.
            """
            name = row_dict.get("name", "") or ""
            sym_kind = row_dict.get("kind", "")
            file_path = row_dict.get("file", "")

            score = match_rung(query, name) * 1000.0
            if not _IDENT_RE.match(name):
                score += 300.0
            if sym_kind not in _PRIMARY_KINDS:
                score += 100.0
            if is_vendored_path(file_path):
                score += 50.0
                row_dict["vendored"] = True
            # bm25 is comparable only within one table/project/query, so clamp it
            # to a band far narrower than a rung and use it as a tiebreak alone.
            base = row_dict.get("rank", 0) or 0
            score += max(-40.0, min(0.0, float(base))) * 0.5
            score += min(len(name), 40) * 0.1
            return score

        # Tier 1+2: FTS5 name search + LIKE fallback per schema
        for batch in self._iter_batches(project_filter=project):
          for schema, project_name in batch:
            # FTS5 on symbol names
            for fts_query in [query, query_tokens]:
                if not fts_query:
                    continue
                try:
                    rows = self._fts_execute(
                        f"""SELECT symbol_id, name, file_path, kind, rank,
                               snippet(symbol_names_fts, 1, '>>>', '<<<', '...', 20) as snippet
                           FROM [{schema}].symbol_names_fts
                           WHERE symbol_names_fts MATCH ?
                           ORDER BY rank LIMIT ?""",
                        (fts_query, limit * 3), schema, "name",
                    )
                    for row in rows:
                        sid = int(row["symbol_id"])
                        key = (project_name, sid)
                        if key in seen_ids:
                            continue
                        if kind and row["kind"] != kind:
                            continue
                        d = {
                            "project": project_name,
                            "symbol_id": sid,
                            "name": row["name"],
                            "file": row["file_path"],
                            "kind": row["kind"],
                            "snippet": row["snippet"],
                            "source": "name",
                            "rank": row["rank"],
                        }
                        d["rank"] = _rank_result(d)
                        results.append(d)
                        seen_ids.add(key)
                except sqlite3.OperationalError as e:
                    self._fts_leg_failed(schema, "name", e)

            # LIKE fallback
            try:
                kind_filter = "AND s.kind = ?" if kind else ""
                like_params: list = [f"%{query}%"]
                if kind:
                    like_params.append(kind)
                like_params.extend([query, limit * 3])

                rows = self.conn.execute(
                    f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                       FROM [{schema}].symbols s
                       JOIN [{schema}].files f ON s.file_id = f.id
                       WHERE s.name LIKE ? COLLATE NOCASE {kind_filter}
                       ORDER BY
                           CASE WHEN s.name = ? THEN 0 ELSE 1 END,
                           length(s.name), s.name
                       LIMIT ?""",
                    like_params,
                ).fetchall()
                for row in rows:
                    sid = int(row["symbol_id"])
                    key = (project_name, sid)
                    if key in seen_ids:
                        continue
                    if kind and row["kind"] != kind:
                        continue
                    d = {
                        "project": project_name,
                        "symbol_id": sid,
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["name"],
                        "source": "name_like",
                        "rank": -15.0,
                    }
                    d["rank"] = _rank_result(d)
                    results.append(d)
                    seen_ids.add(key)
            except sqlite3.OperationalError:
                pass

            # Tier 3: FTS5 on content (trigram)
            try:
                rows = self._fts_execute(
                    f"""SELECT symbol_id, name, file_path, kind, rank,
                           snippet(symbol_content_fts, 0, '>>>', '<<<', '...', 30) as snippet
                       FROM [{schema}].symbol_content_fts
                       WHERE symbol_content_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, limit * 2), schema, "content",
                )
                for row in rows:
                    sid = int(row["symbol_id"])
                    key = (project_name, sid)
                    if key in seen_ids:
                        continue
                    if kind and row["kind"] != kind:
                        continue
                    d = {
                        "project": project_name,
                        "symbol_id": sid,
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["snippet"],
                        "source": "content",
                        "rank": row["rank"],
                    }
                    d["rank"] = _rank_result(d)
                    results.append(d)
                    seen_ids.add(key)
            except sqlite3.OperationalError as e:
                self._fts_leg_failed(schema, "content", e)

            # Tier 4: FTS5 on docs
            try:
                rows = self._fts_execute(
                    f"""SELECT symbol_id, name, file_path, kind, rank,
                           snippet(symbol_docs_fts, 0, '>>>', '<<<', '...', 30) as snippet
                       FROM [{schema}].symbol_docs_fts
                       WHERE symbol_docs_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, limit * 2), schema, "docs",
                )
                for row in rows:
                    sid = int(row["symbol_id"])
                    key = (project_name, sid)
                    if key in seen_ids:
                        continue
                    if kind and row["kind"] != kind:
                        continue
                    d = {
                        "project": project_name,
                        "symbol_id": sid,
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["snippet"],
                        "source": "docs",
                        "rank": row["rank"],
                    }
                    d["rank"] = _rank_result(d)
                    results.append(d)
                    seen_ids.add(key)
            except sqlite3.OperationalError as e:
                self._fts_leg_failed(schema, "docs", e)

        # Collapse repeats. One row per (project, name, kind), carrying how many
        # it stands for. A human's eye skips a duplicate; an agent reads it as
        # corroboration -- `id` returned 20 rows with 2 distinct names, 15 of
        # them one symbol. Hiding the count would trade one distortion for
        # another, so the survivor reports it.
        best: dict[tuple, dict] = {}
        for r in results:
            key = (r.get("project"), r.get("name"), r.get("kind"))
            keep = best.get(key)
            if keep is None or r.get("rank", 0) < keep.get("rank", 0):
                if keep is not None:
                    r["duplicates"] = keep.get("duplicates", 1) + 1
                else:
                    r["duplicates"] = 1
                best[key] = r
            else:
                keep["duplicates"] = keep.get("duplicates", 1) + 1
        results = list(best.values())

        # Sort by rank (lower = better), project code > vendored
        # vendored is a within-rung penalty in the score now, not a primary key.
        # As a primary key it was an INFINITE demotion that made the +20 dead
        # arithmetic -- setting that constant to +/-100000 changed no result.
        results.sort(key=lambda r: (r.get("rank", 0), r.get("project") or "", r.get("name") or ""))
        return results[:limit]

    def codebase_map(self, project: str | None = None) -> dict[str, Any]:
        """Get aggregated stats across all projects (or a single one)."""
        assert self.conn is not None
        stats = self._collect_stats(project_filter=project)

        total_files = total_symbols = total_edges = total_embedded = 0
        all_languages: dict[str, int] = {}
        all_kinds: dict[str, int] = {}
        project_summaries: list[dict] = []
        newest_index: str | None = None
        errors: dict[str, str] = {}

        for project_name in sorted(stats):
            st = stats[project_name]
            if "error" in st:
                errors[project_name] = st["error"]
                continue
            total_files += st["files"]
            total_symbols += st["symbols"]
            total_edges += st["edges"]
            total_embedded += st["embedded"]
            for lang, n in st["languages"].items():
                all_languages[lang] = all_languages.get(lang, 0) + n
            for kind, n in st["kinds"].items():
                all_kinds[kind] = all_kinds.get(kind, 0) + n
            li = st["last_indexed"]
            if li and (newest_index is None or li > newest_index):
                newest_index = li
            project_summaries.append({
                "project": project_name,
                "files": st["files"],
                "symbols": st["symbols"],
                "edges": st["edges"],
                "last_indexed": li,
            })

        return {
            "workspace": self.workspace.name,
            "projects_attached": self.project_count,
            "totals": {
                "files": total_files,
                "symbols": total_symbols,
                "edges": total_edges,
                "embedded": total_embedded,
            },
            "last_indexed": newest_index,
            "projects_errored": len(errors),
            "errors": errors,
            "languages": dict(sorted(all_languages.items(), key=lambda x: -x[1])),
            "symbol_kinds": dict(sorted(all_kinds.items(), key=lambda x: -x[1])),
            "projects": project_summaries,
        }

    @_synchronized
    def get_symbol(self, name: str, project: str | None = None) -> list[dict[str, Any]]:
        """Get full symbol details by name across projects."""
        assert self.conn is not None
        results = []

        for batch in self._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    rows = self.conn.execute(
                        f"""SELECT s.*, f.path as file_path
                           FROM [{schema}].symbols s
                           JOIN [{schema}].files f ON s.file_id = f.id
                           WHERE s.name = ?
                           ORDER BY f.path, s.start_line""",
                        (name,),
                    ).fetchall()
                    if not rows:
                        rows = self.conn.execute(
                            f"""SELECT s.*, f.path as file_path
                               FROM [{schema}].symbols s
                               JOIN [{schema}].files f ON s.file_id = f.id
                               WHERE s.name LIKE ? COLLATE NOCASE
                               ORDER BY f.path, s.start_line
                               LIMIT 20""",
                            (f"%{name}%",),
                        ).fetchall()

                    for row in rows:
                        results.append({
                            "project": project_name,
                            "id": row["id"],
                            "name": row["name"],
                            "qualified_name": row["qualified_name"],
                            "kind": row["kind"],
                            "signature": row["signature"],
                            "file": row["file_path"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "content": row["content"],
                            "doc_comment": row["doc_comment"],
                            "line_count": row["line_count"],
                        })
                except sqlite3.OperationalError:
                    pass

        return results

    @_synchronized
    def _get_project_cache(self, project_name: str):
        """Get or create a VectorCache for a project.

        Returns a loaded VectorCache, or None if no sidecar exists.
        Re-checks for sidecars that may have appeared since last call
        (e.g. after running `srclight index --embed`).
        """
        cached = self._caches.get(project_name)
        if cached is not None and cached.is_loaded():
            return cached

        from .vector_cache import VectorCache

        entry = next(
            (e for e in self._all_indexable if e.name == project_name), None
        )
        if entry is None:
            return None

        srclight_dir = Path(entry.path) / ".srclight"
        cache = VectorCache(srclight_dir)
        if cache.sidecar_exists():
            try:
                cache.load_sidecar()
                if self._sidecar_matches_db(srclight_dir, cache):
                    self._stale_sidecars.discard(project_name)
                else:
                    self._stale_sidecars.add(project_name)
                    logger.warning(
                        "Stale embedding sidecar for %s: built from an older index, "
                        "so semantic search sees only part of it "
                        "(rebuild with `srclight index --embed`)",
                        project_name,
                    )
                self._caches[project_name] = cache
                return cache
            except Exception as e:
                logger.warning("Failed to load sidecar for %s: %s", project_name, e)

        # No sidecar or load failed — return None but don't permanently cache it.
        # Next call will re-check sidecar existence (fast filesystem stat).
        return None

    @staticmethod
    def _is_fts_query_error(exc: Exception) -> bool:
        """True when FTS5 rejected the QUERY TEXT, not the table.

        A user's search string is not a broken index. `get(` raises
        `fts5: syntax error near ""`, `self.conn.execute` raises `syntax error
        near "."`, an unbalanced quote raises `unterminated string`. Counting
        those as a dead leg produced 117 warnings from ONE search across 39
        projects and buried the real signal in /healthz `degraded`.
        """
        msg = str(exc).lower()
        return "syntax error" in msg or "unterminated" in msg

    def _fts_execute(self, sql: str, params: tuple, schema: str, leg: str) -> list:
        """Run one FTS leg. Returns rows; [] if the leg is unavailable.

        When FTS5 rejects the query text, quote it and retry once: someone
        searching `get(` means that literal text, not a boolean expression.
        Quoting is also what makes it work -- bare `get(` errors where `"get("`
        returns rows. Only a genuine leg failure is reported.
        """
        assert self.conn is not None
        try:
            return self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            if not self._is_fts_query_error(e):
                self._fts_leg_failed(schema, leg, e)
                return []
        quoted = '"' + str(params[0]).replace('"', '""') + '"'
        try:
            return self.conn.execute(sql, (quoted, *params[1:])).fetchall()
        except sqlite3.OperationalError as e:
            if not self._is_fts_query_error(e):
                self._fts_leg_failed(schema, leg, e)
            return []

    def _fts_leg_failed(self, schema: str, leg: str, exc: Exception) -> None:
        """Report a search leg that could not run, once per schema and leg.

        These three legs sat behind a bare `except sqlite3.OperationalError:
        pass`. That cannot distinguish "this project predates the FTS tables"
        from "the SQL is malformed against every project forever" — and the
        second is what happened: every leg qualified its table inside WHERE and
        snippet(), which SQLite reads as table.column, so keyword search silently
        fell through to a LIKE on symbols.name for an unknown span of releases.
        Re-warn each pair on an interval below the ring's window, so a
        permanently dead leg stays visible on /healthz without one line per query.
        """
        key = (schema, leg)
        now = time.time()
        last = self._fts_warned.get(key)
        if last is not None and now - last < _FTS_WARN_INTERVAL_SECONDS:
            return
        self._fts_warned[key] = now
        logger.warning(
            "FTS %s search unavailable for [%s]: %s — falling back to LIKE on symbol names",
            leg, schema, exc,
        )

    @staticmethod
    def _sidecar_matches_db(srclight_dir: Path, cache) -> bool:
        """Whether a just-loaded sidecar's version matches the project's index.db.

        vector_search deliberately skips is_valid() per query — a SQLite connect
        per project costs ~60ms across 10 projects — so this is the one place the
        comparison happens: once, when the sidecar is loaded. Without it a sidecar
        left behind by an interrupted re-embed serves a subset of the index for
        the life of the process while /api/embedding_status, which reads the DB,
        reports 100% coverage (intuition-2019, 2026-09-02: 20,648 vs 15,611).
        """
        db_file = srclight_dir / "index.db"
        if not db_file.exists():
            return True
        try:
            conn = sqlite3.connect(_read_only_uri(db_file), uri=True)
            conn.row_factory = sqlite3.Row
            try:
                return cache.is_valid(conn)
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 -- an unreadable db is reported elsewhere
            return True

    def stale_sidecars(self) -> list[str]:
        """Projects whose embedding sidecar is older than their index.db.

        Deliberately NOT @_synchronized: /healthz polls this and must never queue
        behind a search holding the workspace lock.
        """
        return sorted(self._stale_sidecars)

    @_synchronized
    def vector_search(
        self, query_embedding: bytes, dimensions: int,
        project: str | None = None, kind: str | None = None, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search symbols by cosine similarity across workspace projects.

        Fast path: uses per-project VectorCache when sidecars exist (~3ms/project).
        Slow path: fetches all embeddings from SQLite via ATTACH+UNION.
        """
        assert self.conn is not None

        # Try fast path: per-project cache search + merge
        all_candidates: list[tuple[str, int, float, int]] = []  # (proj, row_idx, sim, sym_id)
        cache_miss_projects: list[str] = []

        entries = self._all_indexable
        if project:
            entries = [e for e in entries if e.name == project]

        for entry in entries:
            cache = self._get_project_cache(entry.name)
            if cache is not None and cache.is_loaded():
                # Trust loaded cache — validity is checked on load and after reindex.
                # Skipping per-query SQLite connect saves ~60ms across 10 projects.
                candidates = cache.search(query_embedding, dimensions, limit * 2, kind)
                for row_idx, sim, sym_id in candidates:
                    all_candidates.append((entry.name, row_idx, sim, sym_id))
            elif cache is None and self._caches.get(entry.name) is None:
                # None sentinel — project has no sidecar (and likely no embeddings).
                # Skip it silently.
                pass
            else:
                # Has embeddings but no valid cache — needs slow path
                cache_miss_projects.append(entry.name)

        # If we got cache hits and no misses, use fast enrichment
        if all_candidates and not cache_miss_projects:
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            all_candidates = all_candidates[:limit]
            return self._enrich_workspace_results(all_candidates)

        # Fall back to slow path for any projects without valid caches
        return self._vector_search_slow(
            query_embedding, dimensions, project=project, kind=kind, limit=limit
        )

    def _enrich_workspace_results(
        self, candidates: list[tuple[str, int, float, int]],
    ) -> list[dict[str, Any]]:
        """Fetch full metadata for cache-based search results.

        Groups lookups by project to minimize connection overhead.
        """
        # Group by project
        by_project: dict[str, list[tuple[int, float, int]]] = {}
        for proj_name, row_idx, sim, sym_id in candidates:
            by_project.setdefault(proj_name, []).append((row_idx, sim, sym_id))

        # Fetch metadata per-project (one connection per project)
        enriched: dict[int, dict] = {}  # sym_id -> result dict
        for proj_name, items in by_project.items():
            entry = next(
                (e for e in self._all_indexable if e.name == proj_name), None
            )
            if entry is None:
                continue
            try:
                # Read-only: this runs on the long-lived server in the
                # semantic-search hot path, and a read-write connection
                # checkpoints another process's WAL when it closes.
                proj_conn = sqlite3.connect(_read_only_uri(entry.index_db), uri=True)
                proj_conn.row_factory = sqlite3.Row
                for _row_idx, sim, sym_id in items:
                    row = proj_conn.execute(
                        """SELECT s.name, s.qualified_name, s.kind, s.signature,
                                  f.path as file_path, s.start_line, s.end_line,
                                  s.line_count, s.doc_comment
                           FROM symbols s
                           JOIN files f ON s.file_id = f.id
                           WHERE s.id = ?""",
                        (sym_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    enriched[sym_id] = {
                        "project": proj_name,
                        "symbol_id": sym_id,
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
                    }
                proj_conn.close()
            except Exception as e:
                logger.warning("Error enriching results from %s: %s", proj_name, e)

        # Return in original order (sorted by similarity)
        return [enriched[sym_id] for _, _, _, sym_id in candidates if sym_id in enriched]

    def _vector_search_slow(
        self, query_embedding: bytes, dimensions: int,
        project: str | None = None, kind: str | None = None, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Slow path: fetch all embeddings from SQLite via ATTACH+UNION."""
        assert self.conn is not None
        import struct
        from .vector_math import cosine_top_k, decode_matrix

        n_floats = len(query_embedding) // 4
        query_vec = struct.unpack(f'{n_floats}f', query_embedding)

        all_rows: list[tuple[str, Any]] = []
        for batch in self._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    table_check = self.conn.execute(
                        f"SELECT name FROM [{schema}].sqlite_master "
                        f"WHERE type='table' AND name='symbol_embeddings'"
                    ).fetchone()
                    if not table_check:
                        continue

                    if kind:
                        rows = self.conn.execute(
                            f"""SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                                      s.kind, s.signature, f.path as file_path,
                                      s.start_line, s.end_line, s.line_count, s.doc_comment
                               FROM [{schema}].symbol_embeddings e
                               JOIN [{schema}].symbols s ON e.symbol_id = s.id
                               JOIN [{schema}].files f ON s.file_id = f.id
                               WHERE e.dimensions = ? AND s.kind = ?""",
                            (dimensions, kind),
                        ).fetchall()
                    else:
                        rows = self.conn.execute(
                            f"""SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                                      s.kind, s.signature, f.path as file_path,
                                      s.start_line, s.end_line, s.line_count, s.doc_comment
                               FROM [{schema}].symbol_embeddings e
                               JOIN [{schema}].symbols s ON e.symbol_id = s.id
                               JOIN [{schema}].files f ON s.file_id = f.id
                               WHERE e.dimensions = ?""",
                            (dimensions,),
                        ).fetchall()

                    for row in rows:
                        all_rows.append((project_name, row))
                except sqlite3.OperationalError as e:
                    logger.warning("Vector search error in %s: %s", project_name, e)

        if not all_rows:
            return []

        blobs = [row["embedding"] for _, row in all_rows]
        matrix = decode_matrix(blobs, n_floats)
        top_k = cosine_top_k(query_vec, matrix, limit)

        results = []
        for idx, sim in top_k:
            proj, row = all_rows[idx]
            results.append({
                "project": proj,
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

    def embedding_stats(self, project: str | None = None) -> dict[str, Any]:
        """Get embedding statistics across workspace projects."""
        assert self.conn is not None
        stats = self._collect_stats(project_filter=project)
        total_symbols = total_embedded = 0
        model = dimensions = None
        for project_name in sorted(stats):
            st = stats[project_name]
            if "error" in st:
                continue
            total_symbols += st["symbols"]
            total_embedded += st["embedded"]
            if model is None and st["embedded"]:
                model, dimensions = st["model"], st["dimensions"]
        return {
            "total_symbols": total_symbols,
            "embedded_symbols": total_embedded,
            "coverage_pct": round(total_embedded / total_symbols * 100, 1) if total_symbols else 0,
            "model": model,
            "dimensions": dimensions,
        }
