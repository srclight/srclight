"""Srclight MCP Server.

Exposes code indexing tools to AI agents via the Model Context Protocol.
Supports both single-repo mode and workspace mode (multi-repo via ATTACH+UNION).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import collections
import contextvars
import threading
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer


from .db import Database
from .indexer import IndexConfig, Indexer, resolve_embed_model

logger = logging.getLogger("srclight.server")

_INSTRUCTIONS_TEMPLATE = """Welcome to Srclight — deep code indexing for AI agents.

{dynamic_section}## Getting Started
1. Call `codebase_map()` at the START of every session to orient yourself — it shows project stats, languages, symbol counts, and directory structure.
2. Use `list_projects()` to see all repos in the workspace with file/symbol counts.
3. Use `hybrid_search(query)` to find any code by name, concept, or natural language description.

## Which Search Tool to Use
- **`hybrid_search(query)`** — BEST for most queries. Combines keyword + semantic search via RRF fusion. Use for natural language ("find dictionary lookup code") or keywords.
- **`search_symbols(query)`** — keyword-only search. Faster, good for exact symbol names or code fragments.
- **`semantic_search(query)`** — embedding-only search. Good when you know the concept but not the terminology.

## The `project` Parameter
In workspace mode (multi-repo), many tools accept an optional `project` parameter:
- **Omit it** to search across ALL projects simultaneously.
- **Pass it** to filter results to one specific repo (e.g., `project="nomad-builder"`).
- Graph tools (`get_callers`, `get_callees`, `get_dependents`, `get_implementors`, `get_tests_for`, `get_type_hierarchy`) and git tools (`blame_symbol`, `recent_changes`, `git_hotspots`, `whats_changed`, `changes_to`) REQUIRE `project` in workspace mode.

## Tool Selection Guide
| Need | Tool |
|------|------|
| Overview / orientation | `codebase_map()` |
| Find code by name or concept | `hybrid_search(query)` |
| Full source code of a symbol | `get_symbol(name)` |
| Quick function signature check | `get_signature(name)` |
| List all functions in a file | `symbols_in_file(path, project)` |
| Who calls this function? | `get_callers(symbol, project)` |
| What does this function call? | `get_callees(symbol, project)` |
| What breaks if I change this? | `get_dependents(symbol, project)` |
| What implements this interface? | `get_implementors(interface, project)` |
| Test coverage for a symbol | `get_tests_for(symbol, project)` |
| Class inheritance tree | `get_type_hierarchy(name, project)` |
| Functional module clusters | `get_communities(project)` |
| Which module does this belong to? | `get_community(symbol, project)` |
| Execution paths through code | `get_execution_flows(project)` |
| Risk of changing a symbol | `get_impact(symbol, project)` |
| What did my edits just break? | `detect_changes(project=project)` |
| Who last changed this & why | `blame_symbol(symbol, project)` |
| Recent commit activity | `recent_changes(project=project)` |
| Bug-prone files (churn) | `git_hotspots(project=project)` |
| Uncommitted WIP | `whats_changed(project=project)` |
| What does this file import? | `find_imports(path, project)` |
| Find unused/dead code | `find_dead_code(project)` |
| Search for code patterns (regex) | `find_pattern(pattern, project)` |
| Past decisions/learnings relevant to current work | `relevant_learnings(query)` |
| Record a decision, correction, or discovery | `record_learning(kind, content, reasoning)` |
| Log what this session accomplished | `conversation_summary(session_id, task_summary)` |
| Learning counts and trends | `learning_stats()` |

## Learnings (Conversation Intelligence)
Srclight maintains a workspace-level learnings database — decisions, corrections, discoveries, patterns, and conventions captured across sessions. Use these to build institutional memory:

- **At session start**: Call `relevant_learnings(query)` with the task description to check if prior decisions apply.
- **During work**: Call `record_learning(kind, content, reasoning)` when making important decisions or receiving corrections. Valid kinds: `decision`, `correction`, `discovery`, `pattern`, `blocker`, `convention`.
- **At session end**: Call `conversation_summary(session_id, task_summary)` to log what was accomplished.

## Document Indexing
Srclight indexes non-code files (PDF, DOCX, XLSX, HTML, CSV, email, images, text/RST) alongside source code. Documents become searchable symbols (sections, pages, tables) in the same search indexes.

- **Install**: `pip install 'srclight[docs,pdf]'` for document formats.
- **Scanned PDFs**: Install `pip install 'srclight[pdf,paddleocr]'` + system `poppler-utils` to OCR scanned/image-only PDF pages automatically. Native-text pages are unaffected. If paddleocr is not installed, scanned pages are silently skipped.
- **Image OCR**: Install `pip install 'srclight[docs,ocr]'` + system `tesseract-ocr` for OCR on standalone image files.
- After installing new extras, re-run `srclight index` (or `srclight workspace index`) to pick up documents.

## Adding a New Repo to the Workspace
To index a new repo and add it to the workspace, run these shell commands:
```
srclight workspace add /path/to/repo -w WORKSPACE_NAME
srclight workspace index -w WORKSPACE_NAME -p PROJECT_NAME --embed qwen3-embedding
srclight hook install --workspace WORKSPACE_NAME
```
The server picks up new projects automatically (no restart needed).

## Troubleshooting
- If ALL tools fail with `-32602: Invalid request parameters`, the MCP session is stale (e.g. the srclight service was restarted while this client was connected). Tell the user to **restart their editor/CLI** so the MCP client reconnects. Retrying the same calls will not help.

## Index Coverage
An index answers from the files it read, so what it never read cannot appear in
any result. `index_status()` names both sides: `indexed_extensions` (what it
reads) and `unindexed_extensions` (`{{extension: file count}}` this repo holds
that the last run walked past); `list_projects()` carries the same per project.
- `oversize_skipped` counts files refused on size; `failed_files` counts files
  that could not be read or parsed. Both are gaps like any unread file.
- The tally covers code that was never read: paths ignored on purpose and inert
  suffixes (config, data, manifests, suffixless files) are left out, while a
  document format missing its optional extra is counted.
- `find_pattern` attaches both whenever either is non-zero.
  Its `truncated` field reports **pagination only** — never scan coverage — so
  `truncated: false` alongside a non-empty tally is a partial answer, not a
  complete one. Cross-check those files with grep.
- An extension that should be read can be declared once: `srclight index --ext .inc=cpp`
  (recorded in the index, so the git hooks keep reading it). `--ext .inc=skip`
  declares the opposite: leave it unread, and count it as a gap.

## Prefer Srclight Over Grep
When srclight is available, ALWAYS prefer these tools over grep/find/cat:
- **Instead of grep/rg**: Use `hybrid_search(query)` or `search_symbols(query)` — returns ranked, structured results with file paths, line numbers, and symbol context.
- **Instead of find/ls**: Use `symbols_in_file(path)` — returns a structured table of contents for any file.
- **Instead of reading entire files**: Use `get_symbol(name)` — returns just the function/class you need with full source.
- Srclight results include relationship data (callers, callees, tests) that grep cannot provide.
- Grep sees text. Srclight sees code structure.

## Setup and server control
- `setup_guide()` — Structured instructions for agents: how to add a workspace, connect Cursor, where config lives, how to index with embeddings, hook install. Call when the user or agent needs setup steps.
- `server_stats()` — When the server started and uptime (for "how long has srclight been up").
- `restart_server()` — (SSE only) Exit so a process manager can restart. Allowed by default; set SRCLIGHT_ALLOW_RESTART=0 to disable.
"""


def _build_dynamic_instructions() -> str:
    """Build the dynamic section of the server instructions from current workspace state."""
    lines = []
    try:
        if _is_workspace_mode():
            wdb = _get_workspace_db()
            projects = wdb.list_projects()
            project_count = len(projects)
            total_files = sum(p.get("files", 0) for p in projects)
            total_symbols = sum(p.get("symbols", 0) for p in projects)
            total_edges = sum(p.get("edges", 0) for p in projects)
            project_names = [p.get("name", "?") for p in projects[:10]]

            lines.append(f"## Your Workspace: {_workspace_name}")
            lines.append(f"You have access to **{project_count} indexed project{'s' if project_count != 1 else ''}** "
                         f"containing **{total_files:,} files**, **{total_symbols:,} symbols**, "
                         f"and **{total_edges:,} relationships**.")
            if project_names:
                names_str = ", ".join(project_names)
                if project_count > 10:
                    names_str += f", ... and {project_count - 10} more"
                lines.append(f"Projects: {names_str}")
            lines.append("")
            lines.append("You can search across all projects at once, trace function calls, "
                         "find who changed code and why, and discover relationships between symbols.")
            lines.append("")
        elif _db_path is not None:
            db = _get_db()
            stats = db.stats()
            lines.append("## Your Codebase")
            lines.append(f"You have access to **{stats['files']:,} files**, "
                         f"**{stats['symbols']:,} symbols**, "
                         f"and **{stats['edges']:,} relationships**.")
            if stats.get("languages"):
                lang_list = ", ".join(stats["languages"].keys())
                lines.append(f"Languages: {lang_list}")
            lines.append("")
            lines.append("You can search code, trace function calls, "
                         "find who changed code and why, and discover relationships between symbols.")
            lines.append("")
    except Exception:
        # If we can't get stats (e.g. DB not yet initialized), fall back to generic text
        lines.append("You have access to a code index with searchable symbols, call graphs, and git history.")
        lines.append("")

    return "\n".join(lines)


def _refresh_instructions() -> None:
    """Update the MCP server instructions with current workspace state."""
    try:
        dynamic = _build_dynamic_instructions()
        # `_mcp_server` was the FastMCP handle and does not exist on mcp v2's
        # MCPServer; the AttributeError went into a bare except and the whole
        # workspace walk was discarded for seven releases. `mcp.instructions` is
        # a read-only property, so the settable handle is the lowlevel server —
        # and create_initialization_options() reads it per connection, so a
        # value set here reaches every client that connects afterwards.
        mcp._lowlevel_server.instructions = _INSTRUCTIONS_TEMPLATE.format(
            dynamic_section=dynamic
        )
    except Exception as e:  # noqa: BLE001 -- startup must not die for a string
        logger.warning(
            "Could not install dynamic instructions (%s); clients will see the "
            "generic blob without workspace details", e,
        )


# Shared estate policy, vendored as a single generated file (mcpkit). This REPLACES the
# hand-written _StrictArgsMCP that shipped in 4fa3bee, which carried two defects the shared
# version has since fixed:
#   * it treated an EMPTY property set as "schema unknown" and skipped the check, so all five
#     zero-parameter tools (index_status among them) still accepted anything, silently;
#   * it refused at runtime but never stamped additionalProperties:false, so all 42 tools kept
#     advertising a permissive contract and callers kept sending extras.
# Two independent implementations of one 30-line policy were independently wrong -- which is the
# argument for sharing it rather than copying it.

from ironmcp import strict_server

mcp = strict_server(
    name="srclight",
    reconnect_hint="call index_status and reconnect the srclight MCP",
    instructions=_INSTRUCTIONS_TEMPLATE.format(
        dynamic_section="You have access to a code index with searchable symbols, call graphs, and git history.\n\n"
    ),
)

# Global state — initialized on first tool call or via configure()
_db: Database | None = None
_db_path: Path | None = None
_repo_root: Path | None = None

# Workspace mode state
_workspace_name: str | None = None
_workspace_db = None  # WorkspaceDB instance (lazy import to avoid circular)
_workspace_db_lock = threading.Lock()  # guards (re)creation across web worker threads
_workspace_config_mtime: float = 0.0  # mtime of workspace config at last load

# Vector cache (GPU-resident embedding matrix)
_vector_cache = None  # VectorCache instance (single-repo mode)

# Learnings DB (workspace-level, lazy)
_learnings_db = None  # LearningsDB instance


def _is_workspace_mode() -> bool:
    return _workspace_name is not None


def _read_index_signal(root: Path | None) -> dict | None:
    """Read the last-indexed signal file for a project root."""
    if root is None:
        return None
    signal_file = root / ".srclight" / "last-indexed"
    try:
        if signal_file.exists():
            return json.loads(signal_file.read_text())
    except Exception:
        pass
    return None


def _get_workspace_db():
    """Get or create the WorkspaceDB connection.

    Hot-reloads if the workspace config file has been modified (e.g. after
    `srclight workspace add`). This means you never need to restart the
    MCP server to pick up new repos.
    """
    with _workspace_db_lock:
        return _get_workspace_db_locked()


def _get_workspace_db_locked():
    global _workspace_db, _workspace_config_mtime

    from .workspace import WorkspaceConfig, WorkspaceDB

    config_path = WorkspaceConfig(name=_workspace_name).config_path

    # Check if config has changed since last load
    try:
        current_mtime = config_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    if _workspace_db is not None and current_mtime == _workspace_config_mtime:
        return _workspace_db

    # Config changed (or first load) — (re)create workspace connection
    if _workspace_db is not None:
        logger.info("Workspace config changed, reloading...")
        try:
            _workspace_db.close()
        except Exception:
            pass
        _workspace_db = None

    config = WorkspaceConfig.load(_workspace_name)
    _workspace_db = WorkspaceDB(config)
    _workspace_db.open()
    _workspace_config_mtime = current_mtime
    return _workspace_db


def _get_learnings_db():
    """Get or create the workspace-level LearningsDB."""
    global _learnings_db

    if _learnings_db is not None:
        return _learnings_db

    from .workspace import WorkspaceConfig
    from .learnings import LearningsDB

    if _workspace_name is None:
        raise RuntimeError("Learnings require workspace mode")

    config = WorkspaceConfig.load(_workspace_name)
    _learnings_db = LearningsDB(config.learnings_db_path)
    _learnings_db.open()
    _learnings_db.initialize()
    return _learnings_db


def _get_db() -> Database:
    """Get or create the database connection (single-repo mode)."""
    global _db, _db_path, _repo_root

    if _db is not None:
        return _db

    # Default: look for .srclight/index.db, walk up to find repo root
    # Also checks legacy .codelight/ paths and auto-migrates them.
    if _db_path is None:
        cwd = Path.cwd()
        check = cwd
        while check != check.parent:
            if (check / ".srclight" / "index.db").exists():
                _db_path = check / ".srclight" / "index.db"
                _repo_root = check
                break
            # Legacy: migrate .codelight/ → .srclight/ if found
            legacy = check / ".codelight"
            new_dir = check / ".srclight"
            if (legacy / "index.db").exists() and not new_dir.exists():
                try:
                    legacy.rename(new_dir)
                    logger.info("Migrated %s -> %s", legacy, new_dir)
                    _db_path = new_dir / "index.db"
                    _repo_root = check
                    break
                except OSError:
                    # Fall back to reading from old location
                    _db_path = legacy / "index.db"
                    _repo_root = check
                    break
            if (check / ".srclight.db").exists():
                _db_path = check / ".srclight.db"
                _repo_root = check
                break
            if (check / ".git").exists():
                _db_path = check / ".srclight" / "index.db"
                _repo_root = check
                break
            check = check.parent
        if _db_path is None:
            _db_path = cwd / ".srclight" / "index.db"
            _repo_root = cwd

    _db = Database(_db_path)
    _db.open()

    # Initialize if this database carries no schema. Do NOT test file size:
    # Database.open() runs `PRAGMA journal_mode=WAL`, which writes a 4096-byte
    # header, so `st_size == 0` is already false here and initialize() could
    # never fire for a new index — leaving a 4096-byte file with zero tables and
    # a `no such table: files` on the next statement. Asking the schema also
    # repairs a database left empty by the old guard.
    has_schema = _db.conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='files'"
    ).fetchone()[0]
    if not has_schema:
        _db.initialize()

    return _db


_vector_cache_rebuild_lock = threading.Lock()
_vector_cache_rebuild_failed_at: int | None = None
# True from the moment a rebuild releases our mmap until it has finished.
# Read without the lock on purpose: a reader must not queue behind a
# multi-second build just to learn it should not touch the file.
_vector_cache_rebuilding = False


def _rebuild_vector_cache(db):
    """Rebuild the stale sidecar and publish the result.

    Sync MCP tools run on anyio worker threads, so two searches can arrive
    here at once. Rebuilding is not cheap — a 27K x 4096 index is ~440 MB
    written and ~1.8 GB peak — and VectorCache._atomic_write uses a fixed
    temp name, so concurrent builders rename it out from under each other
    and both fail. One at a time, then.

    Our own mapping has to go before the write, or os.replace hits the same
    WinError 5 the hook did — so the old cache is invalidated in place, and a
    thread that is already inside search() with a reference to it sees an
    empty matrix and returns no hits for that one call. Narrow: a caller
    reaching search() has just passed is_valid(), which is false from the
    moment the writer bumped the version, so it must have passed it before
    that commit landed. Closing the window for good means writing the
    sidecar under a fresh name so the old mapping never blocks the swap —
    that is vector_cache._atomic_write's design, not this branch's.

    The rebuilt cache is published by rebinding, so nothing ever observes a
    half-swapped object. A failure is remembered against the database
    version that provoked it, so a sidecar this process cannot replace —
    another long-lived reader holding it mapped — costs one attempt, not one
    per search forever.
    """
    global _vector_cache, _vector_cache_rebuild_failed_at, _vector_cache_rebuilding

    from .vector_cache import VectorCache

    srclight_dir = _db_path.parent if _db_path else None
    if srclight_dir is None:
        return _vector_cache

    with _vector_cache_rebuild_lock:
        # Another thread may have rebuilt it while we waited.
        if _vector_cache is not None and _vector_cache.is_valid(db.conn):
            return _vector_cache

        version = VectorCache._get_db_version(db.conn)
        if _vector_cache_rebuild_failed_at == version:
            return _vector_cache

        logger.info("Vector cache sidecar is stale — rebuilding in-process")
        stale = _vector_cache
        fresh = VectorCache(srclight_dir)
        try:
            # Announce before releasing: from here to the end of the build,
            # a reader finding an unloaded cache must leave the file alone
            # rather than mapping it again — see _get_vector_cache.
            _vector_cache_rebuilding = True
            if stale is not None:
                stale.invalidate()  # drop our mmap so os.replace can land
            fresh.build_from_db(db.conn)
        except Exception as e:
            # Our mmap is already gone, so there is nothing left to serve from
            # memory. Callers re-check is_valid and take the SQLite scan —
            # slow but correct — and the next call reloads from disk.
            logger.warning("Failed to rebuild vector cache sidecar: %s", e)
            _vector_cache_rebuild_failed_at = version
            _vector_cache = None
            return None
        finally:
            _vector_cache_rebuilding = False

        if not fresh.is_loaded():
            # build_from_db returns without loading anything when the index
            # holds no embeddings at all — a reindex that removed the last
            # embedded file. Publishing that would be publishing a dead
            # cache; the sidecar still on disk describes deleted symbols, so
            # leave it invalid and let the scan answer until embeddings
            # return.
            _vector_cache_rebuild_failed_at = version
            _vector_cache = None
            return None

        _vector_cache_rebuild_failed_at = None
        _vector_cache = fresh
        return _vector_cache


def _get_vector_cache():
    """Get or create the VectorCache (single-repo mode)."""
    global _vector_cache

    from .vector_cache import VectorCache

    db = _get_db()

    if _vector_cache is not None:
        # A flag-less `srclight index` — every git hook run — now embeds, and
        # it rebuilds the sidecar from its own process. On Windows that
        # rebuild cannot land: this server holds embeddings.npy mmap'd for its
        # whole life and os.replace refuses a mapped file, so the hook logs a
        # warning to reindex.log and leaves a sidecar older than the
        # embedding_cache_version it just bumped. Nothing else rebuilds it, so
        # every later semantic_search fell back to a full SQLite scan for the
        # life of the server. We hold the mapping, so we are who can refresh
        # it.
        if not _vector_cache.is_loaded():
            if _vector_cache_rebuilding:
                # A rebuild released this mapping deliberately and is about to
                # replace the file. Re-reading it now is what makes os.replace
                # fail on Windows, so take the SQLite scan for this one call.
                return _vector_cache
            # Empty and nobody is rebuilding: a rebuild dropped our mmap and
            # then could not write, or there was nothing to build from. Never
            # terminal — fall through to the cold path and read the disk
            # again, or the server would serve the SQLite scan for the rest of
            # its life even once a perfectly good sidecar appears.
            _vector_cache = None
        elif not _vector_cache.is_valid(db.conn):
            return _rebuild_vector_cache(db)
        else:
            return _vector_cache

    srclight_dir = _db_path.parent if _db_path else None
    if srclight_dir is None:
        return None

    cache = VectorCache(srclight_dir)
    if cache.sidecar_exists():
        try:
            cache.load_sidecar()
        except Exception as e:
            logger.warning("Failed to load vector cache sidecar: %s", e)
            return None
    _vector_cache = cache
    return _vector_cache


def _symbol_to_dict(sym) -> dict:
    """Convert a SymbolRecord to a clean dict for MCP response."""
    return {
        "id": sym.id,
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "kind": sym.kind,
        "signature": sym.signature,
        "file": sym.file_path,
        "start_line": sym.start_line,
        "end_line": sym.end_line,
        "line_count": sym.line_count,
        "doc_comment": sym.doc_comment,
        "visibility": sym.visibility,
    }


def _stamp_freshness(payload: dict, rel_paths) -> dict:
    """Stamp payload['index_freshness'] for the files this result draws on.

    SINGLE-REPO MODE ONLY: in workspace mode there is no one repo root to stat
    against per row, so the key is OMITTED rather than guessed — an unverifiable
    "fresh" would be the exact lie the freshness feature exists to kill (the
    README documents that absence means "not checked", never "fresh").
    """
    if _is_workspace_mode() or _repo_root is None:
        return payload
    paths = {p for p in rel_paths if p}
    if not paths:
        return payload
    from .freshness import annotate
    return annotate(payload, _get_db(), Path(_repo_root), paths)


def _project_required_error(tool_name: str) -> str:
    """Return a JSON error with the list of valid project names."""
    wdb = _get_workspace_db()
    project_names = sorted(e.name for e in wdb._all_indexable)
    return json.dumps({
        "error": f"In workspace mode, 'project' parameter is required for {tool_name}.",
        "available_projects": project_names,
        "hint": f"Try: {tool_name}(..., project=\"{project_names[0]}\")" if project_names else None,
    })


def _symbol_not_found_error(name: str, project: str | None = None) -> str:
    """Return a JSON error with recovery hints when a symbol lookup fails."""
    ctx = f" in {project}" if project else ""
    return json.dumps({
        "error": f"Symbol '{name}' not found{ctx}",
        "suggestions": [
            f"Try search_symbols(\"{name}\") for fuzzy keyword matching",
            f"Try hybrid_search(\"{name}\") for keyword + semantic matching",
        ],
    })


def _project_not_found_error(project: str) -> str:
    """Return a JSON error with fuzzy 'did you mean' suggestions for project names."""
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        project_names = sorted(e.name for e in wdb._all_indexable)
    else:
        project_names = []
    result: dict[str, object] = {"error": f"Project '{project}' not found"}
    if project_names:
        close = difflib.get_close_matches(project, project_names, n=3, cutoff=0.4)
        if close:
            result["did_you_mean"] = close
        result["available_projects"] = project_names
    return json.dumps(result)


# --- Tier 1: Instant tools ---


@mcp.tool()
def codebase_map(project: str | None = None) -> str:
    """Get a complete overview of the indexed codebase.

    Returns project stats, language breakdown, symbol counts by kind,
    directory structure with symbol counts, and hotspot files.
    Call this FIRST in any new session to orient yourself.

    In workspace mode, returns aggregated stats across all projects.

    Args:
        project: Optional project filter (workspace mode only)
    """
    _record_query(tool="codebase_map", project=project)
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        result = wdb.codebase_map(project=project)
        return json.dumps(result, indent=2)

    db = _get_db()
    stats = db.stats()
    state = db.get_index_state(str(_repo_root)) if _repo_root else None

    result = {
        "repo_root": str(_repo_root),
        "index": {
            "files": stats["files"],
            "symbols": stats["symbols"],
            "edges": stats["edges"],
            "db_size_mb": stats["db_size_mb"],
        },
        "languages": stats["languages"],
        "symbol_kinds": stats["symbol_kinds"],
        "directories": db.directory_summary(max_depth=2),
        "hotspot_files": db.hotspot_files(limit=10),
    }

    if state:
        result["index"]["last_commit"] = state.get("last_commit")
        result["index"]["indexed_at"] = state.get("indexed_at")

    signal = _read_index_signal(_repo_root)
    if signal:
        result["index"]["last_indexed_at"] = signal.get("timestamp")

    return json.dumps(result, indent=2)


@mcp.tool()
def search_symbols(
    query: str, kind: str | None = None, project: str | None = None, limit: int = 20,
) -> str:
    """Search for code symbols (functions, classes, methods, structs, etc.).

    Uses tiered search: symbol names → source code content → documentation.
    In workspace mode, searches across all projects simultaneously.

    Args:
        query: Search query — can be a symbol name, code fragment, or natural language
        kind: Optional filter: 'function', 'class', 'method', 'struct', 'enum', etc.
        project: Optional project filter (workspace mode only, e.g. 'my-app')
        limit: Max results to return (default 20)
    """
    _record_query(tool="search_symbols", query=query, project=project)
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        results = wdb.search_symbols(query, kind=kind, project=project, limit=limit)
    else:
        db = _get_db()
        results = db.search_symbols(query, kind=kind, limit=limit)

    if not results:
        return json.dumps({
            "query": query,
            "result_count": 0,
            "results": [],
            "hint": f"No keyword matches. Try hybrid_search(\"{query}\") for semantic matching.",
        }, indent=2)

    # Wrapped (not a bare list) so the payload can carry index_freshness — and so
    # the hit shape matches the no-hit shape above instead of flip-flopping.
    payload = {"query": query, "result_count": len(results), "results": results}
    _stamp_freshness(payload, (m.get("file") or m.get("file_path")
                               for m in results if isinstance(m, dict)))
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_symbol(name: str, project: str | None = None) -> str:
    """Get full details of a symbol by name.

    Returns the complete source code, signature, documentation,
    file location, and metadata. If multiple symbols share the name,
    all are returned. Falls back to substring matching if no exact match.

    In workspace mode, searches across all projects.

    Args:
        name: Symbol name (e.g., 'Dictionary', 'lookup', 'main')
        project: Optional project filter (workspace mode only)
    """
    _record_query(tool="get_symbol", query=name, project=project)
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        results = wdb.get_symbol(name, project=project)
        if not results:
            return _symbol_not_found_error(name)
        if len(results) == 1:
            return json.dumps(results[0], indent=2)
        return json.dumps({"match_count": len(results), "symbols": results}, indent=2)

    db = _get_db()
    symbols = db.get_symbols_by_name(name)
    if not symbols:
        return _symbol_not_found_error(name)

    if len(symbols) == 1:
        sym = symbols[0]
        result = _symbol_to_dict(sym)
        result["content"] = sym.content
        result["parameters"] = sym.parameters
        result["return_type"] = sym.return_type
        result["metadata"] = sym.metadata
        _stamp_freshness(result, [sym.file_path])
        return json.dumps(result, indent=2)

    results = []
    for sym in symbols:
        d = _symbol_to_dict(sym)
        d["content"] = sym.content
        d["parameters"] = sym.parameters
        d["return_type"] = sym.return_type
        results.append(d)

    payload = {"match_count": len(results), "symbols": results}
    _stamp_freshness(payload, (d.get("file") for d in results))
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_signature(name: str) -> str:
    """Get just the signature of a symbol (lightweight, for planning).

    Returns only the function/method signature without the full body.
    Use this when you need to understand an API without reading all the code.
    Returns all matches if multiple symbols share the name.

    Args:
        name: Symbol name
    """
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        results = wdb.get_symbol(name)
        if not results:
            return _symbol_not_found_error(name)
        sigs = [
            {
                "project": r["project"],
                "name": r["name"],
                "signature": r.get("signature"),
                "kind": r["kind"],
                "file": r["file"],
                "line": r["start_line"],
                "doc": r.get("doc_comment"),
            }
            for r in results
        ]
        if len(sigs) == 1:
            return json.dumps(sigs[0], indent=2)
        return json.dumps({"match_count": len(sigs), "signatures": sigs}, indent=2)

    db = _get_db()
    symbols = db.get_symbols_by_name(name, limit=10)
    if not symbols:
        return _symbol_not_found_error(name)

    results = []
    for sym in symbols:
        results.append({
            "name": sym.name,
            "signature": sym.signature,
            "kind": sym.kind,
            "file": sym.file_path,
            "line": sym.start_line,
            "doc": sym.doc_comment,
        })

    if len(results) == 1:
        _stamp_freshness(results[0], [results[0].get("file")])
        return json.dumps(results[0], indent=2)
    payload = {"match_count": len(results), "signatures": results}
    _stamp_freshness(payload, (r.get("file") for r in results))
    return json.dumps(payload, indent=2)


@mcp.tool()
def symbols_in_file(path: str, project: str | None = None) -> str:
    """List all symbols defined in a specific file.

    Returns a table of contents: every function, class, method, struct, etc.
    in the file, ordered by line number. Use this instead of reading a file
    to understand its structure.

    Args:
        path: Relative file path (e.g., 'src/libdict/dictionary.cpp')
        project: Project name (required in workspace mode if ambiguous)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("symbols_in_file")
        wdb = _get_workspace_db()
        all_results = []
        for batch in wdb._iter_batches(project_filter=project):
            from .workspace import _sanitize_schema_name
            for schema, project_name in batch:
                try:
                    rows = wdb.conn.execute(
                        f"""SELECT s.name, s.qualified_name, s.kind, s.signature, s.start_line,
                                  s.end_line, s.doc_comment
                           FROM [{schema}].symbols s
                           JOIN [{schema}].files f ON s.file_id = f.id
                           WHERE f.path = ?
                           ORDER BY s.start_line""",
                        (path,),
                    ).fetchall()
                    all_results.extend({
                        "name": r["name"],
                        **({"qualified_name": r["qualified_name"]}
                           if r["qualified_name"] and r["qualified_name"] != r["name"] else {}),
                        "kind": r["kind"],
                        "signature": r["signature"],
                        "line": r["start_line"],
                        "end_line": r["end_line"],
                        "doc": r["doc_comment"][:100] if r["doc_comment"] else None,
                    } for r in rows)
                except Exception:
                    pass
        return json.dumps({
            "project": project,
            "file": path,
            "symbol_count": len(all_results),
            "symbols": all_results,
        }, indent=2)

    db = _get_db()
    symbols = db.symbols_in_file(path)
    if not symbols:
        return json.dumps({"error": f"No symbols found in '{path}'"})

    result = []
    for sym in symbols:
        result.append({
            "name": sym.name,
            # Three `CheckFlag`s of three classes read alike without it.
            **({"qualified_name": sym.qualified_name}
               if sym.qualified_name and sym.qualified_name != sym.name else {}),
            "kind": sym.kind,
            "signature": sym.signature,
            "line": sym.start_line,
            "end_line": sym.end_line,
            "doc": sym.doc_comment[:100] if sym.doc_comment else None,
        })

    payload = {"file": path, "symbol_count": len(result), "symbols": result}
    _stamp_freshness(payload, [path])
    return json.dumps(payload, indent=2)


# --- Tier 2: Graph tools ---


def _union_edges(syms, fetch) -> list[dict]:
    """The edges of every symbol a graph query resolved to, each one once.

    Overloads and a declaration/definition pair share their callers: name
    resolution links one call to each of them, and listing it once per
    symbol would repeat the same location.
    """
    seen: set[tuple] = set()
    edges: list[dict] = []
    for sym in syms:
        for edge in fetch(sym.id):
            key = (edge["symbol"].id, edge["edge_type"])
            if key not in seen:
                seen.add(key)
                edges.append(edge)
    return edges


def _matched_symbols(db: Database, syms, incoming: bool = True) -> dict:
    """Name what a qualified name was resolved to, when it was more than one.

    `C::f` reaches `ns::C::f` — and the same class in another namespace, or a
    global class of that name. The merged answer is only honest if it says
    what it merged. For the calls into a symbol, it also says when the graph
    cannot hold them all.
    """
    names = db.graph_entity_names(syms)
    context = {"matched_symbols": names} if len(names) > 1 else {}
    if incoming:
        context.update(_graph_coverage(db, syms))
    return context


def _graph_coverage(db: Database, syms) -> dict:
    """Say when calls to a name are missing from the graph by design.

    An empty caller list reads as "nobody calls this". For a name the graph
    leaves out, or keeps only the calls the evidence decides, it means no
    such thing — and the answer has to say so.
    """
    from .indexer import GRAPH_MAX_FANOUT, graph_name_excluded

    for name in sorted({sym.name.rsplit("::", 1)[-1] for sym in syms}):
        search = f"find_pattern(r'\\b{re.escape(name)}\\s*\\(') lists the call sites."
        if graph_name_excluded(name):
            # A call written with the class, `C::name()`, names one member and
            # is kept.
            from .indexer import _without_template_args
            qualified = sorted({
                "::".join(_without_template_args(sym.qualified_name or sym.name).split("::")[-2:])
                for sym in syms if "::" in (sym.qualified_name or sym.name)})
            if qualified:
                return {"graph_note": (
                    f"Only calls written `{qualified[0]}(...)` are in the graph: `{name}` "
                    f"alone is too short or too common to tell a call from a variable. "
                    f"{search}")}
            return {"graph_note": (
                f"Calls to `{name}` are not in the graph: the name is too short or "
                f"too common to tell a call from a variable. {search}")}
        defined = db.count_graph_targets(name)
        if defined > GRAPH_MAX_FANOUT:
            return {"graph_note": (
                f"`{name}` is defined {defined} times. A call that only the name "
                f"could resolve — `p->{name}()` with the type of `p` unknown — is "
                f"left out, so callers may be missing. {search}")}
    return {}


_CALL_SITE_RE = re.compile(r"(?<![\w$])([A-Za-z_]\w*)\s*\(")


_TYPE_BEFORE_NAME_RE = re.compile(
    r"(?:^|[;{}])\s*(?:(?:const|static|constexpr|volatile|register)\s+)*"
    r"[A-Za-z_][\w:]*(?:\s*<[^;{}()]*>)?(?:\s*[*&]+\s*|\s+)$")
_TYPE_WORD_RE = re.compile(
    r"\s*(?:(?:const|static|constexpr|volatile|register)\s+)*([A-Za-z_][\w:]*)")
# The `:` opening a constructor's initializer list, after its parameters and
# any specifiers: `Foo(int v) noexcept :`.
_INITIALIZER_LIST_RE = re.compile(
    r"\)\s*(?:(?:noexcept|const|override|final|throw\s*\(\s*\))\s*)*:(?!:)")


def _declares_or_initializes(before: str) -> bool:
    """Whether a `name(` preceded by `before` is no call: a variable
    constructed on the spot (`Timer t(5);`, a statement of its own) or the
    first member initializer of a constructor (`Foo(int v) : m(v)`). A `:`
    after `case`, `default` or a `?` and an operator before the name
    (`a && f()`, `x * f()`) leave it a call."""
    from .indexer import _NOT_A_TYPE

    stripped = before.rstrip()
    statement = re.split(r"[;{}]", stripped)[-1]
    # A constructor's initializer list, `Foo(int v) noexcept : m(v), n(w)`:
    # every name opening an initializer at its top level. A `?` in the
    # statement makes the `:` a conditional's, `ok ? f(a) : g(b)`.
    opened = _INITIALIZER_LIST_RE.search(statement)
    if opened and "?" not in statement:
        rest = statement[opened.end():]
        if stripped.endswith(":") and not rest.strip():
            return True
        if stripped.endswith(",") and rest.count("(") == rest.count(")"):
            return True
    if stripped.endswith(":") and not stripped.endswith("::"):
        return False
    match = _TYPE_BEFORE_NAME_RE.search(before)
    if match is None:
        return False
    type_word = _TYPE_WORD_RE.match(before[match.start():].lstrip(";{} \t\r\n"))
    return type_word is not None and type_word.group(1) not in _NOT_A_TYPE


def _callee_coverage(db: Database, syms, result: list[dict]) -> dict:
    """Name the calls a body makes that the graph leaves out by design.

    A callee list that silently skips `update()` and `calc()` reads as a
    body that never calls them. The names it calls but the graph excludes —
    too common, or defined too often to tell which one — are listed. Only
    a callable's body is read, and in C/C++ what reads as a declaration
    (`Timer refresh(5)`) or a member initializer (`: refresh(0)`) is no call.
    """
    from .indexer import _NOT_A_TYPE, GRAPH_MAX_FANOUT, graph_name_excluded
    from .refmask import mask_noncode

    listed = {e["name"].rsplit("::", 1)[-1] for e in result}
    called: set[str] = set()
    for sym in syms:
        if not sym.content or sym.kind not in ("function", "method", "template", "macro"):
            continue
        row = db.conn.execute(
            "SELECT language FROM files WHERE path = ?", (sym.file_path,)).fetchone()
        language = (row[0] if row else "") or ""
        text = mask_noncode(sym.content, language)
        own = sym.name.rsplit("::", 1)[-1]
        head_seen = False
        for m in _CALL_SITE_RE.finditer(text):
            name = m.group(1)
            if name == own and not head_seen:
                head_seen = True  # the definition's own name, not a call
                continue
            if language in ("c", "cpp") and _declares_or_initializes(
                    text[max(0, m.start() - 120):m.start()]):
                continue
            called.add(name)
    missing = []
    for name in sorted(called - listed):
        defined = db.count_graph_targets(name)
        if defined and (graph_name_excluded(name) or defined > GRAPH_MAX_FANOUT):
            missing.append(name)
    if not missing:
        return {}
    shown = ", ".join(f"`{n}`" for n in missing[:15]) + (" …" if len(missing) > 15 else "")
    return {"graph_note": (
        f"Calls to {shown} are not listed: these names are too common, or defined more "
        f"than {GRAPH_MAX_FANOUT} times, for the graph to tell which one is called. "
        f"find_pattern can locate the call sites.")}


def _short_signature(signature: str | None) -> str | None:
    """A signature on one line, cut at 200 characters."""
    if not signature:
        return None
    flat = " ".join(signature.split())
    return flat if len(flat) <= 200 else flat[:197] + "..."


def _edge_name(s) -> str:
    """The name an edge entry is listed under.

    A constructor defined in its class is named like the class: listed by
    its qualified name, `Box::Box`, it stays apart from the class itself.
    A class template's constructor, `Bag<T>::Bag` or `Bag::Bag::Bag` in
    its class, is listed once as `Bag::Bag`.
    """
    from .indexer import _without_template_args

    plain = _without_template_args(s.name or "").rsplit("::", 1)[-1]
    short = re.escape(plain)
    qualified = _without_template_args(getattr(s, "qualified_name", None) or "")
    if (s.kind in ("method", "function", "prototype", "template")
            and re.search(rf"(?:^|::){short}::{short}$", qualified)):
        return re.sub(rf"(?:^|(?<=::)){short}(?:::{short})+$",
                      lambda _: f"{plain}::{plain}", qualified)
    return s.name


def _dedup_edges(edges: list[dict]) -> list[dict]:
    """Deduplicate edges by symbol name, keeping the highest-confidence entry."""
    by_name: dict[str, dict] = {}
    for c in edges:
        s = c["symbol"]
        name = _edge_name(s)
        confidence = c["confidence"]
        entry = {
            "name": name,
            "kind": s.kind,
            "file": s.file_path,
            "line": s.start_line,
            "edge_type": c["edge_type"],
            "confidence": confidence,
        }
        # Which overload the edge reaches: its line alone does not say.
        signature = _short_signature(s.signature)
        if signature:
            entry["signature"] = signature
        # name_only: the call names a symbol of this name, and nothing tells
        # which of its homonyms it is — the receiver's type is not known.
        if c.get("resolution"):
            entry["resolution"] = c["resolution"]
        if name not in by_name:
            by_name[name] = entry
            by_name[name]["_locations"] = [(s.file_path, s.start_line, signature)]
        else:
            if (s.file_path, s.start_line, signature) not in by_name[name]["_locations"]:
                by_name[name]["_locations"].append((s.file_path, s.start_line, signature))
            if confidence > by_name[name]["confidence"]:
                by_name[name].pop("resolution", None)
                by_name[name].pop("signature", None)
                by_name[name].update(entry)
                by_name[name]["_locations"] = by_name[name]["_locations"]

    result = []
    for entry in by_name.values():
        locations = entry.pop("_locations")
        if len(locations) > 1:
            entry["locations"] = [{"file": f, "line": l, **({"signature": g} if g else {})}
                                  for f, l, g in locations]
        result.append(entry)

    result.sort(key=lambda r: (
        0 if r["edge_type"] == "inherits" else 1,
        -r["confidence"],
        r["name"],
    ))
    return result


@mcp.tool()
def get_callers(symbol_name: str, project: str | None = None) -> str:
    """Find all symbols that call or reference a given symbol.

    Answers: "Who calls this function?" / "What depends on this?"
    Note: In workspace mode, requires 'project' to specify which repo's graph to search.

    Args:
        symbol_name: Name of the symbol to find callers for
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("graph queries (get_callers/get_callees)")
        # Use a temporary single-project Database for graph queries
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        syms = db.get_graph_symbols(symbol_name)
        if not syms:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        callers = _union_edges(syms, db.get_callers)
        result = _dedup_edges(callers)
        matched = _matched_symbols(db, syms)
        db.close()
        return json.dumps({
            "project": project,
            "symbol": symbol_name,
            "caller_count": len(result),
            "callers": result,
            **matched,
        }, indent=2)

    db = _get_db()
    # A qualified method name stands for its declaration and its definition:
    # calls land on the one, the scanned body is the other's.
    syms = db.get_graph_symbols(symbol_name)
    if not syms:
        return _symbol_not_found_error(symbol_name)

    callers = _union_edges(syms, db.get_callers)
    result = _dedup_edges(callers)

    payload = {"symbol": symbol_name, "caller_count": len(result), "callers": result,
               **_matched_symbols(db, syms)}
    # A stale caller file makes the whole edge list suspect — stamp the union.
    _stamp_freshness(payload, (c.get("file") or c.get("file_path")
                               for c in result if isinstance(c, dict)))
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_callees(symbol_name: str, project: str | None = None) -> str:
    """Find all symbols that a given symbol calls or references.

    Answers: "What does this function call?" / "What are this symbol's dependencies?"
    Note: In workspace mode, requires 'project' to specify which repo's graph to search.

    Args:
        symbol_name: Name of the symbol to find callees for
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("graph queries (get_callers/get_callees)")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        syms = db.get_graph_symbols(symbol_name)
        if not syms:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        callees = _union_edges(syms, db.get_callees)
        result = _dedup_edges(callees)
        matched = {**_matched_symbols(db, syms, incoming=False),
                   **_callee_coverage(db, syms, result)}
        db.close()
        return json.dumps({
            "project": project,
            "symbol": symbol_name,
            "callee_count": len(result),
            "callees": result,
            **matched,
        }, indent=2)

    db = _get_db()
    syms = db.get_graph_symbols(symbol_name)
    if not syms:
        return _symbol_not_found_error(symbol_name)

    callees = _union_edges(syms, db.get_callees)
    result = _dedup_edges(callees)

    payload = {"symbol": symbol_name, "callee_count": len(result), "callees": result,
               **_matched_symbols(db, syms, incoming=False),
               **_callee_coverage(db, syms, result)}
    _stamp_freshness(payload, (c.get("file") or c.get("file_path")
                               for c in result if isinstance(c, dict)))
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_type_hierarchy(name: str, project: str | None = None) -> str:
    """Get the inheritance hierarchy for a class or struct.

    Shows both base classes (parents) and subclasses (children).
    Note: In workspace mode, requires 'project' to specify which repo.

    Args:
        name: Class or struct name (e.g., 'ICaptureService', 'TtsProvider')
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("get_type_hierarchy")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(name, project)
        base_classes = db.get_base_classes(sym.id)
        subclasses = db.get_subclasses(sym.id)
        db.close()
        return json.dumps({
            "project": project,
            "symbol": {"name": sym.name, "kind": sym.kind, "file": sym.file_path, "line": sym.start_line},
            "base_classes": [{"name": c["symbol"].name, "kind": c["symbol"].kind, "file": c["symbol"].file_path} for c in base_classes],
            "subclasses": [{"name": c["symbol"].name, "kind": c["symbol"].kind, "file": c["symbol"].file_path} for c in subclasses],
        }, indent=2)

    db = _get_db()
    sym = db.get_symbol_by_name(name)
    if sym is None:
        return _symbol_not_found_error(name)

    base_classes = db.get_base_classes(sym.id)
    subclasses = db.get_subclasses(sym.id)

    result = {
        "symbol": {
            "name": sym.name,
            "kind": sym.kind,
            "file": sym.file_path,
            "line": sym.start_line,
        },
        "base_classes": [
            {
                "name": c["symbol"].name,
                "kind": c["symbol"].kind,
                "file": c["symbol"].file_path,
                "line": c["symbol"].start_line,
            }
            for c in base_classes
        ],
        "subclasses": [
            {
                "name": c["symbol"].name,
                "kind": c["symbol"].kind,
                "file": c["symbol"].file_path,
                "line": c["symbol"].start_line,
            }
            for c in subclasses
        ],
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_tests_for(symbol_name: str, project: str | None = None) -> str:
    """Find test functions that cover a given symbol.

    Uses heuristic matching: test file paths + test function names containing
    the symbol name. Also returns any explicit 'tests' edges from the graph.

    Args:
        symbol_name: Name of the symbol to find tests for
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("this tool")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        tests = db.get_tests_for(symbol_name)
        db.close()
    else:
        db = _get_db()
        tests = db.get_tests_for(symbol_name)

    results = []
    for t in tests:
        s = t["symbol"]
        results.append({
            "name": s.name,
            "kind": s.kind,
            "file": s.file_path,
            "line": s.start_line,
            "confidence": t["confidence"],
        })

    return json.dumps({
        "symbol": symbol_name,
        "test_count": len(results),
        "tests": results,
    }, indent=2)


@mcp.tool()
def get_dependents(symbol_name: str, transitive: bool = False, project: str | None = None) -> str:
    """Find all symbols that depend on (call/reference) a given symbol.

    Answers: "What would break if I change this?" / "What's the blast radius?"

    With transitive=True, walks the caller graph recursively to show the full
    impact chain (up to 5 levels deep).

    Args:
        symbol_name: Name of the symbol to find dependents for
        transitive: If True, follow the dependency chain recursively
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("this tool")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        syms = db.get_graph_symbols(symbol_name)
        if not syms:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        deps = _union_edges(syms, lambda i: db.get_dependents(i, transitive=transitive))
        matched = _matched_symbols(db, syms)
        db.close()
    else:
        db = _get_db()
        # The same symbols get_callers answers for, or the two disagree.
        syms = db.get_graph_symbols(symbol_name)
        if not syms:
            return _symbol_not_found_error(symbol_name)
        deps = _union_edges(syms, lambda i: db.get_dependents(i, transitive=transitive))
        matched = _matched_symbols(db, syms)

    result = _dedup_edges(deps)
    return json.dumps({
        "symbol": symbol_name,
        "transitive": transitive,
        "dependent_count": len(result),
        "dependents": result,
        **matched,
    }, indent=2)


@mcp.tool()
def get_implementors(interface_name: str, project: str | None = None) -> str:
    """Find all classes that implement or inherit from an interface/base class.

    Answers: "What classes implement this interface?" / "What are the concrete types?"

    Args:
        interface_name: Name of the interface or base class
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("get_implementors")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(interface_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(interface_name, project)
        impls = db.get_implementors(sym.id)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(interface_name)
        if sym is None:
            return _symbol_not_found_error(interface_name)
        impls = db.get_implementors(sym.id)

    results = [
        {
            "name": c["symbol"].name,
            "kind": c["symbol"].kind,
            "file": c["symbol"].file_path,
            "line": c["symbol"].start_line,
        }
        for c in impls
    ]

    return json.dumps({
        "interface": interface_name,
        "implementor_count": len(results),
        "implementors": results,
    }, indent=2)


@mcp.tool()
def check_freshness(paths: list[str] | None = None, project: str | None = None) -> str:
    """Is the index current for these files (or the whole index)?

    Compares on-disk files against the index (mtime+size fast path, content-hash
    fallback; never writes). Use BEFORE trusting symbol results on a repo under
    active edit, or when a result's `index_freshness` flagged staleness.

    Args:
        paths: Repo-relative paths to check. Omit to check every indexed file
               (cheap: unchanged files cost one stat each).
        project: Project name (required in workspace mode).
    """
    from .freshness import file_freshness, freshness_summary

    if _is_workspace_mode():
        if not project:
            return _project_required_error("check_freshness")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        proj_path = config.projects.get(project)
        if not proj_path:
            return _project_not_found_error(project)
        repo_root = Path(proj_path)
        db_path = repo_root / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        try:
            rels = paths if paths is not None else [
                r["path"] for r in db.conn.execute("SELECT path FROM files")
            ]
            statuses = file_freshness(db, repo_root, rels)
        finally:
            db.close()
    else:
        db = _get_db()
        repo_root = _repo_root or Path.cwd()
        rels = paths if paths is not None else [
            r["path"] for r in db.conn.execute("SELECT path FROM files")
        ]
        statuses = file_freshness(db, repo_root, rels)

    return json.dumps(
        {"index_freshness": freshness_summary(statuses), "checked": len(statuses)},
        indent=2,
    )


def _indexed_extensions(db: Database | None = None) -> list[str]:
    """Every extension this index reads — source, documents, and declared extras.

    A suffix declared unreadable is removed, including a built-in one: this
    is the answer to "what was read", and it cannot name an extension that
    the same payload reports as a gap.
    """
    from .extractors import DOCUMENT_EXTENSIONS
    from .languages import SKIP_LANGUAGE, code_extensions
    exts = set(code_extensions()) | set(DOCUMENT_EXTENSIONS)
    if db is not None:
        overrides = db.get_extension_overrides()
        exts |= {e for e, lang in overrides.items() if lang != SKIP_LANGUAGE}
        exts -= {e for e, lang in overrides.items() if lang == SKIP_LANGUAGE}
    return sorted(exts)


def _unindexed_warning(unindexed: dict[str, int], oversize: int = 0,
                       failed: int = 0) -> dict[str, object]:
    """Fields that keep a result from reading as a complete scan.

    A result whose completeness field says `truncated: false` is taken to
    mean the whole tree was searched. It only ever meant the page was not
    cut short, so when files were never read, the result says so itself.
    """
    if not unindexed and not oversize and not failed:
        return {}
    reasons = []
    if unindexed:
        reasons.append(
            f"{sum(unindexed.values())} file(s) carry an extension this index does not read"
        )
    if oversize:
        reasons.append(f"{oversize} file(s) exceeded the size limit")
    if failed:
        reasons.append(f"{failed} file(s) could not be read or parsed")
    fields: dict[str, object] = {
        "unindexed_note": (
            f"{' and '.join(reasons)}, so they were never scanned and this result is not a "
            f"whole-tree answer; `truncated` reports pagination only. Call index_status() "
            f"for the extensions that are read."
        ),
    }
    if unindexed:
        fields["unindexed_extensions"] = unindexed
    if oversize:
        fields["oversize_skipped"] = oversize
    if failed:
        fields["failed_files"] = failed
    return fields


@mcp.tool()
def index_status() -> str:
    """Check the current state of the code index.

    Reports which extensions the index reads (`indexed_extensions`) and
    which ones this repo holds but the index walked past
    (`unindexed_extensions`), so a gap is visible without comparing a
    result to a grep.

    In workspace mode, shows per-project stats.
    """
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        projects = wdb.list_projects()
        # Every project's declared extensions count as read, since a
        # workspace answer can come from any of them — and `find_pattern`'s
        # note sends the caller here in this mode too.
        from .languages import SKIP_LANGUAGE
        declared: set[str] = set()
        unreadable: set[str] = set()
        for p in projects:
            for ext, lang in (p.get("extension_overrides") or {}).items():
                (unreadable if lang == SKIP_LANGUAGE else declared).add(ext)
        return json.dumps({
            "mode": "workspace",
            "workspace": _workspace_name,
            # A suffix one project declares unreadable stays listed when
            # another reads it — the answer can still come from that one.
            "indexed_extensions": sorted(
                (set(_indexed_extensions()) | declared) - (unreadable - declared)
            ),
            "projects": projects,
        }, indent=2)

    db = _get_db()
    stats = db.stats()
    state = db.get_index_state(str(_repo_root)) if _repo_root else None

    result = {
        "mode": "single",
        "repo_root": str(_repo_root),
        "db_path": str(_db_path),
        "files": stats["files"],
        "symbols": stats["symbols"],
        "edges": stats["edges"],
        "db_size_mb": stats["db_size_mb"],
        "languages": stats["languages"],
        "indexed_extensions": _indexed_extensions(db),
        # What the last run walked past, as {extension: file count}. Empty
        # means the run indexed everything it saw — the one case where a
        # result may be read as covering the whole tree.
        "unindexed_extensions": db.get_unindexed_extensions(),
        # Files srclight recognised but refused on size — its own limit, so a
        # zero here is part of the same affirmative signal.
        "oversize_skipped": db.get_oversize_skipped(),
        # Files the run could not read or parse — unread like any other.
        "failed_files": db.get_failed_files(),
    }

    if state:
        result["last_commit"] = state.get("last_commit")
        result["indexed_at"] = state.get("indexed_at")

    signal = _read_index_signal(_repo_root)
    if signal:
        result["last_indexed_at"] = signal.get("timestamp")

    # Whole-index freshness COUNTS only (stat fast path makes this cheap) — the
    # dashboard number; per-path detail lives in the check_freshness probe.
    if _repo_root is not None:
        from .freshness import FRESH, file_freshness
        rels = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
        statuses = file_freshness(db, Path(_repo_root), rels)
        stale_n = sum(1 for s in statuses.values() if s != FRESH)
        result["index_freshness"] = {"checked": len(statuses), "stale_count": stale_n}

    return json.dumps(result, indent=2)


@mcp.tool()
def list_projects() -> str:
    """List all projects in the workspace with stats.

    Only available in workspace mode. Shows files, symbols, languages,
    and DB size for each project.
    """
    if not _is_workspace_mode():
        return json.dumps({"error": "Not in workspace mode. Start with --workspace NAME"})

    wdb = _get_workspace_db()
    projects = wdb.list_projects()
    return json.dumps({
        "workspace": _workspace_name,
        "project_count": len(projects),
        "projects": projects,
    }, indent=2)


@mcp.tool()
async def reindex(path: str | None = None, embed: bool = True) -> str:
    """Trigger re-indexing of the codebase or a specific path.

    Incrementally updates the index — only re-parses files whose content
    has changed since the last index.

    Args:
        path: Optional specific directory to re-index (default: entire repo)
        embed: Also refresh embeddings, using the model this index already
            holds (or SRCLIGHT_EMBED_MODEL for an index that holds none yet).
            Pass False for a keyword-only refresh when you only need
            search_symbols current: embedding a large backlog calls the
            embedding model and can take minutes. Not free, though — a
            reindex drops the embeddings of every symbol it replaces, and
            with embed=False nothing puts them back, so semantic_search and
            hybrid_search lose coverage on exactly the files being edited.
            Does nothing when no model is configured or recorded.
    """
    global _vector_cache
    # `path` is used as an index ROOT, not a filter: Indexer reads the whole
    # database's file list and DELETES every file not under it. So a foreign
    # path does not pollute an index, it empties it — measured, 3 files to 1
    # with a success JSON. And `_get_db()` resolves the database by walking up
    # from the SERVER'S working directory, not from `path`, so in workspace mode
    # the two are unrelated entirely.
    #
    # Refuse in BOTH modes. An earlier guard covered only workspace mode, which
    # is the safe one; the published plugin runs `serve --transport stdio` with
    # no --workspace, from the user's own repository, so it never fired where it
    # was needed. Sub-path reindexing should return as a FILTER, not a new root.
    if path is not None:
        requested = Path(path).resolve()
        configured = Path(_repo_root).resolve() if _repo_root else None
        if configured is None or requested != configured:
            return json.dumps({
                "error": "reindex does not accept a path",
                "reason": "`path` is treated as a new index root and removes every "
                          "indexed file outside it, which empties the index.",
                "hint": "Reindex the configured repository with reindex(), or index "
                        "another tree from the CLI: `srclight index /path --embed`",
                "configured_root": str(configured) if configured else None,
            })
    root = Path(path) if path else _repo_root
    if root is None:
        return json.dumps({"error": "No repo root configured"})

    root = root.resolve()
    db = _get_db()
    # Release the vector cache BEFORE indexing, not after: the embedding pass
    # rewrites embeddings.npy through os.replace, which fails on Windows while
    # this process still holds the old file mmap'd (np.load(mmap_mode="r")).
    # _build_embeddings swallows that failure, leaving a sidecar whose version
    # no longer matches the bumped embedding_cache_version — every later
    # semantic_search then falls back to a full SQLite scan.
    _vector_cache = None

    config = IndexConfig(root=root, disable_embeddings=not embed)
    # Resolve once and pin, as the CLI does: resolving again after the file
    # pass can disagree with what we report back to the caller.
    config.embed_model = resolve_embed_model(db, config)
    indexer = Indexer(db, config)
    stats = indexer.index(root)

    result = {
        "files_indexed": stats.files_indexed,
        "files_unchanged": stats.files_unchanged,
        "files_removed": stats.files_removed,
        "symbols_extracted": stats.symbols_extracted,
        "errors": stats.errors,
        "elapsed_seconds": round(stats.elapsed_seconds, 2),
        # Say what happened to the embeddings. symbols_embedded 0 with a
        # model configured is ambiguous on purpose — everything was already
        # embedded, or the provider could not be reached; the run log
        # distinguishes them. What it does tell you is that semantic_search
        # sees nothing new from this run.
        "embeddings": {
            "requested": embed,
            "model": config.embed_model,
            "symbols_embedded": stats.symbols_embedded,
        },
    }

    # Send notification to connected clients (MCP logging)
    try:
        ctx = mcp.get_context()
        await ctx.info(
            f"Reindex complete: {stats.files_indexed} files, "
            f"{stats.symbols_extracted} symbols indexed in {stats.elapsed_seconds:.1f}s"
        )
    except Exception:
        pass  # Best effort — client may not support notifications

    return json.dumps(result, indent=2)


# --- Tier 4: Git Change Intelligence ---


def _resolve_repo_root(project: str | None = None) -> Path | None:
    """Resolve repo root for git operations."""
    if _is_workspace_mode() and project:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        return Path(path) if path else None
    return _repo_root


@mcp.tool()
def blame_symbol(symbol_name: str, project: str | None = None) -> str:
    """Get git blame info for a symbol — who changed it, when, and why.

    Returns the last modifier, total unique commits/authors, age in days,
    and the list of commits that touched this symbol's line range.

    Args:
        symbol_name: Name of the symbol to blame
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    # Find the symbol in the index
    if _is_workspace_mode():
        db_path = repo_root / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)

    if sym is None:
        return _symbol_not_found_error(symbol_name)

    result = git_mod.blame_symbol(
        repo_root, sym.file_path, sym.start_line, sym.end_line
    )
    result["symbol"] = symbol_name
    result["file"] = sym.file_path
    result["lines"] = f"{sym.start_line}-{sym.end_line}"

    return json.dumps(result, indent=2)


@mcp.tool()
def recent_changes(
    n: int = 20, author: str | None = None,
    path_filter: str | None = None, project: str | None = None,
) -> str:
    """Get recent git commits with files changed.

    Answers: "What changed recently?" / "What has this author been working on?"

    Args:
        n: Number of commits to return (default 20)
        author: Filter by author name (substring match)
        path_filter: Filter by file path prefix (e.g., 'src/libdict/')
        project: Project name (workspace mode) or uses current repo
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        # Show recent changes across all projects
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        all_changes = []
        for entry in config.get_entries():
            root = Path(entry.path)
            if root.exists():
                commits = git_mod.recent_changes(
                    root, n=n, author=author, path_filter=path_filter
                )
                for c in commits:
                    c["project"] = entry.name
                all_changes.extend(commits)
        # Sort by date descending across all projects
        all_changes.sort(key=lambda c: c.get("date", ""), reverse=True)
        return json.dumps(all_changes[:n], indent=2)

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    commits = git_mod.recent_changes(
        repo_root, n=n, author=author, path_filter=path_filter
    )
    return json.dumps(commits, indent=2)


@mcp.tool()
def git_hotspots(
    n: int = 20, since: str | None = None, project: str | None = None,
) -> str:
    """Find most frequently changed files (churn hotspots / bug magnets).

    Files that change often are more likely to have bugs and be fragile.
    Use this to identify risky areas before making changes.

    Args:
        n: Number of files to return (default 20)
        since: Time period (e.g., '30.days', '3.months', '1.year')
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("git_hotspots")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    spots = git_mod.hotspots(repo_root, n=n, since=since)
    return json.dumps({
        "project": project or str(repo_root),
        "period": since or "all time",
        "hotspot_count": len(spots),
        "hotspots": spots,
    }, indent=2)


@mcp.tool()
def whats_changed(project: str | None = None) -> str:
    """Show uncommitted changes (work in progress).

    Returns staged, unstaged, and untracked files. Use this instead of
    running git status + git diff manually.

    Args:
        project: Project name (workspace mode) or uses current repo
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        # Show changes across all projects
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        all_results = {}
        for entry in config.get_entries():
            root = Path(entry.path)
            if root.exists():
                changes = git_mod.whats_changed(root)
                if changes["total_changes"] > 0:
                    all_results[entry.name] = changes
        return json.dumps({
            "projects_with_changes": len(all_results),
            "projects": all_results,
        }, indent=2)

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    result = git_mod.whats_changed(repo_root)
    return json.dumps(result, indent=2)


@mcp.tool()
def changes_to(symbol_name: str, n: int = 20, project: str | None = None) -> str:
    """Get the change history for a specific symbol's file.

    Shows commits that modified the file containing the symbol.
    Useful for understanding "why is it this way?" and recent activity.

    Args:
        symbol_name: Symbol name to track
        n: Number of commits to return
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    # Find the symbol to get its file
    if _is_workspace_mode():
        db_path = repo_root / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)

    if sym is None:
        return _symbol_not_found_error(symbol_name)

    commits = git_mod.changes_to_file(repo_root, sym.file_path, n=n)
    return json.dumps({
        "symbol": symbol_name,
        "file": sym.file_path,
        "commit_count": len(commits),
        "commits": commits,
    }, indent=2)


# --- Tier 5: Build & Configuration Intelligence ---


@mcp.tool()
def get_build_targets(project: str | None = None) -> str:
    """Get all build targets (libraries, executables) from the build system.

    Parses CMakeLists.txt, .csproj, package.json, Cargo.toml to extract
    targets with their sources, dependencies, and platform conditions.

    Args:
        project: Project name (required in workspace mode)
    """
    from . import build as build_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    info = build_mod.get_build_info(repo_root)
    return json.dumps(info, indent=2)


@mcp.tool()
def get_platform_variants(symbol_name: str, project: str | None = None) -> str:
    """Find platform-specific variants of a symbol.

    Scans C/C++/C# source files for #ifdef platform guards near the symbol.
    Essential for cross-platform projects — shows which platforms have
    specialized implementations.

    Args:
        symbol_name: Symbol name to search for
        project: Project name (required in workspace mode)
    """
    from . import build as build_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    variants = build_mod.get_platform_variants(repo_root, symbol_name)
    return json.dumps({
        "symbol": symbol_name,
        "variant_count": len(variants),
        "variants": variants,
    }, indent=2)


@mcp.tool()
def platform_conditionals(project: str | None = None, platform: str | None = None) -> str:
    """List all platform-conditional code blocks in the project.

    Scans for #ifdef, #if defined(), and similar preprocessor guards.
    Useful for understanding which code is platform-specific.

    Args:
        project: Project name (required in workspace mode)
        platform: Optional filter (e.g., 'windows', 'linux', 'apple', 'android')
    """
    from . import build as build_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    conditionals = build_mod.scan_platform_conditionals(repo_root)

    if platform:
        conditionals = [c for c in conditionals if platform in c["platforms"]]

    # Group by platform for summary
    platform_counts: dict[str, int] = {}
    for c in conditionals:
        for p in c["platforms"]:
            platform_counts[p] = platform_counts.get(p, 0) + 1

    return json.dumps({
        "total": len(conditionals),
        "platform_summary": platform_counts,
        "conditionals": conditionals[:100],  # Cap at 100 for readability
    }, indent=2)


# --- Tier 6: Semantic Search (Embeddings) ---


@mcp.tool()
def semantic_search(
    query: str, kind: str | None = None, project: str | None = None, limit: int = 10,
) -> str:
    """Find semantically similar code using embeddings.

    Unlike search_symbols (keyword-based), this finds conceptually similar
    code even when the exact terms don't match. Good for natural language
    queries like "find code that handles dictionary lookup" or
    "where is the authentication logic".

    Requires embeddings to be generated first (srclight index --embed).

    Args:
        query: Natural language description of what you're looking for
        kind: Optional filter by symbol kind (function, class, method, etc.)
        project: Project name (workspace mode) or uses current repo
        limit: Max results (default 10)
    """
    _record_query(tool="semantic_search", query=query, project=project)
    from .embeddings import get_provider, vector_to_bytes

    # Determine which model was used for embeddings
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        emb_stats = wdb.embedding_stats(project=project)
    else:
        db = _get_db()
        emb_stats = db.embedding_stats()

    if not emb_stats.get("model"):
        return json.dumps({
            "error": "No embeddings found. Run 'srclight index --embed <model>' first.",
            "hint": "Try: srclight index --embed qwen3-embedding",
        })

    model_name = emb_stats["model"]
    dims = emb_stats["dimensions"]

    try:
        provider = get_provider(model_name)
        query_vec = provider.embed_one(query)
        query_bytes = vector_to_bytes(query_vec)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to embed query: {e}",
            "model": model_name,
        })

    if _is_workspace_mode():
        results = wdb.vector_search(query_bytes, dims, project=project, kind=kind, limit=limit)
    else:
        cache = _get_vector_cache()
        results = db.vector_search(query_bytes, dims, kind=kind, limit=limit, cache=cache)

    return json.dumps({
        "query": query,
        "model": model_name,
        "result_count": len(results),
        "results": results,
    }, indent=2)


@mcp.tool()
def hybrid_search(
    query: str, kind: str | None = None, project: str | None = None, limit: int = 20,
) -> str:
    """Search using both keyword matching AND semantic similarity.

    Combines FTS5 text search with embedding-based semantic search using
    Reciprocal Rank Fusion (RRF). This is the most powerful search mode —
    it finds results that match either by exact keywords or by meaning.

    Falls back to keyword-only search if embeddings aren't available.

    Args:
        query: Search query (works with both keywords and natural language)
        kind: Optional filter by symbol kind
        project: Project name (workspace mode) or uses current repo
        limit: Max results (default 20)
    """
    _record_query(tool="hybrid_search", query=query, project=project)
    from .embeddings import get_provider, rrf_merge, vector_to_bytes

    # Get FTS results
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        fts_results = wdb.search_symbols(query, kind=kind, project=project, limit=limit * 2)
    else:
        db = _get_db()
        fts_results = db.search_symbols(query, kind=kind, limit=limit * 2)

    # Try to get embedding results
    embedding_results = []
    model_used = None
    embedding_error: str | None = None

    if _is_workspace_mode():
        emb_stats = wdb.embedding_stats(project=project)
    else:
        emb_stats = db.embedding_stats()

    if emb_stats.get("model"):
        model_name = emb_stats["model"]
        dims = emb_stats["dimensions"]
        try:
            provider = get_provider(model_name)
            query_vec = provider.embed_one(query)
            query_bytes = vector_to_bytes(query_vec)

            if _is_workspace_mode():
                embedding_results = wdb.vector_search(
                    query_bytes, dims, project=project, kind=kind, limit=limit * 2
                )
            else:
                cache = _get_vector_cache()
                embedding_results = db.vector_search(
                    query_bytes, dims, kind=kind, limit=limit * 2, cache=cache
                )
            model_used = model_name
        except Exception as e:
            # Fail fast and report clearly when the embedding provider
            # (e.g. Ollama) is unreachable or misconfigured. We still
            # return FTS-only results, but include the error so clients
            # can surface it instead of silently degrading.
            embedding_error = str(e)
            logger.warning("Embedding search failed, using FTS only: %s", e)

    if embedding_results:
        merged = rrf_merge(fts_results, embedding_results)
        final = merged[:limit]
        # A keyword hit says `line`, an embedding hit `start_line`: give every
        # merged hit both, so neither kind reads as having no position.
        for hit in final:
            if "start_line" in hit:
                hit.setdefault("line", hit["start_line"])
            elif "line" in hit:
                hit["start_line"] = hit["line"]
        payload: dict[str, object] = {
            "query": query,
            "mode": "hybrid (FTS5 + embeddings)",
            "model": model_used,
            "result_count": len(final),
            "results": final,
        }
        if not final:
            payload["hint"] = "No results. Try broadening your query or check that the index is up to date with reindex()."
        _stamp_freshness(payload, (m.get("file") or m.get("file_path")
                                   for m in final if isinstance(m, dict)))
        return json.dumps(payload, indent=2)
    else:
        payload = {
            "query": query,
            "mode": "keyword only (no embeddings available)",
            "result_count": min(len(fts_results), limit),
            "results": fts_results[:limit],
        }
        if not fts_results:
            payload["hint"] = "No results. Try broadening your query or check that the index is up to date with reindex()."
        if embedding_error is not None:
            payload["embedding_error"] = embedding_error
        _stamp_freshness(payload, (m.get("file") or m.get("file_path")
                                   for m in fts_results[:limit] if isinstance(m, dict)))
        return json.dumps(payload, indent=2)


@mcp.tool()
def embedding_status(project: str | None = None) -> str:
    """Check embedding coverage and model info.

    Shows how many symbols have embeddings, which model was used,
    and the coverage percentage.

    In single-repo mode the result also carries `configured_model`: the model
    the next flag-less run (a git hook, or reindex()) will actually use, null
    meaning it will not embed. `model` differs — it names a model already
    present in the rows, which after a switch can be one no run will use
    again. Workspace mode does not report `configured_model`: each project
    records its own, and they need not agree.

    Args:
        project: Project name (workspace mode) or uses current repo
    """
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        stats = wdb.embedding_stats(project=project)
    else:
        db = _get_db()
        stats = db.embedding_stats()
        # stats["model"] comes from an arbitrary embedding row, so after a
        # model switch it can name the old one. This is the model a flag-less
        # run — every git hook, and reindex() — will actually use; null means
        # such a run leaves embeddings alone. Resolved, not just read back:
        # SRCLIGHT_EMBED_MODEL is the third leg of the same chain, and an
        # index with no record still embeds when it is exported.
        stats["configured_model"] = resolve_embed_model(db, IndexConfig())

    if not stats.get("model"):
        stats["hint"] = "Run 'srclight index --embed <model>' to generate embeddings"

    return json.dumps(stats, indent=2)


@mcp.tool()
def embedding_health(project: str | None = None) -> str:
    """Check if the configured embedding provider is reachable.

    Uses embedding_stats() to find the active model, then performs a
    lightweight provider-specific health check (e.g. Ollama /api/tags), and
    for Ollama also reports whether the model is resident in memory (/api/ps).
    Returns a JSON blob with status, model, and any error message so
    clients can surface problems instead of silently degrading.
    """
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        stats = wdb.embedding_stats(project=project)
    else:
        db = _get_db()
        stats = db.embedding_stats()

    if not stats.get("model"):
        return json.dumps({
            "status": "no_embeddings",
            "detail": "No embeddings found in the index. Run 'srclight index --embed <model>' first.",
            "stats": stats,
        }, indent=2)

    model_name = stats["model"]
    from .embeddings import get_provider

    result: dict[str, object] = {
        "status": "unknown",
        "model": model_name,
        "dimensions": stats.get("dimensions"),
    }

    try:
        provider = get_provider(model_name)

        # OllamaProvider exposes is_available(), which hits /api/tags with a short timeout.
        is_available = getattr(provider, "is_available", None)
        if callable(is_available):
            ok = bool(is_available())
            result["provider"] = provider.name
            result["reachable"] = ok
            if ok:
                result["status"] = "ok"
                # Pulled != resident. With OLLAMA_MAX_LOADED_MODELS=1 another
                # client's model evicts ours; the next embed pays a cold load
                # (9 GB qwen3-embedding: 16-18 s from disk) that can exceed
                # the request timeout — and the timeout aborts the load, so
                # back-to-back queries thrash. Surface it so agents don't
                # read "ok" as "warm". (Incident 2026-08-31.)
                is_loaded = getattr(provider, "is_loaded", None)
                if callable(is_loaded):
                    resident = is_loaded()
                    result["resident"] = resident
                    if resident is False:
                        from .embeddings import _embed_request_timeout
                        result["warning"] = (
                            "Model is pulled but not resident in Ollama memory; "
                            "the next embed pays a cold load that may exceed "
                            f"SRCLIGHT_EMBED_REQUEST_TIMEOUT={_embed_request_timeout()}s "
                            "(hybrid_search then falls back to keyword-only). "
                            "Check for OLLAMA_MAX_LOADED_MODELS=1 with other clients "
                            "using different models."
                        )
            else:
                result["status"] = "error"
                result["error"] = "Embedding provider reported is_available() == False"
        else:
            # Fallback: we don't know how to health-check this provider without
            # running a full embed call. Leave status as unknown but include name.
            result["provider"] = getattr(provider, "name", model_name)
            result["status"] = "unknown"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


# --- Tier 7: Import Resolution ---


# Import extraction lives in srclight.imports so the indexer can use it
# without importing the server module. Re-exported here under the old names
# so find_imports and its tests are unchanged.
from .imports import IMPORT_PATTERNS, extract_imports as _extract_imports  # noqa: E402


@mcp.tool()
def find_imports(path: str, project: str | None = None) -> str:
    """Find and resolve import statements in a source file.

    Extracts all import/include/require statements and attempts to resolve
    them to indexed symbols. Answers: "What does this file depend on?"

    Supports: Python (import/from...import), JavaScript/TypeScript (import/require),
    C/C++ (#include), Go (import), Java/Kotlin (import), Dart (import),
    Swift (import), C# (using), PHP (use/require/include).

    Args:
        path: Relative file path (e.g., 'src/srclight/server.py')
        project: Project name (required in workspace mode if ambiguous)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("find_imports")
        wdb = _get_workspace_db()

        # Find the file in the workspace
        file_info = None
        for batch in wdb._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    row = wdb.conn.execute(
                        f"SELECT * FROM [{schema}].files WHERE path = ?",
                        (path,),
                    ).fetchone()
                    if row:
                        file_info = {"language": row["language"], "schema": schema}
                        break
                except Exception:
                    pass
            if file_info:
                break

        if not file_info:
            return json.dumps({"error": f"File '{path}' not found in project '{project}'"})

        language = file_info["language"]
        if not language or language not in IMPORT_PATTERNS:
            return json.dumps({
                "file": path,
                "language": language,
                "import_count": 0,
                "resolved_count": 0,
                "imports": [],
                "note": f"Import extraction not supported for language: {language}",
            }, indent=2)

        # Read file content from disk
        config_entries = [e for e in wdb._all_indexable if e.name == project]
        if not config_entries:
            return _project_not_found_error(project)
        project_root = Path(config_entries[0].path)
        file_path = project_root / path
        try:
            content = file_path.read_text(errors="replace")
        except OSError as e:
            return json.dumps({"error": f"Cannot read file: {e}"})

        raw_imports = _extract_imports(content, language)

        imports = []
        resolved_count = 0
        for imp in raw_imports:
            entry: dict = {
                "statement": imp["statement"],
                "module": imp["module"],
            }
            if imp["names"]:
                entry["names"] = imp["names"]

            resolved_to = None
            names_to_try = imp["names"] if imp["names"] else [imp["module"].split(".")[-1]]
            for name in names_to_try:
                for batch in wdb._iter_batches(project_filter=project):
                    for schema, pname in batch:
                        try:
                            row = wdb.conn.execute(
                                f"""SELECT s.name, s.kind, s.start_line, f.path as file_path
                                    FROM [{schema}].symbols s
                                    JOIN [{schema}].files f ON s.file_id = f.id
                                    WHERE s.name = ?
                                    LIMIT 1""",
                                (name,),
                            ).fetchone()
                            if row:
                                resolved_to = {
                                    "name": row["name"],
                                    "file": row["file_path"],
                                    "line": row["start_line"],
                                    "kind": row["kind"],
                                }
                                break
                        except Exception:
                            pass
                    if resolved_to:
                        break
                if resolved_to:
                    break

            if resolved_to:
                entry["resolved_to"] = resolved_to
                entry["status"] = "resolved"
                resolved_count += 1
            else:
                entry["resolved_to"] = None
                entry["status"] = "external"

            imports.append(entry)

        return json.dumps({
            "file": path,
            "language": language,
            "import_count": len(imports),
            "resolved_count": resolved_count,
            "imports": imports,
        }, indent=2)

    # Single-repo mode
    db = _get_db()
    file_rec = db.get_file(path)
    if not file_rec:
        return json.dumps({"error": f"File '{path}' not found in index"})

    language = file_rec.language
    if not language or language not in IMPORT_PATTERNS:
        return json.dumps({
            "file": path,
            "language": language,
            "import_count": 0,
            "resolved_count": 0,
            "imports": [],
            "note": f"Import extraction not supported for language: {language}",
        }, indent=2)

    if _repo_root:
        file_path = _repo_root / path
    else:
        file_path = Path(path)
    try:
        content = file_path.read_text(errors="replace")
    except OSError as e:
        return json.dumps({"error": f"Cannot read file: {e}"})

    raw_imports = _extract_imports(content, language)

    imports = []
    resolved_count = 0
    for imp in raw_imports:
        entry: dict = {
            "statement": imp["statement"],
            "module": imp["module"],
        }
        if imp["names"]:
            entry["names"] = imp["names"]

        resolved_to = None
        names_to_try = imp["names"] if imp["names"] else [imp["module"].split(".")[-1]]
        for name in names_to_try:
            result = db.resolve_import(name, hint_path=path)
            if result:
                resolved_to = result
                break

        if resolved_to:
            entry["resolved_to"] = resolved_to
            entry["status"] = "resolved"
            resolved_count += 1
        else:
            entry["resolved_to"] = None
            entry["status"] = "external"

        imports.append(entry)

    return json.dumps({
        "file": path,
        "language": language,
        "import_count": len(imports),
        "resolved_count": resolved_count,
        "imports": imports,
    }, indent=2)


# --- Tier 8: Code Analysis ---


@mcp.tool()
def find_dead_code(project: str | None = None, kind: str | None = None) -> str:
    """Find symbols that have no callers or references — potential dead code.

    Returns symbols that are defined but never referenced by any other symbol
    in the indexed codebase. Useful for cleanup and understanding which code
    is actually used.

    Excludes: main/entry points, __init__/__main__, test functions, and
    symbols in vendored/third-party code.

    Args:
        project: Project name (required in workspace mode)
        kind: Filter by symbol kind (e.g., 'function', 'class', 'method')
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("find_dead_code")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        dead = db.get_dead_symbols(kind=kind)
        db.close()
    else:
        db = _get_db()
        dead = db.get_dead_symbols(kind=kind)

    # Group by file for readability
    by_file: dict[str, list[dict]] = {}
    for sym in dead:
        entry = {
            "name": sym.name,
            "kind": sym.kind,
            "line": sym.start_line,
            "signature": sym.signature,
        }
        file_path = sym.file_path or "unknown"
        by_file.setdefault(file_path, []).append(entry)

    result: dict[str, object] = {
        "total_unreferenced": len(dead),
        "file_count": len(by_file),
        "by_file": by_file,
    }
    if project:
        result["project"] = project
    if kind:
        result["kind_filter"] = kind
    if not dead:
        result["hint"] = "No unreferenced symbols found. This may mean the codebase is well-connected, or edges haven't been indexed yet."

    # Dead-code verdicts on drifted files are stale advice — stamp the files involved.
    _stamp_freshness(result, (p for p in by_file if p != "unknown"))
    return json.dumps(result, indent=2)


@mcp.tool()
def find_pattern(
    pattern: str,
    project: str | None = None,
    language: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """Search for structural code patterns in symbol bodies.

    Goes beyond text grep by searching within parsed symbol boundaries.
    Patterns are matched against symbol source code with context about
    the containing function/class/method.

    Unlike grep, results include:
    - The symbol name and kind containing the match
    - File path and line numbers of the symbol
    - The match context within the symbol

    The response reports three counts, and they mean different things:
    `match_count` is how many SYMBOLS are returned (capped by `limit`),
    `matched_lines_total` is how many LINES matched inside them, and
    `truncated` says whether more symbols existed beyond `limit`. When
    `truncated` is true, `matched_lines_total` is a floor over what was
    returned, not a repo-wide total: the search stops once it has enough
    matching symbols to fill the page, so it never sees the rest of the
    index. (A pattern that matches nothing never reaches that point and does
    test every candidate symbol.)

    Pattern supports regex. Common patterns:
    - "Color\\\\(0x" — find raw color literals
    - "requests\\\\.get\\\\(" — find HTTP calls
    - "TODO|FIXME|HACK" — find code annotations
    - "except.*Exception" — find broad exception handlers
    - "sleep\\\\(" — find sleep calls
    - "eval\\\\(|exec\\\\(" — find dynamic code execution

    Args:
        pattern: Regex pattern to search for in symbol source code
        project: Optional project filter (workspace mode: filters to one project)
        language: Filter by language (e.g., 'python', 'javascript')
        kind: Filter by symbol kind (e.g., 'function', 'method')
        limit: Maximum results (default 50)
        offset: Skip this many matching symbols before collecting (default 0).
            With `truncated`, this pages through a result set larger than
            `limit`.
    """
    import re as _re

    # Validate regex
    try:
        _re.compile(pattern)
    except _re.error as e:
        return json.dumps({"error": f"Invalid regex pattern: {e}"}, indent=2)

    # Clamp before use: a negative limit would otherwise make `limit + 1`
    # below request 0 or fewer rows while `len(matches) > limit` stays true
    # for any non-negative match count, reporting `truncated: true` with 0
    # results.
    limit = max(0, limit)

    if _is_workspace_mode():
        if not project:
            return _project_required_error("find_pattern")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        matches = db.find_pattern_in_symbols(
            pattern, language=language, kind=kind, limit=limit + 1, offset=offset
        )
        unindexed = db.get_unindexed_extensions()
        oversize = db.get_oversize_skipped()
        failed = db.get_failed_files()
        db.close()
    else:
        db = _get_db()
        matches = db.find_pattern_in_symbols(
            pattern, language=language, kind=kind, limit=limit + 1, offset=offset
        )
        unindexed = db.get_unindexed_extensions()
        oversize = db.get_oversize_skipped()
        failed = db.get_failed_files()

    # Asking for limit + 1 is how truncation is detected: holding one more than
    # requested proves more exist. Exact, and the scan still stops early.
    truncated = len(matches) > limit
    matches = matches[:limit]
    matched_lines_total = sum(m["match_count"] for m in matches)

    # Group by file for readability
    by_file: dict[str, list[dict]] = {}
    for m in matches:
        file_path = m.pop("file", "unknown")
        by_file.setdefault(file_path, []).append(m)

    result: dict[str, object] = {
        "pattern": pattern,
        "match_count": len(matches),
        "matched_lines_total": matched_lines_total,
        "truncated": truncated,
        "offset": offset,
        "file_count": len(by_file),
        "by_file": by_file,
    }
    result.update(_unindexed_warning(unindexed, oversize, failed))
    if project:
        result["project"] = project
    if language:
        result["language_filter"] = language
    if kind:
        result["kind_filter"] = kind
    if not matches:
        result["hint"] = "No matches found. Try a broader pattern or check that symbols have been indexed."

    return json.dumps(result, indent=2)


# Set when run_server() or first tool runs — for server_stats
_server_start_time: float | None = None

# Query activity tracking (for Flutter app / web dashboard)
_last_query_time: float | None = None
_last_query_client: str | None = None
_query_count: int = 0

# UI event queue — polled by Flutter app via /api/ui_events.
_ui_events: list[dict] = []


# True while a request originates from the web dashboard's own polling. The
# watchman does not sign the guest book: dashboard traffic must never count as
# an agent query, or "last query 3s ago" becomes the page measuring its own pulse.
_dashboard_request: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "srclight_dashboard_request", default=False
)


# The agent ledger: what agents actually asked, newest last. Bounded; never
# holds file contents; queries are truncated. Exposed by /api/recent_queries.
_recent_queries: "collections.deque[dict]" = collections.deque(maxlen=200)
# Set by the --web startup warm-up while it loads stats and vector caches, so
# the dashboard can say "loading embeddings into memory" instead of "healthy".
_warming: str | None = None


def _record_query(
    client: str | None = None,
    *,
    tool: str | None = None,
    query: str | None = None,
    project: str | None = None,
) -> None:
    """Record that a tool was called (timestamp + what was asked)."""
    global _last_query_time, _last_query_client, _query_count
    if _dashboard_request.get():
        return
    _last_query_time = time.time()
    _query_count += 1
    if client:
        _last_query_client = client
    _recent_queries.append({
        "ts": datetime.fromtimestamp(_last_query_time, tz=timezone.utc).isoformat(),
        "tool": tool,
        "query": (query[:80] if isinstance(query, str) else None),
        "project": project,
        "client": client or _last_query_client,
    })


def recent_queries(limit: int = 20) -> list[dict]:
    """Newest first."""
    items = list(_recent_queries)[-max(1, min(limit, 200)):]
    items.reverse()
    return items


def _humanize_seconds(seconds: float) -> str:
    """4m 18s, 2h 14m, 1d 1h -- two units, largest first. Never "7200s"."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


@mcp.tool()
async def server_stats() -> str:
    """Return when this server process started and how long it has been running."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()
    now = time.time()
    uptime = now - _server_start_time
    started_at = datetime.fromtimestamp(_server_start_time, tz=timezone.utc)
    return json.dumps({
        "started_at": started_at.isoformat(),
        "started_at_epoch": _server_start_time,
        "uptime_seconds": round(uptime, 2),
        "uptime_human": _humanize_seconds(uptime),
    }, indent=2)


@mcp.tool()
async def restart_server() -> str:
    """Request the server to exit so a process manager can restart it (SSE only).

    Exits with code 0 so a wrapper can start a fresh process (e.g. loads updated
    code). Client must reconnect after restart. Restart is allowed by default;
    set SRCLIGHT_ALLOW_RESTART=0 to disable.
    """
    allow = os.environ.get("SRCLIGHT_ALLOW_RESTART", "1").strip().lower()
    if allow in ("0", "false", "no"):
        return json.dumps({
            "ok": False,
            "message": "Restart is disabled (SRCLIGHT_ALLOW_RESTART=0). Remove it or set to 1 to allow.",
            "hint": "Example: srclight serve --workspace NAME --transport sse --port 8742",
        }, indent=2)

    def _exit():
        _close_databases()   # os._exit skips atexit, SQLite close and the WAL checkpoint
        os._exit(0)

    asyncio.get_running_loop().call_later(0, _exit)
    return json.dumps({
        "ok": True,
        "message": "Server will exit now. Reconnect after your process manager restarts it.",
    }, indent=2)


@mcp.tool()
async def show_status(message: str = "") -> str:
    """Show the srclight dashboard window and return current status.

    Pops up the desktop app window (if running) and returns indexing stats.
    Use this when a user asks about their indexing status, project health,
    or wants to see what srclight is doing.

    Args:
        message: Optional message to display in the dashboard.
    """
    global _ui_events
    _ui_events.append({
        "type": "show_status",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    # Return current stats so the AI has the data in context too.
    stats: dict = {
        "query_count": _query_count,
    }
    if _last_query_time is not None:
        stats["last_query_ago_seconds"] = round(time.time() - _last_query_time, 1)
    if _last_query_client is not None:
        stats["last_query_client"] = _last_query_client
    try:
        map_result = await codebase_map()
        stats["codebase"] = json.loads(map_result)
    except Exception:
        pass
    return json.dumps(stats, indent=2)


@mcp.tool()
async def setup_guide() -> str:
    """Structured setup instructions for AI agents and users.
    Returns: how to add a workspace, connect Cursor, where config lives, how to index with embeddings, hook install."""
    from .workspace import WORKSPACES_DIR

    return json.dumps({
        "title": "Srclight setup guide for agents",
        "config_location": {
            "workspaces_dir": str(WORKSPACES_DIR),
            "description": "Workspace configs are JSON files: ~/.srclight/workspaces/{name}.json",
        },
        "steps": [
            {
                "step": 1,
                "title": "Create or use a workspace",
                "commands": [
                    "srclight workspace init WORKSPACE_NAME",
                    "srclight workspace add /path/to/repo -w WORKSPACE_NAME",
                ],
            },
            {
                "step": 2,
                "title": "Index the workspace (optionally with embeddings)",
                "commands": [
                    "srclight workspace index -w WORKSPACE_NAME",
                    "srclight workspace index -w WORKSPACE_NAME --embed qwen3-embedding",
                ],
                "notes": "Ollama on localhost:11434 for qwen3-embedding. --embed is passed once: "
                         "each index records its model and reuses it on every later run, git hooks "
                         "included (srclight index --forget-embed-model turns that off). Server "
                         "hot-reloads; no restart needed after indexing.",
            },
            {
                "step": 3,
                "title": "Install git hooks (optional, for auto-reindex)",
                "commands": ["srclight hook install --workspace WORKSPACE_NAME"],
            },
            {
                "step": 4,
                "title": "Start the MCP server and connect Cursor",
                "commands": [
                    "srclight serve --workspace WORKSPACE_NAME",
                    "# Or with web dashboard: srclight serve --workspace WORKSPACE_NAME --web",
                ],
                "notes": "Server binds to 127.0.0.1:8742. In Cursor MCP config use URL http://127.0.0.1:8742 (Streamable HTTP /mcp or SSE /sse). Start server before opening Cursor.",
            },
        ],
        "for_agents": "Call codebase_map() at session start. Use list_projects() to see repos. Use setup_guide() to get these steps for the user.",
        "after_upgrade": "After upgrading srclight (pip install -U srclight), restart the server and then restart your editor/CLI to pick up new tools. Existing MCP sessions only discover tools at connect time.",
    }, indent=2)


# --- Tier 7: Learnings (workspace-level conversation intelligence) ---


@mcp.tool()
def record_learning(
    kind: str,
    content: str,
    reasoning: str | None = None,
    project: str | None = None,
    scope: str = "workspace",
    confidence: float = 1.0,
    ttl_days: int | None = None,
    symbols: list[str] | None = None,
    source_type: str | None = None,
    source_ref: str | None = None,
) -> str:
    """Record a learning — a decision, correction, discovery, pattern, blocker, or convention.

    Learnings persist across sessions and are searchable via relevant_learnings().
    Use this to capture important decisions, corrections from the user, discovered
    patterns, blockers, or coding conventions.

    Args:
        kind: One of 'decision', 'correction', 'discovery', 'pattern', 'blocker', 'convention'
        content: The learning itself (what was decided/discovered/corrected)
        reasoning: Why this learning matters or the context behind it
        project: Project this applies to (omit for cross-project learnings)
        scope: 'workspace', 'project', 'file', or 'symbol'
        confidence: 0.0-1.0 confidence level (default 1.0)
        ttl_days: Auto-expire after N days (omit for permanent)
        symbols: Symbol names this learning relates to
        source_type: 'conversation', 'labbook', 'decisions_md', 'claude_md', 'agent_log'
        source_ref: Reference identifier (session ID, file path, etc.)
    """
    from .learnings import LearningRecord

    ldb = _get_learnings_db()
    rec = LearningRecord(
        kind=kind,
        content=content,
        reasoning=reasoning,
        scope=scope,
        project=project,
        confidence=confidence,
        ttl_days=ttl_days,
    )

    sources = None
    if source_type:
        sources = [{"type": source_type, "ref": source_ref or ""}]

    learning_id = ldb.record_learning(rec, symbols=symbols, sources=sources)
    return json.dumps({"learning_id": learning_id, "status": "recorded"})


@mcp.tool()
def conversation_summary(
    session_id: str,
    task_summary: str,
    project: str | None = None,
    model: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
) -> str:
    """Record a conversation session summary.

    Call at the end of a session to log what was accomplished.

    Args:
        session_id: Unique session identifier
        task_summary: Brief description of what was done
        project: Primary project worked on (if any)
        model: Model used (e.g. 'claude-opus-4-6')
        tokens_in: Input tokens consumed
        tokens_out: Output tokens generated
        cost_usd: Estimated cost in USD
    """
    from .learnings import ConversationRecord

    ldb = _get_learnings_db()
    rec = ConversationRecord(
        session_id=session_id,
        project=project,
        task_summary=task_summary,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
    )
    conv_id = ldb.record_conversation(rec)
    return json.dumps({"conversation_id": conv_id, "status": "recorded"})


@mcp.tool()
def relevant_learnings(
    query: str,
    project: str | None = None,
    kind: str | None = None,
    limit: int = 10,
) -> str:
    """Find relevant learnings using hybrid search (keyword + semantic).

    Searches past decisions, corrections, discoveries, patterns, blockers,
    and conventions. Returns the most relevant learnings for your current context.

    Call this at the START of a session or when making a decision that might
    have been addressed before.

    Args:
        query: What you're looking for (natural language or keywords)
        project: Filter to a specific project (omit for all)
        kind: Filter by kind: 'decision', 'correction', 'discovery', 'pattern', 'blocker', 'convention'
        limit: Max results (default 10)
    """
    ldb = _get_learnings_db()

    # FTS search
    fts_results = ldb.search_fts(query, kind=kind, project=project, limit=limit * 2)

    # Try embedding search if available
    embedding_results = []
    # (Embedding search is a bonus — FTS alone is sufficient)

    if embedding_results:
        results = ldb.hybrid_search(fts_results, embedding_results, limit=limit)
    else:
        results = fts_results[:limit]

    if not results:
        return json.dumps({"results": [], "message": "No relevant learnings found."})

    # Format results
    formatted = []
    for r in results:
        entry = {
            "kind": r["kind"],
            "content": r["content"],
            "project": r["project"],
            "created_at": r["created_at"],
        }
        if r.get("reasoning"):
            entry["reasoning"] = r["reasoning"]
        if r.get("rrf_score"):
            entry["rrf_score"] = r["rrf_score"]
        formatted.append(entry)

    return json.dumps({"results": formatted, "count": len(formatted)}, indent=2)


@mcp.tool()
def learning_stats(
    project: str | None = None,
    days: int | None = None,
) -> str:
    """Get learning statistics — counts by kind over time.

    Shows how many decisions, corrections, discoveries, etc. have been captured.

    Args:
        project: Filter to a specific project (omit for all)
        days: Look back N days (omit for all time)
    """
    ldb = _get_learnings_db()
    s = ldb.stats(project=project, days=days)
    return json.dumps(s, indent=2)


@mcp.tool()
def get_communities(project: str | None = None) -> str:
    """Get detected functional communities (module clusters) in the call graph.

    Communities are auto-detected using the Louvain algorithm on call-graph edges.
    Each community has a TF-IDF auto-label, member count, and cohesion score.

    Args:
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("community analysis")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        communities = db.get_communities()
        db.close()
    else:
        db = _get_db()
        communities = db.get_communities()

    if not communities:
        return json.dumps({"info": "No communities detected. Run reindex to generate.", "communities": []})

    return json.dumps({
        "project": project,
        "community_count": len(communities),
        "communities": communities,
    }, indent=2)


@mcp.tool()
def get_community(symbol_name: str, project: str | None = None) -> str:
    """Get the community that a specific symbol belongs to, with all co-members.

    Answers: "What functional module does this symbol belong to?"

    Args:
        symbol_name: Name of the symbol to look up
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("community lookup")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        comm_id = db.get_community_for_symbol(sym.id)
        if comm_id is None:
            db.close()
            return json.dumps({"symbol": symbol_name, "community": None, "info": "Symbol not assigned to any community"})
        members = db.get_community_members(comm_id)
        communities = db.get_communities()
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            return _symbol_not_found_error(symbol_name)
        comm_id = db.get_community_for_symbol(sym.id)
        if comm_id is None:
            return json.dumps({"symbol": symbol_name, "community": None, "info": "Symbol not assigned to any community"})
        members = db.get_community_members(comm_id)
        communities = db.get_communities()

    # Find the community metadata
    comm_info = next((c for c in communities if c["id"] == comm_id), None)

    return json.dumps({
        "symbol": symbol_name,
        "community_id": comm_id,
        "label": comm_info["label"] if comm_info else "unknown",
        "keywords": comm_info.get("keywords", []) if comm_info else [],
        "cohesion": comm_info["cohesion"] if comm_info else None,
        "member_count": len(members),
        "members": members,
    }, indent=2)


@mcp.tool()
def get_execution_flows(project: str | None = None) -> str:
    """Get traced execution flows through the call graph.

    Flows are paths from entry points (like main, run, handle) through the call graph,
    showing how execution moves across functional communities.

    Args:
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("execution flow analysis")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        flows = db.get_execution_flows()
        db.close()
    else:
        db = _get_db()
        flows = db.get_execution_flows()

    if not flows:
        return json.dumps({"info": "No execution flows traced. Run reindex to generate.", "flows": []})

    return json.dumps({
        "project": project,
        "flow_count": len(flows),
        "flows": flows,
    }, indent=2)


@mcp.tool()
def get_impact(symbol_name: str, project: str | None = None) -> str:
    """Compute blast radius and risk for modifying a symbol.

    Answers: "How risky is it to change this?" Analyzes direct/transitive dependents,
    affected communities, and affected execution flows to assign a risk level
    (LOW / MEDIUM / HIGH / CRITICAL).

    Args:
        symbol_name: Name of the symbol to analyze
        project: Project name (required in workspace mode)
    """
    from .community import compute_impact

    if _is_workspace_mode():
        if not project:
            return _project_required_error("impact analysis")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        syms = db.get_graph_symbols(symbol_name)
        if not syms:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        # Build sym_to_community map from stored data
        communities = db.get_communities()
        sym_to_comm: dict[int, int] = {}
        for c in communities:
            for m in db.get_community_members(c["id"]):
                sym_to_comm[m["id"]] = c["id"]
        flows = db.get_execution_flows()
        # Reconstruct flow step dicts for compute_impact
        flow_dicts = _reconstruct_flows(db, flows)
        result = compute_impact(db, syms[0].id, sym_to_comm, flow_dicts,
                                also=[s.id for s in syms[1:]])
        matched = _matched_symbols(db, syms)
        db.close()
    else:
        db = _get_db()
        syms = db.get_graph_symbols(symbol_name)
        if not syms:
            return _symbol_not_found_error(symbol_name)
        communities = db.get_communities()
        sym_to_comm = {}
        for c in communities:
            for m in db.get_community_members(c["id"]):
                sym_to_comm[m["id"]] = c["id"]
        flows = db.get_execution_flows()
        flow_dicts = _reconstruct_flows(db, flows)
        result = compute_impact(db, syms[0].id, sym_to_comm, flow_dicts,
                                also=[s.id for s in syms[1:]])
        matched = _matched_symbols(db, syms)

    return json.dumps({
        "symbol": symbol_name,
        "project": project,
        **result,
        **matched,
    }, indent=2)


def _reconstruct_flows(db: Database, stored_flows: list[dict]) -> list[dict]:
    """Reconstruct flow dicts with steps from stored flow data."""
    result = []
    for flow in stored_flows:
        steps = db.get_flow_steps(flow["id"])
        result.append({
            "entry_symbol_id": flow["entry_symbol_id"],
            "terminal_symbol_id": flow["terminal_symbol_id"],
            "label": flow["label"],
            "step_count": flow["step_count"],
            "communities_crossed": flow["communities_crossed"],
            "steps": [{"symbol_id": s["symbol_id"], "community_id": s["community_id"], "order": s["step_order"]} for s in steps],
        })
    return result


@mcp.tool()
def detect_changes(
    ref: str | None = None,
    project: str | None = None,
) -> str:
    """Detect which symbols were changed and compute their aggregate blast radius.

    Maps git diff hunks to indexed symbols, then runs impact analysis on each
    to show what breaks. Call this after editing files or before committing to
    understand the full impact of your changes.

    Args:
        ref: Git ref to diff against (default: uncommitted changes vs HEAD).
             Use "HEAD~1" for last commit's impact, or a branch name.
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod
    from .community import compute_impact

    if _is_workspace_mode() and not project:
        return _project_required_error("detect_changes")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    # Get per-project DB
    if _is_workspace_mode():
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
    else:
        db = _get_db()

    # Parse diff into changed file/line ranges
    changed_files = git_mod.detect_changes(repo_root, ref=ref)
    if not changed_files:
        if _is_workspace_mode():
            db.close()
        return json.dumps({"info": "No changes detected", "changed_symbols": []})

    # Map hunks to symbols
    changed_symbols: list[dict] = []
    seen_sym_ids: set[int] = set()

    for file_change in changed_files:
        file_path = file_change["file"]
        symbols = db.symbols_in_file(file_path)
        if not symbols:
            continue

        for sym in symbols:
            if sym.id in seen_sym_ids:
                continue
            # Check if any hunk overlaps this symbol's line range
            for hunk in file_change["hunks"]:
                hunk_start = hunk["new_start"]
                hunk_end = hunk_start + max(hunk["new_count"] - 1, 0)
                if hunk_start <= sym.end_line and hunk_end >= sym.start_line:
                    seen_sym_ids.add(sym.id)
                    changed_symbols.append({
                        "id": sym.id,
                        "name": sym.name,
                        "qualified_name": sym.qualified_name,
                        "kind": sym.kind,
                        "file": file_path,
                        "lines": f"{sym.start_line}-{sym.end_line}",
                    })
                    break

    if not changed_symbols:
        if _is_workspace_mode():
            db.close()
        return json.dumps({
            "info": "Changes detected but no indexed symbols affected",
            "changed_files": [f["file"] for f in changed_files],
            "changed_symbols": [],
        })

    # Load community and flow data for impact analysis
    communities = db.get_communities()
    sym_to_comm: dict[int, int] = {}
    for c in communities:
        for m in db.get_community_members(c["id"]):
            sym_to_comm[m["id"]] = c["id"]

    flows = db.get_execution_flows()
    flow_dicts = _reconstruct_flows(db, flows)

    # Run impact analysis on each changed symbol
    all_affected_comms: set[int] = set()
    all_affected_flows: set[str] = set()
    total_direct = 0
    total_transitive = 0
    max_risk = "LOW"
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    symbol_impacts: list[dict] = []

    for sym_info in changed_symbols:
        impact = compute_impact(db, sym_info["id"], sym_to_comm, flow_dicts)
        sym_info["risk"] = impact["risk"]
        sym_info["direct_dependents"] = impact["direct_dependents"]
        sym_info["affected_flows"] = impact["affected_flows"]
        symbol_impacts.append(sym_info)

        total_direct += impact["direct_dependents"]
        total_transitive += impact["transitive_dependents"]
        all_affected_comms.update(impact["affected_communities"])
        all_affected_flows.update(impact["affected_flows"])
        if risk_order.get(impact["risk"], 0) > risk_order.get(max_risk, 0):
            max_risk = impact["risk"]

    if _is_workspace_mode():
        db.close()

    # Sort by risk descending
    symbol_impacts.sort(key=lambda s: risk_order.get(s["risk"], 0), reverse=True)

    return json.dumps({
        "project": project,
        "ref": ref or "HEAD (uncommitted)",
        "overall_risk": max_risk,
        "changed_symbol_count": len(symbol_impacts),
        "total_direct_dependents": total_direct,
        "total_transitive_dependents": total_transitive,
        "communities_affected": len(all_affected_comms),
        "flows_affected": len(all_affected_flows),
        "changed_symbols": symbol_impacts,
    }, indent=2)


def make_sse_and_streamable_http_app(mount_path: str | None = "/"):
    """Return a Starlette app serving both SSE and Streamable HTTP on one port (Cursor compatibility)."""
    streamable_app = mcp.streamable_http_app()
    sse_app = mcp.sse_app()
    sse_routes = [r for r in sse_app.routes if getattr(r, "path", None) in ("/sse", "/messages")]
    streamable_app.router.routes.extend(sse_routes)
    return streamable_app


def _close_databases() -> None:
    """Close every database handle, checkpointing each WAL on the way out.

    Call before any deliberate process exit. `os._exit()` skips atexit and
    SQLite's own cleanup, so without this the index is left sitting entirely in
    index.db-wal while the main file keeps a bare 4096-byte header — and an
    index.db copied without its sidecar is an empty database (issue #16).
    """
    global _db, _workspace_db, _learnings_db
    for handle, label in ((_db, "index"), (_workspace_db, "workspace"),
                          (_learnings_db, "learnings")):
        if handle is None:
            continue
        try:
            # Shutdown is a write boundary: fold the WAL back so index.db is
            # self-contained at rest. close() no longer does this, because most
            # callers are readers.
            if hasattr(handle, "checkpoint"):
                handle.checkpoint()
            handle.close()
        except Exception:  # noqa: BLE001 -- cleanup must never block an exit
            logger.warning("Failed to close the %s database cleanly", label, exc_info=True)
    _db = None
    _workspace_db = None
    _learnings_db = None


def configure(db_path: Path | None = None, repo_root: Path | None = None) -> None:
    """Configure the server for single-repo mode."""
    global _db_path, _repo_root, _db, _vector_cache, _vector_cache_rebuild_failed_at
    if _db is not None:
        _db.close()
        _db = None
    _vector_cache = None
    # The failure memo is about one database at one version; pointing the
    # server somewhere else must not carry it over.
    _vector_cache_rebuild_failed_at = None
    _db_path = db_path
    _repo_root = repo_root
    _refresh_instructions()


def configure_workspace(workspace_name: str) -> None:
    """Configure the server for workspace (multi-repo) mode."""
    global _workspace_name, _workspace_db, _learnings_db
    _workspace_name = workspace_name
    if _workspace_db is not None:
        _workspace_db.close()
        _workspace_db = None
    if _learnings_db is not None:
        _learnings_db.close()
        _learnings_db = None
    _refresh_instructions()


def run_server(transport: str = "sse", port: int = 8742):
    """Start the MCP server."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()
    if transport == "sse":
        mcp.run(transport=transport, host="127.0.0.1", port=port)
    else:
        mcp.run(transport=transport)
