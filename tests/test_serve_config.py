"""The uvicorn config `srclight serve --web` runs under.

2026-09-02: srclight.service hung for the full 90s TimeoutStopSec and was
SIGKILLed on 9 of its last 18 stops. uvicorn.Server.shutdown() drains open
connections *before* running the ASGI lifespan that closes the MCP session
manager, so an open `GET /mcp` stream is an unbounded wait by construction:
timeout_graceful_shutdown defaults to None -> asyncio.wait_for(..., None).
"""

from srclight.cli import _uvicorn_config

from .test_workspace import _create_indexed_project, ws_dir  # noqa: F401  (fixture re-export)


async def _app(scope, receive, send):  # minimal ASGI callable
    pass


def test_graceful_shutdown_is_bounded():
    """An open SSE stream must never be able to block shutdown forever."""
    config = _uvicorn_config(_app, port=8742, log_level="info")
    assert config.timeout_graceful_shutdown is not None, (
        "unset means asyncio.wait_for(..., timeout=None) — an open /mcp stream "
        "hangs the stop until systemd SIGKILLs the process"
    )
    assert 0 < config.timeout_graceful_shutdown <= 30


def test_binds_loopback_only():
    """The dashboard and MCP endpoint must not be reachable off-box."""
    config = _uvicorn_config(_app, port=8742, log_level="info")
    assert config.host == "127.0.0.1"
    assert config.port == 8742


def test_close_databases_releases_every_handle(tmp_path):
    """restart_server() called os._exit(0), which skipped every database close.

    os._exit skips atexit, SQLite close, and therefore the WAL checkpoint — so
    the dashboard's Restart button left the index sitting entirely in
    index.db-wal every single time it was pressed (issue #16).
    """
    from srclight import server as server_mod

    server_mod.configure(db_path=tmp_path / "index.db", repo_root=tmp_path)
    db = server_mod._get_db()
    assert db.conn is not None
    assert server_mod._db is not None

    server_mod._close_databases()

    assert server_mod._db is None, "server still holds a database handle"
    assert db.conn is None, "the connection was never closed"
    server_mod.configure(db_path=None, repo_root=None)


def test_get_db_initializes_a_genuinely_new_database(tmp_path):
    """`_get_db()` must create the schema for a new index.

    The guard read `if not path.exists() or path.stat().st_size == 0:
    initialize()`, placed AFTER `Database.open()`. open() runs
    `PRAGMA journal_mode=WAL`, which writes a 4096-byte header — so both arms
    were false by the time the check ran and initialize() could never fire. The
    artifact on this machine: ~/.srclight/index.db, 4096 bytes, zero tables,
    created 2026-03-03 and never noticed.
    """
    from srclight import server as server_mod

    db_path = tmp_path / "index.db"
    assert not db_path.exists()

    server_mod.configure(db_path=db_path, repo_root=tmp_path)
    try:
        db = server_mod._get_db()
        tables = {
            r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"files", "symbols"} <= tables, f"schema never created; tables={tables}"
    finally:
        server_mod._close_databases()
        server_mod.configure(db_path=None, repo_root=None)


def test_get_db_heals_an_empty_database_left_by_the_old_guard(tmp_path):
    """A 4096-byte index.db with no tables must be repaired, not trusted."""
    import sqlite3

    from srclight import server as server_mod

    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")   # exactly what open() did
    conn.close()
    assert db_path.stat().st_size > 0
    assert not sqlite3.connect(db_path).execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]

    server_mod.configure(db_path=db_path, repo_root=tmp_path)
    try:
        db = server_mod._get_db()
        tables = {
            r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"files", "symbols"} <= tables, f"empty db not healed; tables={tables}"
    finally:
        server_mod._close_databases()
        server_mod.configure(db_path=None, repo_root=None)


def _call(coro_or_val):
    """Run an async MCP tool the way the rest of the suite does."""
    import asyncio
    return asyncio.run(coro_or_val) if asyncio.iscoroutine(coro_or_val) else coro_or_val


def test_workspace_instructions_actually_reach_the_client(tmp_path, ws_dir):
    """The dynamic instructions must be INSTALLED, not silently dropped.

    _refresh_instructions wrote to `mcp._mcp_server.instructions`, which was
    correct under FastMCP and does not exist on mcp v2's MCPServer. The
    AttributeError went into `except Exception: pass`, so the whole workspace
    walk ran and its result was discarded — every client got the generic
    fallback blob. Broken since the ironmcp v2 cutover (2026-08-31) and shipped
    in seven releases, because nothing ever read the surface back.
    """
    from srclight import server as server_mod
    from srclight.workspace import WorkspaceConfig

    proj = _create_indexed_project(tmp_path, "alpha", [("Dictionary", "class")])
    config = WorkspaceConfig(name="instr-test")
    config.add_project("alpha", str(proj))
    config.save()

    server_mod.configure_workspace("instr-test")
    try:
        text = server_mod.mcp.instructions
        assert "instr-test" in text, f"workspace name missing from instructions: {text[:200]!r}"
        assert "indexed project" in text, "project count missing from instructions"
        # and the value the wire actually carries must be the same one
        opts = server_mod.mcp._lowlevel_server.create_initialization_options()
        assert "instr-test" in opts.instructions
    finally:
        server_mod._close_databases()


def test_reindex_refuses_a_foreign_path_in_both_modes(tmp_path, ws_dir, monkeypatch):
    """`path` is an index ROOT, not a filter, and it deletes what is not under it.

    Indexer reads the WHOLE database's file list and removes every file outside
    the new root, so reindex(path=X) does not pollute an index, it empties it.
    Measured in single-repo mode: 3 files -> 1, files_removed 3, path rewritten,
    and a success JSON with errors 0.

    The first guard only refused in WORKSPACE mode -- but the published plugin
    runs `uvx srclight serve --transport stdio` with NO --workspace, so it never
    fired for a single plugin user, while the comment justifying it cited the
    plugin by name. Refuse a foreign path in BOTH modes.
    """
    import json

    from srclight import server as server_mod
    from srclight.workspace import WorkspaceConfig

    # tests must never reach a real index: _get_db() walks up from the CWD, and
    # an earlier version of this test wrote its temp dir into srclight's own
    # project index twice, removing 70 files.
    monkeypatch.chdir(tmp_path)

    proj = _create_indexed_project(tmp_path, "alpha", [("Dictionary", "class")])
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "x.py").write_text("def stranger():\n    return 1\n")

    # --- workspace mode ---
    config = WorkspaceConfig(name="reindex-test")
    config.add_project("alpha", str(proj))
    config.save()
    server_mod.configure_workspace("reindex-test")
    try:
        result = json.loads(_call(server_mod.reindex(path=str(other))))
        assert "error" in result, f"workspace mode accepted a foreign path: {result}"
    finally:
        server_mod._close_databases()

    # --- single-repo mode: the one the published plugin actually runs ---
    server_mod._workspace_name = None
    server_mod.configure(db_path=proj / ".srclight" / "index.db", repo_root=proj)
    try:
        result = json.loads(_call(server_mod.reindex(path=str(other))))
        assert "error" in result, f"single-repo mode accepted a foreign path: {result}"
        assert not (other / ".srclight" / "index.db").exists()
    finally:
        server_mod._close_databases()
        server_mod.configure(db_path=None, repo_root=None)
