"""The uvicorn config `srclight serve --web` runs under.

2026-09-02: srclight.service hung for the full 90s TimeoutStopSec and was
SIGKILLed on 9 of its last 18 stops. uvicorn.Server.shutdown() drains open
connections *before* running the ASGI lifespan that closes the MCP session
manager, so an open `GET /mcp` stream is an unbounded wait by construction:
timeout_graceful_shutdown defaults to None -> asyncio.wait_for(..., None).
"""

from srclight.cli import _uvicorn_config


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
