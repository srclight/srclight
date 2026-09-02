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
