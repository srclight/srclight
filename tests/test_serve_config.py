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
