"""show_status reports the codebase it serves, not only the query count."""

import asyncio
import json

from srclight import server


def test_show_status_includes_the_codebase_map(monkeypatch):
    # codebase_map is a plain function. show_status awaited it, the TypeError
    # went into a bare except, and every status came back without "codebase".
    monkeypatch.setattr(server, "codebase_map", lambda: json.dumps({"files": 3}))

    status = json.loads(asyncio.run(server.show_status()))

    assert status["codebase"] == {"files": 3}


def test_show_status_survives_a_codebase_map_that_fails(monkeypatch):
    def broken():
        raise RuntimeError("index not open")

    monkeypatch.setattr(server, "codebase_map", broken)

    status = json.loads(asyncio.run(server.show_status()))

    assert "query_count" in status
    assert "codebase" not in status
