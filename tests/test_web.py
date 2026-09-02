"""Tests for the web dashboard + REST API (srclight serve --web)."""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from srclight import __version__
from srclight.workspace import WorkspaceConfig, WorkspaceDB

from .test_workspace import _create_indexed_project, ws_dir  # noqa: F401  (fixture re-export)


@pytest.fixture
def client(tmp_path, ws_dir):
    """A TestClient over the real MCP+web Starlette app in workspace mode."""
    from srclight import server as server_mod
    from srclight.web import add_web_routes

    config = WorkspaceConfig(name="web-test")
    for i in range(2):
        proj = _create_indexed_project(tmp_path, f"proj{i}", [
            (f"Class{i}", "class"),
            (f"method{i}", "method"),
        ])
        config.add_project(f"proj{i}", str(proj))
    config.save()

    server_mod.configure_workspace("web-test")
    server_mod._server_start_time = time.time()  # as `srclight serve --web` does
    app = server_mod.make_sse_and_streamable_http_app(mount_path="/")
    add_web_routes(app)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    server_mod.configure_workspace("web-test")  # close the workspace db


def test_dashboard_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "srclight" in r.text


def test_healthz_is_honest_and_machine_readable(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["name"] == "srclight"
    assert d["version"] == __version__
    assert d["workspace"] == "web-test"
    assert d["projects"] == 2
    assert d["files"] == 2
    assert d["symbols"] == 4
    assert isinstance(d["uptime_seconds"], (int, float))
    assert d["mcp"] == "/mcp"
    # Embedding health is reported, never omitted: absence must not read as "fine".
    assert "embeddings" in d
    assert d["embeddings"]["status"]


def test_favicon_is_served(client):
    r = client.get("/favicon.ico")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in r.text


def test_list_projects_reports_freshness_and_coverage(client):
    d = client.get("/api/list_projects").json()
    assert d["project_count"] == 2
    for p in d["projects"]:
        # ISO-8601 timestamp of the newest indexed file: lets the dashboard say "indexed 3h ago".
        assert p["last_indexed"] and p["last_indexed"].startswith("20")
        # No embeddings in these fixtures -> coverage is 0.0, but the key is always present.
        assert p["embedding_coverage"] == 0.0
        assert p["embedded_symbols"] == 0


def test_healthz_and_codebase_map_report_last_indexed(client):
    d = client.get("/api/codebase_map").json()
    assert d["last_indexed"] and d["last_indexed"].startswith("20")
    for p in d["projects"]:
        assert p["last_indexed"] and p["last_indexed"].startswith("20")
    h = client.get("/healthz").json()
    assert h["last_indexed"] == d["last_indexed"]


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (42, "42s"), (258.4, "4m 18s"), (3600, "1h 0m"),
    (8040, "2h 14m"), (90000, "1d 1h"), (10 * 86400 + 3600 * 5, "10d 5h"),
])
def test_humanize_uptime(seconds, expected):
    from srclight.server import _humanize_seconds
    assert _humanize_seconds(seconds) == expected


def test_server_stats_uptime_human_is_readable(client):
    d = client.get("/api/server_stats").json()
    assert not d["uptime_human"].endswith("00s") or "m" in d["uptime_human"]
    assert d["uptime_human"] in ("0s",) or any(u in d["uptime_human"] for u in "smhd")


def test_healthz_carries_query_activity_and_embedded_total(client):
    h = client.get("/healthz").json()
    assert h["queries"]["count"] >= 0
    assert "last_ago_seconds" in h["queries"]
    assert h["embedded"] == 0  # fixtures carry no embeddings


@pytest.mark.parametrize("path", ["/dashboard", "/web"])
def test_dashboard_aliases_redirect_to_root(client, path):
    r = client.get(path, follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"] == "/"


def test_dashboard_polls_do_not_count_as_agent_queries(client):
    """The page's own /healthz and /api/* polls must not sign the guest book.

    TOTO/STUBBY/SAM/GROMIT (pack review 2026-09-01): /healthz -> codebase_map()
    -> _record_query() made "last query 3s ago" the dashboard's own pulse and
    "connected" theatre.
    """
    before = client.get("/api/stats").json()["query_count"]
    client.get("/healthz"); client.get("/api/codebase_map"); client.get("/api/list_projects")
    client.get("/api/search?mode=keyword&q=Class")
    after = client.get("/api/stats").json()["query_count"]
    assert after == before


def test_dashboard_routes_reject_foreign_host_and_origin(client):
    """DNS-rebinding guard, same line the MCP SDK draws for /sse and /mcp (SAM)."""
    assert client.get("/healthz", headers={"host": "attacker.example"}).status_code == 421
    assert client.get("/api/list_projects", headers={"host": "attacker.example:8742"}).status_code == 421
    assert client.get("/healthz", headers={"host": "localhost:8742"}).status_code == 200
    # Cross-origin simple POSTs (form / no-cors fetch) carry Origin: refuse them.
    r = client.post("/api/switch_workspace", headers={"origin": "http://evil.example"},
                    content=b'{"workspace":"x"}', )
    assert r.status_code == 403
    r = client.post("/api/switch_workspace", headers={"origin": "http://127.0.0.1"},
                    json={"workspace": "web-test"})
    assert r.status_code == 200


def test_corrupt_index_costs_one_row_not_the_workspace(tmp_path, ws_dir):
    """BARRY path 7: one bad index.db must be one red pill, not a dead workspace."""
    from srclight import server as server_mod
    from srclight.web import add_web_routes
    config = WorkspaceConfig(name="corrupt-test")
    for i in range(2):
        proj = _create_indexed_project(tmp_path, f"p{i}", [(f"C{i}", "class")])
        config.add_project(f"p{i}", str(proj))
    config.save()
    (tmp_path / "p1" / ".srclight" / "index.db").write_bytes(b"this is not a database" * 100)

    server_mod.configure_workspace("corrupt-test")
    server_mod._server_start_time = time.time()
    app = server_mod.make_sse_and_streamable_http_app(mount_path="/")
    add_web_routes(app)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        projects = {p["project"]: p for p in c.get("/api/list_projects").json()["projects"]}
        assert projects["p0"]["symbols"] == 1
        assert "error" in projects["p1"]
        h = c.get("/healthz").json()
        assert h["status"] == "ok"                 # process is alive
        assert h["projects_errored"] == 1
        assert any("unreadable" in d for d in h["degraded"])
    server_mod.configure_workspace("corrupt-test")


def test_last_indexed_prefers_the_index_run_signal(tmp_path, ws_dir):
    """"Indexed 160d ago" must mean the last index RUN, not the last file re-parse (TOTO)."""
    import json as _json
    config = WorkspaceConfig(name="signal-test")
    proj = _create_indexed_project(tmp_path, "sig", [("A", "class")])
    config.add_project("sig", str(proj))
    (proj / ".srclight" / "last-indexed").write_text(_json.dumps({
        "timestamp": "2031-01-01T00:00:00+00:00", "files": 1, "symbols": 1}))
    with WorkspaceDB(config) as wdb:
        assert wdb.list_projects()[0]["last_indexed"].startswith("2031-01-01")
        assert wdb.codebase_map()["last_indexed"].startswith("2031-01-01")


def test_recent_queries_ledger_records_agent_calls_not_dashboard_polls(client):
    """TOGO: the one thing only an agent index can show is what agents asked."""
    from srclight import server as server_mod
    # An MCP-style call (not through the dashboard context) lands in the ledger.
    server_mod.search_symbols("Class0")
    server_mod.get_symbol("Class1", project="proj1")
    d = client.get("/api/recent_queries").json()
    tools = [i["tool"] for i in d["items"]]
    assert tools[:2] == ["get_symbol", "search_symbols"]        # newest first
    assert d["items"][1]["query"] == "Class0"
    assert d["items"][0]["project"] == "proj1"
    assert all("ts" in i for i in d["items"])
    # The dashboard's own search is not an agent call.
    client.get("/api/search?mode=keyword&q=Class")
    assert len(client.get("/api/recent_queries").json()["items"]) == len(d["items"])
    h = client.get("/healthz").json()
    assert h["queries"]["recent"][0]["tool"] == "get_symbol"
    assert "warming" in h


def test_warm_stats_do_not_wait_for_the_workspace_lock(tmp_path, ws_dir):
    """K9 (SEV-1): searches hold the workspace lock for seconds; a warm /healthz
    must answer from the cache without queueing behind them."""
    import threading
    config = WorkspaceConfig(name="convoy-test")
    proj = _create_indexed_project(tmp_path, "c", [("A", "class")])
    config.add_project("c", str(proj))
    with WorkspaceDB(config) as wdb:
        wdb.list_projects()  # warm
        wdb._lock.acquire()  # simulate a long search walk on another thread
        try:
            result: list = []
            t = threading.Thread(target=lambda: result.append(wdb.codebase_map()["totals"]["symbols"]))
            t.start(); t.join(timeout=3)
            assert not t.is_alive(), "codebase_map blocked on the lock with a warm cache"
            assert result == [1]
        finally:
            wdb._lock.release()


def test_project_indexed_after_open_is_picked_up(tmp_path, ws_dir):
    """K9 (SEV-2): the app's add-then-index flow must not need a restart."""
    config = WorkspaceConfig(name="late-index-test")
    proj = _create_indexed_project(tmp_path, "early", [("A", "class")])
    config.add_project("early", str(proj))
    late_dir = tmp_path / "late"; late_dir.mkdir()
    config.add_project("late", str(late_dir))
    with WorkspaceDB(config) as wdb:
        assert {p["project"]: p.get("indexed") for p in wdb.list_projects()} == {"early": True, "late": False}
        late_dir.rmdir()
        _create_indexed_project(tmp_path, "late", [("B", "class")])
        assert {p["project"]: p.get("indexed") for p in wdb.list_projects()} == {"early": True, "late": True}
        assert wdb.codebase_map()["totals"]["symbols"] == 2


def test_find_imports_works_in_workspace_mode(client):
    """K9 (SEV-1): ProjectEntry has .path, not .root; the tool raised AttributeError."""
    from srclight import server as server_mod
    import json as _json
    out = _json.loads(server_mod.find_imports("src/proj0.cs", project="proj0"))
    # The fixture writes rows, not files: reaching the file read is the proof.
    assert "AttributeError" not in _json.dumps(out)
    assert "imports" in out or "Cannot read file" in out.get("error", "")


def test_stale_sidecar_reaches_healthz_degraded(tmp_path, ws_dir):
    """A sidecar serving only part of its index must reach the header.

    intuition-2019 (found live 2026-09-02): index.db held 20,648 embeddings and
    the sidecar 15,611. /api/embedding_status reads the DB and reported 100%
    coverage while semantic search reads the sidecar and saw 76% of the repo.
    Two surfaces, one of them silently wrong, and `degraded` was empty.
    """
    from srclight import server as server_mod
    from srclight.db import Database
    from srclight.embeddings import vector_to_bytes
    from srclight.vector_cache import VectorCache
    from srclight.web import add_web_routes

    config = WorkspaceConfig(name="stale-test")
    proj = _create_indexed_project(tmp_path, "alpha", [("Dictionary", "class")])
    config.add_project("alpha", str(proj))
    config.save()

    db = Database(proj / ".srclight" / "index.db")
    db.open()
    for i, row in enumerate(db.conn.execute("SELECT id FROM symbols ORDER BY id")):
        db.upsert_embedding(row["id"], "mock:test", 8,
                            vector_to_bytes([0.1 * (i + 1)] * 8), f"h{i}")
    db.commit()
    VectorCache(proj / ".srclight").build_from_db(db.conn)
    db.conn.execute("INSERT OR REPLACE INTO schema_info (key, value) "
                    "VALUES ('embedding_cache_version', '9999')")
    db.commit()
    db.close()

    server_mod.configure_workspace("stale-test")
    server_mod._server_start_time = time.time()
    app = server_mod.make_sse_and_streamable_http_app(mount_path="/")
    add_web_routes(app)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        server_mod._get_workspace_db()._get_project_cache("alpha")  # as startup does
        h = c.get("/healthz").json()
        assert h["status"] == "ok"                      # liveness is unaffected
        assert any("sidecar" in d for d in h["degraded"]), h["degraded"]
