"""Tests for the web dashboard + REST API (srclight serve --web)."""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from srclight import __version__
from srclight.workspace import WorkspaceConfig

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
    with TestClient(app) as c:
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
