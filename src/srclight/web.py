"""
Web dashboard and REST API for srclight (optional, used with --web when serving SSE).

Local-only: server binds to 127.0.0.1. No secrets in responses; config paths only.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

if TYPE_CHECKING:
    from starlette.applications import Starlette


def _dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Srclight</title>
  <style>
    :root { --bg: #0f1117; --surface: #161b22; --border: #30363d; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --success: #3fb950; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 1rem; line-height: 1.5; }
    h1 { font-size: 1.5rem; margin: 0 0 1rem; }
    h2 { font-size: 1.1rem; margin: 1rem 0 0.5rem; color: var(--muted); }
    section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    button { background: var(--accent); color: #fff; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
    button.secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
    .loading { color: var(--muted); }
    .error { color: #f85149; font-size: 0.9rem; }
    .flex { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
    pre { white-space: pre-wrap; font-size: 0.85rem; max-height: 300px; overflow: auto; margin: 0.5rem 0 0; }
    p.note { font-size: 0.9rem; color: var(--success); margin: 0 0 1rem; }
  </style>
</head>
<body>
  <h1>Srclight</h1>
  <p class="note"><strong>Local only.</strong> Server listens on 127.0.0.1. Same server as MCP (SSE at <code>/sse</code>, Streamable HTTP at <code>/mcp</code>).</p>

  <section>
    <h2>Status</h2>
    <div class="flex">
      <button type="button" id="btnServerStats">Server stats</button>
      <button type="button" id="btnIndexStatus">Index status</button>
      <button type="button" id="btnEmbeddingStatus">Embedding status</button>
      <button type="button" id="btnEmbeddingHealth">Embedding health</button>
    </div>
    <div id="statusOutput" class="loading" style="white-space: pre-wrap; font-size: 0.85rem;">Click a button to load.</div>
  </section>

  <section>
    <h2>Workspace</h2>
    <div class="flex">
      <button type="button" id="btnListProjects">List projects</button>
      <button type="button" id="btnCodebaseMap">Codebase map</button>
    </div>
    <pre id="workspaceOutput"></pre>
  </section>

  <section>
    <h2>Setup</h2>
    <div class="flex">
      <button type="button" id="btnSetupGuide">Setup guide (for agents)</button>
    </div>
    <pre id="setupOutput"></pre>
  </section>

  <section>
    <h2>Server control</h2>
    <div class="flex">
      <button type="button" id="btnRestart" class="secondary">Request restart (SRCLIGHT_ALLOW_RESTART=1)</button>
    </div>
    <div id="restartOutput" class="loading"></div>
  </section>

  <script>
    const api = (path) => fetch(path, { headers: { Accept: 'application/json' } }).then(r => r.json());
    const set = (id, text, isError) => { const el = document.getElementById(id); el.textContent = text; el.className = isError ? 'error' : ''; };
    const setPre = (id, text) => { document.getElementById(id).textContent = text; };

    document.getElementById('btnServerStats').onclick = async () => {
      set('statusOutput', 'Loading…');
      try {
        const data = await api('/api/server_stats');
        set('statusOutput', `Started: ${data.started_at} · Uptime: ${data.uptime_human} (${data.uptime_seconds}s)`);
      } catch (e) { set('statusOutput', e.message, true); }
    };
    document.getElementById('btnIndexStatus').onclick = async () => {
      set('statusOutput', 'Loading…');
      try {
        const data = await api('/api/index_status');
        set('statusOutput', JSON.stringify(data, null, 2));
      } catch (e) { set('statusOutput', e.message, true); }
    };
    document.getElementById('btnEmbeddingStatus').onclick = async () => {
      set('statusOutput', 'Loading…');
      try {
        const data = await api('/api/embedding_status');
        set('statusOutput', JSON.stringify(data, null, 2));
      } catch (e) { set('statusOutput', e.message, true); }
    };
    document.getElementById('btnEmbeddingHealth').onclick = async () => {
      set('statusOutput', 'Loading…');
      try {
        const data = await api('/api/embedding_health');
        set('statusOutput', JSON.stringify(data, null, 2));
      } catch (e) { set('statusOutput', e.message, true); }
    };
    document.getElementById('btnListProjects').onclick = async () => {
      try {
        const data = await api('/api/list_projects');
        setPre('workspaceOutput', JSON.stringify(data, null, 2));
      } catch (e) { setPre('workspaceOutput', e.message); }
    };
    document.getElementById('btnCodebaseMap').onclick = async () => {
      try {
        const data = await api('/api/codebase_map');
        setPre('workspaceOutput', JSON.stringify(data, null, 2));
      } catch (e) { setPre('workspaceOutput', e.message); }
    };
    document.getElementById('btnSetupGuide').onclick = async () => {
      try {
        const data = await api('/api/setup_guide');
        setPre('setupOutput', JSON.stringify(data, null, 2));
      } catch (e) { setPre('setupOutput', e.message); }
    };
    document.getElementById('btnRestart').onclick = async () => {
      set('restartOutput', 'Calling restart…');
      try {
        const data = await api('/api/restart_server');
        set('restartOutput', JSON.stringify(data, null, 2));
      } catch (e) { set('restartOutput', e.message, true); }
    };
    document.getElementById('btnServerStats').click();
  </script>
</body>
</html>
"""


async def _run_sync(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def _api_list_projects(_request: Request) -> Response:
    from .server import list_projects
    body = await _run_sync(list_projects)
    return JSONResponse(json.loads(body))


async def _api_codebase_map(request: Request) -> Response:
    from .server import codebase_map
    project = request.query_params.get("project") or None
    body = await _run_sync(codebase_map, project)
    return JSONResponse(json.loads(body))


async def _api_index_status(_request: Request) -> Response:
    from .server import index_status
    body = await _run_sync(index_status)
    return JSONResponse(json.loads(body))


async def _api_embedding_status(request: Request) -> Response:
    from .server import embedding_status
    project = request.query_params.get("project") or None
    body = await _run_sync(embedding_status, project)
    return JSONResponse(json.loads(body))


async def _api_embedding_health(request: Request) -> Response:
    from .server import embedding_health
    project = request.query_params.get("project") or None
    body = await _run_sync(embedding_health, project)
    return JSONResponse(json.loads(body))


async def _api_setup_guide(_request: Request) -> Response:
    from .server import setup_guide
    body = await setup_guide()
    return JSONResponse(json.loads(body))


async def _api_server_stats(_request: Request) -> Response:
    from .server import server_stats
    body = await server_stats()
    return JSONResponse(json.loads(body))


async def _api_restart_server(_request: Request) -> Response:
    from .server import restart_server
    body = await restart_server()
    return JSONResponse(json.loads(body))


async def _dashboard(_request: Request) -> Response:
    return HTMLResponse(_dashboard_html())


def add_web_routes(app: "Starlette") -> None:
    """Add dashboard and REST API routes to a Starlette app (e.g. from make_sse_and_streamable_http_app)."""
    from starlette.routing import Route
    routes = [
        Route("/", _dashboard, methods=["GET"]),
        Route("/api/list_projects", _api_list_projects, methods=["GET"]),
        Route("/api/codebase_map", _api_codebase_map, methods=["GET"]),
        Route("/api/index_status", _api_index_status, methods=["GET"]),
        Route("/api/embedding_status", _api_embedding_status, methods=["GET"]),
        Route("/api/embedding_health", _api_embedding_health, methods=["GET"]),
        Route("/api/setup_guide", _api_setup_guide, methods=["GET"]),
        Route("/api/server_stats", _api_server_stats, methods=["GET"]),
        Route("/api/restart_server", _api_restart_server, methods=["POST"]),
    ]
    for r in routes:
        app.router.routes.append(r)
