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
    .muted { color: var(--muted); }
    .error { color: #f85149; font-size: 0.9rem; }
    .workspace-alert { background: rgba(248,81,73,0.15); border: 1px solid #f85149; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.9rem; }
    .flex { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
    pre { white-space: pre-wrap; font-size: 0.85rem; max-height: 300px; overflow: auto; margin: 0.5rem 0 0; }
    p.note { font-size: 0.9rem; color: var(--success); margin: 0 0 1rem; }
  </style>
</head>
<body>
  <h1>Srclight</h1>
  <p class="note"><strong>Local only.</strong> Server listens on 127.0.0.1. Same server as MCP (SSE at <code>/sse</code>, Streamable HTTP at <code>/mcp</code>).</p>

  <div id="workspaceAlert" class="workspace-alert" style="display: none;"></div>

  <section>
    <h2>Workspace</h2>
    <div class="flex" style="align-items: center; gap: 0.5rem;">
      <label for="workspaceSelect" style="margin: 0; color: var(--muted);">Current workspace:</label>
      <select id="workspaceSelect" style="max-width: 200px; padding: 0.4rem 0.5rem;">
        <option value="">Loading…</option>
      </select>
      <span id="workspaceSwitchResult" class="muted" style="font-size: 0.85rem;"></span>
    </div>
    <p style="margin: 0.5rem 0 0; font-size: 0.85rem; color: var(--muted);">Switch workspace to change which projects List projects / Codebase map use. Create workspaces with <code>srclight workspace init NAME</code>.</p>
  </section>

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
    <h2>Projects &amp; map</h2>
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
    const api = (path, opts = {}) => fetch(path, { headers: { Accept: 'application/json', ...opts.headers }, ...opts }).then(async r => { const data = await r.json(); if (!r.ok && data && data.error) throw new Error(data.error); if (!r.ok) throw new Error(r.statusText || 'Request failed'); return data; });
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
        const data = await api('/api/restart_server', { method: 'POST' });
        set('restartOutput', JSON.stringify(data, null, 2));
      } catch (e) { set('restartOutput', e.message, true); }
    };

    const workspaceSelect = document.getElementById('workspaceSelect');
    const workspaceSwitchResult = document.getElementById('workspaceSwitchResult');
    const workspaceAlert = document.getElementById('workspaceAlert');
    async function loadWorkspaceSelector() {
      try {
        const [workspaces, current] = await Promise.all([api('/api/workspaces'), api('/api/current_workspace')]);
        const cur = current.current_workspace;
        const available = workspaces.workspaces || [];
        workspaceSelect.innerHTML = '';
        if (!cur) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = '— single-repo mode (no workspace) —';
          workspaceSelect.appendChild(opt);
        }
        if (available.length === 0 && cur) {
          const opt = document.createElement('option');
          opt.value = cur;
          opt.textContent = cur + ' (config not found — create workspace or restart)';
          opt.selected = true;
          workspaceSelect.appendChild(opt);
          workspaceAlert.textContent = "Workspace '" + cur + "' not found. Create it with srclight workspace init " + cur + " (and add projects), or restart the server with an existing workspace: srclight serve --workspace NAME --web.";
          workspaceAlert.style.display = 'block';
        } else {
          available.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            workspaceSelect.appendChild(opt);
          });
          if (cur && available.includes(cur)) {
            workspaceSelect.value = cur;
            workspaceAlert.style.display = 'none';
          } else if (cur) {
            const opt = document.createElement('option');
            opt.value = cur;
            opt.textContent = cur + ' (not found — create or restart server)';
            opt.selected = true;
            workspaceSelect.insertBefore(opt, workspaceSelect.firstChild);
            workspaceAlert.textContent = "Workspace '" + cur + "' not found. Select an existing workspace above, or create it with srclight workspace init " + cur + ", or restart with: srclight serve --workspace NAME --web.";
            workspaceAlert.style.display = 'block';
          }
        }
      } catch (e) {
        workspaceSelect.innerHTML = '<option value="">Error loading workspaces</option>';
        workspaceSwitchResult.textContent = e.message;
        workspaceSwitchResult.className = 'error';
        workspaceAlert.style.display = 'none';
      }
    }
    workspaceSelect.addEventListener('change', async () => {
      const name = workspaceSelect.value;
      if (!name) return;
      const sel = workspaceSelect.selectedOptions[0];
      if (sel && sel.textContent.includes('(not found')) return;
      workspaceSwitchResult.textContent = 'Switching…';
      workspaceSwitchResult.className = '';
      try {
        await api('/api/switch_workspace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ workspace: name }) });
        workspaceSwitchResult.textContent = 'Switched to ' + name;
        workspaceSwitchResult.className = '';
      } catch (e) {
        workspaceSwitchResult.textContent = e.message;
        workspaceSwitchResult.className = 'error';
      }
    });
    loadWorkspaceSelector();

    document.getElementById('btnServerStats').click();
  </script>
</body>
</html>
"""


async def _run_sync(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def _api_list_projects(_request: Request) -> Response:
    try:
        from .server import list_projects
        body = await _run_sync(list_projects)
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_codebase_map(request: Request) -> Response:
    try:
        from .server import codebase_map
        project = request.query_params.get("project") or None
        body = await _run_sync(codebase_map, project)
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_index_status(_request: Request) -> Response:
    try:
        from .server import index_status
        body = await _run_sync(index_status)
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_embedding_status(request: Request) -> Response:
    try:
        from .server import embedding_status
        project = request.query_params.get("project") or None
        body = await _run_sync(embedding_status, project)
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_embedding_health(request: Request) -> Response:
    try:
        from .server import embedding_health
        project = request.query_params.get("project") or None
        body = await _run_sync(embedding_health, project)
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_setup_guide(_request: Request) -> Response:
    try:
        from .server import setup_guide
        body = await setup_guide()
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_server_stats(_request: Request) -> Response:
    try:
        from .server import server_stats
        body = await server_stats()
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_restart_server(_request: Request) -> Response:
    try:
        from .server import restart_server
        body = await restart_server()
        return JSONResponse(json.loads(body))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_workspaces(_request: Request) -> Response:
    try:
        from .workspace import WorkspaceConfig
        names = WorkspaceConfig.list_all()
        return JSONResponse({"workspaces": names})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_current_workspace(_request: Request) -> Response:
    try:
        from . import server as server_mod
        current = getattr(server_mod, "_workspace_name", None)
        return JSONResponse({"current_workspace": current})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_switch_workspace(request: Request) -> Response:
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    name = data.get("workspace") if isinstance(data, dict) else None
    if not name or not isinstance(name, str):
        return JSONResponse({"error": "workspace name required"}, status_code=400)
    try:
        from .workspace import WorkspaceConfig
        WorkspaceConfig.load(name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    try:
        from . import server as server_mod
        server_mod.configure_workspace(name)
        return JSONResponse({"ok": True, "workspace": name})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _dashboard(_request: Request) -> Response:
    return HTMLResponse(_dashboard_html())


def add_web_routes(app: "Starlette") -> None:
    """Add dashboard and REST API routes to a Starlette app (e.g. from make_sse_and_streamable_http_app)."""
    from starlette.routing import Route
    routes = [
        Route("/", _dashboard, methods=["GET"]),
        Route("/api/workspaces", _api_workspaces, methods=["GET"]),
        Route("/api/current_workspace", _api_current_workspace, methods=["GET"]),
        Route("/api/switch_workspace", _api_switch_workspace, methods=["POST"]),
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
