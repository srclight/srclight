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
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Srclight Dashboard</title>
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      --bg: #0a0a0f;
      --bg-card: #12121a;
      --bg-card-hover: #1a1a25;
      --amber: #f59e0b;
      --amber-light: #fbbf24;
      --amber-dim: rgba(245, 158, 11, 0.15);
      --green: #3fb950;
      --green-dim: rgba(63, 185, 80, 0.15);
      --red: #f85149;
      --red-dim: rgba(248, 81, 73, 0.15);
      --text: #e4e4e7;
      --text-dim: #9ca3af;
      --border: #1e1e2a;
      --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', 'Liberation Mono', monospace;
      --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, Helvetica, Arial, sans-serif;
    }

    html { font-size: 16px; }
    body {
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }

    .container { max-width: 1000px; margin: 0 auto; padding: 0 24px; }

    /* -- Header -- */
    header {
      padding: 20px 0;
      border-bottom: 1px solid var(--border);
    }
    .header-inner {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .wordmark {
      font-family: var(--mono);
      font-size: 1.4rem;
      font-weight: 700;
      text-decoration: none;
      letter-spacing: -0.02em;
    }
    .wordmark .src { color: var(--amber); }
    .wordmark .light { color: #fff; }
    .badge {
      font-size: 0.7rem;
      font-family: var(--mono);
      color: var(--text-dim);
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 3px 8px;
      border-radius: 4px;
    }
    .header-right {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .header-right select {
      background: var(--bg-card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.85rem;
      font-family: var(--sans);
      cursor: pointer;
    }
    .header-right select:focus { border-color: var(--amber); outline: none; }
    .status-dot {
      width: 8px; height: 8px; border-radius: 50%;
      display: inline-block;
    }
    .status-dot.ok { background: var(--green); box-shadow: 0 0 6px rgba(63,185,80,0.5); }
    .status-dot.err { background: var(--red); box-shadow: 0 0 6px rgba(248,81,73,0.5); }
    .server-status {
      display: flex; align-items: center; gap: 6px;
      font-size: 0.8rem; color: var(--text-dim);
    }

    /* -- Alert -- */
    .alert {
      background: var(--red-dim);
      border: 1px solid var(--red);
      border-radius: 8px;
      padding: 12px 16px;
      margin-top: 16px;
      font-size: 0.85rem;
      display: none;
    }

    /* -- Stats bar -- */
    .stats {
      padding: 28px 0;
      border-bottom: 1px solid var(--border);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 20px;
      text-align: center;
    }
    .stat-value {
      font-family: var(--mono);
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--amber);
    }
    .stat-label {
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-top: 2px;
    }

    /* -- Section -- */
    .section {
      padding: 32px 0;
    }
    .section + .section {
      border-top: 1px solid var(--border);
    }
    .section-title {
      font-size: 1.15rem;
      font-weight: 600;
      color: #fff;
      margin-bottom: 20px;
    }

    /* -- Project cards -- */
    .project-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }
    .project-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      transition: border-color 0.15s, background 0.15s;
    }
    .project-card:hover {
      border-color: rgba(245, 158, 11, 0.3);
      background: var(--bg-card-hover);
    }
    .project-name {
      font-family: var(--mono);
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--amber);
      margin-bottom: 10px;
    }
    .project-stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 16px;
      font-size: 0.8rem;
    }
    .project-stat {
      display: flex;
      justify-content: space-between;
    }
    .project-stat-label { color: var(--text-dim); }
    .project-stat-value { color: var(--text); font-family: var(--mono); font-weight: 500; }
    .project-langs {
      margin-top: 10px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .lang-tag {
      font-size: 0.7rem;
      font-family: var(--mono);
      color: var(--amber);
      background: var(--amber-dim);
      padding: 2px 8px;
      border-radius: 4px;
    }
    .embed-bar {
      margin-top: 10px;
    }
    .embed-bar-label {
      font-size: 0.7rem;
      color: var(--text-dim);
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
    }
    .embed-bar-track {
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      overflow: hidden;
    }
    .embed-bar-fill {
      height: 100%;
      background: var(--amber);
      border-radius: 2px;
      transition: width 0.3s;
    }

    /* -- Search -- */
    .search-box {
      display: flex;
      gap: 10px;
      margin-bottom: 16px;
    }
    .search-input {
      flex: 1;
      background: var(--bg-card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 16px;
      font-size: 0.9rem;
      font-family: var(--sans);
    }
    .search-input:focus { border-color: var(--amber); outline: none; }
    .search-input::placeholder { color: var(--text-dim); }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
      font-family: var(--sans);
    }
    .btn-primary {
      background: var(--amber);
      color: #0a0a0f;
    }
    .btn-primary:hover { background: var(--amber-light); }
    .btn-secondary {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover {
      border-color: var(--amber);
      color: var(--amber);
    }
    .btn-danger {
      background: transparent;
      color: var(--red);
      border: 1px solid rgba(248,81,73,0.3);
    }
    .btn-danger:hover {
      background: var(--red-dim);
      border-color: var(--red);
    }

    .search-results {
      font-size: 0.85rem;
    }
    .search-result {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 8px;
      transition: border-color 0.15s;
    }
    .search-result:hover {
      border-color: rgba(245, 158, 11, 0.3);
    }
    .sr-header {
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 4px;
    }
    .sr-name {
      font-family: var(--mono);
      font-weight: 600;
      color: var(--amber);
    }
    .sr-kind {
      font-size: 0.7rem;
      color: var(--text-dim);
      background: var(--bg);
      padding: 1px 6px;
      border-radius: 3px;
    }
    .sr-project {
      font-size: 0.7rem;
      color: var(--amber);
      background: var(--amber-dim);
      padding: 1px 6px;
      border-radius: 3px;
    }
    .sr-file {
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--text-dim);
    }
    .sr-sig {
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--text-dim);
      margin-top: 4px;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .search-meta {
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-bottom: 12px;
    }

    /* -- Server info -- */
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
    }
    .info-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .info-label {
      font-size: 0.75rem;
      color: var(--text-dim);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .info-value {
      font-family: var(--mono);
      font-size: 0.9rem;
      color: var(--text);
      word-break: break-all;
    }

    /* -- Footer -- */
    footer {
      padding: 24px 0;
      border-top: 1px solid var(--border);
      text-align: center;
    }
    .footer-links {
      display: flex;
      gap: 20px;
      justify-content: center;
      margin-bottom: 8px;
    }
    .footer-links a {
      color: var(--text-dim);
      text-decoration: none;
      font-size: 0.8rem;
      transition: color 0.15s;
    }
    .footer-links a:hover { color: var(--amber); }
    .footer-copy {
      font-size: 0.75rem;
      color: var(--text-dim);
      opacity: 0.6;
    }

    .hidden { display: none; }

    /* -- Responsive -- */
    @media (max-width: 768px) {
      .stats-grid { grid-template-columns: repeat(3, 1fr); }
      .project-grid { grid-template-columns: 1fr; }
      .header-inner { flex-direction: column; gap: 12px; }
    }
    @media (max-width: 480px) {
      .stats-grid { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="container header-inner">
      <div class="header-left">
        <span class="wordmark"><span class="src">src</span><span class="light">light</span></span>
        <span class="badge" id="versionBadge">...</span>
      </div>
      <div class="header-right">
        <select id="workspaceSelect" title="Switch workspace">
          <option value="">Loading...</option>
        </select>
        <div class="server-status">
          <span class="status-dot" id="statusDot"></span>
          <span id="uptimeText">connecting...</span>
        </div>
      </div>
    </div>
    <div class="container">
      <div class="alert" id="alertBox"></div>
    </div>
  </header>

  <!-- Stats -->
  <section class="stats">
    <div class="container">
      <div class="stats-grid">
        <div>
          <div class="stat-value" id="statProjects">-</div>
          <div class="stat-label">Projects</div>
        </div>
        <div>
          <div class="stat-value" id="statFiles">-</div>
          <div class="stat-label">Files</div>
        </div>
        <div>
          <div class="stat-value" id="statSymbols">-</div>
          <div class="stat-label">Symbols</div>
        </div>
        <div>
          <div class="stat-value" id="statEdges">-</div>
          <div class="stat-label">Edges</div>
        </div>
        <div>
          <div class="stat-value" id="statEmbeddings">-</div>
          <div class="stat-label">Embeddings</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Projects -->
  <div class="container">
    <div class="section" id="projectsSection">
      <div class="section-title">Projects</div>
      <div class="project-grid" id="projectGrid">
        <div style="color: var(--text-dim); font-size: 0.85rem;">Loading projects...</div>
      </div>
    </div>

    <!-- Search -->
    <div class="section">
      <div class="section-title">Search symbols</div>
      <div class="search-box">
        <input type="text" class="search-input" id="searchInput" placeholder="Search symbols by name, code, or concept..." autocomplete="off">
        <button class="btn btn-primary" id="btnSearch" type="button">Search</button>
        <select id="searchMode" title="Search mode" style="background: var(--bg-card); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 0.85rem;">
          <option value="hybrid">Hybrid</option>
          <option value="keyword">Keyword</option>
        </select>
      </div>
      <div class="search-meta hidden" id="searchMeta"></div>
      <div class="search-results" id="searchResults"></div>
    </div>

    <!-- Server Info -->
    <div class="section">
      <div class="section-title">Server</div>
      <div class="info-grid" id="infoGrid">
        <div class="info-card">
          <div class="info-label">Uptime</div>
          <div class="info-value" id="infoUptime">-</div>
        </div>
        <div class="info-card">
          <div class="info-label">SSE endpoint</div>
          <div class="info-value">/sse</div>
        </div>
        <div class="info-card">
          <div class="info-label">Streamable HTTP</div>
          <div class="info-value">/mcp</div>
        </div>
        <div class="info-card">
          <div class="info-label">Embedding provider</div>
          <div class="info-value" id="infoEmbedModel">-</div>
        </div>
        <div class="info-card">
          <div class="info-label">Embedding health</div>
          <div class="info-value" id="infoEmbedHealth">-</div>
        </div>
        <div class="info-card" style="display: flex; align-items: center; justify-content: center;">
          <button class="btn btn-danger" id="btnRestart" type="button">Restart server</button>
        </div>
      </div>
      <div id="restartMsg" style="font-size: 0.8rem; color: var(--text-dim); margin-top: 8px;"></div>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <div class="container">
      <div class="footer-links">
        <a href="https://github.com/srclight/srclight">GitHub</a>
        <a href="https://pypi.org/project/srclight/">PyPI</a>
        <a href="https://github.com/srclight/srclight/issues">Issues</a>
        <a href="https://srclight.dev">srclight.dev</a>
      </div>
      <p class="footer-copy">Local only. Server listens on 127.0.0.1.</p>
    </div>
  </footer>

  <script>
    /* ---- helpers ---- */
    const api = (path, opts = {}) =>
      fetch(path, { headers: { Accept: 'application/json', ...opts.headers }, ...opts })
        .then(async r => {
          const data = await r.json();
          if (!r.ok && data && data.error) throw new Error(data.error);
          if (!r.ok) throw new Error(r.statusText || 'Request failed');
          return data;
        });

    const $ = id => document.getElementById(id);
    const fmt = n => n == null ? '-' : n.toLocaleString();
    const showAlert = (msg) => { const a = $('alertBox'); a.textContent = msg; a.style.display = 'block'; };
    const hideAlert = () => { $('alertBox').style.display = 'none'; };

    /* ---- version ---- */
    api('/api/version').then(d => {
      $('versionBadge').textContent = 'v' + d.version;
    }).catch(() => {});

    /* ---- server status ---- */
    async function loadServerStatus() {
      try {
        const d = await api('/api/server_stats');
        $('statusDot').className = 'status-dot ok';
        $('uptimeText').textContent = d.uptime_human;
        $('infoUptime').textContent = d.uptime_human;
      } catch {
        $('statusDot').className = 'status-dot err';
        $('uptimeText').textContent = 'unreachable';
      }
    }

    /* ---- workspace selector ---- */
    const ws = $('workspaceSelect');
    async function loadWorkspaces() {
      try {
        const [wsList, curWs] = await Promise.all([api('/api/workspaces'), api('/api/current_workspace')]);
        const cur = curWs.current_workspace;
        const avail = wsList.workspaces || [];
        ws.innerHTML = '';
        if (!cur && avail.length === 0) {
          ws.innerHTML = '<option value="">single-repo mode</option>';
          return;
        }
        avail.forEach(n => {
          const o = document.createElement('option');
          o.value = n; o.textContent = n;
          if (n === cur) o.selected = true;
          ws.appendChild(o);
        });
        if (cur && !avail.includes(cur)) {
          const o = document.createElement('option');
          o.value = cur; o.textContent = cur + ' (not found)'; o.selected = true;
          ws.prepend(o);
          showAlert("Workspace '" + cur + "' config not found. Select another workspace or create it with: srclight workspace init " + cur);
        } else {
          hideAlert();
        }
      } catch (e) {
        ws.innerHTML = '<option value="">error</option>';
      }
    }
    ws.addEventListener('change', async () => {
      const name = ws.value;
      if (!name) return;
      try {
        await api('/api/switch_workspace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ workspace: name }) });
        hideAlert();
        loadStats();
        loadProjects();
      } catch (e) {
        showAlert(e.message);
      }
    });

    /* ---- stats ---- */
    async function loadStats() {
      try {
        const d = await api('/api/codebase_map');
        if (d.projects) {
          // workspace mode aggregated
          let files = 0, symbols = 0, edges = 0;
          const projs = d.projects;
          Object.values(projs).forEach(p => {
            files += p.files || 0;
            symbols += p.symbols || 0;
            edges += p.edges || 0;
          });
          $('statProjects').textContent = fmt(Object.keys(projs).length);
          $('statFiles').textContent = fmt(files);
          $('statSymbols').textContent = fmt(symbols);
          $('statEdges').textContent = fmt(edges);
        } else if (d.index) {
          $('statProjects').textContent = '1';
          $('statFiles').textContent = fmt(d.index.files);
          $('statSymbols').textContent = fmt(d.index.symbols);
          $('statEdges').textContent = fmt(d.index.edges);
        }
      } catch {}
      try {
        const e = await api('/api/embedding_status');
        if (e.total_embedded != null) {
          $('statEmbeddings').textContent = fmt(e.total_embedded);
        } else if (e.embedded_symbols != null) {
          $('statEmbeddings').textContent = fmt(e.embedded_symbols);
        } else {
          $('statEmbeddings').textContent = '-';
        }
      } catch {}
    }

    /* ---- projects ---- */
    async function loadProjects() {
      const grid = $('projectGrid');
      try {
        const d = await api('/api/list_projects');
        const projects = d.projects || d;
        if (!projects || (Array.isArray(projects) && projects.length === 0)) {
          grid.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem;">No projects found. Add repos with <code>srclight workspace add</code>.</div>';
          return;
        }
        // Handle both array and object formats
        const items = Array.isArray(projects) ? projects : Object.entries(projects).map(([k,v]) => ({name: k, ...v}));
        grid.innerHTML = '';
        for (const p of items) {
          const name = p.name || p.project || '?';
          const files = p.files || 0;
          const symbols = p.symbols || 0;
          const edges = p.edges || 0;
          const dbMb = p.db_size_mb != null ? p.db_size_mb.toFixed(1) + ' MB' : '-';
          const langs = p.languages || {};
          const embCov = p.embedding_coverage;

          let langHtml = '';
          const langEntries = Object.entries(langs).sort((a,b) => b[1] - a[1]).slice(0, 5);
          langEntries.forEach(([lang, count]) => {
            langHtml += `<span class="lang-tag">${lang} ${count}</span>`;
          });

          let embedHtml = '';
          if (embCov != null) {
            const pct = Math.round(embCov * 100);
            embedHtml = `
              <div class="embed-bar">
                <div class="embed-bar-label">
                  <span>Embeddings</span>
                  <span>${pct}%</span>
                </div>
                <div class="embed-bar-track">
                  <div class="embed-bar-fill" style="width: ${pct}%"></div>
                </div>
              </div>`;
          }

          grid.innerHTML += `
            <div class="project-card">
              <div class="project-name">${name}</div>
              <div class="project-stats">
                <div class="project-stat"><span class="project-stat-label">Files</span><span class="project-stat-value">${fmt(files)}</span></div>
                <div class="project-stat"><span class="project-stat-label">Symbols</span><span class="project-stat-value">${fmt(symbols)}</span></div>
                <div class="project-stat"><span class="project-stat-label">Edges</span><span class="project-stat-value">${fmt(edges)}</span></div>
                <div class="project-stat"><span class="project-stat-label">DB size</span><span class="project-stat-value">${dbMb}</span></div>
              </div>
              ${langHtml ? '<div class="project-langs">' + langHtml + '</div>' : ''}
              ${embedHtml}
            </div>`;
        }
      } catch (e) {
        grid.innerHTML = `<div style="color: var(--red); font-size: 0.85rem;">${e.message}</div>`;
      }
    }

    /* ---- search ---- */
    async function doSearch() {
      const q = $('searchInput').value.trim();
      if (!q) return;
      const mode = $('searchMode').value;
      const meta = $('searchMeta');
      const results = $('searchResults');
      results.innerHTML = '<div style="color: var(--text-dim);">Searching...</div>';
      meta.classList.add('hidden');
      try {
        const endpoint = mode === 'hybrid' ? '/api/search?mode=hybrid&q=' : '/api/search?mode=keyword&q=';
        const d = await api(endpoint + encodeURIComponent(q));

        const items = d.results || [];
        const count = d.result_count ?? items.length;
        const modeLabel = d.mode || mode;
        meta.textContent = `${count} result${count !== 1 ? 's' : ''} via ${modeLabel}`;
        meta.classList.remove('hidden');

        if (items.length === 0) {
          results.innerHTML = '<div style="color: var(--text-dim); padding: 12px 0;">No results found.' + (d.hint ? ' ' + d.hint : '') + '</div>';
          return;
        }

        results.innerHTML = '';
        for (const r of items) {
          const name = r.name || r.qualified_name || '?';
          const kind = r.kind || '';
          const file = r.file || r.file_path || '';
          const line = r.start_line || r.line || '';
          const sig = r.signature || '';
          const proj = r.project || '';
          const loc = file + (line ? ':' + line : '');

          results.innerHTML += `
            <div class="search-result">
              <div class="sr-header">
                <span class="sr-name">${name}</span>
                ${kind ? '<span class="sr-kind">' + kind + '</span>' : ''}
                ${proj ? '<span class="sr-project">' + proj + '</span>' : ''}
              </div>
              <div class="sr-file">${loc}</div>
              ${sig ? '<div class="sr-sig">' + sig.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>' : ''}
            </div>`;
        }
      } catch (e) {
        results.innerHTML = `<div style="color: var(--red);">${e.message}</div>`;
      }
    }
    $('btnSearch').onclick = doSearch;
    $('searchInput').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

    /* ---- embedding info ---- */
    async function loadEmbeddingInfo() {
      try {
        const d = await api('/api/embedding_status');
        $('infoEmbedModel').textContent = d.model || 'none';
      } catch { $('infoEmbedModel').textContent = '-'; }
      try {
        const d = await api('/api/embedding_health');
        const ok = d.status === 'ok' || d.healthy === true;
        $('infoEmbedHealth').textContent = ok ? 'healthy' : (d.error || d.status || 'unknown');
        $('infoEmbedHealth').style.color = ok ? 'var(--green)' : 'var(--red)';
      } catch { $('infoEmbedHealth').textContent = '-'; }
    }

    /* ---- restart ---- */
    $('btnRestart').onclick = async () => {
      if (!confirm('Restart the srclight server?')) return;
      $('restartMsg').textContent = 'Requesting restart...';
      try {
        const d = await api('/api/restart_server', { method: 'POST' });
        $('restartMsg').textContent = d.message || 'Restart requested.';
      } catch (e) {
        $('restartMsg').textContent = e.message;
        $('restartMsg').style.color = 'var(--red)';
      }
    };

    /* ---- init ---- */
    loadServerStatus();
    loadWorkspaces();
    loadStats();
    loadProjects();
    loadEmbeddingInfo();

    // Refresh server status periodically
    setInterval(loadServerStatus, 30000);
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


async def _api_version(_request: Request) -> Response:
    from . import __version__
    return JSONResponse({"version": __version__})


async def _api_search(request: Request) -> Response:
    q = request.query_params.get("q", "").strip()
    if not q:
        return JSONResponse({"error": "query parameter 'q' is required"}, status_code=400)
    mode = request.query_params.get("mode", "hybrid")
    try:
        if mode == "keyword":
            from .server import search_symbols
            body = await _run_sync(search_symbols, q)
        else:
            from .server import hybrid_search
            body = await _run_sync(hybrid_search, q)
        return JSONResponse(json.loads(body))
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
        Route("/api/version", _api_version, methods=["GET"]),
        Route("/api/search", _api_search, methods=["GET"]),
    ]
    for r in routes:
        app.router.routes.append(r)
