"""
Web dashboard and REST API for srclight (optional, used with --web when serving SSE).

Local-only: server binds to 127.0.0.1. No secrets in responses; config paths only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger("srclight.web")


def _dashboard_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>srclight</title>
  <link rel="icon" href="/favicon.ico" type="image/svg+xml">
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
      --text-faint: #6b7280;
      --border: #1e1e2a;
      --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', 'Liberation Mono', monospace;
      --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, Helvetica, Arial, sans-serif;
    }

    html { font-size: 15px; }
    body {
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }
    a { color: var(--amber); }
    button, input, select { font-family: var(--sans); font-size: 0.85rem; }
    :focus-visible { outline: none; box-shadow: 0 0 0 2px var(--amber-dim), 0 0 0 3px var(--amber); border-radius: 6px; }

    .container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

    /* -- Header -- */
    header { padding: 16px 0; border-bottom: 1px solid var(--border); }
    .header-inner { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
    .header-left { display: flex; align-items: center; gap: 14px; }
    .wordmark { font-family: var(--mono); font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
    .wordmark .src { color: var(--amber); }
    .wordmark .light { color: #fff; }
    .badge {
      font-size: 0.7rem; font-family: var(--mono); color: var(--text-dim);
      background: var(--bg-card); border: 1px solid var(--border); padding: 3px 8px; border-radius: 4px;
    }
    .header-right { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .header-right select {
      background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 6px 10px; cursor: pointer;
    }
    .health {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 5px 12px; border-radius: 999px; border: 1px solid var(--border);
      background: var(--bg-card); font-size: 0.8rem; color: var(--text);
    }
    .health.ok { border-color: rgba(63,185,80,0.35); }
    .health.warn { border-color: rgba(245,158,11,0.45); }
    .health.err { border-color: rgba(248,81,73,0.45); }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: var(--text-faint); }
    .status-dot.ok { background: var(--green); box-shadow: 0 0 6px rgba(63,185,80,0.5); }
    .status-dot.warn { background: var(--amber); box-shadow: 0 0 6px rgba(245,158,11,0.5); }
    .status-dot.err { background: var(--red); box-shadow: 0 0 6px rgba(248,81,73,0.5); }
    .header-meta { font-size: 0.78rem; color: var(--text-dim); font-family: var(--mono); }
    .header-meta b { color: var(--text); font-weight: 500; }

    /* -- Alert -- */
    .alert {
      position: sticky; top: 8px; z-index: 50;
      display: none; align-items: center; justify-content: space-between; gap: 12px;
      background: var(--red-dim); border: 1px solid var(--red); border-radius: 8px;
      padding: 10px 14px; margin-top: 14px; font-size: 0.85rem;
    }
    .alert.show { display: flex; }
    .alert.warn { background: var(--amber-dim); border-color: var(--amber); }
    .alert details { font-size: 0.75rem; color: var(--text-dim); margin-top: 4px; }
    .alert code { font-family: var(--mono); word-break: break-all; }

    /* -- Stats bar -- */
    .stats { padding: 24px 0 18px; border-bottom: 1px solid var(--border); }
    .stats-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; text-align: center; }
    .stat-value { font-family: var(--mono); font-size: 1.7rem; font-weight: 600; color: var(--text); line-height: 1.2; }
    .stat-value.ok { color: var(--green); }
    .stat-value.warn { color: var(--amber); }
    .stat-value.err { color: var(--red); }
    .stat-value.is-loading { color: var(--text-faint); animation: pulse 1.2s ease-in-out infinite; }
    .stat-label { font-size: 0.72rem; color: var(--text-dim); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
    .stat-sub { font-size: 0.72rem; color: var(--text-faint); font-family: var(--mono); margin-top: 2px; min-height: 1em; }
    .stats-sub { text-align: center; font-size: 0.78rem; color: var(--text-dim); margin-top: 14px; font-family: var(--mono); }
    .stats-sub b { color: var(--text); font-weight: 500; }
    @keyframes pulse { 0%, 100% { opacity: 0.35; } 50% { opacity: 0.9; } }
    @media (prefers-reduced-motion: reduce) {
      .stat-value.is-loading { animation: none; }
      .status-dot { box-shadow: none !important; }
    }

    /* -- Section -- */
    .section { padding: 26px 0; }
    .section + .section, details.section { border-top: 1px solid var(--border); }
    .section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
    .section-title {
      font-size: 0.75rem; font-weight: 600; color: var(--text-dim);
      text-transform: uppercase; letter-spacing: 0.08em;
    }
    .section-note { font-size: 0.78rem; color: var(--text-faint); }
    .section-note b { color: var(--text-dim); font-weight: 500; }
    .section-note .warn { color: var(--amber); }

    /* -- Buttons -- */
    .btn {
      display: inline-flex; align-items: center; gap: 6px; padding: 9px 16px; border-radius: 8px;
      border: 1px solid transparent; font-weight: 500; cursor: pointer; transition: all 0.15s;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-primary { background: var(--amber); color: #0a0a0f; }
    .btn-primary:hover:not(:disabled) { background: var(--amber-light); }
    .btn-secondary { background: transparent; color: var(--text); border-color: var(--border); }
    .btn-secondary:hover:not(:disabled), .btn-secondary[aria-pressed="true"] { border-color: var(--amber); color: var(--amber); }
    .btn-text { background: none; border: none; color: var(--text-dim); padding: 4px 6px; cursor: pointer; font-size: 0.8rem; }
    .btn-text:hover:not(:disabled) { color: var(--red); }
    .btn-sm { padding: 5px 10px; font-size: 0.78rem; }

    /* -- Search -- */
    .search-box { display: flex; gap: 10px; flex-wrap: wrap; }
    .search-input {
      flex: 1 1 320px; background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 14px; font-size: 0.9rem;
    }
    .search-input:focus { border-color: var(--amber); }
    .search-input::placeholder { color: var(--text-faint); }
    .search-box select {
      background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 12px;
    }
    .search-help { font-size: 0.75rem; color: var(--text-faint); margin-top: 8px; }
    .search-help kbd { font-family: var(--mono); background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; padding: 0 5px; }
    .search-meta { font-size: 0.78rem; color: var(--text-dim); margin: 12px 0 10px; font-family: var(--mono); }
    .search-results { font-size: 0.85rem; }
    .search-result {
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 14px; margin-bottom: 8px;
    }
    .sr-header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 3px; flex-wrap: wrap; }
    .sr-name { font-family: var(--mono); font-weight: 600; color: var(--amber); }
    .sr-kind { font-size: 0.7rem; color: var(--text-dim); background: var(--bg); padding: 1px 6px; border-radius: 3px; }
    .sr-project { font-size: 0.7rem; color: var(--amber); background: var(--amber-dim); padding: 1px 6px; border-radius: 3px; }
    .sr-sim { font-size: 0.7rem; color: var(--text-faint); font-family: var(--mono); margin-left: auto; }
    .sr-file { font-family: var(--mono); font-size: 0.75rem; color: var(--text-dim); cursor: copy; }
    .sr-file:hover { color: var(--text); }
    .sr-sig { font-family: var(--mono); font-size: 0.78rem; color: var(--text-dim); margin-top: 4px; white-space: pre-wrap; word-break: break-all; }

    /* -- Projects -- */
    .proj-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .proj-toolbar input, .proj-toolbar select {
      background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 6px 10px; font-size: 0.8rem;
    }
    .proj-toolbar input { flex: 1 1 200px; max-width: 320px; }
    .proj-toolbar input:focus { border-color: var(--amber); }
    .proj-table { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--bg-card); }
    .proj-row, .proj-head {
      display: grid;
      grid-template-columns: minmax(160px, 2fr) repeat(3, minmax(70px, 1fr)) minmax(110px, 1.2fr) minmax(90px, 1fr) minmax(110px, 1fr);
      gap: 12px; align-items: center; padding: 8px 14px;
    }
    .proj-head {
      font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint);
      border-bottom: 1px solid var(--border); background: var(--bg);
    }
    .proj-row { border-bottom: 1px solid var(--border); cursor: pointer; font-size: 0.82rem; }
    .proj-row:last-of-type { border-bottom: none; }
    .proj-row:hover { background: var(--bg-card-hover); }
    .proj-name { font-family: var(--mono); font-weight: 600; color: var(--amber); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .num { font-family: var(--mono); text-align: right; color: var(--text); }
    .num.zero { color: var(--text-faint); }
    .embed-cell { display: flex; align-items: center; gap: 8px; }
    .embed-bar-track { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
    .embed-bar-fill { height: 100%; background: var(--green); border-radius: 2px; }
    .embed-bar-fill.warn { background: var(--amber); }
    .embed-pct { font-family: var(--mono); font-size: 0.75rem; color: var(--text-dim); min-width: 3.5em; text-align: right; }
    .rel { font-family: var(--mono); font-size: 0.75rem; color: var(--text-dim); }
    .pill { font-size: 0.68rem; font-family: var(--mono); padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
    .pill.ok { color: var(--green); background: var(--green-dim); }
    .pill.warn { color: var(--amber); background: var(--amber-dim); }
    .pill.err { color: var(--red); background: var(--red-dim); }
    .pill.dim { color: var(--text-dim); background: var(--bg); border: 1px solid var(--border); }
    .proj-detail {
      display: none; grid-column: 1 / -1; padding: 6px 0 4px 0; font-size: 0.78rem; color: var(--text-dim);
    }
    .proj-row.open .proj-detail { display: block; }
    .proj-detail .path { font-family: var(--mono); color: var(--text); word-break: break-all; }
    .proj-langs { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
    .lang-tag { font-size: 0.7rem; font-family: var(--mono); color: var(--amber); background: var(--amber-dim); padding: 1px 7px; border-radius: 4px; }
    .lang-tag.other { color: var(--text-dim); background: var(--bg); border: 1px solid var(--border); }
    .proj-empty { padding: 18px 14px; color: var(--text-dim); font-size: 0.85rem; }
    .proj-empty code { font-family: var(--mono); color: var(--text); }

    /* -- Connect -- */
    details.section summary { list-style: none; cursor: pointer; }
    details.section summary::-webkit-details-marker { display: none; }
    details.section summary .section-title::before { content: '▸ '; color: var(--text-faint); }
    details.section[open] summary .section-title::before { content: '▾ '; }
    .connect-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 14px; }
    .connect-detail { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
    .connect-path { font-size: 0.78rem; color: var(--text-dim); margin-bottom: 8px; }
    .connect-path code { font-family: var(--mono); color: var(--text); }
    .connect-snippet {
      font-family: var(--mono); font-size: 0.8rem; color: var(--text);
      white-space: pre-wrap; word-break: break-all; margin: 0; user-select: all;
    }
    .connect-after { font-size: 0.78rem; color: var(--text-dim); margin-top: 10px; }
    .copy-msg { font-size: 0.8rem; color: var(--green); margin-left: 8px; }

    /* -- Server info -- */
    .info-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
    .info-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
    .info-label { font-size: 0.68rem; color: var(--text-faint); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
    .info-value { font-family: var(--mono); font-size: 0.85rem; color: var(--text); word-break: break-all; }
    .info-value.ok { color: var(--green); }
    .info-value.warn { color: var(--amber); }
    .info-value.err { color: var(--red); }
    .server-actions { display: flex; align-items: center; gap: 10px; margin-top: 12px; font-size: 0.8rem; color: var(--text-dim); }

    /* -- Footer -- */
    footer { padding: 22px 0; border-top: 1px solid var(--border); text-align: center; }
    .footer-links { display: flex; gap: 20px; justify-content: center; margin-bottom: 6px; }
    .footer-links a { color: var(--text-dim); text-decoration: none; font-size: 0.78rem; }
    .footer-links a:hover { color: var(--amber); }
    .footer-copy { font-size: 0.72rem; color: var(--text-faint); }

    .hidden { display: none !important; }
    body.is-down .stat-value, body.is-down .info-value, body.is-down .stats-sub { opacity: 0.4; }
    .dim { color: var(--text-dim); }

    /* -- Responsive -- */
    @media (max-width: 900px) {
      .proj-row, .proj-head { grid-template-columns: minmax(120px, 2fr) repeat(2, minmax(60px, 1fr)) minmax(100px, 1fr) minmax(90px, 1fr); }
      .col-edges, .col-indexed { display: none; }
    }
    @media (max-width: 768px) {
      .stats-grid { grid-template-columns: repeat(3, 1fr); }
      .header-inner { flex-direction: column; align-items: flex-start; }
    }
    @media (max-width: 560px) {
      .stats-grid { grid-template-columns: repeat(2, 1fr); }
      .proj-row, .proj-head { grid-template-columns: minmax(100px, 2fr) minmax(60px, 1fr) minmax(90px, 1fr); }
      .col-files, .col-embed { display: none; }
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="container header-inner">
      <div class="header-left">
        <span class="wordmark"><span class="src">src</span><span class="light">light</span></span>
        <span class="badge" id="versionBadge">v?</span>
      </div>
      <div class="header-right">
        <select id="workspaceSelect" aria-label="Workspace" title="Switch workspace">
          <option value="">Loading…</option>
        </select>
        <div class="health" id="healthPill" role="status" aria-live="polite">
          <span class="status-dot" id="statusDot"></span>
          <span id="healthText">connecting…</span>
        </div>
        <div class="header-meta" id="headerMeta"></div>
      </div>
    </div>
    <div class="container">
      <div class="alert" id="alertBox" role="alert" aria-live="polite">
        <div>
          <div id="alertText"></div>
          <details id="alertDetails" class="hidden"><summary>details</summary><code id="alertRaw"></code></details>
        </div>
        <button class="btn btn-secondary btn-sm" id="alertRetry" type="button">Retry</button>
      </div>
    </div>
  </header>

  <!-- Stats -->
  <section class="stats">
    <div class="container">
      <div class="stats-grid">
        <div>
          <div class="stat-value is-loading" id="statProjects">—</div>
          <div class="stat-label">Projects</div>
          <div class="stat-sub" id="statProjectsSub"></div>
        </div>
        <div>
          <div class="stat-value is-loading" id="statFiles">—</div>
          <div class="stat-label">Files</div>
          <div class="stat-sub"></div>
        </div>
        <div>
          <div class="stat-value is-loading" id="statSymbols">—</div>
          <div class="stat-label">Symbols</div>
          <div class="stat-sub"></div>
        </div>
        <div>
          <div class="stat-value is-loading" id="statEdges">—</div>
          <div class="stat-label">Edges</div>
          <div class="stat-sub"></div>
        </div>
        <div>
          <div class="stat-value is-loading" id="statEmbedded">—</div>
          <div class="stat-label">Embedded</div>
          <div class="stat-sub" id="statEmbeddedSub"></div>
        </div>
      </div>
      <div class="stats-sub" id="statsSub"></div>
    </div>
  </section>

  <main class="container">

    <!-- Search -->
    <section class="section" id="searchSection">
      <div class="section-head">
        <div class="section-title">Search</div>
        <div class="section-note">One hit here is proof the index answers.</div>
      </div>
      <div class="search-box">
        <input type="text" class="search-input" id="searchInput" aria-label="Search symbols"
               placeholder="Symbol name, code fragment, or a concept in plain words" autocomplete="off">
        <select id="searchMode" aria-label="Search mode" title="Search mode">
          <option value="hybrid">Hybrid</option>
          <option value="keyword">Keyword</option>
        </select>
        <button class="btn btn-primary" id="btnSearch" type="button">Search</button>
      </div>
      <div class="search-help">
        <b>Hybrid</b> = full-text + embeddings (needs the embedding provider). <b>Keyword</b> = full-text only, works even if it is down.
        Press <kbd>/</kbd> to focus. Click a file:line to copy it.
      </div>
      <div class="search-meta hidden" id="searchMeta"></div>
      <div class="search-results" id="searchResults"></div>
    </section>

    <!-- Projects -->
    <section class="section" id="projectsSection">
      <div class="section-head">
        <div class="section-title">Projects</div>
        <div class="section-note" id="projectsNote"></div>
      </div>
      <div class="proj-toolbar" id="projToolbar">
        <input type="search" id="projectFilter" aria-label="Filter projects" placeholder="Filter by name or path" autocomplete="off">
        <select id="projectSort" aria-label="Sort projects" title="Sort">
          <option value="attention">Needs attention first</option>
          <option value="symbols">Most symbols</option>
          <option value="files">Most files</option>
          <option value="edges">Most edges</option>
          <option value="embed">Lowest embedding coverage</option>
          <option value="indexed">Least recently indexed</option>
          <option value="size">Largest DB</option>
          <option value="name">Name</option>
        </select>
      </div>
      <div class="proj-table" id="projectTable" style="margin-top: 12px;">
        <div class="proj-head">
          <div>Project</div>
          <div class="num col-files">Files</div>
          <div class="num">Symbols</div>
          <div class="num col-edges">Edges</div>
          <div class="col-embed">Embedded</div>
          <div class="col-indexed">Indexed</div>
          <div>Status</div>
        </div>
        <div id="projectList"><div class="proj-empty dim">Loading projects…</div></div>
      </div>
    </section>

    <!-- Connect Your AI -->
    <details class="section" id="connectSection">
      <summary>
        <div class="section-head" style="margin-bottom: 0;">
          <div class="section-title">Connect your AI</div>
          <div class="section-note" id="connectNote"></div>
        </div>
      </summary>
      <p class="section-note" style="margin-top: 10px;">
        Add srclight to your AI tool so it can search this index, trace call graphs, and read git history.
        Pick your tool, copy the snippet into the config file shown, restart the tool.
      </p>
      <div class="connect-tabs" id="connectTabs">
        <button class="btn btn-secondary btn-sm connect-tab" data-client="claude_code" type="button" aria-pressed="false" disabled>Claude Code</button>
        <button class="btn btn-secondary btn-sm connect-tab" data-client="claude_desktop" type="button" aria-pressed="false" disabled>Claude Desktop</button>
        <button class="btn btn-secondary btn-sm connect-tab" data-client="cursor" type="button" aria-pressed="false" disabled>Cursor</button>
        <button class="btn btn-secondary btn-sm connect-tab" data-client="vscode" type="button" aria-pressed="false" disabled>VS Code</button>
        <button class="btn btn-secondary btn-sm connect-tab" data-client="windsurf" type="button" aria-pressed="false" disabled>Windsurf</button>
      </div>
      <div id="connectDetail" class="connect-detail hidden">
        <div class="connect-path" id="connectPath"></div>
        <pre class="connect-snippet" id="connectSnippet"></pre>
        <div style="margin-top: 10px;">
          <button class="btn btn-primary btn-sm" id="btnCopySnippet" type="button">Copy to clipboard</button>
          <span class="copy-msg" id="copyMsg"></span>
        </div>
        <div class="connect-after" id="connectAfter"></div>
      </div>
      <div id="connectError" class="section-note hidden" style="color: var(--red); margin-top: 8px;"></div>
    </details>

    <!-- Server -->
    <section class="section" id="serverSection">
      <div class="section-head">
        <div class="section-title">Server</div>
        <div class="section-note" id="serverNote"></div>
      </div>
      <div class="info-grid">
        <div class="info-card">
          <div class="info-label">MCP endpoint</div>
          <div class="info-value" id="infoMcpUrl">—</div>
        </div>
        <div class="info-card">
          <div class="info-label">Embedding model</div>
          <div class="info-value" id="infoEmbedModel">—</div>
        </div>
        <div class="info-card">
          <div class="info-label">Embedding health</div>
          <div class="info-value" id="infoEmbedHealth">—</div>
        </div>
        <div class="info-card">
          <div class="info-label">Index freshness</div>
          <div class="info-value" id="infoFreshness">—</div>
        </div>
        <div class="info-card">
          <div class="info-label">Build</div>
          <div class="info-value" id="infoBuild">—</div>
        </div>
      </div>
      <div class="server-actions">
        <button class="btn-text" id="btnRestart" type="button">Restart server</button>
        <span id="restartMsg"></span>
      </div>
    </section>

  </main>

  <!-- Footer -->
  <footer>
    <div class="container">
      <div class="footer-links">
        <a href="https://github.com/srclight/srclight">GitHub</a>
        <a href="https://pypi.org/project/srclight/">PyPI</a>
        <a href="https://github.com/srclight/srclight/issues">Issues</a>
        <a href="https://srclight.dev">srclight.dev</a>
        <a href="/healthz">/healthz</a>
      </div>
      <p class="footer-copy">Local only. Server listens on 127.0.0.1.</p>
    </div>
  </footer>

  <script>
    /* ================= helpers ================= */
    const $ = id => document.getElementById(id);
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt = n => (n == null || Number.isNaN(n)) ? '—' : Number(n).toLocaleString();
    const pct = (num, den) => den ? Math.round(num / den * 100) : 0;

    function humanSecs(s) {
      if (s == null) return '—';
      s = Math.floor(s);
      if (s < 60) return s + 's';
      const m = Math.floor(s / 60), r = s % 60;
      if (m < 60) return m + 'm ' + r + 's';
      const h = Math.floor(m / 60), mm = m % 60;
      if (h < 24) return h + 'h ' + mm + 'm';
      return Math.floor(h / 24) + 'd ' + (h % 24) + 'h';
    }
    function relTime(iso) {
      if (!iso) return null;
      const t = Date.parse(iso.endsWith('Z') || /[+-]\d\d:\d\d$/.test(iso) ? iso : iso + 'Z');
      if (Number.isNaN(t)) return null;
      const s = Math.max(0, (Date.now() - t) / 1000);
      if (s < 60) return 'just now';
      if (s < 3600) return Math.floor(s / 60) + 'm ago';
      if (s < 86400) return Math.floor(s / 3600) + 'h ago';
      return Math.floor(s / 86400) + 'd ago';
    }

    /* Friendly copy for backend failures. Raw text is kept for the details fold. */
    class ApiError extends Error {
      constructor(message, { status = 0, raw = '', transient = false } = {}) {
        super(message); this.status = status; this.raw = raw; this.transient = transient;
      }
    }
    function classify(rawMsg, status) {
      const m = (rawMsg || '').toLowerCase();
      if (/misuse|database is locked|busy|too many attached/.test(m)) return { text: 'Index is busy.', transient: true };
      if (status === 0) return { text: 'Server unreachable.', transient: true };
      if (status >= 500) return { text: 'Server error.', transient: false };
      if (status === 404) return { text: 'Not found.', transient: false };
      return { text: rawMsg || 'Request failed.', transient: false };
    }
    async function api(path, opts = {}) {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), opts.timeout || 20000);
      let r, text;
      try {
        r = await fetch(path, { ...opts, signal: ctrl.signal, headers: { Accept: 'application/json', ...(opts.headers || {}) } });
        text = await r.text();
      } catch (e) {
        clearTimeout(timer);
        const timedOut = e && e.name === 'AbortError';
        throw new ApiError(timedOut ? 'Request timed out.' : 'Server unreachable.', { status: 0, raw: String(e), transient: true });
      }
      clearTimeout(timer);
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = null; }
      if (!r.ok) {
        const raw = (data && data.error) || text || r.statusText;
        const c = classify(raw, r.status);
        throw new ApiError(c.text, { status: r.status, raw, transient: c.transient });
      }
      if (data && data.error && opts.strict !== false) {
        const c = classify(data.error, 200);
        throw new ApiError(c.text, { status: 200, raw: data.error, transient: c.transient });
      }
      return data;
    }
    async function withRetry(fn, tries = 3) {
      let last;
      for (let i = 0; i < tries; i++) {
        try { return await fn(i); } catch (e) { last = e; if (!(e instanceof ApiError) || !e.transient) throw e; await new Promise(r => setTimeout(r, 400 * (i + 1))); }
      }
      throw last;
    }

    /* ================= alert (single error surface) ================= */
    // What to type when the process is gone (BARRY): the page cannot start it.
    function rescueLine() {
      const wsName = (state.health && state.health.workspace) || (ws && ws.value) || '<workspace>';
      return 'If nothing supervises it, start it with: srclight serve --workspace ' + wsName + ' --web  (or: systemctl --user start srclight)';
    }
    let _retryFn = null, _alertPriority = 0;
    // priority: 3 = server unreachable, 2 = workspace/index, 1 = a single pane. A lower
    // priority never overwrites a higher one, so "create it with srclight workspace init"
    // is not clobbered by the next poll's generic index error.
    function showAlert(msg, { raw = '', level = 'err', retry = null, priority = 1 } = {}) {
      const a = $('alertBox');
      if (a.classList.contains('show') && priority < _alertPriority) return;
      _alertPriority = priority;
      $('alertText').textContent = msg;
      $('alertRaw').textContent = raw || '';
      $('alertDetails').classList.toggle('hidden', !raw);
      a.classList.toggle('warn', level === 'warn');
      a.classList.add('show');
      _retryFn = retry;
      $('alertRetry').classList.toggle('hidden', !retry);
    }
    function hideAlert() { $('alertBox').classList.remove('show'); _retryFn = null; _alertPriority = 0; }
    $('alertRetry').onclick = () => { if (_retryFn) { hideAlert(); _retryFn(); } };

    /* ================= state ================= */
    const state = { gen: 0, health: null, cmap: null, projects: [], lastGoodWs: null, connection: null, filter: '', sort: 'attention', open: new Set() };
    const setLoading = (id, on) => $(id).classList.toggle('is-loading', on);

    /* ================= health (drives the header) ================= */
    // The one rule that decides the colour of the dot. Tweak here, nowhere else.
    function composeHealth(h) {
      if (!h) return { level: 'err', text: 'unreachable' };
      if (h.status === 'error' || h.index_error) return { level: 'err', text: 'index error' };
      // The server spells out every reason a monitor would alert on; the
      // header shows the first so human and machine agree (STUBBY).
      const d = Array.isArray(h.degraded) ? h.degraded : [];
      if (d.length) return { level: 'warn', text: 'degraded · ' + d[0] + (d.length > 1 ? ' (+' + (d.length - 1) + ')' : '') };
      return { level: 'ok', text: 'healthy' };
    }
    function paintHealth(h) {
      const c = composeHealth(h);
      $('statusDot').className = 'status-dot ' + c.level;
      $('healthPill').className = 'health ' + c.level;
      $('healthText').textContent = c.text;
      $('healthPill').title = h && h.degraded && h.degraded.length ? h.degraded.join('\n') : '';
      document.body.classList.toggle('is-down', !h);
      if (!h) { $('headerMeta').innerHTML = '<span class="dim">showing last known values</span>'; return; }
      const q = h.queries || {};
      const lastQ = q.last_ago_seconds != null ? '<b>' + esc(humanSecs(q.last_ago_seconds)) + ' ago</b>' : '<b>never</b>';
      $('headerMeta').innerHTML = 'up ' + esc(humanSecs(h.uptime_seconds)) + ' · last query ' + lastQ + ' · ' + fmt(q.count || 0) + ' queries';
      if (h.version) { $('versionBadge').textContent = 'v' + h.version; }
      document.title = 'srclight' + (h.workspace ? ' · ' + h.workspace : '');

      // Embedded stat + server cards come from the same payload.
      const e = h.embeddings || {};
      const embOk = e.status === 'ok';
      const cov = pct(h.embedded || 0, h.symbols || 0);
      const stat = $('statEmbedded');
      setLoading('statEmbedded', false);
      if (!embOk && !(h.embedded > 0)) {
        stat.textContent = 'Keyword only'; stat.className = 'stat-value err';
        $('statEmbeddedSub').textContent = e.error ? 'provider ' + (e.status || 'error') : 'no embeddings';
      } else {
        stat.textContent = fmt(h.embedded);
        stat.className = 'stat-value ' + (cov >= 95 ? 'ok' : 'warn');
        $('statEmbeddedSub').textContent = cov + '% of symbols' + (embOk ? '' : ' · provider ' + (e.status || 'down'));
      }
      const model = e.model || e.provider || null;
      $('infoEmbedModel').textContent = model || 'none configured';
      const healthEl = $('infoEmbedHealth');
      if (embOk) {
        healthEl.textContent = e.resident === false ? 'reachable · model not loaded' : 'healthy' + (e.dimensions ? ' · ' + e.dimensions + 'd' : '');
        healthEl.className = 'info-value ' + (e.resident === false ? 'warn' : 'ok');
      } else {
        healthEl.textContent = (e.status || 'unknown') + (e.error ? ' · ' + e.error : '') + (e.hint ? ' — ' + e.hint : '');
        healthEl.className = 'info-value err';
      }
      const fresh = relTime(h.last_indexed);
      $('infoFreshness').textContent = fresh ? 'indexed ' + fresh : 'no index yet';
      $('infoBuild').textContent = 'v' + (h.version || '?') + (h.code_sha && h.code_sha !== 'unknown' ? ' @ ' + h.code_sha : '') + (h.mcp_sdk ? ' · mcp ' + h.mcp_sdk : '');
      $('infoMcpUrl').textContent = location.origin + (h.mcp || '/mcp');
      $('statsSub').innerHTML = [
        model ? '<b>' + esc(model) + '</b>' : 'no embedding model',
        fresh ? 'indexed <b>' + esc(fresh) + '</b>' : 'no index yet',
        h.workspace ? 'workspace <b>' + esc(h.workspace) + '</b>' : 'single repo',
      ].join(' · ');

      // Connect: expand for the person who has never been queried.
      const hasQueries = (q.count || 0) > 0;
      $('connectNote').textContent = hasQueries ? 'connected · ' + fmt(q.count) + ' queries served' : 'no queries yet — paste the snippet into your tool';
      if (!state._connectDecided) { $('connectSection').open = !hasQueries; state._connectDecided = true; }
    }
    async function loadHealth() {
      const wasDown = state.down === true;
      try {
        const h = await api('/healthz', { timeout: 30000 });
        state.health = h; state.down = false; paintHealth(h);
        if (h.index_error) { showAlert('The index could not be read.', { raw: h.index_error, retry: reloadAll, priority: 2 }); return; }
        // Back after an outage or restart: clear the red state and rebuild every pane.
        if (wasDown) { hideAlert(); loadStats(); loadProjects(); loadWorkspaces(); loadConnectionInfo(); }
      } catch (e) {
        state.health = null; state.down = true; paintHealth(null);
        showAlert('Cannot reach the srclight server. Retrying every few seconds… ' + rescueLine(), { raw: e.raw || e.message, retry: reloadAll, priority: 3 });
      }
    }

    /* ================= workspace selector ================= */
    const ws = $('workspaceSelect');
    async function loadWorkspaces() {
      try {
        const [wsList, curWs] = await Promise.all([api('/api/workspaces'), api('/api/current_workspace')]);
        const cur = curWs.current_workspace;
        const avail = wsList.workspaces || [];
        ws.innerHTML = '';
        if (!cur && avail.length === 0) { ws.innerHTML = '<option value="">single-repo mode</option>'; ws.disabled = true; return; }
        avail.forEach(n => { const o = document.createElement('option'); o.value = n; o.textContent = n; if (n === cur) o.selected = true; ws.appendChild(o); });
        if (cur && !avail.includes(cur)) {
          const o = document.createElement('option'); o.value = cur; o.textContent = cur + ' (not found)'; o.selected = true; ws.prepend(o);
          showAlert("Workspace '" + cur + "' has no config. Pick another, or create it with: srclight workspace init " + cur, { priority: 2 });
        }
        state.lastGoodWs = ws.value;
      } catch (e) {
        ws.innerHTML = '<option value="">workspaces unavailable</option>';
      }
    }
    ws.addEventListener('change', async () => {
      const name = ws.value;
      if (!name || name === state.lastGoodWs) return;
      if (!confirm('Switch the server to workspace "' + name + '"? Every connected AI tool will see the new workspace.')) { ws.value = state.lastGoodWs; return; }
      const gen = ++state.gen;
      ['statProjects','statFiles','statSymbols','statEdges','statEmbedded'].forEach(id => { $(id).textContent = '—'; setLoading(id, true); });
      $('projectList').innerHTML = '<div class="proj-empty dim">Switching to ' + esc(name) + '…</div>';
      $('searchResults').innerHTML = ''; $('searchMeta').classList.add('hidden');
      try {
        await api('/api/switch_workspace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ workspace: name }) });
        if (gen !== state.gen) return;
        state.lastGoodWs = name; hideAlert(); state._connectDecided = false;
        await reloadAll();
      } catch (e) {
        ws.value = state.lastGoodWs;
        showAlert('Could not switch workspace: ' + e.message, { raw: e.raw, priority: 2 });
        reloadAll();
      }
    });

    /* ================= stats ================= */
    async function loadStats() {
      const gen = state.gen;
      try {
        const d = await withRetry(() => api('/api/codebase_map', { timeout: 30000 }));
        if (gen !== state.gen) return;
        state.cmap = d;
        let projects, files, symbols, edges;
        if (d.totals) {            // workspace mode: trust the server's totals
          projects = d.projects_attached; files = d.totals.files; symbols = d.totals.symbols; edges = d.totals.edges;
        } else if (d.index) {      // single-repo mode
          projects = 1; files = d.index.files; symbols = d.index.symbols; edges = d.index.edges;
        }
        $('statProjects').textContent = fmt(projects);
        $('statFiles').textContent = fmt(files);
        $('statSymbols').textContent = fmt(symbols);
        $('statEdges').textContent = fmt(edges);
        ['statProjects','statFiles','statSymbols','statEdges'].forEach(id => setLoading(id, false));
      } catch (e) {
        if (gen !== state.gen) return;
        ['statProjects','statFiles','statSymbols','statEdges'].forEach(id => { $(id).textContent = '—'; setLoading(id, false); });
        showAlert('Could not read index totals. ' + e.message, { raw: e.raw, retry: reloadAll });
      }
    }

    /* ================= projects ================= */
    const CODE_LANGS = new Set(['c','cpp','csharp','python','dart','javascript','typescript','java','kotlin','swift','go','rust','php','ruby','bash','cmake','sql','groovy','objc','scala','lua','r','perl']);
    function projectStatus(p, wsHasEmbeddings) {
      if (p.error) return { level: 'err', text: 'read failed', rank: 0 };
      if (p.indexed === false) return { level: 'dim', text: 'not indexed', rank: 1 };
      if ((p.files || 0) > 0 && (p.symbols || 0) === 0) return { level: 'warn', text: 'no symbols', rank: 2 };
      if ((p.symbols || 0) > 0 && (p.edges || 0) === 0) return { level: 'warn', text: 'no edges', rank: 3 };
      if (wsHasEmbeddings && (p.symbols || 0) > 0 && (p.embedding_coverage || 0) < 0.95) return { level: 'warn', text: 'partial embed', rank: 4 };
      return { level: 'ok', text: 'ok', rank: 9 };
    }
    function renderProjects() {
      const list = $('projectList');
      const items = state.projects;
      if (!items.length) {
        list.innerHTML = '<div class="proj-empty">No projects yet. Run <code>srclight workspace add /path/to/repo</code>, or use the desktop app.</div>';
        $('projectsNote').textContent = ''; return;
      }
      const wsHasEmb = !!(state.health && state.health.embedded > 0);
      const q = state.filter.trim().toLowerCase();
      const rows = items.map(p => ({ p, st: projectStatus(p, wsHasEmb) }))
        .filter(({ p }) => !q || (p.project || '').toLowerCase().includes(q) || (p.path || '').toLowerCase().includes(q));
      const key = state.sort;
      const num = (v) => (v == null ? -1 : Number(v));
      rows.sort((a, b) => {
        if (key === 'attention') { if (a.st.rank !== b.st.rank) return a.st.rank - b.st.rank; return num(b.p.symbols) - num(a.p.symbols); }
        if (key === 'name') return (a.p.project || '').localeCompare(b.p.project || '');
        if (key === 'embed') return num(a.p.embedding_coverage) - num(b.p.embedding_coverage);
        if (key === 'indexed') return (a.p.last_indexed || '').localeCompare(b.p.last_indexed || '');
        if (key === 'size') return num(b.p.db_size_mb) - num(a.p.db_size_mb);
        return num(b.p[key]) - num(a.p[key]);
      });
      const attention = items.filter(p => { const r = projectStatus(p, wsHasEmb).rank; return r < 9; }).length;
      $('projectsNote').innerHTML = '<b>' + fmt(items.length) + '</b> projects' + (q ? ' · <b>' + rows.length + '</b> shown' : '') +
        (attention ? ' · <span class="warn">' + attention + ' need attention</span>' : ' · all healthy');
      const html = rows.map(({ p, st }) => {
        const name = p.project || p.name || '?';
        const cov = p.embedding_coverage != null ? Math.round(p.embedding_coverage * 100) : null;
        const numCell = (v, cls = '') => p.error
          ? '<div class="num ' + cls + ' zero">—</div>'
          : '<div class="num ' + cls + (v ? '' : ' zero') + '">' + fmt(v ?? 0) + '</div>';
        const embed = (cov == null || p.error) ? '<div class="col-embed rel">—</div>' :
          '<div class="col-embed embed-cell"><div class="embed-bar-track"><div class="embed-bar-fill' + (cov < 95 ? ' warn' : '') + '" style="width:' + cov + '%"></div></div><span class="embed-pct">' + cov + '%</span></div>';
        const rel = relTime(p.last_indexed);
        const langs = Object.entries(p.languages || {}).sort((a, b) => b[1] - a[1]);
        const code = langs.filter(([l]) => CODE_LANGS.has(l)).slice(0, 6);
        const other = langs.filter(([l]) => !CODE_LANGS.has(l));
        const otherN = other.reduce((s, [, n]) => s + n, 0);
        const langHtml = code.map(([l, n]) => '<span class="lang-tag">' + esc(l) + ' ' + fmt(n) + '</span>').join('') +
          (otherN ? '<span class="lang-tag other">+' + fmt(otherN) + ' ' + esc(other.slice(0, 3).map(([l]) => l).join('/')) + (other.length > 3 ? '…' : '') + '</span>' : '');
        return '<div class="proj-row' + (st.rank < 9 ? ' problem' : '') + (state.open.has(name) ? ' open' : '') + '" data-name="' + esc(name) + '" tabindex="0" role="button" aria-expanded="' + (state.open.has(name) ? 'true' : 'false') + '">' +
          '<div class="proj-name" title="' + esc(p.path || '') + '">' + esc(name) + '</div>' +
          numCell(p.files, 'col-files') + numCell(p.symbols) + numCell(p.edges, 'col-edges') + embed +
          '<div class="col-indexed rel">' + esc(rel || '—') + '</div>' +
          '<div><span class="pill ' + st.level + '">' + esc(st.text) + '</span></div>' +
          '<div class="proj-detail">' +
            '<div class="path">' + esc(p.path || '') + '</div>' +
            '<div style="margin-top:4px;">' + (p.db_size_mb != null ? 'DB ' + p.db_size_mb.toFixed(1) + ' MB · ' : '') +
              (p.embedded_symbols != null ? fmt(p.embedded_symbols) + ' of ' + fmt(p.symbols) + ' symbols embedded · ' : '') +
              (p.last_indexed ? 'indexed ' + esc(new Date(p.last_indexed.endsWith('Z') ? p.last_indexed : p.last_indexed + 'Z').toLocaleString()) : 'never indexed') +
              (p.error ? ' · <span style="color:var(--red)">' + esc(p.error) + '</span>' : '') + '</div>' +
            (langHtml ? '<div class="proj-langs">' + langHtml + '</div>' : '') +
          '</div>' +
        '</div>';
      }).join('');
      list.innerHTML = html || '<div class="proj-empty dim">No project matches “' + esc(state.filter) + '”.</div>';
    }
    async function loadProjects() {
      const gen = state.gen;
      try {
        const d = await withRetry(() => api('/api/list_projects', { timeout: 30000 }));
        if (gen !== state.gen) return;
        const projects = d.projects || (Array.isArray(d) ? d : []);
        state.projects = Array.isArray(projects) ? projects : Object.entries(projects).map(([k, v]) => ({ project: k, ...v }));
        renderProjects();
      } catch (e) {
        if (gen !== state.gen) return;
        $('projectList').innerHTML = '<div class="proj-empty" style="color: var(--red);">Could not load projects. ' + esc(e.message) + '</div>';
        showAlert('Could not load the project list. ' + e.message, { raw: e.raw, retry: reloadAll });
      }
    }
    $('projectList').addEventListener('click', ev => {
      const row = ev.target.closest('.proj-row'); if (!row) return;
      const name = row.dataset.name;
      if (state.open.has(name)) state.open.delete(name); else state.open.add(name);
      row.classList.toggle('open'); row.setAttribute('aria-expanded', row.classList.contains('open'));
    });
    $('projectList').addEventListener('keydown', ev => { if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.classList.contains('proj-row')) { ev.preventDefault(); ev.target.click(); } });
    $('projectFilter').addEventListener('input', ev => { state.filter = ev.target.value; renderProjects(); });
    $('projectSort').addEventListener('change', ev => { state.sort = ev.target.value; renderProjects(); });

    /* ================= search ================= */
    async function doSearch() {
      const q = $('searchInput').value.trim();
      const meta = $('searchMeta'), results = $('searchResults');
      if (!q) { meta.textContent = 'Type a symbol name or a concept.'; meta.classList.remove('hidden'); return; }
      const mode = $('searchMode').value;
      results.innerHTML = '<div class="dim">Searching…</div>';
      meta.classList.add('hidden');
      $('btnSearch').disabled = true;
      try {
        const d = await api('/api/search?mode=' + encodeURIComponent(mode) + '&q=' + encodeURIComponent(q), { timeout: 45000, strict: false });
        if (d && d.error && !d.results) throw new ApiError(classify(d.error, 200).text, { raw: d.error });
        const items = d.results || [];
        const count = d.result_count ?? items.length;
        const served = d.mode || mode;
        meta.textContent = count + ' result' + (count !== 1 ? 's' : '') + ' via ' + served + (served !== mode ? ' (fell back from ' + mode + ')' : '');
        meta.classList.remove('hidden');
        if (!items.length) { results.innerHTML = '<div class="dim" style="padding: 8px 0;">No results.' + (d.hint ? ' ' + esc(d.hint) : '') + '</div>'; return; }
        results.innerHTML = items.map(r => {
          const name = r.name || r.qualified_name || '?';
          const file = r.file || r.file_path || '';
          const line = r.start_line || r.line || '';
          const loc = file + (line ? ':' + line : '');
          const sim = r.similarity != null ? 'sim ' + Number(r.similarity).toFixed(2) : (r.score != null ? 'score ' + Number(r.score).toFixed(2) : '');
          return '<div class="search-result">' +
            '<div class="sr-header"><span class="sr-name">' + esc(name) + '</span>' +
              (r.kind ? '<span class="sr-kind">' + esc(r.kind) + '</span>' : '') +
              (r.project ? '<span class="sr-project">' + esc(r.project) + '</span>' : '') +
              (sim ? '<span class="sr-sim">' + esc(sim) + '</span>' : '') + '</div>' +
            (loc ? '<div class="sr-file" data-loc="' + esc(loc) + '" title="Click to copy">' + esc(loc) + '</div>' : '') +
            (r.signature ? '<div class="sr-sig">' + esc(r.signature) + '</div>' : '') +
          '</div>';
        }).join('');
      } catch (e) {
        meta.classList.add('hidden');
        results.innerHTML = '<div style="color: var(--red);">' + esc(e.message) + (e.raw ? ' <span class="dim">(' + esc(e.raw).slice(0, 160) + ')</span>' : '') + '</div>';
      } finally {
        $('btnSearch').disabled = false;
      }
    }
    $('btnSearch').onclick = doSearch;
    $('searchInput').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
    document.addEventListener('keydown', e => {
      if (e.key === '/' && !/input|select|textarea/i.test(document.activeElement.tagName)) { e.preventDefault(); $('searchInput').focus(); }
    });
    $('searchResults').addEventListener('click', ev => {
      const f = ev.target.closest('.sr-file'); if (!f) return;
      copyText(f.dataset.loc).then(ok => { const t = f.textContent; f.textContent = ok ? 'copied' : t; setTimeout(() => { f.textContent = t; }, 900); });
    });

    /* ================= connect ================= */
    function copyText(text) {
      if (navigator.clipboard && window.isSecureContext !== false) {
        return navigator.clipboard.writeText(text).then(() => true).catch(() => fallbackCopy(text));
      }
      return Promise.resolve(fallbackCopy(text));
    }
    function fallbackCopy(text) {
      const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      let ok = false; try { ok = document.execCommand('copy'); } catch { ok = false; }
      document.body.removeChild(ta); return ok;
    }
    function pickClient(client) {
      const info = state.connection && state.connection.clients && state.connection.clients[client];
      if (!info) return;
      let path = info.config_path || '';
      if (client === 'claude_desktop' && / \(macOS\) or /.test(path)) {
        const isWin = /Windows/i.test(navigator.userAgent);
        const parts = path.split(' or ');
        const mac = parts[0].replace(' (macOS)', ''), win = (parts[1] || '').replace(' (Windows)', '');
        path = isWin ? win + '  (macOS: ' + mac + ')' : mac + '  (Windows: ' + win + ')';
      }
      $('connectPath').innerHTML = 'Add to <code>' + esc(path) + '</code>';
      $('connectSnippet').textContent = JSON.stringify(info.snippet, null, 2);
      $('connectDetail').classList.remove('hidden');
      $('copyMsg').textContent = '';
      $('connectAfter').textContent = 'Then restart ' + info.name + '. When it connects, the header shows a last-query time.';
      document.querySelectorAll('.connect-tab').forEach(b => b.setAttribute('aria-pressed', b.dataset.client === client ? 'true' : 'false'));
    }
    async function loadConnectionInfo() {
      try {
        state.connection = await api('/api/connection_info?port=' + encodeURIComponent(location.port || 80));
        document.querySelectorAll('.connect-tab').forEach(b => { b.disabled = !state.connection.clients[b.dataset.client]; });
        $('connectError').classList.add('hidden');
        pickClient('claude_code');
      } catch (e) {
        $('connectError').textContent = 'Could not load client config snippets. ' + e.message;
        $('connectError').classList.remove('hidden');
      }
    }
    document.querySelectorAll('.connect-tab').forEach(btn => btn.addEventListener('click', () => pickClient(btn.dataset.client)));
    $('btnCopySnippet').addEventListener('click', () => {
      copyText($('connectSnippet').textContent).then(ok => {
        $('copyMsg').textContent = ok ? 'Copied.' : 'Copy failed — select the snippet and copy it by hand.';
        $('copyMsg').style.color = ok ? 'var(--green)' : 'var(--red)';
        setTimeout(() => { $('copyMsg').textContent = ''; }, 2500);
      });
    });

    /* ================= restart ================= */
    $('btnRestart').onclick = async () => {
      if (!confirm('Restart the srclight server? Connected AI tools will reconnect.')) return;
      const btn = $('btnRestart'), msg = $('restartMsg');
      btn.disabled = true; msg.style.color = ''; msg.textContent = 'Requesting restart…';
      try {
        const d = await api('/api/restart_server', { method: 'POST', timeout: 8000 });
        if (d && d.ok === false) { msg.style.color = 'var(--amber)'; msg.textContent = d.message || 'Restart is disabled on this server.'; return; }
        msg.textContent = (d && d.message) || 'Restarting…';
        // Burst-poll until the process answers again, then rebuild the page.
        const started = state.health && state.health.uptime_seconds;
        let back = false;
        for (let i = 0; i < 30; i++) {
          await new Promise(r => setTimeout(r, 1000));
          try { const h = await api('/healthz', { timeout: 2000 }); if (started == null || h.uptime_seconds < started) { back = true; break; } } catch {}
          msg.textContent = 'Waiting for the server… ' + (i + 1) + 's';
        }
        if (back) { msg.style.color = 'var(--green)'; msg.textContent = 'Back up.'; hideAlert(); await reloadAll(); }
        else { msg.style.color = 'var(--red)'; msg.textContent = 'srclight exited but nothing restarted it within 30s. ' + rescueLine(); }
      } catch (e) {
        msg.style.color = 'var(--red)'; msg.textContent = e.message + (e.raw ? ' (' + e.raw + ')' : '');
      } finally { btn.disabled = false; }
    };

    /* ================= init + polling ================= */
    async function reloadAll() {
      hideAlert();
      await Promise.all([loadHealth(), loadStats(), loadProjects()]);
      renderProjects();
    }
    loadWorkspaces();
    loadConnectionInfo();
    reloadAll();
    setInterval(loadHealth, 10000);
    setInterval(() => { if (state.down) loadHealth(); }, 3000);
    setInterval(() => { if (!document.hidden) { loadStats(); loadProjects(); } }, 30000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) loadHealth(); });
  </script>
</body>
</html>
"""


async def _run_sync(fn, *args, **kwargs):
    """Run a sync tool function on a worker thread, flagged as dashboard traffic.

    The context var is copied into the thread by asyncio.to_thread, so
    _record_query skips it: the page's polls never count as agent queries.
    """
    from . import server as server_mod
    token = server_mod._dashboard_request.set(True)
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    finally:
        server_mod._dashboard_request.reset(token)


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _host_of(value: str) -> str:
    """Hostname part of a Host/Origin value: strips scheme and port, keeps [::1]."""
    v = value.strip().lower()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0]
    if v.startswith("["):
        return v.split("]", 1)[0] + "]"
    return v.rsplit(":", 1)[0] if ":" in v else v


def _local_only(endpoint):
    """Draw the same line for the dashboard that the MCP SDK draws for /mcp.

    SAM (pack review 2026-09-01): mcp>=2 rejects foreign Host headers on /sse
    and /mcp (DNS-rebinding guard) but routes appended by add_web_routes sat
    outside it, so a rebinding page could read /api/list_projects and POST to
    /api/restart_server. A foreign Host gets 421; a POST with a foreign Origin
    (form / no-cors fetch) gets 403. No Origin at all (curl, same-origin GET)
    is allowed.
    """
    async def guarded(request: Request) -> Response:
        host = request.headers.get("host", "")
        if _host_of(host) not in _LOCAL_HOSTS:
            return JSONResponse({"error": "Invalid Host header"}, status_code=421)
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if origin and _host_of(origin) not in _LOCAL_HOSTS:
                return JSONResponse({"error": "Cross-origin request refused"}, status_code=403)
        return await endpoint(request)
    guarded.__name__ = getattr(endpoint, "__name__", "guarded")
    return guarded


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
        # configure_workspace closes the current WorkspaceDB under its lock;
        # off the event loop so an in-flight walk cannot freeze MCP sessions (K9).
        await _run_sync(server_mod.configure_workspace, name)
        return JSONResponse({"ok": True, "workspace": name})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _api_version(_request: Request) -> Response:
    from . import __version__
    return JSONResponse({"version": __version__})


async def _api_connection_info(request: Request) -> Response:
    """Return MCP config snippets for each AI client."""
    try:
        port = int(request.query_params.get("port", 8742))
    except (TypeError, ValueError):
        port = 8742
    base = f"http://127.0.0.1:{port}"
    mcp_url = f"{base}/mcp"
    sse_url = f"{base}/sse"

    clients = {
        "claude_desktop": {
            "name": "Claude Desktop",
            "config_path": "~/Library/Application Support/Claude/claude_desktop_config.json (macOS) or %APPDATA%/Claude/claude_desktop_config.json (Windows)",
            "snippet": {
                "mcpServers": {
                    "srclight": {
                        "url": mcp_url,
                    }
                }
            },
        },
        "claude_code": {
            "name": "Claude Code",
            "config_path": "~/.claude/settings.json",
            "snippet": {
                "mcpServers": {
                    "srclight": {
                        "type": "sse",
                        "url": sse_url,
                    }
                }
            },
        },
        "cursor": {
            "name": "Cursor",
            "config_path": ".cursor/mcp.json",
            "snippet": {
                "mcpServers": {
                    "srclight": {
                        "url": sse_url,
                    }
                }
            },
        },
        "vscode": {
            "name": "VS Code",
            "config_path": ".vscode/settings.json",
            "snippet": {
                "mcp": {
                    "servers": {
                        "srclight": {
                            "type": "sse",
                            "url": sse_url,
                        }
                    }
                }
            },
        },
        "windsurf": {
            "name": "Windsurf",
            "config_path": "~/.codeium/windsurf/mcp_config.json",
            "snippet": {
                "mcpServers": {
                    "srclight": {
                        "serverUrl": sse_url,
                    }
                }
            },
        },
    }

    return JSONResponse({
        "clients": clients,
        "server_url": mcp_url,
        "sse_url": sse_url,
    })


async def _api_stats(_request: Request) -> Response:
    """Return query activity stats for the Flutter app / dashboard."""
    try:
        from . import server as server_mod
        last_time = getattr(server_mod, "_last_query_time", None)
        last_client = getattr(server_mod, "_last_query_client", None)
        query_count = getattr(server_mod, "_query_count", 0)
        result: dict = {"query_count": query_count}
        if last_time is not None:
            result["last_query_time"] = last_time
            result["last_query_ago_seconds"] = round(time.time() - last_time, 1)
        if last_client is not None:
            result["last_query_client"] = last_client
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


# Bulb + code brackets, amber on transparent -- the srclight mark.
_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<circle cx="16" cy="13" r="8" fill="#f59e0b"/>
<rect x="12" y="21" width="8" height="3" rx="1" fill="#fbbf24"/>
<rect x="13" y="25" width="6" height="2" rx="1" fill="#9ca3af"/>
<path d="M7 8l-4 5 4 5M25 8l4 5-4 5" fill="none" stroke="#e4e4e7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


async def _favicon(_request: Request) -> Response:
    return Response(
        _FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _healthz_payload() -> dict:
    """Liveness + the numbers an operator (or agent) needs to trust the index.

    Runs on a worker thread: it walks the workspace for totals and asks the
    embedding provider whether it is reachable. Every section is present
    even when degraded, so a missing key can never be mistaken for "fine".
    """
    from ironmcp.health import health_payload

    from . import __version__
    from . import server as server_mod

    payload = health_payload("srclight", __version__)
    started = server_mod._server_start_time
    payload["uptime_seconds"] = round(time.time() - started, 1) if started else None
    payload["mcp"] = "/mcp"
    payload["workspace"] = server_mod._workspace_name
    payload.update({
        "projects": None, "files": None, "symbols": None, "edges": None,
        "embedded": None, "last_indexed": None, "projects_errored": 0, "errors": {},
    })
    last_q = getattr(server_mod, "_last_query_time", None)
    payload["queries"] = {
        "count": getattr(server_mod, "_query_count", 0),
        "last_ago_seconds": round(time.time() - last_q, 1) if last_q else None,
        "recent": server_mod.recent_queries(10),
    }
    payload["warming"] = server_mod._warming

    try:
        cmap = json.loads(server_mod.codebase_map())
        if "totals" in cmap:  # workspace mode
            payload["projects"] = cmap.get("projects_attached")
            payload.update({k: cmap["totals"].get(k) for k in ("files", "symbols", "edges", "embedded")})
            payload["last_indexed"] = cmap.get("last_indexed")
            payload["projects_errored"] = cmap.get("projects_errored", 0)
            payload["errors"] = cmap.get("errors", {})
        elif "index" in cmap:  # single-repo mode
            payload["projects"] = 1
            payload.update({k: cmap["index"].get(k) for k in ("files", "symbols", "edges")})
    except Exception as e:  # noqa: BLE001 -- health must answer, not raise
        payload["status"] = "error"
        payload["index_error"] = str(e)

    try:
        payload["embeddings"] = json.loads(server_mod.embedding_health())
    except Exception as e:  # noqa: BLE001
        payload["embeddings"] = {"status": "error", "error": str(e)}
    if not payload["embeddings"].get("status"):
        payload["embeddings"]["status"] = "unknown"

    # `status` stays "ok" while the process can answer (liveness). Anything a
    # monitor or the header should alert on is spelled out in `degraded`, so
    # human and machine disagree about nothing (STUBBY).
    from .workspace import warning_ring
    warnings = warning_ring.since(3600)
    payload["warnings_last_hour"] = len(warnings)
    degraded: list[str] = []
    emb = payload["embeddings"]
    if emb.get("status") != "ok":
        degraded.append(f"embeddings {emb.get('status')}" + (f": {emb['error']}" if emb.get("error") else ""))
    elif emb.get("reachable") is False:
        degraded.append("embedding provider unreachable")
    elif emb.get("resident") is False:
        degraded.append("embedding model not loaded")
    if payload.get("embedded") == 0 and (payload.get("symbols") or 0) > 0:
        degraded.append("no embeddings: keyword search only")
    if payload["projects_errored"]:
        degraded.append(f"{payload['projects_errored']} project(s) unreadable")
    if warnings:
        degraded.append(f"{len(warnings)} workspace warning(s) in the last hour")
    payload["degraded"] = degraded
    return payload


async def _api_recent_queries(request: Request) -> Response:
    """The agent ledger: what tools were called, with what, newest first."""
    from . import server as server_mod
    try:
        limit = int(request.query_params.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    return JSONResponse({"items": server_mod.recent_queries(limit)})


async def _redirect_root(_request: Request) -> Response:
    from starlette.responses import RedirectResponse
    return RedirectResponse("/", status_code=302)


async def _healthz(_request: Request) -> Response:
    try:
        return JSONResponse(await _run_sync(_healthz_payload))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "error", "name": "srclight", "error": str(e)}, status_code=500)


def add_web_routes(app: "Starlette") -> None:
    """Add dashboard and REST API routes to a Starlette app (e.g. from make_sse_and_streamable_http_app)."""
    from starlette.routing import Route
    routes = [
        Route("/", _dashboard, methods=["GET"]),
        Route("/healthz", _healthz, methods=["GET"]),
        Route("/dashboard", _redirect_root, methods=["GET"]),
        Route("/web", _redirect_root, methods=["GET"]),
        Route("/favicon.ico", _favicon, methods=["GET"]),
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
        Route("/api/connection_info", _api_connection_info, methods=["GET"]),
        Route("/api/stats", _api_stats, methods=["GET"]),
        Route("/api/recent_queries", _api_recent_queries, methods=["GET"]),
    ]
    for r in routes:
        r.endpoint = _local_only(r.endpoint)
        r.app = _local_only_app(r)
        app.router.routes.append(r)

    async def _warm_stats() -> None:
        """Prime the per-project stats cache off the request path so the first
        /healthz after a restart answers in milliseconds, not 15 s (TOTO)."""
        import threading
        from . import server as server_mod

        def run() -> None:
            try:
                if not server_mod._is_workspace_mode():
                    return
                server_mod._warming = "index stats"
                wdb = server_mod._get_workspace_db()
                wdb.codebase_map()
                # First hybrid search after a restart otherwise pays for every
                # sidecar load (a minute on a 39-project workspace). Load them
                # now, one project at a time so the lock is never held long.
                names = [e.name for e in wdb._all_indexable]
                for i, name in enumerate(names, 1):
                    server_mod._warming = f"vector caches ({i}/{len(names)})"
                    try:
                        wdb._get_project_cache(name)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("vector cache warm-up failed for %s: %s", name, e)
            except Exception as e:  # noqa: BLE001
                logger.warning("warm-up failed: %s", e)
            finally:
                server_mod._warming = None

        threading.Thread(target=run, name="srclight-stats-warm", daemon=True).start()

    app.router.on_startup.append(_warm_stats)


def _local_only_app(route):
    """Starlette caches `route.app` at construction; rebuild it from the guarded endpoint."""
    from starlette.routing import request_response
    return request_response(route.endpoint)
