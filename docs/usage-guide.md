# Srclight Usage Guide

## Deployment Model

Srclight runs as a single MCP server process. It indexes repos on local filesystems and serves them to any MCP client (Claude Code, Cursor, etc.).

```
MCP Client ──stdio/sse──→ srclight serve --workspace myworkspace
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
           project-a       project-b       project-c
           .srclight/      .srclight/      .srclight/
             index.db        index.db        index.db
             embeddings.npy  embeddings.npy  embeddings.npy
```

Each repo has its own `.srclight/` directory with:
- `index.db` — SQLite FTS5 index (write path, per-symbol CRUD)
- `embeddings.npy` — float32 matrix snapshot for GPU/CPU search
- `embeddings_norms.npy` — pre-computed row norms
- `embeddings_meta.json` — symbol mapping + cache version

## Setup

### 1. Install Srclight

```bash
# From PyPI
pip install srclight

# From source
git clone https://github.com/srclight/srclight.git
cd srclight
pip install -e .
```

### 2. Add as MCP Server

Srclight supports two transport modes:

- **stdio** — one server process per client session (simple, no setup)
- **SSE** — one persistent server, many clients (recommended for workspaces)

#### Option A: Stdio (simplest)

Each Claude Code session spawns its own srclight process:

```bash
# Add for current project
claude mcp add srclight -- srclight serve --workspace myworkspace

# Add globally (available in all projects)
claude mcp add --scope user srclight -- srclight serve --workspace myworkspace
```

#### Option B: SSE with systemd (recommended)

Run srclight as a persistent background service. This is faster (no cold start per session), supports multiple concurrent clients, and survives restarts.

**Create the service file** (`~/.config/systemd/user/srclight.service`):
```ini
[Unit]
Description=Srclight MCP Server (workspace: myworkspace)
After=network.target

[Service]
Type=simple
ExecStart=/path/to/srclight-venv/bin/srclight serve --workspace myworkspace
Restart=on-failure
RestartSec=3
Environment=PATH=/path/to/srclight-venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
```

**Enable and start:**
```bash
systemctl --user daemon-reload
systemctl --user enable srclight
systemctl --user start srclight

# Verify it's running
systemctl --user status srclight
curl -s http://127.0.0.1:8742/sse  # should stream SSE events
```

**Connect Claude Code to the SSE server:**
```bash
claude mcp add --transport sse srclight http://127.0.0.1:8742/sse
```

**WSL + Windows Claude Code:** If Claude Code runs on Windows but srclight runs in WSL, the same `localhost:8742` URL works — WSL2 forwards localhost ports to Windows automatically:
```bash
# Run this in Windows Claude Code (cmd/PowerShell terminal)
claude mcp add --transport sse srclight http://127.0.0.1:8742/sse
```

#### Option C: Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "srclight": {
      "command": "srclight",
      "args": ["serve", "--workspace", "myworkspace"]
    }
  }
}
```

### 3. Add to OpenClaw

[OpenClaw](https://openclaw.ai) connects to srclight via its built-in [mcporter](https://mcporter.dev) MCP tool server.

**Prerequisite:** Srclight must be running as an SSE server (Option B above).

```bash
# 1. Add srclight to mcporter config
mcporter config add srclight http://127.0.0.1:8742/sse \
  --transport sse --scope home \
  --description "Srclight deep code indexing"

# 2. Verify the connection
mcporter call srclight.list_projects

# 3. Restart the OpenClaw gateway to pick up the new server
systemctl --user restart openclaw-gateway
```

The OpenClaw agent uses srclight tools via the `mcporter` skill and `exec`:
```
mcporter call srclight.list_projects
mcporter call srclight.search_symbols query="MyClass"
mcporter call srclight.get_callers symbol_name="lookup" project="my-repo"
mcporter call srclight.hybrid_search query="authentication logic"
```

All 25 srclight tools are available as `srclight.<tool_name>` through mcporter.

### 4. Verify

Start a new session and ask:
```
What projects are in the srclight workspace?
```
The agent should call `list_projects()` and show your repos.

## Day-to-Day Usage

### Searching Across Projects

Once the MCP server is active, just ask naturally:

| Question | What happens |
|----------|-------------|
| "Compare dictionary lookup in project-a vs project-b" | `hybrid_search("dictionary lookup", project="project-a")` + same for project-b |
| "Show me the TTS architecture" | `semantic_search("text to speech provider")` across all projects |
| "Map the project-a codebase" | `codebase_map(project="project-a")` |
| "Who calls `lookup` in project-c?" | `get_callers("lookup", project="project-c")` |
| "What changed recently across all repos?" | `recent_changes()` |

The `project` parameter filters to one repo. Omit it to search all.

### What Happens on Git Commit

1. You commit in any repo with hooks installed
2. The `post-commit` hook fires (background, non-blocking)
3. `srclight index .` runs with `flock` (prevents concurrent re-indexes)
4. Changed files are re-parsed (tree-sitter), FTS5 indexes updated
5. Output logged to `.srclight/reindex.log`

**Note**: The hook does NOT re-embed. FTS5 search (`search_symbols`, keyword part of `hybrid_search`) is always fresh. Semantic search for new/changed symbols requires a manual embed pass (see below).

### What Happens on Branch Switch

1. `git checkout other-branch` triggers `post-checkout` hook
2. Only fires on branch checkouts (not file checkouts) and only when HEAD changes
3. Same background `srclight index .` as post-commit
4. FTS5 indexes updated for all files that differ between branches

### Re-Embedding After Significant Changes

After major refactors, new branches with many new files, or initial setup:

```bash
# Re-embed a single project
cd /path/to/repo
srclight index --embed qwen3-embedding

# Re-embed all projects in workspace
srclight workspace index -w myworkspace --embed qwen3-embedding

# Re-embed just one project via workspace command
srclight workspace index -w myworkspace -p project-name --embed qwen3-embedding
```

Embedding is incremental — only symbols whose `body_hash` changed get re-embedded. The `.npy` sidecar is rebuilt automatically after embedding.

## Adding a New Repo

```bash
# 1. Add to workspace
srclight workspace add /path/to/new-repo -w myworkspace
srclight workspace add /path/to/new-repo -w myworkspace -n custom-name  # optional custom name

# 2. Index with embeddings
srclight workspace index -w myworkspace -p new-repo --embed qwen3-embedding

# 3. Install git hooks
cd /path/to/new-repo
srclight hook install
# Or install across entire workspace (safe — skips already-installed):
srclight hook install --workspace myworkspace

# 4. Verify
srclight workspace status -w myworkspace
```

The new repo is immediately searchable. The MCP server picks up new projects on the next tool call (no restart needed — workspace config is re-read).

**Note:** Both `srclight index` and `srclight hook install` automatically add `.srclight/` to the repo's `.gitignore`. The index databases and embedding files can be large (hundreds of MB) and should never be committed.

## Removing a Repo

```bash
# Remove from workspace config
srclight workspace remove project-name -w myworkspace

# Optionally remove hooks
cd /path/to/repo
srclight hook uninstall

# The .srclight/ directory in the repo is left on disk (safe to delete manually)
```

## Checking Status

```bash
# Workspace overview (all projects)
srclight workspace status -w myworkspace

# List all workspaces
srclight workspace list

# Hook status for current repo
srclight hook status

# Hook status for all repos in workspace
srclight hook status --workspace myworkspace
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Semantic search (workspace, 27K vectors) | ~105ms warm | GPU-resident .npy cache |
| Semantic search (single repo, 15K vectors) | ~12ms warm | |
| Cold start (first query after server start) | ~300ms | Loads .npy to GPU VRAM |
| FTS5 search | <10ms | SQLite, always fast |
| Incremental re-index (post-commit) | 1-5s | Background, non-blocking |
| Full re-embed (27K symbols) | ~15 min | Ollama qwen3-embedding, one-time |

## Troubleshooting

### MCP server not responding
```bash
# Check if srclight binary works
srclight workspace status -w myworkspace

# Restart by removing and re-adding
claude mcp remove srclight
claude mcp add --scope user srclight -- srclight serve --workspace myworkspace
```

### Semantic search returning stale results
```bash
# Check embedding status via CLI
cd /path/to/repo
srclight index --embed qwen3-embedding

# Or ask the agent: "What's the embedding status?"
# → calls embedding_status() tool
```

### Hook not firing
```bash
# Check hook status
cd /path/to/repo
srclight hook status

# Re-install if needed
srclight hook install

# Check hook log
cat .srclight/reindex.log
```

### Repo moved or renamed
If a repo changes location on disk, update the workspace:
```bash
srclight workspace remove old-name -w myworkspace
srclight workspace add /new/path/to/repo -w myworkspace
srclight workspace index -w myworkspace -p new-name --embed qwen3-embedding
```

## Claude Code Custom Agents (Subagents)

Claude Code supports [custom agents](https://docs.anthropic.com/en/docs/claude-code/agents) defined in `.claude/agents/*.md`. These agents run as subprocesses with their own tool access, controlled by the `tools:` frontmatter field.

### The Problem: MCP Tools in Subagents

Custom agents defined in `.claude/agents/` **cannot access MCP tools**. This is a [known bug](https://github.com/anthropics/claude-code/issues/13605) ([#25200](https://github.com/anthropics/claude-code/issues/25200)) as of Claude Code v2.1.52 (Feb 2026).

The tool injection code has two paths: built-in agents receive MCP tools, custom agents do not. None of these workarounds help:
- Adding MCP tool names to `tools:` frontmatter
- Adding `ToolSearch` to `tools:` frontmatter
- Adding `mcpServers:` to frontmatter
- Omitting `tools:` entirely (should inherit all — doesn't)

| Agent Type | Tools | MCP Access |
|---|---|---|
| `general-purpose` | `*` (all) | **Yes** |
| `Explore` | All except Task/Edit/Write | **Yes** |
| `Plan` | All except Task/Edit/Write | **Yes** |
| Custom agents (`.claude/agents/`) | Core tools only | **No** — bug [#13605](https://github.com/anthropics/claude-code/issues/13605) |

### Solution: Use `general-purpose` Agent Type

Until the bug is fixed, the only way to give a subagent access to srclight is to use `general-purpose` as the `subagent_type`. It has `(Tools: *)` which includes ToolSearch and all MCP tools:

```
Task(
  subagent_type="general-purpose",
  prompt="You are a UI design reviewer. Use srclight MCP tools for code analysis..."
)
```

The agent must call `ToolSearch("srclight")` before using any `mcp__srclight__*` tool. Include this instruction in the prompt.

**Tradeoff:** `general-purpose` agents also have write access (Edit, Write), which is more permissive than a read-only reviewer needs. The agent's system prompt can instruct it not to modify files.

### Example: UI Design Reviewer with Srclight

Since custom agents can't access MCP ([#13605](https://github.com/anthropics/claude-code/issues/13605)), invoke via `general-purpose` and include your review instructions in the prompt:

```
Task(
  subagent_type="general-purpose",
  prompt="You are a senior UI/UX designer reviewing a Flutter app.

  ## srclight Code Index (MCP)

  Use ToolSearch to load srclight tools before calling them. Key tools:

  | Tool | Use |
  |------|-----|
  | mcp__srclight__symbols_in_file(path, project) | Widget/class outline |
  | mcp__srclight__get_callers(symbol, project)    | Consistency checks |
  | mcp__srclight__search_symbols(query, project)  | Find exact names |

  Workflow: Use symbols_in_file to get outlines, then Read sections.
  Use get_callers to verify token usage consistency. Use Grep for pattern
  violations (raw Color literals, bare EdgeInsets) that srclight can't catch.

  DO NOT modify any files. This is a read-only review."
)

## Architecture Notes

- **One server, one workspace**: The MCP server runs in workspace mode serving all repos. Each project's `.srclight/index.db` is ATTACHed to a `:memory:` database at query time via SQLite's ATTACH mechanism.
- **ATTACH limit**: SQLite allows max 10 ATTACHed databases. >10 projects are handled by batch detach/reattach in `_iter_batches()`.
- **GPU cache**: Each project gets its own `VectorCache` loaded to GPU VRAM (cupy) or CPU RAM (numpy). Caches are loaded lazily on first semantic search and invalidated when `embedding_cache_version` in the DB changes.
- **No network**: Everything runs locally. Ollama is on `localhost:11434`. No cloud APIs unless you opt into Voyage Code 3.
