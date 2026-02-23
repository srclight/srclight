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
# From PyPI (when published)
pip install srclight

# From source
git clone https://github.com/srclight/srclight.git
cd srclight
pip install -e .
```

### 2. Add as MCP Server

**Claude Code (stdio — simplest):**
```bash
claude mcp add srclight -- srclight serve --workspace myworkspace
```

**Claude Code (global, user-scoped):**
```bash
claude mcp add --scope user srclight -- srclight serve --workspace myworkspace
```

**Claude Desktop (`claude_desktop_config.json`):**
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

**WSL note:** If your MCP client runs on Windows but srclight is installed in WSL, use `wsl` as the command:
```json
"srclight": {
  "type": "stdio",
  "command": "wsl",
  "args": [
    "/path/to/srclight-mcp/.venv/bin/srclight",
    "serve", "--workspace", "myworkspace"
  ]
}
```

### 3. Verify

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

## Architecture Notes

- **One server, one workspace**: The MCP server runs in workspace mode serving all repos. Each project's `.srclight/index.db` is ATTACHed to a `:memory:` database at query time via SQLite's ATTACH mechanism.
- **ATTACH limit**: SQLite allows max 10 ATTACHed databases. >10 projects are handled by batch detach/reattach in `_iter_batches()`.
- **GPU cache**: Each project gets its own `VectorCache` loaded to GPU VRAM (cupy) or CPU RAM (numpy). Caches are loaded lazily on first semantic search and invalidated when `embedding_cache_version` in the DB changes.
- **No network**: Everything runs locally. Ollama is on `localhost:11434`. No cloud APIs unless you opt into Voyage Code 3.
