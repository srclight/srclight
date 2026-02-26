# CLAUDE.md — Srclight Project Instructions

## What Is This

Srclight is a deep code indexing MCP server for AI agents. SQLite FTS5 + tree-sitter + embeddings + ATTACH+UNION for multi-repo workspaces.

## Commands

```sh
# Dev install (from repo root)
pip install -e .

# Run tests
python -m pytest tests/ -v

# Index a single repo
srclight index /path/to/repo

# Index with embeddings (requires Ollama on localhost:11434)
srclight index --embed qwen3-embedding /path/to/repo

# Index all repos in workspace
srclight workspace index -w myworkspace

# Index workspace with embeddings
srclight workspace index -w myworkspace --embed qwen3-embedding

# Start MCP server (SSE, default — multi-session safe)
srclight serve --workspace myworkspace

# Start MCP server (stdio, for single-session/debugging)
srclight serve --workspace myworkspace --transport stdio

# Hook management
srclight hook install --workspace myworkspace
srclight hook status --workspace myworkspace
```

## Architecture

### DB-Per-Repo + ATTACH
Each repo gets `.srclight/index.db`. Workspace mode ATTACHes all DBs to `:memory:` and UNIONs across schemas. SQLite's 10-ATTACH limit handled by batch detach/reattach in `_iter_batches()`.

### Key Modules
| Module | Purpose |
|--------|---------|
| `db.py` | SQLite schema (v4), FTS5 indexes, symbol CRUD, graph queries, vector search |
| `indexer.py` | tree-sitter parsing, symbol extraction, edge building, embedding generation |
| `embeddings.py` | Embedding providers (Ollama, Voyage), RRF hybrid search |
| `vector_math.py` | cupy/numpy vectorized cosine similarity (GPU or CPU backend) |
| `vector_cache.py` | GPU/CPU-resident embedding matrix with `.npy` sidecar files |
| `workspace.py` | WorkspaceConfig (JSON), WorkspaceDB (ATTACH+UNION), per-project vector cache |
| `server.py` | FastMCP server, all 25 MCP tools |
| `git.py` | Git blame/log/diff parsing, change intelligence |
| `build.py` | CMake/.csproj/package.json parsing, platform conditionals |
| `cli.py` | Click CLI, workspace commands, hook install/uninstall |
| `languages.py` | tree-sitter language configs, symbol queries |

### FTS5 Design (3 indexes per DB)
1. **symbol_names_fts** — unicode61 tokenizer, splits CamelCase/snake_case
2. **symbol_content_fts** — trigram tokenizer for substring matching
3. **symbol_docs_fts** — porter stemmer for natural language in docstrings

### Embeddings Design
- `symbol_embeddings` table: symbol_id, model, dimensions, embedding (BLOB), body_hash
- Embedding text = qualified_name + signature + doc_comment + content[:2000]
- Providers: **OllamaProvider** (HTTP to localhost:11434), **VoyageProvider** (API)
- **Vectorized math**: cupy (GPU) or numpy (CPU) backend auto-detected at import
- **GPU-resident vector cache**: `.npy` sidecar files loaded to GPU VRAM once, all queries served from VRAM (~3ms/query for 27K vectors on a modern GPU)
- **Cache invalidation**: `embedding_cache_version` in `schema_info` bumped on every `upsert_embedding()`; cache auto-rebuilds after reindex
- **Sidecar files**: `embeddings.npy` (matrix), `embeddings_norms.npy` (row norms), `embeddings_meta.json` (symbol mapping)
- Hybrid search via Reciprocal Rank Fusion (RRF, k=60)
- Incremental: only re-embeds symbols whose body_hash changed

### Edge Types
Currently extracted: `calls`, `inherits`. Graph queries: callers, callees, dependents (transitive), implementors, tests_for, type_hierarchy.

## MCP Tools (25 total)

### Tier 1: Instant
- `codebase_map()` — project overview, call first every session
- `search_symbols(query)` — FTS5 across names/content/docs
- `get_symbol(name)` — full source + metadata
- `get_signature(name)` — just the signature (lightweight)
- `symbols_in_file(path)` — file table of contents
- `list_projects()` — workspace project list with stats

### Tier 2: Graph
- `get_callers(symbol)` — who calls this
- `get_callees(symbol)` — what does this call
- `get_dependents(symbol, transitive)` — blast radius
- `get_implementors(interface)` — all implementations
- `get_tests_for(symbol)` — test functions covering this
- `get_type_hierarchy(symbol)` — inheritance tree

### Tier 3: Git (Change Intelligence)
- `blame_symbol(symbol)` — who/when/why changed this
- `recent_changes(n)` — commit feed, cross-project in workspace
- `git_hotspots(n, since)` — churn/bug magnets
- `whats_changed()` — uncommitted WIP
- `changes_to(symbol)` — commit history for a symbol's file

### Tier 4: Build & Config
- `get_build_targets()` — CMake/.csproj targets with deps
- `get_platform_variants(symbol)` — #ifdef platform guards
- `platform_conditionals()` — all platform-conditional blocks

### Tier 5: Semantic Search (Embeddings)
- `semantic_search(query)` — find code by meaning (natural language)
- `hybrid_search(query)` — keyword + semantic with RRF fusion (best mode)
- `embedding_status()` — embedding coverage and model info

### Meta
- `index_status()` — index freshness
- `reindex()` — trigger incremental re-index

## Embedding Models

| Model | Provider | Dims | Quality | Notes |
|-------|----------|------|---------|-------|
| `qwen3-embedding` | Ollama | varies | Best local | Default. Needs ~6GB VRAM |
| `nomic-embed-text` | Ollama | 768 | Good local | Lighter, 137M params |
| `text-embedding-3-small` | OpenAI | 1536 | Good | Requires OPENAI_API_KEY |
| `text-embedding-3-large` | OpenAI | 3072 | Better | Requires OPENAI_API_KEY |
| `embed-v4.0` | Cohere | 1024 | Good | Requires COHERE_API_KEY |
| `voyage-code-3` | Voyage API | 1024 | Best overall | Requires VOYAGE_API_KEY |

### OpenAI-compatible providers
Any service speaking the `/v1/embeddings` format works with the `openai:` prefix.
Set `OPENAI_BASE_URL` to point at the provider:
```sh
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://api.together.xyz  # Together, Fireworks, etc.
srclight index --embed openai:your-model /path/to/repo
```

## Server Robustness
- **Workspace config hot-reload**: `_get_workspace_db()` checks config file mtime — adding repos via `srclight workspace add` takes effect immediately, no server restart needed
- **VectorCache re-discovery**: `_get_project_cache()` re-checks for sidecars that appear after `srclight index --embed` — no permanent None sentinel
- **MCP instructions**: FastMCP `instructions` field includes session protocol, tool selection guide, and `project` parameter docs — agents get guidance automatically
- **Error messages**: All "project required" errors include `available_projects` list with valid project names

## Version Bumps

Version is tracked in **two files** — both MUST be updated together:
- `pyproject.toml` → `version = "X.Y.Z"`
- `src/srclight/__init__.py` → `__version__ = "X.Y.Z"`

Semantic versioning:
- **Patch** (0.8.x): bug fixes, minor improvements
- **Minor** (0.x.0): new features, new MCP tools, new language support
- **Major** (x.0.0): breaking changes to MCP tool signatures or DB schema

## Do NOT

- Do NOT remove `.srclight/` directories — they contain repo indexes + embeddings
- Do NOT modify `workspace.py` `_sanitize_schema_name` without checking reserved name list
- Do NOT assume more than 10 ATTACHed databases — always use `_iter_batches()`
- Do NOT run builds without the venv — `tree-sitter` bindings are in `.venv/`
- Do NOT add torch/transformers as hard dependencies — numpy is optional (for vector cache), cupy is optional (for GPU acceleration)

## Release Process

### Git Flow
```sh
# Feature development
git checkout develop && git checkout -b feature/xxx
# ... work ...
git checkout develop && git merge feature/xxx
git checkout master && git merge develop
git tag -a vX.Y.Z -m "vX.Y.Z description"
git push origin master develop --tags
gh release create vX.Y.Z --title "vX.Y.Z — Description" --notes "Notes"
```

### Automated Publishing (GitHub Actions)
On `release: [published]`, `.github/workflows/publish.yml` runs:
1. **PyPI** — trusted publisher via OIDC, no secrets needed
2. **MCP Registry** — `mcp-publisher` with OIDC auth, patches version from git tag

### Post-Release Checklist
1. Reinstall dev package: `pip install -e .`
2. Reindex workspace: `srclight workspace index -w <workspace> --embed qwen3-embedding`
3. Verify `index_status()` shows new `indexer_version`

### server.json
Updated automatically during CI from git tag. Description should be keyword-rich for AI discovery. Current keywords: "code indexing", "MCP tools", "FTS5", "embedding search", "call graphs", "git blame", "multi-repo", "fully local".

## Documentation Strategy

This is a public repo. Keep a clear split between public docs and private strategic notes.

### Public — `docs/` in this repo
- **Usage guides** (setup, deployment, troubleshooting) — `docs/usage-guide.md`
- **Architecture** for contributors (how the code works, how to extend)
- **API reference** (MCP tool docs, CLI reference)
- **Changelog** and release notes

### Private — Obsidian Vault (Dropbox)
Strategic, competitive, and personal notes live in Tim's Vault:

| Location | Contents |
|----------|----------|
| `Areas/Srclight/Marketing/` | AI discovery strategy, draft posts, launch playbook, competitive analysis |
| `Areas/Srclight/performance-analysis.md` | Embedding bottleneck analysis, GPU/platform support |
| `Areas/Code Indexing/` | Architecture research, embedding model analysis, FTS5 lessons |
| `Projects/srclight-personal-claude.md` | Personal dev environment: machine paths, hardware, workspace config, systemd, MCP setup |

### CLAUDE.md Backup
A copy of this file is kept in the private Vault (`Projects/srclight-CLAUDE-md-backup.md`). When making significant changes to this CLAUDE.md, update the backup copy too (see `srclight-personal-claude.md` for the exact path/command).

### Rule of thumb
- Mentions competitors, pricing, market strategy, or "why to build this" → **Vault**
- Marketing plans, registry strategy, posting schedules, AI-persuasion techniques → **Vault** (`Areas/Srclight/Marketing/`)
- Personal paths, hardware specs, workspace names → **Vault** (`srclight-personal-claude.md`)
- Helps an external user or contributor use/understand srclight → **`docs/`**
- Release process (git flow, CI) is fine in public CLAUDE.md
- Do NOT put strategic analysis, marketing plans, or personal infra details in the public repo

## Testing

113 tests: db (8), indexer (9), features (14), workspace (7), hooks (11), git (9), build (6), embeddings (27), vector_math (7), vector_cache (11), workspace batch (1).

```sh
python -m pytest tests/ -v
```
