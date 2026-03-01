<!-- mcp-name: io.github.srclight/srclight -->

# Srclight

[![PyPI](https://img.shields.io/pypi/v/srclight)](https://pypi.org/project/srclight/)
[![License](https://img.shields.io/github/license/srclight/srclight)](https://github.com/srclight/srclight/blob/master/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/srclight)](https://pypi.org/project/srclight/)

**Deep code indexing for AI agents.** SQLite FTS5 + tree-sitter + embeddings + MCP.

Srclight builds a rich, searchable index of your codebase that AI coding agents can query instantly — replacing dozens of grep/glob calls with precise, structured lookups. It is the most comprehensive code intelligence MCP server available: 29 tools covering symbol search, relationship graphs, git change intelligence, semantic search, build system awareness, and document extraction — capabilities no other single MCP server combines. Fully local and private: your code never leaves your machine.

## Why?

AI coding agents (Claude Code, Cursor, etc.) spend **40-60% of their tokens on orientation** — searching for files, reading code to understand structure, hunting for callers and callees. Srclight eliminates this waste.

| Without Srclight | With Srclight |
|---|---|
| 8-12 grep rounds to find callers | `get_callers("lookup")` — one call |
| Read 5 files to understand module | `codebase_map()` — instant overview |
| "Find code that does X" → 20 greps | `semantic_search("dictionary lookup")` — one call |
| 15-25 tool calls per bug fix | 5-8 tool calls per bug fix |

## Features

- **Minimal dependencies** — single SQLite file per repo, no Docker/Redis/vector DB
- **Fully offline** — no API calls, works air-gapped (Ollama local embeddings)
- **Incremental** — only re-indexes changed files (content hash detection)
- **11 languages** — Python, C, C++, C#, JavaScript, TypeScript, PHP, Dart, Swift, Kotlin, Java, Go
- **10 document formats** — PDF, DOCX, XLSX, HTML, CSV/TSV, email (.eml), images (PNG/JPG/SVG/etc.), plain text, RST, Markdown
- **Optional OCR** — PaddleOCR for scanned/image-only PDF pages; pytesseract for images
- **4 search modes** — symbol names, source code (trigram), documentation (stemmed), semantic (embeddings)
- **Hybrid search** — RRF fusion of keyword + semantic results for best accuracy
- **Multi-repo workspaces** — search across all your repos simultaneously via SQLite ATTACH+UNION
- **MCP server** — works with Claude Code, Cursor, and any MCP client
- **CLI** — index, search, and inspect from the terminal
- **Auto-reindex** — git post-commit/post-checkout hooks keep indexes fresh

## Requirements

- **Python 3.11+**
- **Git** (for change intelligence and auto-reindex hooks)
- **Ollama** (optional, for semantic search / embeddings) — [ollama.com](https://ollama.com)
- **NVIDIA GPU + cupy** (optional, for GPU-accelerated vector search)
- **Poppler** (optional, for PaddleOCR scanned-PDF support) — `apt install poppler-utils` / `brew install poppler`

## Quick Start

```bash
# Install from PyPI
pip install srclight

# Install from source
git clone https://github.com/srclight/srclight.git
cd srclight
pip install -e .

# Optional: document format support (PDF, DOCX, XLSX, HTML, images)
pip install 'srclight[docs,pdf]'

# Optional: OCR for scanned PDFs (also needs poppler-utils on your system)
pip install 'srclight[pdf,paddleocr]'

# Optional: OCR for images (needs tesseract on your system)
pip install 'srclight[docs,ocr]'

# Optional: GPU-accelerated vector search (requires CUDA 12.x)
pip install 'srclight[gpu]'

# Everything (docs + pdf + ocr + paddleocr + gpu)
pip install 'srclight[all]'

# Index your project
cd /path/to/your/project
srclight index

# Index with embeddings (requires Ollama running)
srclight index --embed qwen3-embedding

# Search
srclight search "lookup"
srclight search --kind function "parse"
srclight symbols src/main.py

# Start MCP server (for Claude Code / Cursor)
srclight serve
```

> **Note:** `srclight index` automatically adds `.srclight/` to your `.gitignore`. Index databases and embedding files can be large and should never be committed.

## Semantic Search (Embeddings)

Srclight supports embedding-based semantic search for natural language queries like "find code that handles authentication" or "where is the database connection pool".

### Setup

```bash
# Install Ollama (https://ollama.com)
# Pull an embedding model
ollama pull qwen3-embedding       # Best quality (8B params, needs ~6GB VRAM)
ollama pull nomic-embed-text      # Lighter alternative (137M params)

# Index with embeddings
srclight index --embed qwen3-embedding

# Or index workspace with embeddings
srclight workspace index -w myworkspace --embed qwen3-embedding
```

### How It Works

1. Each symbol's name + signature + docstring + content is embedded as a float vector
2. Vectors are stored as BLOBs in `symbol_embeddings` table (SQLite)
3. After indexing, a `.npy` sidecar snapshot is built and loaded to **GPU VRAM** (cupy) or CPU RAM (numpy) for fast search
4. `semantic_search(query)` embeds the query and runs cosine similarity against the GPU-resident matrix (~3ms for 27K vectors on a modern GPU)
5. `hybrid_search(query)` combines FTS5 keyword results + embedding results via Reciprocal Rank Fusion (RRF)

### Embedding Providers

| Provider | Model | Quality | Local? | Notes |
|----------|-------|---------|--------|-------|
| **Ollama** (default) | `qwen3-embedding` | Best local | Yes | Needs ~6GB VRAM |
| Ollama | `nomic-embed-text` | Good | Yes | Lighter, works on 8GB VRAM |
| **Voyage AI** (API) | `voyage-code-3` | Best overall | No | Requires `VOYAGE_API_KEY` |

```bash
# Use Voyage Code 3 (API, highest quality)
VOYAGE_API_KEY=your-key srclight index --embed voyage-code-3
```

### Storage

Embeddings are stored in `symbol_embeddings` table in `.srclight/index.db`. After indexing, a `.npy` sidecar snapshot is built for fast GPU loading:

| File | Purpose |
|------|---------|
| `index.db` | Write path — per-symbol CRUD during indexing |
| `embeddings.npy` | Read path — contiguous float32 matrix for GPU/CPU search |
| `embeddings_norms.npy` | Pre-computed row norms (avoids recomputation per query) |
| `embeddings_meta.json` | Symbol ID mapping, model info, version for cache invalidation |

For ~27K symbols at 4096 dims (qwen3-embedding), that's ~428 MB on disk, ~450 MB in VRAM. Incremental: only re-embeds symbols whose content changed; sidecar rebuilt after each indexing run.

## Multi-Repo Workspaces

Search across multiple repos simultaneously. Each repo keeps its own `.srclight/index.db`; at query time, srclight ATTACHes them all and UNIONs across schemas.

```bash
# Create a workspace
srclight workspace init myworkspace

# Add repos
srclight workspace add /path/to/repo1 -w myworkspace
srclight workspace add /path/to/repo2 -w myworkspace -n custom-name

# Index all repos (with optional embeddings)
srclight workspace index -w myworkspace
srclight workspace index -w myworkspace --embed qwen3-embedding

# Search across all repos
srclight workspace search "Dictionary" -w myworkspace
srclight workspace search "Dictionary" -w myworkspace --project repo1

# Status
srclight workspace status -w myworkspace
srclight workspace list

# Start MCP server in workspace mode
srclight serve --workspace myworkspace
```

**Git submodules** are not indexed automatically — `git ls-files` does not recurse into them. To index a submodule, clone it separately and add it as its own workspace project. See [docs/usage-guide.md](docs/usage-guide.md#git-submodules) for details.

## MCP Integration

Srclight supports two transport modes: **stdio** (one server per session) and **SSE** (persistent server, multiple sessions). SSE is recommended for workspaces.

### Claude Code

**Stdio (simplest — one server per session):**
```bash
# Single repo
claude mcp add srclight -- srclight serve

# Workspace mode
claude mcp add srclight -- srclight serve --workspace myworkspace

# Make it available in all projects (user scope)
claude mcp add --scope user srclight -- srclight serve --workspace myworkspace
```

**SSE (persistent server — recommended for workspaces):**

Run srclight as a long-lived server, then point Claude Code at it:

```bash
# Start the server (default: http://127.0.0.1:8742/sse)
srclight serve --workspace myworkspace &

# Or install as a systemd user service (Linux/WSL)
# See docs/usage-guide.md for the service file

# Connect Claude Code to the running server
claude mcp add --transport sse srclight http://127.0.0.1:8742/sse
```

SSE mode supports multiple concurrent sessions and survives Claude Code restarts.

### Cursor

**SSE (recommended):** Run srclight once, then connect Cursor to it. Best for responsiveness and no cold-start per session.

Start the server: `srclight serve --workspace myworkspace` (default SSE on port 8742).

- **UI:** Settings → Tools & MCP → Add new MCP server → Type: `streamableHttp`, URL: `http://127.0.0.1:8742/sse`.
- **JSON** (project `.cursor/mcp.json` or global `~/.cursor/mcp.json`):

```json
"srclight": {
  "url": "http://127.0.0.1:8742/sse"
}
```

**Stdio (alternative):** One server process per Cursor session.

- **UI:** Type: `command`, Command: `srclight`, Args: `serve --workspace myworkspace` (or `serve` for single-repo).
- **JSON:**

```json
"srclight": {
  "command": "srclight",
  "args": ["serve", "--workspace", "myworkspace"]
}
```

For single-repo: `"args": ["serve"]`. Restart Cursor completely after adding the server.

**Verify:** In Cursor chat, ask "What projects are in the srclight workspace?" or "List srclight tools" — the agent should call `list_projects()` or show srclight tools.

### OpenClaw

OpenClaw connects to srclight via [mcporter](https://mcporter.dev), its built-in MCP tool server CLI.

```bash
# 1. Add srclight to mcporter's home config
mcporter config add srclight http://127.0.0.1:8742/sse \
  --transport sse --scope home \
  --description "Srclight deep code indexing"

# 2. Verify the connection
mcporter call srclight.list_projects

# 3. Restart the OpenClaw gateway to pick up the new server
systemctl --user restart openclaw-gateway  # if using systemd
# or: openclaw daemon restart
```

The OpenClaw agent can then use srclight tools via the `mcporter` skill:
```
mcporter call srclight.search_symbols query="my_function"
mcporter call srclight.get_callers symbol_name="MyClass" project="my-repo"
mcporter call srclight.hybrid_search query="authentication logic"
```

> **Prerequisite:** Srclight must be running as an SSE server (see above). OpenClaw's mcporter connects over HTTP — stdio mode is not supported.

### Claude Desktop (`claude_desktop_config.json`)

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

### Any MCP Client (SSE)

Any MCP-compatible client can connect to the SSE endpoint:
```
http://127.0.0.1:8742/sse
```

## MCP Tools (29)

Srclight exposes 29 MCP tools organized in six tiers. The MCP server includes built-in instructions that guide AI agents on which tool to use and when — agents receive a session protocol, tool selection guide, and `project` parameter documentation automatically on connection.

### Tier 1: Instant Orientation
| Tool | What it does |
|------|-------------|
| `codebase_map()` | Full project overview — call first every session |
| `search_symbols(query)` | Search across symbol names, code, and docs |
| `get_symbol(name)` | Full source code + metadata for a symbol |
| `get_signature(name)` | Just the signature (lightweight) |
| `symbols_in_file(path)` | Table of contents for a file |
| `list_projects()` | All projects in workspace with stats |

### Tier 2: Relationship Graph
| Tool | What it does |
|------|-------------|
| `get_callers(name)` | Who calls this symbol? |
| `get_callees(name)` | What does this symbol call? |
| `get_dependents(name, transitive)` | Blast radius — what breaks if I change this? |
| `get_implementors(interface)` | All classes implementing an interface |
| `get_tests_for(name)` | Test functions covering a symbol |
| `get_type_hierarchy(name)` | Inheritance tree (base classes + subclasses) |

### Tier 3: Git Change Intelligence
| Tool | What it does |
|------|-------------|
| `blame_symbol(name)` | Who changed this, when, and why |
| `recent_changes(n)` | Commit feed (cross-project in workspace) |
| `git_hotspots(n, since)` | Most frequently changed files (bug magnets) |
| `whats_changed()` | Uncommitted work in progress |
| `changes_to(name)` | Commit history for a symbol's file |

### Tier 4: Build & Config
| Tool | What it does |
|------|-------------|
| `get_build_targets()` | CMake/.csproj/npm targets with dependencies |
| `get_platform_variants(name)` | #ifdef platform guards around a symbol |
| `platform_conditionals()` | All platform-conditional code blocks |

### Tier 5: Semantic Search (Embeddings)
| Tool | What it does |
|------|-------------|
| `semantic_search(query)` | Find code by meaning (natural language) |
| `hybrid_search(query)` | Best of both: keyword + semantic with RRF fusion |
| `embedding_status()` | Embedding coverage and model info |

### Tier 6: Meta & Server
| Tool | What it does |
|------|-------------|
| `index_status()` | Index freshness and stats |
| `reindex()` | Trigger incremental re-index |
| `embedding_health()` | Check if the embedding provider (Ollama, etc.) is reachable |
| `setup_guide()` | Structured setup instructions for agents and users |
| `server_stats()` | Server uptime and process info |
| `restart_server()` | Request server restart (SSE only) |

In workspace mode, `search_symbols`, `get_symbol`, `codebase_map`, and `hybrid_search` accept an optional `project` filter. Graph/git/build tools require `project` in workspace mode.

## Deployment Guide

See **[docs/usage-guide.md](docs/usage-guide.md)** for the full deployment and usage guide, including:
- Setting up srclight as a global MCP server for Claude Code
- Adding/removing repos from workspaces
- What happens on commits and branch switches
- Re-embedding workflows
- Troubleshooting

## Auto-Reindex (Git Hook)

Keep indexes fresh automatically:

```bash
# Install post-commit + post-checkout hooks in current repo
srclight hook install

# Install across all repos in a workspace
srclight hook install --workspace myworkspace

# Remove hooks
srclight hook uninstall
```

The hooks run `srclight index` in the background after each commit and branch switch.

## How It Works

1. **tree-sitter** parses every source file into an AST
2. **Document extractors** handle non-code files (PDF, DOCX, XLSX, HTML, CSV, images, email, text) — extracting headings, tables, pages, and metadata as searchable symbols. Scanned PDF pages are optionally OCR'd via PaddleOCR.
3. Symbols (functions, classes, methods, structs, etc.) are extracted with full metadata
4. Three **SQLite FTS5** indexes are built with different tokenization strategies:
   - **Names**: code-aware tokenization (splits `camelCase`, handles `::`, `->`)
   - **Content**: trigram index for substring matching
   - **Docs**: Porter stemming for natural language in docstrings
5. Optional: **embedding vectors** are generated via Ollama or Voyage API and stored as BLOBs
6. A `.npy` **sidecar snapshot** is built and loaded to **GPU VRAM** (cupy) or CPU RAM (numpy) for fast search
7. The **MCP server** exposes structured query tools that AI agents call instead of grep
8. **Hybrid search** merges keyword (FTS5) and semantic (embedding) results via RRF

### Architecture (Workspace Mode)

```
repo1/.srclight/index.db  ──┐
repo2/.srclight/index.db  ──┼── ATTACH ──→ :memory: ──→ UNION ALL queries
repo3/.srclight/index.db  ──┘
```

Each repo is indexed independently. At query time, SQLite's ATTACH mechanism joins them into a single searchable namespace. Handles >10 repos via automatic batching (SQLite's ATTACH limit).

## How Srclight Compares

A survey of 50+ MCP code intelligence servers across all major registries (Official MCP Registry, Smithery, Glama, mcp.so, awesome-mcp-servers) found that no other single server combines srclight's full capabilities:

| Capability | srclight | grep/glob (default) | CodeMCP (SCIP) | Claude Context (Zilliz) |
|-----------|----------|---------------------|----------------|------------------------|
| Symbol search (FTS5) | 3 indexes (name, content, docs) | None | SCIP-based | BM25 |
| Semantic search (embeddings) | GPU-accelerated, ~3ms | None | None | OpenAI API + Milvus |
| Hybrid search (keyword + semantic) | RRF fusion | None | None | BM25 + vector |
| Relationship graph (callers, callees) | tree-sitter edges | None | SCIP edges | None |
| Git change intelligence | blame, hotspots, WIP | None | None | None |
| Build system awareness | CMake, .csproj, #ifdef | None | None | None |
| Multi-repo workspace | ATTACH+UNION | None | None | None |
| Infrastructure required | `pip install`, SQLite | None | SCIP indexer | Docker, Milvus, OpenAI API |
| Fully local / private | Yes, zero API calls | Yes | Yes | No (needs OpenAI) |
| Languages | 11 | Any (regex) | 5 (SCIP) | Any (chunking) |
| MCP tools | 29 | 2 (grep, glob) | 80+ | ~10 |

Unlike grep-based tools, srclight builds a persistent index with structured lookups. Unlike cloud-based solutions, everything runs locally — your code never leaves your machine. Unlike IDE plugins, srclight works with any MCP client.

## Roadmap

### Done
- [x] Symbol intelligence + 3x FTS5 search
- [x] Relationship graph: callers, callees, hierarchy
- [x] Blast radius, test discovery, implementors
- [x] Git change intelligence: blame, hotspots, recent changes
- [x] Build system awareness: CMake, .csproj, platform conditionals
- [x] Semantic search: embeddings via Ollama/Voyage, hybrid RRF
- [x] GPU-accelerated vector search: `.npy` sidecar, cupy/numpy vectorized math
- [x] Multi-repo workspaces (ATTACH+UNION)
- [x] Auto-reindex git hooks (post-commit + post-checkout)
- [x] Document extraction: PDF, DOCX, XLSX, HTML, CSV, email, images, text (heading detection, tables, metadata)
- [x] Optional OCR: PaddleOCR for scanned PDFs, pytesseract for images
- [x] MCP agent guidance: comprehensive instructions, tool selection guide, session protocol
- [x] Workspace config hot-reload (no server restart needed to add repos)
- [x] VectorCache sidecar re-discovery (no restart needed after embedding)
- [x] Project name suggestions in error messages

### Next
- [ ] Cross-language concept mapping (explicit edges between equivalent symbols across languages)
- [ ] Pattern intelligence (convention detection, coding pattern extraction)
- [ ] AI pre-computation (symbol summaries via cheap LLM)

## License

MIT — Gig8 LLC
