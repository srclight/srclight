# Community Detection & Execution Flow Tracing

**Date:** 2026-04-06
**Status:** Approved
**Scope:** New module `community.py`, schema v5, 4 MCP tools, ~15-20 tests

## Motivation

Code indexing gives agents symbols and call relationships, but not structural understanding. Community detection clusters symbols into functional modules automatically (e.g., "auth", "database", "routing"). Execution flow tracing finds end-to-end paths through the call graph. Together they answer: "what module does this belong to?" and "what execution flows pass through here?"

Inspired by academic research (Louvain: Blondel et al. 2008, Leiden: Traag et al. 2019) and competitive analysis of graph-based code intelligence tools.

## Architecture

### Approach: Communities as Computed Metadata

Community detection runs as a post-indexing phase in `indexer.py`. Results are stored persistently in the per-repo SQLite database (schema v5). This fits our existing architecture: SQLite per-repo, workspace ATTACH+UNION across DBs, incremental reindex.

### New Dependency

`networkx` (BSD-3-Clause) — provides `louvain_communities()`. Optional at runtime (graceful degradation if not installed). ~15MB installed size.

## Schema Changes (v4 -> v5)

```sql
CREATE TABLE IF NOT EXISTS communities (
    id INTEGER PRIMARY KEY,
    label TEXT,
    symbol_count INTEGER DEFAULT 0,
    cohesion REAL,
    keywords TEXT,           -- JSON array
    metadata TEXT             -- JSON
);

CREATE TABLE IF NOT EXISTS symbol_communities (
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    PRIMARY KEY (symbol_id, community_id)
);

CREATE TABLE IF NOT EXISTS execution_flows (
    id INTEGER PRIMARY KEY,
    label TEXT,
    entry_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    terminal_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    step_count INTEGER,
    communities_crossed INTEGER DEFAULT 0,
    metadata TEXT             -- JSON
);

CREATE TABLE IF NOT EXISTS flow_steps (
    flow_id INTEGER NOT NULL REFERENCES execution_flows(id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    community_id INTEGER,
    PRIMARY KEY (flow_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_sym_comm_community ON symbol_communities(community_id);
CREATE INDEX IF NOT EXISTS idx_flow_steps_symbol ON flow_steps(symbol_id);
CREATE INDEX IF NOT EXISTS idx_flows_entry ON execution_flows(entry_symbol_id);
```

Migration: detect schema_version=4, CREATE new tables, UPDATE to 5.

## New Module: `community.py`

### Community Detection

1. Load all `symbol_edges` where `edge_type='calls'` into a networkx DiGraph
2. Convert to undirected for Louvain (community structure is undirected)
3. Run `networkx.community.louvain_communities(G, resolution=1.0, seed=42)`
4. Auto-label each community:
   - Extract symbol names from members
   - Split identifiers (CamelCase/snake_case) into tokens
   - Score tokens by TF-IDF: frequency in community / frequency globally
   - Top 2-3 distinctive tokens become the label
5. Compute cohesion as modularity contribution per community
6. Store in `communities` + `symbol_communities` tables

### Execution Flow Tracing

1. Score entry points: symbols with high out-degree / low in-degree on calls edges
   - Bonus for heuristic names: `main`, `handle_*`, `route_*`, `setup_*`, `init_*`, `run_*`
   - Exclude test files
2. BFS from top-N entry points (N=50), max depth 10, max branching 4
3. Deduplicate: remove subset flows, keep longest for same entry+terminal pair
4. Retain top 75 flows sorted by step count
5. Annotate each step with its community ID
6. Count community boundary crossings per flow
7. Store in `execution_flows` + `flow_steps` tables

### Impact Analysis

`compute_impact(db, symbol_id)` returns:
- Direct dependents (callers)
- Transitive dependents (depth 3)
- Affected communities (set of community IDs touched)
- Affected execution flows (flows containing this symbol)
- Risk level: LOW / MEDIUM / HIGH / CRITICAL

Risk criteria:
| Risk | Criteria |
|------|----------|
| LOW | <=3 direct dependents, all same community |
| MEDIUM | 4-10 dependents, or crosses 1 community boundary |
| HIGH | 11-25 dependents, or crosses 2+ community boundaries |
| CRITICAL | >25 dependents, or symbol is execution flow entry point |

## MCP Tools (4 new)

### `get_communities(project?)`
List all detected communities with stats: label, symbol_count, cohesion, keywords.

### `get_community(symbol, project)`
Which community a symbol belongs to, plus its co-members and inter-community connections.

### `get_execution_flows(symbol?, project)`
If symbol given: flows containing that symbol. Otherwise: top flows in the project.

### `get_impact(symbol, project)`
Blast radius with risk scoring. Returns dependents, affected communities, affected flows, risk level.

## Integration

- **`indexer.py`**: Call community detection + flow tracing after edge extraction
- **`db.py`**: CRUD for communities/flows, schema migration
- **`server.py`**: 4 new MCP tool handlers
- **`workspace.py`**: Add to `_iter_batches()` UNION queries

## Testing

Target: ~15-20 tests in `tests/test_community.py`

- Louvain on known small graph (expected clusters)
- Entry point scoring (high out-degree + low in-degree = entry)
- BFS flow tracing (depth limit, branching limit, dedup)
- Community labeling (keyword extraction)
- Impact scoring (risk levels)
- Full pipeline on srclight's own index (1008 symbols, 4049 edges)
- Schema migration v4 -> v5
- Graceful degradation when networkx unavailable

## Lab Book

Experiments and observations recorded in `docs/labbook/community-detection.md`.
