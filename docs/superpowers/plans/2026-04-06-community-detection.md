# Community Detection & Execution Flow Tracing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Louvain community detection and BFS execution flow tracing to Srclight, stored persistently in per-repo SQLite (schema v5), exposed via 4 new MCP tools.

**Architecture:** Post-indexing phase runs networkx Louvain on call-graph edges, stores communities + flows in new tables. Results survive restarts and are queryable across workspace via existing ATTACH+UNION. Impact scoring combines graph depth with community boundary crossings.

**Tech Stack:** networkx (BSD-3, Louvain), SQLite (existing), tree-sitter (existing)

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `src/srclight/community.py` | Community detection, flow tracing, impact analysis | Create |
| `src/srclight/db.py` | Schema v5 migration, CRUD for communities/flows | Modify |
| `src/srclight/indexer.py` | Post-index hook for community/flow computation | Modify |
| `src/srclight/server.py` | 4 new MCP tools | Modify |
| `src/srclight/workspace.py` | UNION queries for communities/flows | Modify |
| `tests/test_community.py` | All community/flow/impact tests | Create |
| `docs/labbook/community-detection.md` | Experiment results | Modify |

---

### Task 1: Install networkx and Write First Test

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_community.py`
- Create: `src/srclight/community.py`

- [ ] **Step 1: Add networkx dependency**

In `pyproject.toml`, add `networkx` to the dependencies list:

```toml
"networkx>=3.0",
```

- [ ] **Step 2: Install**

Run: `cd /home/tim/Projects/srclight/srclight && .venv/bin/pip install -e .`

- [ ] **Step 3: Write failing test for community detection on a small graph**

Create `tests/test_community.py`:

```python
"""Tests for community detection, execution flow tracing, and impact analysis."""

import tempfile
from pathlib import Path

import pytest

from srclight.db import Database, FileRecord, SymbolRecord, EdgeRecord


@pytest.fixture
def db(tmp_path):
    """Create a temporary database with schema v5."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def _build_test_graph(db):
    """Build a small call graph with two clear clusters.

    Cluster A (auth): login -> validate_password -> hash_password
    Cluster B (db):   query_users -> connect_db -> execute_sql
    Cross-cluster:    login -> query_users (bridge edge)
    """
    # Create two files
    f1 = db.upsert_file(FileRecord(
        path="src/auth.py", content_hash="a1", mtime=1.0,
        language="python", size=100, line_count=20,
    ))
    f2 = db.upsert_file(FileRecord(
        path="src/database.py", content_hash="b1", mtime=1.0,
        language="python", size=100, line_count=20,
    ))

    # Cluster A symbols
    sym_login = db.upsert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="login",
        qualified_name="auth.login", start_line=1, end_line=10,
        content="def login(): validate_password(); query_users()",
    ))
    sym_validate = db.upsert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="validate_password",
        qualified_name="auth.validate_password", start_line=11, end_line=20,
        content="def validate_password(): hash_password()",
    ))
    sym_hash = db.upsert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="hash_password",
        qualified_name="auth.hash_password", start_line=21, end_line=30,
        content="def hash_password(): pass",
    ))

    # Cluster B symbols
    sym_query = db.upsert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="query_users",
        qualified_name="database.query_users", start_line=1, end_line=10,
        content="def query_users(): connect_db()",
    ))
    sym_connect = db.upsert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="connect_db",
        qualified_name="database.connect_db", start_line=11, end_line=20,
        content="def connect_db(): execute_sql()",
    ))
    sym_exec = db.upsert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="execute_sql",
        qualified_name="database.execute_sql", start_line=21, end_line=30,
        content="def execute_sql(): pass",
    ))

    # Edges: Cluster A internal
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_validate, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_validate, target_id=sym_hash, edge_type="calls"))

    # Edges: Cluster B internal
    db.insert_edge(EdgeRecord(source_id=sym_query, target_id=sym_connect, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_connect, target_id=sym_exec, edge_type="calls"))

    # Bridge edge: login -> query_users
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_query, edge_type="calls"))

    db.commit()

    return {
        "login": sym_login, "validate_password": sym_validate,
        "hash_password": sym_hash, "query_users": sym_query,
        "connect_db": sym_connect, "execute_sql": sym_exec,
    }


def test_detect_communities_two_clusters(db):
    """Louvain should detect two communities in a graph with two clear clusters."""
    from srclight.community import detect_communities

    syms = _build_test_graph(db)
    communities = detect_communities(db)

    # Should find exactly 2 communities
    assert len(communities) >= 2

    # login and validate_password should be in the same community
    login_comm = None
    query_comm = None
    for c in communities:
        member_ids = {m["id"] for m in c["members"]}
        if syms["login"] in member_ids:
            login_comm = c
        if syms["query_users"] in member_ids:
            query_comm = c

    assert login_comm is not None
    assert query_comm is not None
    # The two clusters should be different communities
    assert login_comm["id"] != query_comm["id"]
```

- [ ] **Step 4: Create minimal community.py stub**

Create `src/srclight/community.py`:

```python
"""Community detection, execution flow tracing, and impact analysis.

Uses Louvain algorithm (networkx) on call-graph edges to cluster
symbols into functional communities. BFS from entry points traces
execution flows.

References:
- Blondel et al. 2008, "Fast unfolding of communities in large networks"
- Traag et al. 2019, "From Louvain to Leiden" (future upgrade path)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .db import Database

logger = logging.getLogger("srclight.community")


def detect_communities(db: Database) -> list[dict[str, Any]]:
    """Detect communities in the call graph using Louvain algorithm.

    Returns list of community dicts with keys:
        id, label, symbol_count, cohesion, keywords, members
    """
    raise NotImplementedError("TODO")
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_detect_communities_two_clusters -v`
Expected: FAIL with "NotImplementedError"

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/srclight/community.py tests/test_community.py
git commit -m "feat: scaffold community detection with first failing test"
```

---

### Task 2: Implement Community Detection

**Files:**
- Modify: `src/srclight/community.py`

- [ ] **Step 1: Implement detect_communities**

Replace the `detect_communities` stub in `src/srclight/community.py` with:

```python
def detect_communities(db: Database) -> list[dict[str, Any]]:
    """Detect communities in the call graph using Louvain algorithm.

    Returns list of community dicts with keys:
        id, label, symbol_count, cohesion, keywords, members
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        logger.warning("networkx not installed — skipping community detection")
        return []

    assert db.conn is not None

    # Load call edges
    rows = db.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'"
    ).fetchall()

    if not rows:
        return []

    # Build undirected graph for community detection
    G = nx.Graph()
    for row in rows:
        src, tgt = row["source_id"], row["target_id"]
        if G.has_edge(src, tgt):
            # Increase weight for repeated edges
            G[src][tgt]["weight"] = G[src][tgt].get("weight", 1) + 1
        else:
            G.add_edge(src, tgt, weight=1)

    if G.number_of_nodes() < 2:
        return []

    # Run Louvain
    partition = louvain_communities(G, resolution=1.0, seed=42)

    # Load symbol names for labeling
    all_ids = set()
    for community_set in partition:
        all_ids.update(community_set)

    id_to_name = {}
    if all_ids:
        placeholders = ",".join("?" * len(all_ids))
        for row in db.conn.execute(
            f"SELECT id, name, qualified_name, kind FROM symbols WHERE id IN ({placeholders})",
            list(all_ids),
        ):
            id_to_name[row["id"]] = {
                "id": row["id"],
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "kind": row["kind"],
            }

    # Compute global token frequencies for TF-IDF labeling
    global_freq = Counter()
    for info in id_to_name.values():
        tokens = _tokenize_name(info["name"] or "")
        global_freq.update(set(tokens))  # set() for document frequency

    # Build community results
    communities = []
    for i, community_set in enumerate(partition):
        members = [id_to_name[sid] for sid in community_set if sid in id_to_name]
        if not members:
            continue

        label = _label_community(
            [m["name"] for m in members if m["name"]],
            global_freq,
            len(partition),
        )
        keywords = _extract_keywords(
            [m["name"] for m in members if m["name"]],
            global_freq,
            len(partition),
        )

        communities.append({
            "id": i,
            "label": label,
            "symbol_count": len(members),
            "cohesion": _community_cohesion(community_set, G),
            "keywords": keywords,
            "members": members,
        })

    # Sort by size descending
    communities.sort(key=lambda c: c["symbol_count"], reverse=True)
    # Re-number IDs after sorting
    for i, c in enumerate(communities):
        c["id"] = i

    return communities


def _tokenize_name(name: str) -> list[str]:
    """Split a symbol name into lowercase tokens."""
    import re
    if not name:
        return []
    # Split on :: . -> _
    parts = re.split(r"::|->|\.|_", name)
    tokens = []
    for part in parts:
        if not part:
            continue
        # Split CamelCase
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        tokens.extend(t.lower() for t in s.split() if t)
    return tokens


def _label_community(
    names: list[str], global_freq: Counter, n_communities: int,
) -> str:
    """Auto-label a community from its member symbol names using TF-IDF."""
    local_freq = Counter()
    for name in names:
        tokens = _tokenize_name(name)
        local_freq.update(tokens)

    if not local_freq:
        return "unnamed"

    # TF-IDF-like: local frequency * inverse document frequency
    import math
    scored = {}
    for token, count in local_freq.items():
        tf = count / len(names)
        # IDF: log(total_communities / communities_containing_token)
        # Approximate: use global_freq as doc frequency
        df = global_freq.get(token, 1)
        idf = math.log(max(n_communities, 2) / max(df, 1)) + 1
        scored[token] = tf * idf

    # Filter very short or very common tokens
    scored = {t: s for t, s in scored.items() if len(t) > 1}

    # Top 2-3 tokens
    top = sorted(scored, key=scored.get, reverse=True)[:3]
    if not top:
        return "unnamed"
    return " ".join(top).title()


def _extract_keywords(
    names: list[str], global_freq: Counter, n_communities: int,
) -> list[str]:
    """Extract top keywords for a community."""
    local_freq = Counter()
    for name in names:
        tokens = _tokenize_name(name)
        local_freq.update(tokens)

    import math
    scored = {}
    for token, count in local_freq.items():
        if len(token) <= 1:
            continue
        tf = count / max(len(names), 1)
        df = global_freq.get(token, 1)
        idf = math.log(max(n_communities, 2) / max(df, 1)) + 1
        scored[token] = tf * idf

    top = sorted(scored, key=scored.get, reverse=True)[:5]
    return top


def _community_cohesion(members: set[int], G) -> float:
    """Compute cohesion as ratio of internal edges to possible edges."""
    if len(members) < 2:
        return 1.0
    internal = sum(
        1 for u in members for v in G.neighbors(u) if v in members
    )
    # Each undirected edge counted twice (once from each endpoint)
    internal //= 2
    possible = len(members) * (len(members) - 1) // 2
    return round(internal / possible, 4) if possible > 0 else 0.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_detect_communities_two_clusters -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/srclight/community.py
git commit -m "feat: implement Louvain community detection"
```

---

### Task 3: Test Community Labeling and Edge Cases

**Files:**
- Modify: `tests/test_community.py`

- [ ] **Step 1: Write tests for labeling and edge cases**

Append to `tests/test_community.py`:

```python
def test_community_labels_are_meaningful(db):
    """Community labels should reflect member symbol names."""
    from srclight.community import detect_communities

    _build_test_graph(db)
    communities = detect_communities(db)

    all_labels = [c["label"] for c in communities]
    # Labels should not all be "unnamed"
    assert any(label != "unnamed" for label in all_labels)
    # Each community should have keywords
    for c in communities:
        assert isinstance(c["keywords"], list)


def test_community_cohesion_range(db):
    """Cohesion should be between 0 and 1."""
    from srclight.community import detect_communities

    _build_test_graph(db)
    communities = detect_communities(db)

    for c in communities:
        assert 0.0 <= c["cohesion"] <= 1.0


def test_detect_communities_empty_graph(db):
    """No edges -> no communities."""
    from srclight.community import detect_communities

    communities = detect_communities(db)
    assert communities == []


def test_detect_communities_single_edge(db):
    """Single edge -> one community with 2 members."""
    from srclight.community import detect_communities

    f1 = db.upsert_file(FileRecord(
        path="a.py", content_hash="x", mtime=1.0,
        language="python", size=10, line_count=5,
    ))
    s1 = db.upsert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="foo",
        qualified_name="foo", start_line=1, end_line=5, content="def foo(): bar()",
    ))
    s2 = db.upsert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="bar",
        qualified_name="bar", start_line=6, end_line=10, content="def bar(): pass",
    ))
    db.insert_edge(EdgeRecord(source_id=s1, target_id=s2, edge_type="calls"))
    db.commit()

    communities = detect_communities(db)
    assert len(communities) == 1
    assert communities[0]["symbol_count"] == 2


def test_tokenize_name():
    """Identifier tokenization handles CamelCase, snake_case, qualifiers."""
    from srclight.community import _tokenize_name

    assert _tokenize_name("getUserName") == ["get", "user", "name"]
    assert _tokenize_name("get_user_name") == ["get", "user", "name"]
    assert _tokenize_name("HTTPClient") == ["http", "client"]
    assert _tokenize_name("auth::validate") == ["auth", "validate"]
    assert _tokenize_name("") == []
```

- [ ] **Step 2: Run all community tests**

Run: `.venv/bin/python -m pytest tests/test_community.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_community.py
git commit -m "test: community detection edge cases and labeling"
```

---

### Task 4: Implement Execution Flow Tracing

**Files:**
- Modify: `src/srclight/community.py`
- Modify: `tests/test_community.py`

- [ ] **Step 1: Write failing test for flow tracing**

Append to `tests/test_community.py`:

```python
def test_trace_execution_flows(db):
    """BFS should find flows from entry points through the call graph."""
    from srclight.community import detect_communities, trace_execution_flows

    syms = _build_test_graph(db)
    communities = detect_communities(db)

    # Build symbol->community mapping
    sym_to_comm = {}
    for c in communities:
        for m in c["members"]:
            sym_to_comm[m["id"]] = c["id"]

    flows = trace_execution_flows(db, sym_to_comm)

    # login has highest out-degree (2) and zero in-degree — should be entry point
    assert len(flows) >= 1

    # At least one flow should start from login
    entry_ids = {f["entry_symbol_id"] for f in flows}
    assert syms["login"] in entry_ids

    # Flows should have steps
    for f in flows:
        assert f["step_count"] >= 2
        assert len(f["steps"]) == f["step_count"]


def test_flow_communities_crossed(db):
    """Flows crossing community boundaries should be counted."""
    from srclight.community import detect_communities, trace_execution_flows

    syms = _build_test_graph(db)
    communities = detect_communities(db)

    sym_to_comm = {}
    for c in communities:
        for m in c["members"]:
            sym_to_comm[m["id"]] = c["id"]

    flows = trace_execution_flows(db, sym_to_comm)

    # The flow login -> query_users -> connect_db -> execute_sql crosses a boundary
    cross_flows = [f for f in flows if f["communities_crossed"] > 0]
    assert len(cross_flows) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_trace_execution_flows -v`
Expected: FAIL with "ImportError" (trace_execution_flows not defined)

- [ ] **Step 3: Implement trace_execution_flows**

Add to `src/srclight/community.py`:

```python
def trace_execution_flows(
    db: Database,
    sym_to_community: dict[int, int],
    max_entry_points: int = 50,
    max_depth: int = 10,
    max_branching: int = 4,
    max_flows: int = 75,
) -> list[dict[str, Any]]:
    """Trace execution flows via BFS from entry points along call edges.

    Args:
        db: Database with call edges
        sym_to_community: symbol_id -> community_id mapping
        max_entry_points: number of entry points to trace from
        max_depth: maximum BFS depth per flow
        max_branching: maximum callees to follow per node
        max_flows: maximum flows to return

    Returns list of flow dicts with keys:
        entry_symbol_id, terminal_symbol_id, label, step_count,
        communities_crossed, steps (list of {symbol_id, community_id, order})
    """
    assert db.conn is not None

    # Build adjacency list from call edges
    adjacency: dict[int, list[int]] = {}
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}

    rows = db.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'"
    ).fetchall()

    for row in rows:
        src, tgt = row["source_id"], row["target_id"]
        adjacency.setdefault(src, []).append(tgt)
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    all_nodes = set(out_degree.keys()) | set(in_degree.keys())
    if not all_nodes:
        return []

    # Load symbol names for labeling and entry point heuristics
    placeholders = ",".join("?" * len(all_nodes))
    id_to_info: dict[int, dict] = {}
    for row in db.conn.execute(
        f"SELECT id, name, file_id FROM symbols WHERE id IN ({placeholders})",
        list(all_nodes),
    ):
        id_to_info[row["id"]] = {"name": row["name"], "file_id": row["file_id"]}

    # Load file paths to detect test files
    file_ids = {info["file_id"] for info in id_to_info.values()}
    if file_ids:
        fp = ",".join("?" * len(file_ids))
        test_file_ids = set()
        for row in db.conn.execute(
            f"SELECT id, path FROM files WHERE id IN ({fp})", list(file_ids)
        ):
            if "test" in row["path"].lower():
                test_file_ids.add(row["id"])
    else:
        test_file_ids = set()

    # Score entry points
    ENTRY_HEURISTICS = {"main", "run", "start", "init", "setup", "execute", "handle", "serve"}
    entry_scores: list[tuple[int, float]] = []

    for node_id in all_nodes:
        info = id_to_info.get(node_id)
        if not info:
            continue
        # Skip test files
        if info["file_id"] in test_file_ids:
            continue

        out_d = out_degree.get(node_id, 0)
        in_d = in_degree.get(node_id, 0)

        if out_d == 0:
            continue  # leaf nodes aren't entry points

        # Score: high out-degree, low in-degree
        score = out_d / max(in_d, 0.5)

        # Bonus for heuristic names
        name = (info["name"] or "").lower()
        for h in ENTRY_HEURISTICS:
            if name.startswith(h) or name == h:
                score *= 2.0
                break

        entry_scores.append((node_id, score))

    entry_scores.sort(key=lambda x: x[1], reverse=True)
    entry_points = [ep[0] for ep in entry_scores[:max_entry_points]]

    # BFS from each entry point
    raw_flows = []
    for entry_id in entry_points:
        flows_from_entry = _bfs_flows(
            entry_id, adjacency, max_depth, max_branching
        )
        raw_flows.extend(flows_from_entry)

    # Deduplicate: remove subset flows
    raw_flows.sort(key=lambda f: len(f), reverse=True)
    unique_flows = []
    seen_step_sets: list[set[int]] = []
    for flow in raw_flows:
        flow_set = set(flow)
        is_subset = any(flow_set.issubset(existing) for existing in seen_step_sets)
        if not is_subset:
            unique_flows.append(flow)
            seen_step_sets.append(flow_set)

    # Deduplicate by entry+terminal pair: keep longest
    pair_best: dict[tuple[int, int], list[int]] = {}
    for flow in unique_flows:
        pair = (flow[0], flow[-1])
        if pair not in pair_best or len(flow) > len(pair_best[pair]):
            pair_best[pair] = flow
    unique_flows = list(pair_best.values())

    # Sort by length and take top N
    unique_flows.sort(key=len, reverse=True)
    unique_flows = unique_flows[:max_flows]

    # Build result dicts
    results = []
    for flow in unique_flows:
        steps = []
        prev_comm = None
        crossings = 0
        for order, sym_id in enumerate(flow):
            comm_id = sym_to_community.get(sym_id)
            steps.append({
                "symbol_id": sym_id,
                "community_id": comm_id,
                "order": order,
            })
            if prev_comm is not None and comm_id is not None and comm_id != prev_comm:
                crossings += 1
            prev_comm = comm_id

        entry_name = id_to_info.get(flow[0], {}).get("name", "?")
        terminal_name = id_to_info.get(flow[-1], {}).get("name", "?")
        label = f"{entry_name} -> {terminal_name}"

        results.append({
            "entry_symbol_id": flow[0],
            "terminal_symbol_id": flow[-1],
            "label": label,
            "step_count": len(flow),
            "communities_crossed": crossings,
            "steps": steps,
        })

    return results


def _bfs_flows(
    start: int,
    adjacency: dict[int, list[int]],
    max_depth: int,
    max_branching: int,
) -> list[list[int]]:
    """BFS from a single entry point. Returns list of paths (each a list of symbol IDs)."""
    flows = []
    # Stack: (current_path, depth)
    stack = [([start], 0)]

    while stack:
        path, depth = stack.pop()
        current = path[-1]

        if depth >= max_depth:
            flows.append(path)
            continue

        callees = adjacency.get(current, [])
        if not callees:
            # Leaf — this is a complete flow
            if len(path) >= 2:
                flows.append(path)
            continue

        # Limit branching
        extended = False
        for callee in callees[:max_branching]:
            if callee in path:
                continue  # avoid cycles
            stack.append((path + [callee], depth + 1))
            extended = True

        if not extended and len(path) >= 2:
            flows.append(path)

    return flows
```

- [ ] **Step 4: Run flow tracing tests**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_trace_execution_flows tests/test_community.py::test_flow_communities_crossed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/srclight/community.py tests/test_community.py
git commit -m "feat: implement BFS execution flow tracing"
```

---

### Task 5: Implement Impact Analysis

**Files:**
- Modify: `src/srclight/community.py`
- Modify: `tests/test_community.py`

- [ ] **Step 1: Write failing test for impact analysis**

Append to `tests/test_community.py`:

```python
def test_compute_impact_low_risk(db):
    """Leaf node with few dependents should be LOW risk."""
    from srclight.community import detect_communities, trace_execution_flows, compute_impact

    syms = _build_test_graph(db)
    communities = detect_communities(db)
    sym_to_comm = {}
    for c in communities:
        for m in c["members"]:
            sym_to_comm[m["id"]] = c["id"]

    flows = trace_execution_flows(db, sym_to_comm)

    # hash_password is a leaf with 1 caller, same community
    result = compute_impact(db, syms["hash_password"], sym_to_comm, flows)
    assert result["risk"] == "LOW"
    assert result["direct_dependents"] >= 1


def test_compute_impact_higher_risk_for_bridge(db):
    """Symbol that bridges communities should have higher risk."""
    from srclight.community import detect_communities, trace_execution_flows, compute_impact

    syms = _build_test_graph(db)
    communities = detect_communities(db)
    sym_to_comm = {}
    for c in communities:
        for m in c["members"]:
            sym_to_comm[m["id"]] = c["id"]

    flows = trace_execution_flows(db, sym_to_comm)

    # login is an entry point that bridges communities
    result = compute_impact(db, syms["login"], sym_to_comm, flows)
    # Should be at least MEDIUM due to being an entry point in flows
    assert result["risk"] in ("MEDIUM", "HIGH", "CRITICAL")
    assert len(result["affected_flows"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_compute_impact_low_risk -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Implement compute_impact**

Add to `src/srclight/community.py`:

```python
def compute_impact(
    db: Database,
    symbol_id: int,
    sym_to_community: dict[int, int],
    flows: list[dict[str, Any]],
    max_depth: int = 3,
) -> dict[str, Any]:
    """Compute blast radius and risk for modifying a symbol.

    Returns dict with keys:
        risk (LOW/MEDIUM/HIGH/CRITICAL),
        direct_dependents, transitive_dependents,
        affected_communities, affected_flows,
        details
    """
    # Get direct callers
    direct = db.get_callers(symbol_id)
    direct_ids = {d["id"] for d in direct}

    # Get transitive dependents
    transitive = db.get_dependents(symbol_id, transitive=True, max_depth=max_depth)
    transitive_ids = {d["id"] for d in transitive}

    # Affected communities
    my_comm = sym_to_community.get(symbol_id)
    affected_comms = set()
    for dep_id in transitive_ids:
        comm = sym_to_community.get(dep_id)
        if comm is not None and comm != my_comm:
            affected_comms.add(comm)

    # Affected flows
    affected_flow_labels = []
    is_entry_point = False
    for flow in flows:
        step_ids = {s["symbol_id"] for s in flow["steps"]}
        if symbol_id in step_ids:
            affected_flow_labels.append(flow["label"])
            if flow["entry_symbol_id"] == symbol_id:
                is_entry_point = True

    # Risk scoring
    n_direct = len(direct_ids)
    n_comm_crossings = len(affected_comms)

    if n_direct > 25 or is_entry_point:
        risk = "CRITICAL"
    elif n_direct > 10 or n_comm_crossings >= 2:
        risk = "HIGH"
    elif n_direct > 3 or n_comm_crossings >= 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "risk": risk,
        "direct_dependents": n_direct,
        "transitive_dependents": len(transitive_ids),
        "affected_communities": list(affected_comms),
        "affected_flows": affected_flow_labels,
        "is_entry_point": is_entry_point,
        "details": f"{n_direct} direct callers, {len(transitive_ids)} transitive, "
                   f"{n_comm_crossings} community boundaries crossed, "
                   f"{len(affected_flow_labels)} flows affected",
    }
```

- [ ] **Step 4: Run impact tests**

Run: `.venv/bin/python -m pytest tests/test_community.py -k "impact" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/srclight/community.py tests/test_community.py
git commit -m "feat: implement impact analysis with risk scoring"
```

---

### Task 6: Schema v5 Migration and DB CRUD

**Files:**
- Modify: `src/srclight/db.py`

- [ ] **Step 1: Write failing test for schema migration**

Append to `tests/test_community.py`:

```python
def test_schema_v5_migration(tmp_path):
    """DB should migrate from v4 to v5, adding community/flow tables."""
    db_path = tmp_path / "migrate.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    # Verify new tables exist
    tables = {row[0] for row in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    assert "communities" in tables
    assert "symbol_communities" in tables
    assert "execution_flows" in tables
    assert "flow_steps" in tables

    # Verify schema version is 5
    version = db.conn.execute(
        "SELECT value FROM schema_info WHERE key = 'schema_version'"
    ).fetchone()["value"]
    assert version == "5"

    db.close()


def test_store_and_retrieve_communities(db):
    """Can store and retrieve community data."""
    syms = _build_test_graph(db)

    from srclight.community import detect_communities
    communities = detect_communities(db)

    # Store communities
    db.store_communities(communities)
    db.commit()

    # Retrieve
    stored = db.get_communities()
    assert len(stored) >= 2

    # Check symbol-community mapping
    comm_id = db.get_community_for_symbol(syms["login"])
    assert comm_id is not None


def test_store_and_retrieve_flows(db):
    """Can store and retrieve execution flow data."""
    from srclight.community import detect_communities, trace_execution_flows

    syms = _build_test_graph(db)
    communities = detect_communities(db)
    sym_to_comm = {}
    for c in communities:
        for m in c["members"]:
            sym_to_comm[m["id"]] = c["id"]

    flows = trace_execution_flows(db, sym_to_comm)
    db.store_communities(communities)
    db.store_execution_flows(flows)
    db.commit()

    # Retrieve all flows
    stored = db.get_execution_flows()
    assert len(stored) >= 1

    # Retrieve flows for a specific symbol
    login_flows = db.get_flows_for_symbol(syms["login"])
    assert len(login_flows) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_schema_v5_migration -v`
Expected: FAIL (tables don't exist yet, version still 4)

- [ ] **Step 3: Update SCHEMA_SQL and SCHEMA_VERSION in db.py**

In `src/srclight/db.py`, change `SCHEMA_VERSION = 4` to `SCHEMA_VERSION = 5`.

After the `index_state` table definition (around line 174), add:

```sql
-- Communities (Louvain clusters of call-graph symbols)
CREATE TABLE IF NOT EXISTS communities (
    id INTEGER PRIMARY KEY,
    label TEXT,
    symbol_count INTEGER DEFAULT 0,
    cohesion REAL,
    keywords TEXT,
    metadata TEXT
);

-- Junction: which symbols belong to which community
CREATE TABLE IF NOT EXISTS symbol_communities (
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    PRIMARY KEY (symbol_id, community_id)
);

CREATE INDEX IF NOT EXISTS idx_sym_comm_community ON symbol_communities(community_id);

-- Execution flows (BFS traces from entry points)
CREATE TABLE IF NOT EXISTS execution_flows (
    id INTEGER PRIMARY KEY,
    label TEXT,
    entry_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    terminal_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    step_count INTEGER,
    communities_crossed INTEGER DEFAULT 0,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_flows_entry ON execution_flows(entry_symbol_id);

-- Ordered steps in a flow
CREATE TABLE IF NOT EXISTS flow_steps (
    flow_id INTEGER NOT NULL REFERENCES execution_flows(id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    community_id INTEGER,
    PRIMARY KEY (flow_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_flow_steps_symbol ON flow_steps(symbol_id);
```

- [ ] **Step 4: Add CRUD methods to Database class**

In `src/srclight/db.py`, add these methods to the `Database` class:

```python
    def store_communities(self, communities: list[dict]) -> None:
        """Store detected communities and their symbol memberships."""
        assert self.conn is not None
        # Clear existing
        self.conn.execute("DELETE FROM symbol_communities")
        self.conn.execute("DELETE FROM communities")

        for c in communities:
            self.conn.execute(
                """INSERT INTO communities (id, label, symbol_count, cohesion, keywords, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (c["id"], c["label"], c["symbol_count"], c["cohesion"],
                 json.dumps(c.get("keywords", [])), json.dumps(c.get("metadata"))),
            )
            for member in c.get("members", []):
                self.conn.execute(
                    "INSERT OR IGNORE INTO symbol_communities (symbol_id, community_id) VALUES (?, ?)",
                    (member["id"], c["id"]),
                )

    def get_communities(self) -> list[dict]:
        """Get all communities with stats."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT * FROM communities ORDER BY symbol_count DESC"
        ).fetchall()
        return [
            {
                "id": r["id"], "label": r["label"],
                "symbol_count": r["symbol_count"], "cohesion": r["cohesion"],
                "keywords": json.loads(r["keywords"] or "[]"),
            }
            for r in rows
        ]

    def get_community_for_symbol(self, symbol_id: int) -> int | None:
        """Get the community ID for a symbol, or None."""
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT community_id FROM symbol_communities WHERE symbol_id = ?",
            (symbol_id,),
        ).fetchone()
        return row["community_id"] if row else None

    def get_community_members(self, community_id: int) -> list[dict]:
        """Get all symbols in a community."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.id, s.name, s.qualified_name, s.kind, f.path as file_path
               FROM symbol_communities sc
               JOIN symbols s ON sc.symbol_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE sc.community_id = ?
               ORDER BY s.name""",
            (community_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def store_execution_flows(self, flows: list[dict]) -> None:
        """Store execution flows and their steps."""
        assert self.conn is not None
        self.conn.execute("DELETE FROM flow_steps")
        self.conn.execute("DELETE FROM execution_flows")

        for flow in flows:
            cursor = self.conn.execute(
                """INSERT INTO execution_flows
                   (label, entry_symbol_id, terminal_symbol_id, step_count, communities_crossed)
                   VALUES (?, ?, ?, ?, ?)""",
                (flow["label"], flow["entry_symbol_id"], flow["terminal_symbol_id"],
                 flow["step_count"], flow["communities_crossed"]),
            )
            flow_id = cursor.lastrowid
            for step in flow["steps"]:
                self.conn.execute(
                    """INSERT INTO flow_steps (flow_id, step_order, symbol_id, community_id)
                       VALUES (?, ?, ?, ?)""",
                    (flow_id, step["order"], step["symbol_id"], step.get("community_id")),
                )

    def get_execution_flows(self, limit: int = 50) -> list[dict]:
        """Get top execution flows ordered by step count."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT ef.*, s1.name as entry_name, s2.name as terminal_name
               FROM execution_flows ef
               LEFT JOIN symbols s1 ON ef.entry_symbol_id = s1.id
               LEFT JOIN symbols s2 ON ef.terminal_symbol_id = s2.id
               ORDER BY ef.step_count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_flows_for_symbol(self, symbol_id: int) -> list[dict]:
        """Get all execution flows that pass through a symbol."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT ef.*, s1.name as entry_name, s2.name as terminal_name
               FROM flow_steps fs
               JOIN execution_flows ef ON fs.flow_id = ef.id
               LEFT JOIN symbols s1 ON ef.entry_symbol_id = s1.id
               LEFT JOIN symbols s2 ON ef.terminal_symbol_id = s2.id
               WHERE fs.symbol_id = ?
               ORDER BY ef.step_count DESC""",
            (symbol_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_flow_steps(self, flow_id: int) -> list[dict]:
        """Get ordered steps for a specific flow."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT fs.step_order, fs.symbol_id, fs.community_id,
                      s.name, s.qualified_name, s.kind, f.path as file_path
               FROM flow_steps fs
               JOIN symbols s ON fs.symbol_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE fs.flow_id = ?
               ORDER BY fs.step_order""",
            (flow_id,),
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 5: Add migration logic to initialize()**

In the `initialize()` method of `Database`, after the existing `ALTER TABLE` migration, add:

```python
        # Migrate v4 -> v5: add community/flow tables
        try:
            self.conn.execute("SELECT 1 FROM communities LIMIT 1")
        except Exception:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS communities (
                    id INTEGER PRIMARY KEY, label TEXT,
                    symbol_count INTEGER DEFAULT 0, cohesion REAL,
                    keywords TEXT, metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS symbol_communities (
                    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                    community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
                    PRIMARY KEY (symbol_id, community_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sym_comm_community ON symbol_communities(community_id);
                CREATE TABLE IF NOT EXISTS execution_flows (
                    id INTEGER PRIMARY KEY, label TEXT,
                    entry_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
                    terminal_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
                    step_count INTEGER, communities_crossed INTEGER DEFAULT 0, metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_flows_entry ON execution_flows(entry_symbol_id);
                CREATE TABLE IF NOT EXISTS flow_steps (
                    flow_id INTEGER NOT NULL REFERENCES execution_flows(id) ON DELETE CASCADE,
                    step_order INTEGER NOT NULL,
                    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                    community_id INTEGER,
                    PRIMARY KEY (flow_id, step_order)
                );
                CREATE INDEX IF NOT EXISTS idx_flow_steps_symbol ON flow_steps(symbol_id);
            """)
```

- [ ] **Step 6: Run all community tests**

Run: `.venv/bin/python -m pytest tests/test_community.py -v`
Expected: All PASS

- [ ] **Step 7: Run existing tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/test_db.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/srclight/db.py tests/test_community.py
git commit -m "feat: schema v5 with community/flow tables and CRUD"
```

---

### Task 7: Integrate with Indexer

**Files:**
- Modify: `src/srclight/indexer.py`
- Modify: `tests/test_community.py`

- [ ] **Step 1: Write test that indexer triggers community detection**

Append to `tests/test_community.py`:

```python
def test_indexer_runs_community_detection(tmp_path):
    """The indexer should compute communities after building edges."""
    from srclight.indexer import Indexer, IndexerConfig

    # Create a minimal Python project with two files
    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.py").write_text(
        "def login():\n    validate()\n\ndef validate():\n    hash_pw()\n\ndef hash_pw():\n    pass\n"
    )
    (src / "db.py").write_text(
        "def query():\n    connect()\n\ndef connect():\n    execute()\n\ndef execute():\n    pass\n"
    )
    # Initialize git repo so indexer can use git ls-files
    import subprocess
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "init"],
        capture_output=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_COMMITTER_NAME": "test",
             "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t"},
    )

    db_path = tmp_path / ".srclight" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    db.open()
    db.initialize()

    config = IndexerConfig(root=tmp_path, db=db)
    indexer = Indexer(config)
    stats = indexer.index()

    # Communities should have been computed
    communities = db.get_communities()
    assert len(communities) >= 1

    # Flows should have been computed
    flows = db.get_execution_flows()
    # May or may not find flows depending on edge extraction — just check it ran
    assert isinstance(flows, list)

    db.close()
```

- [ ] **Step 2: Add community detection call to indexer.index()**

In `src/srclight/indexer.py`, after the edge building block (line ~603, after `stats.edges_created += self._build_inheritance_edges()`), add:

```python
            # Community detection and execution flow tracing (post-edge phase)
            try:
                from .community import detect_communities, trace_execution_flows
                communities = detect_communities(self.db)
                if communities:
                    sym_to_comm = {}
                    for c in communities:
                        for m in c["members"]:
                            sym_to_comm[m["id"]] = c["id"]
                    flows = trace_execution_flows(self.db, sym_to_comm)
                    self.db.store_communities(communities)
                    self.db.store_execution_flows(flows)
                    logger.info(
                        "Detected %d communities, %d execution flows",
                        len(communities), len(flows),
                    )
            except ImportError:
                logger.debug("networkx not available — skipping community detection")
            except Exception:
                logger.warning("Community detection failed", exc_info=True)
```

- [ ] **Step 3: Run test**

Run: `.venv/bin/python -m pytest tests/test_community.py::test_indexer_runs_community_detection -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/srclight/indexer.py tests/test_community.py
git commit -m "feat: integrate community detection into indexer pipeline"
```

---

### Task 8: Add MCP Tools to Server

**Files:**
- Modify: `src/srclight/server.py`
- Modify: `tests/test_community.py`

- [ ] **Step 1: Read server.py to find the tool registration area**

Read `src/srclight/server.py` to identify where to add new `@mcp.tool()` definitions — typically after the existing graph tools like `get_callers`, `get_callees`.

- [ ] **Step 2: Add 4 new MCP tools**

Add these tool definitions to `src/srclight/server.py` after the existing graph tool section:

```python
@mcp.tool()
def get_communities(project: str | None = None) -> str:
    """List all detected code communities (functional clusters).

    Communities are auto-detected via Louvain clustering on the call graph.
    Each community represents a group of symbols that frequently call each other.

    Returns community labels, member counts, cohesion scores, and keywords.

    WHEN TO USE: To understand the high-level structure of a codebase —
    what functional modules exist and how big they are.
    AFTER THIS: Use get_community() on a specific symbol to see its neighbors.

    Args:
        project: Project name (required in workspace mode)
    """
    wdb = _get_workspace_db()
    if wdb is None:
        return "Error: no workspace configured"

    project = _resolve_project(project)
    if isinstance(project, str) and project.startswith("Error"):
        return project

    try:
        results = []
        for batch in wdb._iter_batches(project_filter=project):
            for schema, proj_name in batch:
                try:
                    rows = wdb.conn.execute(
                        f"SELECT * FROM [{schema}].communities ORDER BY symbol_count DESC"
                    ).fetchall()
                    for r in rows:
                        results.append({
                            "project": proj_name,
                            "id": r["id"],
                            "label": r["label"],
                            "symbol_count": r["symbol_count"],
                            "cohesion": r["cohesion"],
                            "keywords": json.loads(r["keywords"] or "[]"),
                        })
                except Exception:
                    pass  # table may not exist in older indexes

        if not results:
            return "No communities detected. Run `srclight index` to generate them."

        lines = [f"Found {len(results)} communities:\n"]
        for c in results:
            proj = f"[{c['project']}] " if c.get("project") else ""
            kw = ", ".join(c["keywords"][:3]) if c["keywords"] else ""
            lines.append(
                f"  {proj}#{c['id']} {c['label']} — {c['symbol_count']} symbols, "
                f"cohesion {c['cohesion']:.2f}"
                + (f" ({kw})" if kw else "")
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_community(symbol_name: str, project: str | None = None) -> str:
    """Find which community a symbol belongs to and who its co-members are.

    Shows the functional cluster a symbol is part of, plus other symbols
    in the same community. Useful for understanding what module a function
    belongs to and finding related code.

    Args:
        symbol_name: Name of the symbol to look up
        project: Project name (required in workspace mode)
    """
    wdb = _get_workspace_db()
    if wdb is None:
        return "Error: no workspace configured"

    project = _resolve_project(project)
    if isinstance(project, str) and project.startswith("Error"):
        return project

    try:
        # Find the symbol first
        symbols = wdb.search_symbols(symbol_name, limit=1, project=project)
        if not symbols:
            return f"Symbol '{symbol_name}' not found."

        sym = symbols[0]
        sym_id = sym["id"]
        proj_name = sym.get("project", project)

        # Find community membership via workspace DB
        for batch in wdb._iter_batches(project_filter=proj_name):
            for schema, pname in batch:
                if pname != proj_name:
                    continue
                try:
                    row = wdb.conn.execute(
                        f"""SELECT c.* FROM [{schema}].symbol_communities sc
                            JOIN [{schema}].communities c ON sc.community_id = c.id
                            WHERE sc.symbol_id = ?""",
                        (sym_id,),
                    ).fetchone()
                    if row is None:
                        return f"'{symbol_name}' is not in any detected community."

                    # Get co-members
                    members = wdb.conn.execute(
                        f"""SELECT s.name, s.kind, f.path as file_path
                            FROM [{schema}].symbol_communities sc
                            JOIN [{schema}].symbols s ON sc.symbol_id = s.id
                            JOIN [{schema}].files f ON s.file_id = f.id
                            WHERE sc.community_id = ?
                            ORDER BY s.name LIMIT 25""",
                        (row["id"],),
                    ).fetchall()

                    keywords = json.loads(row["keywords"] or "[]")
                    lines = [
                        f"Community #{row['id']}: {row['label']}",
                        f"  Cohesion: {row['cohesion']:.2f}",
                        f"  Keywords: {', '.join(keywords[:5])}",
                        f"  Members ({row['symbol_count']} total):",
                    ]
                    for m in members:
                        marker = " <--" if m["name"] == sym["name"] else ""
                        lines.append(f"    {m['kind']} {m['name']} ({m['file_path']}){marker}")

                    return "\n".join(lines)
                except Exception:
                    pass

        return f"No community data found for '{symbol_name}'."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_execution_flows(
    symbol_name: str | None = None, project: str | None = None, limit: int = 10,
) -> str:
    """Show execution flows (call chains) through the codebase.

    If symbol_name is given, shows flows that pass through that symbol.
    Otherwise, shows the top flows in the project by length.

    Each flow is a traced path from an entry point through the call graph,
    with community boundary crossings marked.

    Args:
        symbol_name: Optional symbol to filter flows by
        project: Project name (required in workspace mode)
        limit: Maximum flows to return (default 10)
    """
    wdb = _get_workspace_db()
    if wdb is None:
        return "Error: no workspace configured"

    project = _resolve_project(project)
    if isinstance(project, str) and project.startswith("Error"):
        return project

    try:
        results = []
        target_sym_id = None

        if symbol_name:
            symbols = wdb.search_symbols(symbol_name, limit=1, project=project)
            if symbols:
                target_sym_id = symbols[0]["id"]
                project = symbols[0].get("project", project)

        for batch in wdb._iter_batches(project_filter=project):
            for schema, proj_name in batch:
                try:
                    if target_sym_id:
                        rows = wdb.conn.execute(
                            f"""SELECT ef.* FROM [{schema}].flow_steps fs
                                JOIN [{schema}].execution_flows ef ON fs.flow_id = ef.id
                                WHERE fs.symbol_id = ?
                                ORDER BY ef.step_count DESC LIMIT ?""",
                            (target_sym_id, limit),
                        ).fetchall()
                    else:
                        rows = wdb.conn.execute(
                            f"""SELECT * FROM [{schema}].execution_flows
                                ORDER BY step_count DESC LIMIT ?""",
                            (limit,),
                        ).fetchall()

                    for r in rows:
                        # Get steps
                        steps = wdb.conn.execute(
                            f"""SELECT fs.step_order, s.name, s.kind, fs.community_id
                                FROM [{schema}].flow_steps fs
                                JOIN [{schema}].symbols s ON fs.symbol_id = s.id
                                WHERE fs.flow_id = ?
                                ORDER BY fs.step_order""",
                            (r["id"],),
                        ).fetchall()

                        results.append({
                            "project": proj_name,
                            "label": r["label"],
                            "step_count": r["step_count"],
                            "communities_crossed": r["communities_crossed"],
                            "steps": [dict(s) for s in steps],
                        })
                except Exception:
                    pass

        if not results:
            msg = "No execution flows found."
            if not symbol_name:
                msg += " Run `srclight index` to generate them."
            return msg

        lines = [f"Found {len(results)} execution flow(s):\n"]
        for f in results[:limit]:
            proj = f"[{f['project']}] " if f.get("project") else ""
            lines.append(
                f"  {proj}{f['label']} ({f['step_count']} steps, "
                f"{f['communities_crossed']} community crossings)"
            )
            prev_comm = None
            for step in f["steps"]:
                boundary = ""
                if prev_comm is not None and step["community_id"] is not None and step["community_id"] != prev_comm:
                    boundary = " [boundary]"
                lines.append(f"    {step['step_order']}. {step['kind']} {step['name']}{boundary}")
                prev_comm = step["community_id"]
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_impact(symbol_name: str, project: str | None = None) -> str:
    """Analyze the blast radius of modifying a symbol.

    Shows direct and transitive dependents, affected communities,
    affected execution flows, and an overall risk level
    (LOW/MEDIUM/HIGH/CRITICAL).

    WHEN TO USE: Before modifying a function, class, or method —
    understand what could break.

    Args:
        symbol_name: Name of the symbol to analyze
        project: Project name (required in workspace mode)
    """
    wdb = _get_workspace_db()
    if wdb is None:
        return "Error: no workspace configured"

    project = _resolve_project(project)
    if isinstance(project, str) and project.startswith("Error"):
        return project

    try:
        symbols = wdb.search_symbols(symbol_name, limit=1, project=project)
        if not symbols:
            return f"Symbol '{symbol_name}' not found."

        sym = symbols[0]
        sym_id = sym["id"]
        proj_name = sym.get("project", project)

        # Need per-project DB for impact analysis
        for batch in wdb._iter_batches(project_filter=proj_name):
            for schema, pname in batch:
                if pname != proj_name:
                    continue
                try:
                    # Build sym_to_comm mapping
                    sym_to_comm = {}
                    for row in wdb.conn.execute(
                        f"SELECT symbol_id, community_id FROM [{schema}].symbol_communities"
                    ).fetchall():
                        sym_to_comm[row["symbol_id"]] = row["community_id"]

                    # Get flows
                    flow_rows = wdb.conn.execute(
                        f"SELECT * FROM [{schema}].execution_flows"
                    ).fetchall()
                    flows = []
                    for fr in flow_rows:
                        steps = wdb.conn.execute(
                            f"""SELECT step_order, symbol_id, community_id
                                FROM [{schema}].flow_steps WHERE flow_id = ?
                                ORDER BY step_order""",
                            (fr["id"],),
                        ).fetchall()
                        flows.append({
                            "label": fr["label"],
                            "entry_symbol_id": fr["entry_symbol_id"],
                            "steps": [dict(s) for s in steps],
                        })

                    # Use the project's individual DB for impact computation
                    from .community import compute_impact
                    entry = wdb._config.projects.get(proj_name)
                    if not entry:
                        continue
                    from pathlib import Path
                    db_path = Path(entry) / ".srclight" / "index.db"
                    if not db_path.exists():
                        continue

                    from .db import Database
                    with Database(db_path) as proj_db:
                        result = compute_impact(proj_db, sym_id, sym_to_comm, flows)

                    risk_emoji = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}
                    lines = [
                        f"Impact Analysis: {symbol_name}",
                        f"  Risk: {risk_emoji.get(result['risk'], result['risk'])}",
                        f"  {result['details']}",
                    ]
                    if result["affected_flows"]:
                        lines.append(f"  Affected flows:")
                        for fl in result["affected_flows"][:5]:
                            lines.append(f"    - {fl}")
                    if result["affected_communities"]:
                        lines.append(f"  Crosses into communities: {result['affected_communities']}")
                    if result["is_entry_point"]:
                        lines.append(f"  WARNING: This is an execution flow entry point")

                    return "\n".join(lines)
                except Exception as e:
                    logger.debug("Impact analysis error for %s: %s", pname, e)

        return f"Could not compute impact for '{symbol_name}'."
    except Exception as e:
        return f"Error: {e}"
```

- [ ] **Step 3: Add json import if not already present at top of server.py**

Check that `import json` is at the top of `server.py`. Add it if missing.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS including existing ones

- [ ] **Step 5: Commit**

```bash
git add src/srclight/server.py
git commit -m "feat: add 4 MCP tools for communities, flows, and impact"
```

---

### Task 9: Lab Book Experiments on Real Data

**Files:**
- Modify: `docs/labbook/community-detection.md`

- [ ] **Step 1: Run community detection on srclight's own index**

```bash
cd /home/tim/Projects/srclight/srclight
.venv/bin/python -c "
from srclight.db import Database
from srclight.community import detect_communities, trace_execution_flows
import time

db = Database('.srclight/index.db')
db.open()
db.initialize()  # ensure v5 tables exist

t0 = time.monotonic()
communities = detect_communities(db)
t_comm = time.monotonic() - t0

sym_to_comm = {}
for c in communities:
    for m in c['members']:
        sym_to_comm[m['id']] = c['id']

t0 = time.monotonic()
flows = trace_execution_flows(db, sym_to_comm)
t_flow = time.monotonic() - t0

print(f'Communities: {len(communities)} ({t_comm:.3f}s)')
for c in communities:
    print(f'  #{c[\"id\"]} {c[\"label\"]} — {c[\"symbol_count\"]} symbols, cohesion={c[\"cohesion\"]:.3f}')
    print(f'    Keywords: {c[\"keywords\"][:5]}')

print(f'\nFlows: {len(flows)} ({t_flow:.3f}s)')
for f in flows[:10]:
    print(f'  {f[\"label\"]} ({f[\"step_count\"]} steps, {f[\"communities_crossed\"]} crossings)')

db.close()
"
```

- [ ] **Step 2: Run on bitcoin repo (196K edges, scale test)**

```bash
.venv/bin/python -c "
from srclight.db import Database
from srclight.community import detect_communities
import time

db = Database('/home/tim/Projects/loqu8/bitcoin/.srclight/index.db')
db.open()

t0 = time.monotonic()
communities = detect_communities(db)
elapsed = time.monotonic() - t0

print(f'Bitcoin: {len(communities)} communities in {elapsed:.3f}s')
for c in communities[:15]:
    print(f'  #{c[\"id\"]} {c[\"label\"]} — {c[\"symbol_count\"]} symbols, cohesion={c[\"cohesion\"]:.3f}')

db.close()
"
```

- [ ] **Step 3: Record results in lab book**

Update `docs/labbook/community-detection.md` with actual experiment results from steps 1 and 2: community counts, labels, timing, and observations about whether the clusters match known module structure.

- [ ] **Step 4: Commit**

```bash
git add docs/labbook/community-detection.md
git commit -m "docs: lab book results from community detection experiments"
```

---

### Task 10: Update CLAUDE.md and Version

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml`
- Modify: `src/srclight/__init__.py`

- [ ] **Step 1: Update CLAUDE.md**

Add `community.py` to the Key Modules table. Update the MCP Tools count from 29 to 33. Add a "Tier 7: Community & Flows" section to the MCP Tools list:

```markdown
### Tier 7: Community Detection & Execution Flows
- `get_communities(project?)` — auto-detected functional clusters
- `get_community(symbol, project)` — community membership + co-members
- `get_execution_flows(symbol?, project)` — call chains through the codebase
- `get_impact(symbol, project)` — blast radius with risk scoring
```

Update the Edge Types section to mention communities and flows.

- [ ] **Step 2: Bump version**

This adds 4 new MCP tools = minor version bump. Update both files:

- `pyproject.toml`: bump version
- `src/srclight/__init__.py`: bump `__version__`

- [ ] **Step 3: Run full test suite one final time**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md pyproject.toml src/srclight/__init__.py
git commit -m "docs: update CLAUDE.md with community detection tools, bump version"
```
