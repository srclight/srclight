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
import math
import re
from collections import Counter
from typing import Any

from .db import Database

logger = logging.getLogger("srclight.community")


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

        names = [m["name"] for m in members if m["name"]]
        label = _label_community(names, global_freq, len(partition))
        keywords = _extract_keywords(names, global_freq, len(partition))

        communities.append({
            "id": i,
            "label": label,
            "symbol_count": len(members),
            "cohesion": _community_cohesion(community_set, G),
            "keywords": keywords,
            "members": members,
        })

    # Sort by size descending, re-number
    communities.sort(key=lambda c: c["symbol_count"], reverse=True)
    for i, c in enumerate(communities):
        c["id"] = i

    return communities


def _tokenize_name(name: str) -> list[str]:
    """Split a symbol name into lowercase tokens."""
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

    scored = {}
    for token, count in local_freq.items():
        tf = count / len(names)
        df = global_freq.get(token, 1)
        idf = math.log(max(n_communities, 2) / max(df, 1)) + 1
        scored[token] = tf * idf

    # Filter very short tokens
    scored = {t: s for t, s in scored.items() if len(t) > 1}

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

    scored = {}
    for token, count in local_freq.items():
        if len(token) <= 1:
            continue
        tf = count / max(len(names), 1)
        df = global_freq.get(token, 1)
        idf = math.log(max(n_communities, 2) / max(df, 1)) + 1
        scored[token] = tf * idf

    return sorted(scored, key=scored.get, reverse=True)[:5]


def _community_cohesion(members: set[int], G) -> float:
    """Compute cohesion as ratio of internal edges to possible edges."""
    if len(members) < 2:
        return 1.0
    internal = sum(
        1 for u in members for v in G.neighbors(u) if v in members
    )
    # Each undirected edge counted twice
    internal //= 2
    possible = len(members) * (len(members) - 1) // 2
    return round(internal / possible, 4) if possible > 0 else 0.0


def trace_execution_flows(
    db: Database,
    sym_to_community: dict[int, int],
    max_entry_points: int = 20,
    max_depth: int = 8,
    max_branching: int = 3,
    max_flows: int = 50,
) -> list[dict[str, Any]]:
    """Trace execution flows via BFS from entry points along call edges.

    Returns list of flow dicts with keys:
        entry_symbol_id, terminal_symbol_id, label, step_count,
        communities_crossed, steps
    """
    assert db.conn is not None

    # Container kinds are not behavioral — skip as entry points and targets
    CONTAINER_KINDS = {"class", "mixin", "enum", "struct", "extension", "section"}

    # Build adjacency list from call edges
    adjacency: dict[int, list[int]] = {}
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}

    rows = db.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'"
    ).fetchall()

    # Load symbol info (kind, name, file_id) for all nodes in edges
    edge_node_ids: set[int] = set()
    for row in rows:
        edge_node_ids.add(row["source_id"])
        edge_node_ids.add(row["target_id"])

    if not edge_node_ids:
        return []

    placeholders = ",".join("?" * len(edge_node_ids))
    id_to_info: dict[int, dict] = {}
    for row in db.conn.execute(
        f"SELECT id, name, kind, file_id FROM symbols WHERE id IN ({placeholders})",
        list(edge_node_ids),
    ):
        id_to_info[row["id"]] = {
            "name": row["name"], "kind": row["kind"], "file_id": row["file_id"],
        }

    # Filter edges: skip edges where source or target is a container kind
    for row in rows:
        src, tgt = row["source_id"], row["target_id"]
        src_kind = id_to_info.get(src, {}).get("kind", "")
        tgt_kind = id_to_info.get(tgt, {}).get("kind", "")
        if src_kind in CONTAINER_KINDS or tgt_kind in CONTAINER_KINDS:
            continue
        adjacency.setdefault(src, []).append(tgt)
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    all_nodes = set(out_degree.keys()) | set(in_degree.keys())
    if not all_nodes:
        return []

    # Detect test files
    file_ids = {info["file_id"] for info in id_to_info.values() if info.get("file_id")}
    test_file_ids: set[int] = set()
    if file_ids:
        fp = ",".join("?" * len(file_ids))
        for row in db.conn.execute(
            f"SELECT id, path FROM files WHERE id IN ({fp})", list(file_ids)
        ):
            if "test" in row["path"].lower():
                test_file_ids.add(row["id"])

    # Score entry points (functions/methods only, not containers)
    ENTRY_HEURISTICS = {"main", "run", "start", "init", "setup", "execute", "handle", "serve"}
    entry_scores: list[tuple[int, float]] = []

    for node_id in all_nodes:
        info = id_to_info.get(node_id)
        if not info:
            continue
        if info.get("kind") in CONTAINER_KINDS:
            continue
        if info.get("file_id") in test_file_ids:
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

    # BFS from each entry point (with global iteration cap)
    raw_flows: list[list[int]] = []
    max_total_iterations = 5000
    total_iterations = 0
    for entry_id in entry_points:
        flows_from_entry, iters = _bfs_flows(
            entry_id, adjacency, max_depth, max_branching,
            iteration_budget=max_total_iterations - total_iterations,
        )
        total_iterations += iters
        raw_flows.extend(flows_from_entry)
        if total_iterations >= max_total_iterations:
            break

    # Deduplicate: remove subset flows
    raw_flows.sort(key=len, reverse=True)
    unique_flows: list[list[int]] = []
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
    iteration_budget: int = 5000,
) -> tuple[list[list[int]], int]:
    """BFS from a single entry point. Returns (list of paths, iteration count)."""
    flows: list[list[int]] = []
    stack = [([start], 0)]
    iterations = 0

    while stack:
        iterations += 1
        if iterations > iteration_budget:
            break

        path, depth = stack.pop()
        current = path[-1]

        if depth >= max_depth:
            flows.append(path)
            continue

        callees = adjacency.get(current, [])
        if not callees:
            if len(path) >= 2:
                flows.append(path)
            continue

        extended = False
        for callee in callees[:max_branching]:
            if callee in path:
                continue  # avoid cycles
            stack.append((path + [callee], depth + 1))
            extended = True

        if not extended and len(path) >= 2:
            flows.append(path)

    return flows, iterations


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
        is_entry_point, details
    """
    # Get direct callers (returns list of {"symbol": SymbolRecord, "edge_type": ...})
    direct = db.get_callers(symbol_id)
    direct_ids = {d["symbol"].id for d in direct}

    # Get transitive dependents (same format)
    transitive = db.get_dependents(symbol_id, transitive=True, max_depth=max_depth)
    transitive_ids = {d["symbol"].id for d in transitive}

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
