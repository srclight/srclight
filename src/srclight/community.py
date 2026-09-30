"""Community detection, execution flow tracing, and impact analysis.

Uses Louvain algorithm (networkx) on call-graph edges to cluster
symbols into functional communities. BFS from entry points traces
execution flows.

References:
- Blondel et al. 2008, "Fast unfolding of communities in large networks"
- Traag et al. 2019, "From Louvain to Leiden" (future upgrade path)
"""

from __future__ import annotations

import functools
import logging
import math
import re
from collections import Counter
from typing import Any, Iterable

from .db import Database

logger = logging.getLogger("srclight.community")


# SQLite caps bound parameters per statement (default 999). Batch large
# `WHERE id IN (...)` lookups so community detection works on 50k+ symbol repos.
_IN_CHUNK = 500


def _chunked_in_select(conn, columns: str, table: str, ids, *, chunk: int = _IN_CHUNK):
    """Run ``SELECT {columns} FROM {table} WHERE id IN (...)`` over ``ids`` in chunks.

    ``columns`` and ``table`` are internal literal strings (not user input).
    Yields matching rows. Avoids SQLite's bound-parameter limit on large repos.
    """
    id_list = list(ids)
    for start in range(0, len(id_list), chunk):
        batch = id_list[start:start + chunk]
        placeholders = ",".join("?" * len(batch))
        yield from conn.execute(
            f"SELECT {columns} FROM {table} WHERE id IN ({placeholders})", batch,
        )


# Part of the call graph fingerprint: a change to how communities are found
# or how execution flows are traced must not be skipped as "graph unchanged"
# on an index that has not changed. Bump it with any change to either.
_GRAPH_ANALYSIS_METHOD = "louvain-1.0-seed42-sorted/flows-20-8-3-50"


def call_graph_edges(db: Database) -> list[tuple[int, int, int]]:
    """The undirected call graph community detection works on, as sorted
    (low id, high id, weight) triples: the weight counts the call edges
    between the two symbols, in either direction.

    Sorted, so the graph built from it is the same whatever order the edges
    were written in. Louvain visits nodes in insertion order, and the order
    of the edge rows changes from one index run to the next: built in that
    order, the same graph gave different communities each time.
    """
    assert db.conn is not None
    weights: Counter = Counter()
    for src, tgt in db.conn.execute(
            "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'"):
        weights[(src, tgt) if src <= tgt else (tgt, src)] += 1
    return sorted((low, high, w) for (low, high), w in weights.items())


def call_graph_fingerprint(db: Database) -> str:
    """A digest of everything communities and execution flows are found
    from: the calls, what the symbols making them are called, what they are
    and where, and the methods.

    The calls are counted with their direction: communities only need the
    undirected graph, but execution flows follow calls from caller to
    callee. And a symbol id is no identity on its own: a file re-parsed gets
    its symbols re-inserted, and SQLite can hand the same ids back to
    symbols that are renamed or moved. So each one's name, qualified name,
    kind and file path are part of it.
    """
    import hashlib

    assert db.conn is not None
    calls = sorted(Counter(tuple(row) for row in db.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'")).items())
    nodes = sorted({n for (src, tgt), _ in calls for n in (src, tgt)})
    symbols = sorted(tuple(row) for row in _chunked_in_select(
        db.conn, "id, name, qualified_name, kind, file_id", "symbols", nodes))
    paths = dict(tuple(row) for row in _chunked_in_select(
        db.conn, "id, path", "files", {row[4] for row in symbols}))
    digest = hashlib.blake2b(_GRAPH_ANALYSIS_METHOD.encode(), digest_size=16)
    for (src, tgt), count in calls:
        digest.update(b"%d>%d*%d;" % (src, tgt, count))
    for sid, name, qualified, kind, file_id in symbols:
        digest.update(repr((sid, name, qualified, kind, paths.get(file_id))).encode())
    return digest.hexdigest()


def detect_communities(db: Database,
                       edges: list[tuple[int, int, int]] | None = None) -> list[dict[str, Any]]:
    """Detect communities in the call graph using Louvain algorithm.

    `edges` is call_graph_edges(db), read here when not given.

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

    if edges is None:
        edges = call_graph_edges(db)
    if not edges:
        return []

    # Build undirected graph for community detection, in one pass
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

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
        for row in _chunked_in_select(
            db.conn, "id, name, qualified_name, kind", "symbols", all_ids
        ):
            id_to_name[row["id"]] = {
                "id": row["id"],
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "kind": row["kind"],
            }

    # Each name is split once, not once per use: the global frequencies, the
    # label and the keywords all need it.
    tokenize = functools.lru_cache(maxsize=None)(_tokenize_name)

    # Compute global token frequencies for TF-IDF labeling
    global_freq = Counter()
    for info in id_to_name.values():
        tokens = tokenize(info["name"] or "")
        global_freq.update(set(tokens))  # set() for document frequency

    # Build community results
    communities = []
    for i, community_set in enumerate(partition):
        members = [id_to_name[sid] for sid in community_set if sid in id_to_name]
        if not members:
            continue

        names = [m["name"] for m in members if m["name"]]
        label = _label_community(names, global_freq, len(partition), tokenize)
        keywords = _extract_keywords(names, global_freq, len(partition), tokenize)

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
    names: list[str], global_freq: Counter, n_communities: int, tokenize=None,
) -> str:
    """Auto-label a community from its member symbol names using TF-IDF."""
    tokenize = tokenize or _tokenize_name
    local_freq = Counter()
    for name in names:
        tokens = tokenize(name)
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
    names: list[str], global_freq: Counter, n_communities: int, tokenize=None,
) -> list[str]:
    """Extract top keywords for a community."""
    tokenize = tokenize or _tokenize_name
    local_freq = Counter()
    for name in names:
        tokens = tokenize(name)
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

    # In a fixed order: each symbol's callees are followed in the order they
    # are listed, and only the first few, so the order the edge rows came
    # back in — which changes from one index run to the next — chose which.
    rows = sorted(
        db.conn.execute(
            "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'"
        ).fetchall(),
        key=lambda row: (row[0], row[1]),
    )

    # Load symbol info (kind, name, file_id) for all nodes in edges
    edge_node_ids: set[int] = set()
    for row in rows:
        edge_node_ids.add(row["source_id"])
        edge_node_ids.add(row["target_id"])

    if not edge_node_ids:
        return []

    id_to_info: dict[int, dict] = {}
    for row in _chunked_in_select(
        db.conn, "id, name, kind, file_id", "symbols", edge_node_ids
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
        for row in _chunked_in_select(db.conn, "id, path", "files", file_ids):
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

    # Ties go to the lower symbol id, not to the order the nodes were met in.
    entry_scores.sort(key=lambda x: (-x[1], x[0]))
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
    also: Iterable[int] = (),
) -> dict[str, Any]:
    """Compute blast radius and risk for modifying a symbol.

    `also` names other symbols that are the same entity — a C++ method's
    declaration and definition: calls land on either, so the impact is that
    of all of them together.

    Returns dict with keys:
        risk (LOW/MEDIUM/HIGH/CRITICAL),
        direct_dependents, transitive_dependents,
        affected_communities, affected_flows,
        is_entry_point, details
    """
    ids = {symbol_id, *also}
    # Edges between the halves of one entity (a definition calling its own
    # declaration) are not dependents. With a single symbol nothing is taken
    # away, so a recursive call still counts, as it always has.
    internal = ids if also else set()

    # Get direct callers (returns list of {"symbol": SymbolRecord, "edge_type": ...})
    direct_ids = {d["symbol"].id for i in ids for d in db.get_callers(i)} - internal

    # Get transitive dependents (same format)
    transitive_ids = {
        d["symbol"].id for i in ids
        for d in db.get_dependents(i, transitive=True, max_depth=max_depth)
    } - internal

    # Affected communities
    my_comms = {sym_to_community.get(i) for i in ids}
    affected_comms = set()
    for dep_id in transitive_ids:
        comm = sym_to_community.get(dep_id)
        if comm is not None and comm not in my_comms:
            affected_comms.add(comm)

    # Affected flows
    affected_flow_labels = []
    is_entry_point = False
    for flow in flows:
        step_ids = {s["symbol_id"] for s in flow["steps"]}
        # Several ids can sit in one flow, and flows can share a label: the
        # label is listed once, but every flow is checked for its entry.
        if ids & step_ids:
            if flow["entry_symbol_id"] in ids:
                is_entry_point = True
            if flow["label"] not in affected_flow_labels:
                affected_flow_labels.append(flow["label"])

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
