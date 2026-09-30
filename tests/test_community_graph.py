"""The graph communities are found in: built the same whatever order its edges
were written in, and not searched again when it has not changed."""
import logging
import random

import pytest

from srclight.community import call_graph_edges, call_graph_fingerprint, detect_communities
from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord
from srclight.indexer import IndexConfig, Indexer


def _db_with_graph(path, order_seed):
    """A call graph with no clear-cut clusters — where the order Louvain
    visits nodes in decides what it finds — its edges written in an order
    that depends on `order_seed`, some twice and some both ways."""
    db = Database(path)
    db.open()
    db.initialize()
    fid = db.upsert_file(FileRecord(path="m.py", content_hash="h", mtime=1.0,
                                    language="python", size=1, line_count=1))
    names = [f"{verb}_{noun}_{i}" for verb in ("read", "write", "check", "load")
             for noun in ("config", "frame", "entry", "index", "state") for i in range(2)]
    ids = [db.insert_symbol(SymbolRecord(file_id=fid, kind="function", name=n,
                                         start_line=k + 1, end_line=k + 1,
                                         content=f"def {n}(): pass"), "m.py")
           for k, n in enumerate(names)]
    rng = random.Random(7)
    edges = [(a, b) for a in ids for b in ids if a < b and rng.random() < 0.12]
    edges += [(b, a) for a, b in edges[:5]] + edges[5:8]  # reversed and repeated
    random.Random(order_seed).shuffle(edges)
    for a, b in edges:
        db.insert_edge(EdgeRecord(source_id=a, target_id=b, edge_type="calls"))
    db.commit()
    return db


def _partition(communities):
    return {frozenset(m["id"] for m in c["members"]) for c in communities}


def test_the_same_graph_gives_the_same_communities_whatever_the_edge_order(tmp_path):
    """Louvain visits nodes in insertion order. Built in the order the edge
    rows came back — which changes from one index run to the next — the same
    call graph gave different communities each time."""
    first = _db_with_graph(tmp_path / "a.db", order_seed=1)
    second = _db_with_graph(tmp_path / "b.db", order_seed=2)
    assert call_graph_edges(first) == call_graph_edges(second)
    one, other = detect_communities(first), detect_communities(second)
    assert len(one) >= 2
    assert _partition(one) == _partition(other)
    assert [(c["label"], c["keywords"], c["cohesion"]) for c in one] == \
           [(c["label"], c["keywords"], c["cohesion"]) for c in other]
    first.close()
    second.close()


def test_edges_are_undirected_and_weighted_by_their_count(tmp_path):
    db = _db_with_graph(tmp_path / "a.db", order_seed=3)
    edges = call_graph_edges(db)
    assert all(low <= high for low, high, _ in edges)
    assert edges == sorted(edges)
    raw = db.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls'").fetchall()
    assert sum(w for _, _, w in edges) == len(raw)
    assert any(w > 1 for _, _, w in edges)
    db.close()


def test_the_fingerprint_follows_the_calls_and_their_symbols(tmp_path):
    a = _db_with_graph(tmp_path / "a.db", order_seed=4)
    b = _db_with_graph(tmp_path / "b.db", order_seed=5)
    same = call_graph_fingerprint(a)
    assert same == call_graph_fingerprint(b), "the order the edges were written in"

    src, tgt = a.conn.execute(
        "SELECT source_id, target_id FROM symbol_edges WHERE edge_type = 'calls' "
        "ORDER BY rowid LIMIT 1").fetchone()
    # One more call, between two symbols not linked yet.
    linked = {tuple(r) for r in a.conn.execute("SELECT source_id, target_id FROM symbol_edges")}
    ids = [r[0] for r in a.conn.execute("SELECT id FROM symbols ORDER BY id")]
    extra = next((x, y) for x in ids for y in ids
                 if x != y and (x, y) not in linked and (y, x) not in linked)
    a.insert_edge(EdgeRecord(source_id=extra[0], target_id=extra[1], edge_type="calls"))
    assert call_graph_fingerprint(a) != same
    a.conn.execute("DELETE FROM symbol_edges WHERE source_id = ? AND target_id = ?", extra)
    assert call_graph_fingerprint(a) == same

    # The same call the other way round: the same undirected graph, so the
    # same communities, but not the same execution flows.
    undirected = call_graph_edges(a)
    a.conn.execute("UPDATE symbol_edges SET source_id = ?, target_id = ? "
                   "WHERE source_id = ? AND target_id = ?", (tgt, src, src, tgt))
    assert call_graph_edges(a) == undirected
    assert call_graph_fingerprint(a) != same
    a.conn.execute("UPDATE symbol_edges SET source_id = ?, target_id = ? "
                   "WHERE source_id = ? AND target_id = ?", (src, tgt, tgt, src))
    assert call_graph_fingerprint(a) == same

    # The same id, another symbol: renamed where it stands.
    a.conn.execute("UPDATE symbols SET name = 'renamed' WHERE id = ?", (src,))
    assert call_graph_fingerprint(a) != same
    a.close()
    b.close()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "flow.py").write_text(
        "def load_config():\n    return parse_config()\n\n\n"
        "def parse_config():\n    return check_config()\n\n\n"
        "def check_config():\n    return 1\n")
    (root / "alone.py").write_text("def unrelated_helper():\n    return 2\n")
    return root


def _index(root, db_path):
    db = Database(db_path)
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index(root)
    return db


def test_an_unchanged_call_graph_is_not_searched_again(project, tmp_path, caplog):
    """A run that re-parses a file without calls rebuilds the same graph:
    the stored communities still hold, and Louvain is not run again."""
    caplog.set_level(logging.INFO, logger="srclight.indexer")
    db_path = tmp_path / "index.db"
    _index(project, db_path).close()
    stored = Database(db_path)
    stored.open()
    before = stored.get_communities()
    stored.close()
    assert before

    (project / "alone.py").write_text("def unrelated_helper():\n    return 3\n")
    caplog.clear()
    db = _index(project, db_path)
    assert "Communities unchanged" in caplog.text
    assert db.get_communities() == before
    db.close()

    (project / "flow.py").write_text(
        "def load_config():\n    return check_config()\n\n\n"
        "def check_config():\n    return 1\n")
    caplog.clear()
    db = _index(project, db_path)
    assert "Communities unchanged" not in caplog.text
    assert "Detected" in caplog.text
    db.close()


def _members_missing(db):
    """Symbols with call edges but no community."""
    return db.conn.execute(
        """SELECT COUNT(*) FROM (SELECT source_id AS id FROM symbol_edges WHERE edge_type = 'calls'
                                 UNION SELECT target_id FROM symbol_edges WHERE edge_type = 'calls')
           WHERE id NOT IN (SELECT symbol_id FROM symbol_communities)""").fetchone()[0]


def test_symbols_re_parsed_under_the_same_ids_get_their_communities_back(project, tmp_path, caplog):
    """Re-parsing a file deletes its symbols, and their community and flow
    rows with them (ON DELETE CASCADE). SQLite can hand the same ids back to
    the re-inserted symbols, and the graph then looks the same: skipped, the
    run left them in no community at all."""
    caplog.set_level(logging.INFO, logger="srclight.indexer")
    db_path = tmp_path / "index.db"

    def flow_ids(db):
        return [r[0] for r in db.conn.execute(
            "SELECT s.id FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE f.path = 'flow.py' ORDER BY s.id")]

    db = _index(project, db_path)
    ids_before = flow_ids(db)
    assert _members_missing(db) == 0
    db.close()

    # A body changed, no call: the last file indexed, so its ids come back.
    flow = project / "flow.py"
    flow.write_text(flow.read_text().replace("return 1", "return 2"))
    caplog.clear()
    db = _index(project, db_path)
    assert flow_ids(db) == ids_before, "the test needs SQLite to reuse the ids"
    assert "Communities unchanged" not in caplog.text
    assert _members_missing(db) == 0
    db.close()


def test_flows_that_failed_to_store_are_found_again(project, tmp_path, caplog, monkeypatch):
    """What the communities were found from is recorded once they and the
    flows are stored: a failure in between leaves nothing to skip on."""
    caplog.set_level(logging.INFO, logger="srclight.indexer")
    db_path = tmp_path / "index.db"

    def locked(self, flows):
        raise RuntimeError("database is locked")

    with monkeypatch.context() as m:
        m.setattr(Database, "store_execution_flows", locked)
        _index(project, db_path).close()
    assert "Community detection failed" in caplog.text

    (project / "alone.py").write_text("def unrelated_helper():\n    return 4\n")
    caplog.clear()
    db = _index(project, db_path)
    assert "Communities unchanged" not in caplog.text
    assert db.conn.execute("SELECT COUNT(*) FROM execution_flows").fetchone()[0] > 0
    db.close()


def test_the_same_graph_gives_the_same_execution_flows_whatever_the_edge_order(tmp_path):
    """Flows follow only the first few callees of each symbol, in the order
    they are listed: listed in edge row order, the same graph gave other
    flows from one run to the next."""
    from srclight.community import trace_execution_flows

    def flows(db):
        found = trace_execution_flows(db, {}, max_branching=2)
        return [[s["symbol_id"] for s in f["steps"]] for f in found]

    runs = []
    for seed in (1, 2, 3, 4):
        db = _db_with_graph(tmp_path / f"{seed}.db", order_seed=seed)
        runs.append(flows(db))
        db.close()
    assert runs[0]
    assert all(run == runs[0] for run in runs)


def test_a_record_that_is_not_a_dict_is_not_trusted(tmp_path):
    db = _db_with_graph(tmp_path / "a.db", order_seed=1)
    db.conn.execute("INSERT OR REPLACE INTO schema_info (key, value) "
                    "VALUES ('communities_state', '[1, 2]')")
    assert db.communities_still_hold(call_graph_fingerprint(db)) is False
    db.close()
