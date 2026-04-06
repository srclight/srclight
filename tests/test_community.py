"""Tests for community detection, execution flow tracing, and impact analysis."""

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
    f1 = db.upsert_file(FileRecord(
        path="src/auth.py", content_hash="a1", mtime=1.0,
        language="python", size=100, line_count=20,
    ))
    f2 = db.upsert_file(FileRecord(
        path="src/database.py", content_hash="b1", mtime=1.0,
        language="python", size=100, line_count=20,
    ))

    sym_login = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="login",
        qualified_name="auth.login", start_line=1, end_line=10,
        content="def login(): validate_password(); query_users()",
    ), file_path="src/auth.py")
    sym_validate = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="validate_password",
        qualified_name="auth.validate_password", start_line=11, end_line=20,
        content="def validate_password(): hash_password()",
    ), file_path="src/auth.py")
    sym_hash = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="hash_password",
        qualified_name="auth.hash_password", start_line=21, end_line=30,
        content="def hash_password(): pass",
    ), file_path="src/auth.py")

    sym_query = db.insert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="query_users",
        qualified_name="database.query_users", start_line=1, end_line=10,
        content="def query_users(): connect_db()",
    ), file_path="src/database.py")
    sym_connect = db.insert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="connect_db",
        qualified_name="database.connect_db", start_line=11, end_line=20,
        content="def connect_db(): execute_sql()",
    ), file_path="src/database.py")
    sym_exec = db.insert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="execute_sql",
        qualified_name="database.execute_sql", start_line=21, end_line=30,
        content="def execute_sql(): pass",
    ), file_path="src/database.py")

    # Cluster A internal (triangle for strong internal connectivity)
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_validate, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_validate, target_id=sym_hash, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_hash, edge_type="calls"))
    # Cluster B internal (triangle)
    db.insert_edge(EdgeRecord(source_id=sym_query, target_id=sym_connect, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_connect, target_id=sym_exec, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_query, target_id=sym_exec, edge_type="calls"))
    # Bridge (single weak link)
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

    assert len(communities) >= 2

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
    assert login_comm["id"] != query_comm["id"]


def test_community_labels_are_meaningful(db):
    """Community labels should reflect member symbol names."""
    from srclight.community import detect_communities

    _build_test_graph(db)
    communities = detect_communities(db)

    all_labels = [c["label"] for c in communities]
    assert any(label != "unnamed" for label in all_labels)
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
    s1 = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="foo",
        qualified_name="foo", start_line=1, end_line=5, content="def foo(): bar()",
    ), file_path="a.py")
    s2 = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="bar",
        qualified_name="bar", start_line=6, end_line=10, content="def bar(): pass",
    ), file_path="a.py")
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


def test_trace_execution_flows(db):
    """BFS should find flows from entry points through the call graph."""
    from srclight.community import detect_communities, trace_execution_flows

    syms = _build_test_graph(db)
    communities = detect_communities(db)

    sym_to_comm = {}
    for c in communities:
        for m in c["members"]:
            sym_to_comm[m["id"]] = c["id"]

    flows = trace_execution_flows(db, sym_to_comm)

    # login has highest out-degree and zero in-degree — should be entry point
    assert len(flows) >= 1

    entry_ids = {f["entry_symbol_id"] for f in flows}
    assert syms["login"] in entry_ids

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

    # The flow login -> query_users crosses a community boundary
    cross_flows = [f for f in flows if f["communities_crossed"] > 0]
    assert len(cross_flows) >= 1


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
    assert result["risk"] in ("MEDIUM", "HIGH", "CRITICAL")
    assert len(result["affected_flows"]) >= 1


def test_schema_v5_migration(tmp_path):
    """DB should migrate from v4 to v5, adding community/flow tables."""
    db_path = tmp_path / "migrate.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    tables = {row[0] for row in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    assert "communities" in tables
    assert "symbol_communities" in tables
    assert "execution_flows" in tables
    assert "flow_steps" in tables

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

    db.store_communities(communities)
    db.commit()

    stored = db.get_communities()
    assert len(stored) >= 2

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

    stored = db.get_execution_flows()
    assert len(stored) >= 1

    login_flows = db.get_flows_for_symbol(syms["login"])
    assert len(login_flows) >= 1
