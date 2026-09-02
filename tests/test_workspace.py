"""Tests for workspace (multi-repo) functionality."""

import json
from pathlib import Path

import pytest

from srclight.db import Database, FileRecord, SymbolRecord
from srclight.workspace import WorkspaceConfig, WorkspaceDB, _sanitize_schema_name


@pytest.fixture
def ws_dir(tmp_path):
    """Override workspaces dir for testing."""
    import srclight.workspace as ws_mod
    orig = ws_mod.WORKSPACES_DIR
    ws_mod.WORKSPACES_DIR = tmp_path / "workspaces"
    yield tmp_path / "workspaces"
    ws_mod.WORKSPACES_DIR = orig


def _create_indexed_project(tmp_path: Path, name: str, symbols: list[tuple[str, str]]):
    """Create a project dir with a .srclight/index.db populated with symbols.

    symbols: list of (name, kind) tuples
    """
    project_dir = tmp_path / name
    project_dir.mkdir()
    db_dir = project_dir / ".srclight"
    db_dir.mkdir()
    db_path = db_dir / "index.db"

    db = Database(db_path)
    db.open()
    db.initialize()

    file_id = db.upsert_file(FileRecord(
        path=f"src/{name}.cs",
        content_hash="abc123",
        mtime=1000.0,
        language="csharp",
        size=500,
        line_count=50,
    ))

    for i, (sym_name, sym_kind) in enumerate(symbols):
        db.insert_symbol(SymbolRecord(
            file_id=file_id,
            kind=sym_kind,
            name=sym_name,
            qualified_name=f"{name}.{sym_name}",
            signature=f"{sym_kind} {sym_name}()" if sym_kind in ("method", "function") else sym_name,
            start_line=i * 10 + 1,
            end_line=i * 10 + 8,
            content=f"{sym_kind} {sym_name} {{ }}",
            line_count=8,
        ), f"src/{name}.cs")

    db.commit()
    db.close()
    return project_dir


def test_sanitize_schema_name():
    assert _sanitize_schema_name("nomad-builder") == "nomad_builder"
    assert _sanitize_schema_name("qi") == "qi"
    assert _sanitize_schema_name("123bad") == "_123bad"
    assert _sanitize_schema_name("hello.world") == "hello_world"
    # Reserved names get prefixed
    assert _sanitize_schema_name("main") == "p_main"
    assert _sanitize_schema_name("temp") == "p_temp"
    assert _sanitize_schema_name("") == "_unnamed"


def test_workspace_config_crud(ws_dir):
    """Create, save, load, and modify workspace config."""
    config = WorkspaceConfig(name="test")
    config.save()
    assert config.config_path.exists()

    config.add_project("repo1", "/tmp/repo1")
    config.add_project("repo2", "/tmp/repo2")

    loaded = WorkspaceConfig.load("test")
    assert loaded.name == "test"
    assert len(loaded.projects) == 2
    assert "repo1" in loaded.projects

    config.remove_project("repo1")
    loaded = WorkspaceConfig.load("test")
    assert len(loaded.projects) == 1

    names = WorkspaceConfig.list_all()
    assert "test" in names


def test_workspace_db_attach_and_search(tmp_path, ws_dir):
    """WorkspaceDB attaches multiple project DBs and searches across them."""
    # Create two indexed projects
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        ("Dictionary", "class"),
        ("lookup", "method"),
    ])
    proj2 = _create_indexed_project(tmp_path, "beta", [
        ("Dictionary", "class"),
        ("translate", "method"),
        ("Parser", "class"),
    ])

    # Create workspace config
    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("beta", str(proj2))

    with WorkspaceDB(config) as wdb:
        assert wdb.project_count == 2

        # Search across both projects
        results = wdb.search_symbols("Dictionary")
        assert len(results) >= 2
        projects = {r["project"] for r in results}
        assert "alpha" in projects
        assert "beta" in projects

        # Search with project filter
        results = wdb.search_symbols("Dictionary", project="alpha")
        assert all(r["project"] == "alpha" for r in results)

        # Search for something only in beta
        results = wdb.search_symbols("Parser")
        assert any(r["name"] == "Parser" for r in results)
        assert all(r["project"] == "beta" for r in results if r["name"] == "Parser")


def test_workspace_db_codebase_map(tmp_path, ws_dir):
    """codebase_map aggregates stats across projects."""
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        ("Foo", "class"), ("bar", "method"),
    ])
    proj2 = _create_indexed_project(tmp_path, "beta", [
        ("Baz", "class"), ("qux", "function"), ("quux", "function"),
    ])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("beta", str(proj2))

    with WorkspaceDB(config) as wdb:
        stats = wdb.codebase_map()
        assert stats["workspace"] == "test"
        assert stats["projects_attached"] == 2
        assert stats["totals"]["files"] == 2
        assert stats["totals"]["symbols"] == 5


def test_workspace_db_list_projects(tmp_path, ws_dir):
    """list_projects shows stats for each project."""
    proj1 = _create_indexed_project(tmp_path, "alpha", [("Foo", "class")])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("missing", "/nonexistent/path")

    with WorkspaceDB(config) as wdb:
        projects = wdb.list_projects()
        # alpha should be indexed with stats
        alpha = next(p for p in projects if p["project"] == "alpha")
        assert alpha["files"] == 1
        assert alpha["symbols"] == 1
        # missing should show as unindexed
        missing = next(p for p in projects if p["project"] == "missing")
        assert missing.get("indexed") is False or missing.get("files", 0) == 0


def test_workspace_db_get_symbol(tmp_path, ws_dir):
    """get_symbol returns details from across projects."""
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        ("Dictionary", "class"),
    ])
    proj2 = _create_indexed_project(tmp_path, "beta", [
        ("Dictionary", "class"),
    ])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("beta", str(proj2))

    with WorkspaceDB(config) as wdb:
        results = wdb.get_symbol("Dictionary")
        assert len(results) == 2
        projects = {r["project"] for r in results}
        assert projects == {"alpha", "beta"}

        # Filter by project
        results = wdb.get_symbol("Dictionary", project="beta")
        assert len(results) == 1
        assert results[0]["project"] == "beta"


def test_workspace_db_batch_over_10_projects(tmp_path, ws_dir):
    """Batch iteration handles >10 projects (SQLite ATTACH limit)."""
    import srclight.workspace as ws_mod
    # Temporarily lower the limit to test batching with fewer projects
    orig_limit = ws_mod.MAX_ATTACH
    ws_mod.MAX_ATTACH = 3

    try:
        # Create 5 projects (will need 2 batches of 3)
        projects = {}
        for i in range(5):
            name = f"proj{i}"
            proj = _create_indexed_project(tmp_path, name, [
                (f"Class{i}", "class"),
                (f"method{i}", "method"),
            ])
            projects[name] = proj

        config = WorkspaceConfig(name="batch-test")
        for name, proj_dir in projects.items():
            config.add_project(name, str(proj_dir))

        with WorkspaceDB(config) as wdb:
            assert wdb.project_count == 5

            # list_projects should see all 5
            all_projects = wdb.list_projects()
            indexed = [p for p in all_projects if p.get("files", 0) > 0]
            assert len(indexed) == 5

            # codebase_map should aggregate across all
            stats = wdb.codebase_map()
            assert stats["totals"]["symbols"] == 10  # 2 per project * 5

            # search_symbols should find results across batches
            results = wdb.search_symbols("Class")
            assert len(results) >= 5
            found_projects = {r["project"] for r in results}
            assert len(found_projects) == 5

            # Project filter should work across batch boundaries
            results = wdb.search_symbols("Class4", project="proj4")
            assert len(results) >= 1
            assert all(r["project"] == "proj4" for r in results)

            # get_symbol across batches
            for i in range(5):
                results = wdb.get_symbol(f"Class{i}")
                assert len(results) >= 1
                assert results[0]["project"] == f"proj{i}"
    finally:
        ws_mod.MAX_ATTACH = orig_limit


def test_workspace_db_concurrent_batch_walks_do_not_poison_connection(tmp_path, ws_dir, caplog):
    """Concurrent callers must not corrupt the shared ATTACH state.

    Regression for the web dashboard incident (2026-09-01): the dashboard
    fires several /api/* fetches in parallel, each running a WorkspaceDB
    method on its own thread. Two batch walks interleaving on the one
    connection drift the `_attached` dict away from what SQLite really has
    attached, after which every attach fails with "too many attached
    databases" and every project reports 0 files -- for MCP callers too.
    """
    import logging
    import threading
    import srclight.workspace as ws_mod

    orig_limit = ws_mod.MAX_ATTACH
    ws_mod.MAX_ATTACH = 3
    try:
        config = WorkspaceConfig(name="concurrent-test")
        for i in range(8):  # 3 batches of 3
            proj = _create_indexed_project(tmp_path, f"proj{i}", [
                (f"Class{i}", "class"),
                (f"method{i}", "method"),
            ])
            config.add_project(f"proj{i}", str(proj))

        with WorkspaceDB(config) as wdb:
            errors: list[BaseException] = []
            results: list[list[dict]] = []
            lock = threading.Lock()

            def walk():
                try:
                    for _ in range(5):
                        r = wdb.list_projects()
                        wdb.codebase_map()
                        wdb.search_symbols("Class")
                        with lock:
                            results.append(r)
                except BaseException as e:  # noqa: BLE001
                    with lock:
                        errors.append(e)

            with caplog.at_level(logging.WARNING, logger="srclight.workspace"):
                threads = [threading.Thread(target=walk) for _ in range(6)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            assert not errors, f"concurrent calls raised: {errors[:3]}"
            attach_warnings = [
                r.getMessage() for r in caplog.records if "Failed to attach" in r.getMessage()
            ]
            assert not attach_warnings, attach_warnings[:5]

            # Every concurrent result must be complete.
            for r in results:
                assert sum(1 for p in r if p.get("files", 0) > 0) == 8

            # And the connection must not be left poisoned for later callers.
            after = wdb.list_projects()
            assert sum(1 for p in after if p.get("files", 0) > 0) == 8
            assert wdb.codebase_map()["totals"]["symbols"] == 16
    finally:
        ws_mod.MAX_ATTACH = orig_limit


def test_workspace_stats_are_cached_until_the_index_file_changes(tmp_path, ws_dir):
    """Repeated stat calls must not re-walk (ATTACH + COUNT) 39 databases.

    The dashboard polls /healthz every 10s; on a 39-project workspace a walk
    cost ~4s and held the lock the whole time. Stats are cached per project,
    keyed on the index file's (mtime, size), so a poll is free and a reindex
    is still picked up on the next call.
    """
    import os
    from srclight.db import Database, FileRecord, SymbolRecord

    config = WorkspaceConfig(name="cache-test")
    proj = _create_indexed_project(tmp_path, "cached", [("Alpha", "class")])
    config.add_project("cached", str(proj))

    with WorkspaceDB(config) as wdb:
        assert wdb.list_projects()[0]["symbols"] == 1
        assert wdb.codebase_map()["totals"]["symbols"] == 1

        attaches = []
        real_attach = wdb._attach_batch
        def spy(entries):
            attaches.extend(e.name for e in entries)
            return real_attach(entries)
        wdb._attach_batch = spy

        # Warm cache: no ATTACH, no walk.
        wdb.list_projects(); wdb.codebase_map(); wdb.embedding_stats()
        assert attaches == []

        # Reindex the project: a new symbol lands and the file changes.
        db_path = proj / ".srclight" / "index.db"
        db = Database(db_path); db.open()
        file_id = db.upsert_file(FileRecord(path="src/new.cs", content_hash="zz", mtime=2.0,
                                            language="csharp", size=1, line_count=1))
        db.insert_symbol(SymbolRecord(file_id=file_id, kind="class", name="Beta", qualified_name="cached.Beta",
                                      signature="Beta", start_line=1, end_line=2, content="class Beta {}", line_count=2),
                         "src/new.cs")
        db.commit(); db.close()
        st = db_path.stat()
        os.utime(db_path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

        assert wdb.list_projects()[0]["symbols"] == 2
        assert wdb.codebase_map()["totals"]["files"] == 2


def test_stale_sidecar_is_reported_not_silently_trusted(tmp_path, ws_dir):
    """A sidecar older than its DB must be surfaced, never trusted in silence.

    workspace.vector_search skips is_valid() to save a SQLite connect per query,
    so a sidecar left behind by an interrupted re-embed serves a subset of the
    index for the life of the process while the dashboard, which reads the DB,
    reports 100% coverage. Found live on 2026-09-02: intuition-2019 had 20,648
    embeddings in index.db and 15,611 rows in the sidecar.
    """
    from srclight.embeddings import vector_to_bytes
    from srclight.vector_cache import VectorCache

    proj = _create_indexed_project(tmp_path, "alpha", [
        ("Dictionary", "class"), ("lookup", "method"),
    ])

    db = Database(proj / ".srclight" / "index.db")
    db.open()
    sym_ids = [r["id"] for r in db.conn.execute("SELECT id FROM symbols ORDER BY id")]
    for i, sid in enumerate(sym_ids):
        db.upsert_embedding(sid, "mock:test", 8, vector_to_bytes([0.1 * (i + 1)] * 8), f"h{i}")
    db.commit()
    VectorCache(proj / ".srclight").build_from_db(db.conn)

    # Advance the DB's embedding version, as an interrupted re-embed leaves it.
    db.conn.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('embedding_cache_version', '9999')"
    )
    db.commit()
    db.close()

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj))
    with WorkspaceDB(config) as wdb:
        wdb._get_project_cache("alpha")
        assert wdb.stale_sidecars() == ["alpha"]


def test_fresh_sidecar_is_not_reported_stale(tmp_path, ws_dir):
    """The staleness check must not cry wolf on a sidecar that matches its DB."""
    from srclight.embeddings import vector_to_bytes
    from srclight.vector_cache import VectorCache

    proj = _create_indexed_project(tmp_path, "alpha", [("Dictionary", "class")])
    db = Database(proj / ".srclight" / "index.db")
    db.open()
    sym_ids = [r["id"] for r in db.conn.execute("SELECT id FROM symbols ORDER BY id")]
    for i, sid in enumerate(sym_ids):
        db.upsert_embedding(sid, "mock:test", 8, vector_to_bytes([0.1 * (i + 1)] * 8), f"h{i}")
    db.commit()
    VectorCache(proj / ".srclight").build_from_db(db.conn)
    db.close()

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj))
    with WorkspaceDB(config) as wdb:
        wdb._get_project_cache("alpha")
        assert wdb.stale_sidecars() == []


def _project_with_content(tmp_path: Path, name: str, sym_name: str, body: str) -> Path:
    """A project whose symbol BODY carries a token absent from its name."""
    from srclight.db import Database, FileRecord, SymbolRecord
    project_dir = tmp_path / name
    project_dir.mkdir()
    db_dir = project_dir / ".srclight"
    db_dir.mkdir()
    db = Database(db_dir / "index.db")
    db.open()
    db.initialize()
    fid = db.upsert_file(FileRecord(path=f"src/{name}.py", content_hash="h", mtime=1.0,
                                    language="python", size=len(body), line_count=3))
    db.insert_symbol(SymbolRecord(file_id=fid, kind="function", name=sym_name,
                                  qualified_name=f"{name}.{sym_name}",
                                  start_line=1, end_line=3, content=body,
                                  body_hash="bh"), f"src/{name}.py")
    db.commit()
    db.close()
    return project_dir


def test_workspace_keyword_search_reaches_fts_not_only_the_like_fallback(tmp_path, ws_dir):
    """Workspace search must actually run FTS5, not silently degrade to LIKE.

    All three legs qualified the table in the WHERE clause —
    `WHERE [schema].symbol_content_fts MATCH ?`. SQLite reads a two-part name
    there as table.column, raises `no such column`, and a bare
    `except sqlite3.OperationalError: pass` swallowed it. Every keyword hit came
    from the LIKE fallback on symbols.name, so searching for a token that
    appears only in a symbol's BODY returned nothing at all.
    """
    proj = _project_with_content(
        tmp_path, "alpha", "handler", "def handler():\n    return ZORBLAXSENTINEL\n"
    )
    config = WorkspaceConfig(name="fts-test")
    config.add_project("alpha", str(proj))

    with WorkspaceDB(config) as wdb:
        hits = wdb.search_symbols("ZORBLAXSENTINEL")

    assert hits, "content search returned nothing — the FTS leg never ran"
    assert any(h["source"] == "content" for h in hits), (
        f"no FTS content hit; sources were {[h['source'] for h in hits]}"
    )


def test_a_dead_fts_leg_is_reported_not_silently_swallowed(tmp_path, ws_dir, caplog):
    """A search leg that cannot run must say so once, not degrade in silence.

    The qualifier bug hid for releases behind `except sqlite3.OperationalError:
    pass` in the primary search path. A bare pass cannot tell "this project has
    no FTS tables" from "my SQL is malformed against every project forever", and
    the LIKE fallback answered plausibly enough that nothing ever looked red.
    """
    import logging

    from srclight.db import Database

    proj = _project_with_content(
        tmp_path, "alpha", "handler", "def handler():\n    return ZORBLAXSENTINEL\n"
    )
    db = Database(proj / ".srclight" / "index.db")
    db.open()
    db.conn.execute("DROP TABLE symbol_content_fts")
    db.commit()
    db.close()

    config = WorkspaceConfig(name="fts-warn-test")
    config.add_project("alpha", str(proj))

    with WorkspaceDB(config) as wdb:
        with caplog.at_level(logging.WARNING, logger="srclight.workspace"):
            wdb.search_symbols("ZORBLAXSENTINEL")

    assert any("fts" in r.getMessage().lower() for r in caplog.records), (
        f"dead FTS leg logged nothing; records were {[r.getMessage() for r in caplog.records]}"
    )


def test_semantic_enrichment_does_not_write_to_the_project_index(tmp_path, ws_dir):
    """The semantic-search hot path must open project indexes read-only.

    workspace.py opened `sqlite3.connect(str(entry.index_db))` per project to
    enrich hits — read-write, on the long-lived server. A SELECT-only read-write
    connection is still not a reader: since v0.22.2 Database.close() checkpoints,
    and plain SQLite checkpoints on last-connection close, so an ordinary search
    rewrote index.db and truncated a WAL left by someone else.
    """
    import subprocess
    import sys

    proj = _project_with_content(
        tmp_path, "alpha", "handler", "def handler():\n    return ZORBLAXSENTINEL\n"
    )
    db_path = proj / ".srclight" / "index.db"

    # Leave a hot WAL: a writer that exits without closing.
    subprocess.run(
        [sys.executable, "-c",
         "import sqlite3,os,sys\n"
         "c=sqlite3.connect(sys.argv[1])\n"
         "c.execute('PRAGMA journal_mode=WAL')\n"
         "c.execute('CREATE TABLE IF NOT EXISTS scratch(x)')\n"
         "c.execute('INSERT INTO scratch VALUES (1)')\n"
         "c.commit()\n"
         "os._exit(0)\n", str(db_path)],
        check=True,
    )
    wal = db_path.with_name("index.db-wal")
    assert wal.exists() and wal.stat().st_size > 0, "fixture failed to leave a hot WAL"
    before = (db_path.stat().st_size, wal.stat().st_size)

    config = WorkspaceConfig(name="ro-enrich-test")
    config.add_project("alpha", str(proj))
    with WorkspaceDB(config) as wdb:
        wdb._enrich_workspace_results([("alpha", 0, 0.9, 1)])

    after = (db_path.stat().st_size, wal.stat().st_size if wal.exists() else 0)
    assert after == before, (
        f"enrichment wrote to the project index: {before} -> {after}"
    )


@pytest.mark.parametrize("dirname", [
    "plain", "has space", "issue#42", "what?now", "a+b", "v1.0-100%done", "café",
])
def test_read_only_uri_survives_metacharacters(tmp_path, dirname):
    """The URI must never be built by f-string.

    '#' truncates as a fragment and '?' steals the query string, so mode=ro is
    silently dropped and SQLite opens — or CREATES — a different file,
    read-write, with no error raised. ATTACH then succeeds against an empty
    schema and nothing goes red.
    """
    import sqlite3

    from srclight.workspace import _read_only_uri

    d = tmp_path / dirname
    d.mkdir()
    db_path = d / "index.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE marker(x)")
    conn.execute("INSERT INTO marker VALUES (1)")
    conn.commit()
    conn.close()

    c = sqlite3.connect(":memory:")
    c.execute("ATTACH DATABASE ? AS p", (_read_only_uri(db_path),))
    # the attachment must resolve to the file we meant
    attached = {r[1]: r[2] for r in c.execute("PRAGMA database_list")}
    assert attached["p"] == str(db_path.resolve()), (
        f"attached the wrong file: {attached['p']!r}"
    )
    # and it must actually be read-only
    assert c.execute("SELECT count(*) FROM p.marker").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        c.execute("INSERT INTO p.marker VALUES (2)")
    c.close()


def test_a_missing_project_index_is_an_error_not_a_silently_created_empty_db(tmp_path, ws_dir):
    """A vanished project index must go red, not be recreated empty.

    A plain-path ATTACH of a missing file silently CREATES a zero-byte database
    and attaches an empty schema with no error, so the project reports as
    attached and serves zero symbols. Reachable in normal operation: MAX_ATTACH
    is 10 against 39 projects, so detach/attach runs constantly, and deleting or
    moving a project's index while the server is up made srclight write a fresh
    empty one back into that repo.
    """
    proj = _project_with_content(
        tmp_path, "alpha", "handler", "def handler():\n    return ZORBLAXSENTINEL\n"
    )
    db_path = proj / ".srclight" / "index.db"

    config = WorkspaceConfig(name="missing-index-test")
    config.add_project("alpha", str(proj))

    with WorkspaceDB(config) as wdb:
        entry = next(e for e in wdb._all_indexable if e.name == "alpha")
        wdb._detach_all()
        db_path.unlink()
        for sidecar in ("index.db-wal", "index.db-shm"):
            q = db_path.with_name(sidecar)
            if q.exists():
                q.unlink()

        wdb._attach_batch([entry])

        assert not db_path.exists(), "srclight recreated a deleted project index"
        assert "alpha" in wdb._attach_errors, (
            f"missing index was not reported; errors={wdb._attach_errors}"
        )
