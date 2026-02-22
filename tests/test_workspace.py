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
