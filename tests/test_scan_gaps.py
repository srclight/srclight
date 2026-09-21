"""What the index did NOT scan, and the tools saying so.

An incomplete answer that declares itself incomplete is usable; an
incomplete answer that declares itself complete is not. The indexer records
the extensions it walked past, and the tools that report completeness carry
that record.
"""

import asyncio
import json

import pytest

import srclight.server as server
from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


def _run(coro_or_val):
    return asyncio.run(coro_or_val) if asyncio.iscoroutine(coro_or_val) else coro_or_val


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def repo_with_a_gap(tmp_path, db, monkeypatch):
    """A repo holding one indexed file and two files of an unknown extension."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (root / "frames.dat").write_text("0 1 2\n")
    (root / "palette.dat").write_text("3 4 5\n")

    Indexer(db, IndexConfig(root=root)).index()

    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_db_path", tmp_path / "index.db")
    monkeypatch.setattr(server, "_repo_root", root)
    monkeypatch.setattr(server, "_workspace_name", None)
    return root


def test_indexing_records_the_extensions_it_walked_past(repo_with_a_gap, db):
    assert db.get_unindexed_extensions() == {".dat": 2}


def test_indexing_records_nothing_when_every_file_was_indexed(tmp_path, db):
    root = tmp_path / "clean"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")

    Indexer(db, IndexConfig(root=root)).index()

    assert db.get_unindexed_extensions() == {}


def test_a_later_run_replaces_the_previous_record(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "frames.dat").write_text("0 1 2\n")
    Indexer(db, IndexConfig(root=root)).index()

    (root / "frames.dat").unlink()
    Indexer(db, IndexConfig(root=root)).index()

    assert db.get_unindexed_extensions() == {}


def test_index_status_names_what_it_indexes_and_what_it_skipped(repo_with_a_gap):
    res = json.loads(_run(server.index_status()))

    assert res["unindexed_extensions"] == {".dat": 2}
    # The supported list is what makes the gap discoverable without a grep.
    assert ".py" in res["indexed_extensions"]
    assert ".inc" in res["indexed_extensions"]
    assert ".dat" not in res["indexed_extensions"]


def test_find_pattern_reports_the_files_it_never_scanned(repo_with_a_gap):
    res = json.loads(_run(server.find_pattern(pattern="draw_outline")))

    assert res["truncated"] is False
    assert res["unindexed_extensions"] == {".dat": 2}
    assert "truncated" in res["unindexed_note"]


def test_a_workspace_project_carries_its_own_scan_gaps(tmp_path, monkeypatch):
    """Workspace mode is where a project's gap has to travel per project."""
    import srclight.workspace as ws_mod
    monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", tmp_path / "workspaces")

    project = tmp_path / "alpha"
    project.mkdir()
    (project / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (project / "frames.dat").write_text("0 1 2\n")
    (project / ".srclight").mkdir()
    project_db = Database(project / ".srclight" / "index.db")
    project_db.open()
    project_db.initialize()
    Indexer(project_db, IndexConfig(root=project)).index()
    project_db.close()

    config = ws_mod.WorkspaceConfig(name="gaps")
    config.add_project("alpha", str(project))
    with ws_mod.WorkspaceDB(config) as wdb:
        rows = wdb.list_projects()

    assert rows[0]["unindexed_extensions"] == {".dat": 1}


def test_find_pattern_stays_quiet_when_nothing_was_skipped(tmp_path, db, monkeypatch):
    root = tmp_path / "clean"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    Indexer(db, IndexConfig(root=root)).index()
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_repo_root", root)
    monkeypatch.setattr(server, "_workspace_name", None)

    res = json.loads(_run(server.find_pattern(pattern="draw_outline")))

    assert "unindexed_extensions" not in res
    assert "unindexed_note" not in res
