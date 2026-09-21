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


def test_index_status_names_the_indexed_extensions_in_workspace_mode(tmp_path, monkeypatch):
    """`find_pattern`'s note sends the caller here, in either mode."""
    import srclight.workspace as ws_mod
    monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", tmp_path / "workspaces")

    project = tmp_path / "alpha"
    project.mkdir()
    (project / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (project / ".srclight").mkdir()
    project_db = Database(project / ".srclight" / "index.db")
    project_db.open()
    project_db.initialize()
    Indexer(project_db, IndexConfig(root=project)).index()
    project_db.close()

    config = ws_mod.WorkspaceConfig(name="gaps")
    config.add_project("alpha", str(project))
    monkeypatch.setattr(server, "_workspace_name", "gaps")
    monkeypatch.setattr(server, "_workspace_db", None)
    monkeypatch.setattr(server, "_workspace_config_mtime", None)

    res = json.loads(_run(server.index_status()))

    assert ".py" in res["indexed_extensions"]


def test_a_file_too_big_to_index_is_reported_as_a_gap(tmp_path, db):
    """The size limit is srclight's own choice, not the project's."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (root / "generated.py").write_text("x = 1\n" * 500)

    Indexer(db, IndexConfig(root=root, max_file_size=100)).index()

    assert db.get_oversize_skipped() == 1


def test_index_status_reports_oversize_files(repo_with_a_gap, db):
    res = json.loads(_run(server.index_status()))

    assert res["oversize_skipped"] == 0


def test_find_pattern_reports_oversize_files_it_never_read(tmp_path, db, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (root / "generated.py").write_text("def draw_outline():\n    return 1\n" * 200)
    Indexer(db, IndexConfig(root=root, max_file_size=200)).index()
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_repo_root", root)
    monkeypatch.setattr(server, "_workspace_name", None)

    res = json.loads(_run(server.find_pattern(pattern="draw_outline")))

    assert res["oversize_skipped"] == 1
    assert "truncated" in res["unindexed_note"]


def test_config_and_extensionless_files_are_not_gaps(tmp_path, db):
    """A gap means code that was never read, not repo furniture.

    Every repo carries a LICENSE, a lockfile and some YAML. Counting them
    leaves the tally non-empty everywhere, which puts the warning on every
    answer and buries the extensions that genuinely hold unread code.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (root / "LICENSE").write_text("MIT\n")
    (root / "Dockerfile").write_text("FROM python\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "settings.json").write_text("{}\n")
    (root / "ci.yml").write_text("on: push\n")
    (root / "outline.wgsl").write_text("fn main() {}\n")

    Indexer(db, IndexConfig(root=root)).index()

    assert db.get_unindexed_extensions() == {".wgsl": 1}


def test_workspace_index_status_reports_oversize_files(tmp_path, monkeypatch):
    """`find_pattern`'s note points at index_status in workspace mode too."""
    import srclight.workspace as ws_mod
    monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", tmp_path / "workspaces")

    project = tmp_path / "alpha"
    project.mkdir()
    (project / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (project / "generated.py").write_text("x = 1\n" * 500)
    (project / ".srclight").mkdir()
    project_db = Database(project / ".srclight" / "index.db")
    project_db.open()
    project_db.initialize()
    Indexer(project_db, IndexConfig(root=project, max_file_size=100)).index()
    project_db.close()

    config = ws_mod.WorkspaceConfig(name="gaps")
    config.add_project("alpha", str(project))
    monkeypatch.setattr(server, "_workspace_name", "gaps")
    monkeypatch.setattr(server, "_workspace_db", None)
    monkeypatch.setattr(server, "_workspace_config_mtime", None)

    res = json.loads(_run(server.index_status()))

    assert res["projects"][0]["oversize_skipped"] == 1


def test_a_document_format_this_install_cannot_read_is_a_gap(tmp_path, db, monkeypatch):
    """`pip install srclight` without the extras still has to say so.

    The extras are how PDF/DOCX/XLSX/HTML get read. Without them those
    files are neither indexed nor — since their patterns stay in the ignore
    list — counted, so the default install reported a whole-tree answer
    over a repo whose documents it never opened.
    """
    from srclight import extractors

    monkeypatch.delitem(extractors.DOCUMENT_EXTENSIONS, ".pdf", raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (root / "manual.pdf").write_bytes(b"%PDF-1.4\n")

    Indexer(db, IndexConfig(root=root)).index()

    assert db.get_unindexed_extensions() == {".pdf": 1}


def test_an_unreadable_document_inside_an_ignored_tree_is_not_a_gap(tmp_path, db, monkeypatch):
    """The extractor exemption must not defeat directory-level exclusions."""
    from srclight import extractors

    monkeypatch.delitem(extractors.DOCUMENT_EXTENSIONS, ".pdf", raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    vendored = root / "third_party" / "zlib"
    vendored.mkdir(parents=True)
    (vendored / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    (root / "own.pdf").write_bytes(b"%PDF-1.4\n")

    Indexer(db, IndexConfig(root=root)).index()

    assert db.get_unindexed_extensions() == {".pdf": 1}


def test_a_skipped_extension_inside_an_ignored_tree_is_not_a_gap(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    vendored = root / "third_party" / "zlib"
    vendored.mkdir(parents=True)
    (vendored / "table.inc").write_text("0, 1, 2\n")

    Indexer(db, IndexConfig(root=root, extension_overrides={".inc": "skip"})).index()

    assert db.get_unindexed_extensions() == {}


def test_every_conditionally_ignored_format_has_a_pattern():
    """The list claims these suffixes carry an ignore pattern. Check it.

    An entry with no pattern is never reached through the ignore path and
    only muddles the two reasons a document goes unread.
    """
    from srclight.extractors import CONDITIONALLY_IGNORED_DOCUMENT_EXTENSIONS
    from srclight.indexer import DEFAULT_IGNORE

    for ext in CONDITIONALLY_IGNORED_DOCUMENT_EXTENSIONS:
        assert f"*{ext}" in DEFAULT_IGNORE, f"{ext} is listed but nothing ignores it"


def test_no_readable_document_format_is_left_ignored():
    """An installed extractor's format must not stay hidden behind a pattern.

    Catches the drift the hand-maintained list invites: a new extractor
    whose extension sits in DEFAULT_IGNORE, with nothing to lift it.
    """
    from srclight.extractors import (
        CONDITIONALLY_IGNORED_DOCUMENT_EXTENSIONS,
        DOCUMENT_EXTENSIONS,
    )
    from srclight.indexer import DEFAULT_IGNORE, IndexConfig, Indexer

    db_patterns = Indexer(None, IndexConfig()).config.ignore_patterns
    for ext in DOCUMENT_EXTENSIONS:
        if f"*{ext}" in DEFAULT_IGNORE:
            assert f"*{ext}" not in db_patterns or ext in CONDITIONALLY_IGNORED_DOCUMENT_EXTENSIONS
