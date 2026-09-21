"""Declaring extra extensions for an index.

Default detection stays conservative, so a project with a house extension
needs a way to say what it holds — once, not on every run: the git hooks
reindex without flags, and an override that did not survive them would be
lost on the next commit.
"""

import asyncio
import json

import pytest

import srclight.server as server
from srclight.cli import parse_extension_overrides
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
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "helpers.zz").write_text("def draw_outline():\n    return 1\n")
    return root


def test_a_declared_extension_is_indexed_as_its_language(repo, db):
    Indexer(db, IndexConfig(root=repo, extension_overrides={".zz": "python"})).index()

    rows = db.conn.execute(
        """SELECT s.name FROM symbols s JOIN files f ON s.file_id = f.id
           WHERE f.path = 'helpers.zz'"""
    ).fetchall()

    assert {r["name"] for r in rows} == {"draw_outline"}


def test_a_declared_extension_is_not_reported_as_a_gap(repo, db):
    Indexer(db, IndexConfig(root=repo, extension_overrides={".zz": "python"})).index()

    assert db.get_unindexed_extensions() == {}


def test_an_undeclared_extension_is_still_skipped(repo, db):
    Indexer(db, IndexConfig(root=repo)).index()

    assert db.get_unindexed_extensions() == {".zz": 1}


def test_the_declaration_survives_a_later_run_that_passes_no_flags(repo, db):
    Indexer(db, IndexConfig(root=repo, extension_overrides={".zz": "python"})).index()

    (repo / "more.zz").write_text("def draw_frame():\n    return 2\n")
    Indexer(db, IndexConfig(root=repo)).index()

    rows = db.conn.execute(
        """SELECT s.name FROM symbols s JOIN files f ON s.file_id = f.id
           WHERE f.path = 'more.zz'"""
    ).fetchall()

    assert {r["name"] for r in rows} == {"draw_frame"}


def test_index_status_lists_the_declared_extension(repo, db, tmp_path, monkeypatch):
    Indexer(db, IndexConfig(root=repo, extension_overrides={".zz": "python"})).index()
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_db_path", tmp_path / "index.db")
    monkeypatch.setattr(server, "_repo_root", repo)
    monkeypatch.setattr(server, "_workspace_name", None)

    res = json.loads(_run(server.index_status()))

    assert ".zz" in res["indexed_extensions"]


def test_parse_extension_overrides_normalizes_the_extension():
    parsed = parse_extension_overrides((".zz=python", "INC=cpp"))

    assert parsed == {".zz": "python", ".inc": "cpp"}


def test_an_empty_declaration_clears_the_stored_one(repo, db):
    Indexer(db, IndexConfig(root=repo, extension_overrides={".zz": "python"})).index()

    Indexer(db, IndexConfig(root=repo, extension_overrides={})).index()

    assert db.get_extension_overrides() == {}
    assert db.get_unindexed_extensions() == {".zz": 1}


def test_parse_extension_overrides_clears_on_none():
    assert parse_extension_overrides(("none",)) == {}


def test_parse_extension_overrides_rejects_an_unknown_language():
    with pytest.raises(ValueError, match="klingon"):
        parse_extension_overrides((".zz=klingon",))


def test_parse_extension_overrides_rejects_a_malformed_pair():
    with pytest.raises(ValueError, match="EXT=LANGUAGE"):
        parse_extension_overrides((".zz",))


def test_a_declaration_made_through_the_api_is_normalized(repo, db):
    """The CLI normalizes; IndexConfig is public and must agree with it."""
    Indexer(db, IndexConfig(root=repo, extension_overrides={"ZZ": "python"})).index()

    rows = db.conn.execute(
        """SELECT s.name FROM symbols s JOIN files f ON s.file_id = f.id
           WHERE f.path = 'helpers.zz'"""
    ).fetchall()

    assert {r["name"] for r in rows} == {"draw_outline"}
    assert db.get_extension_overrides() == {".zz": "python"}


def test_an_extension_can_be_declared_unreadable(tmp_path, db):
    """Sniffing gives `.inc` a language; a project must be able to say no."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "rules.inc").write_text("include config.mk\nall: build\n")

    Indexer(db, IndexConfig(root=root, extension_overrides={".inc": "skip"})).index()

    assert db.conn.execute("SELECT COUNT(*) n FROM files").fetchone()["n"] == 0
    assert db.get_unindexed_extensions() == {".inc": 1}


def test_parse_extension_overrides_accepts_skip():
    assert parse_extension_overrides((".inc=skip",)) == {".inc": "skip"}
