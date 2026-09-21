"""Scan gaps as measured on a git repo — the walk almost every index takes.

`git ls-files` supplies the file list whenever one is available, and that
branch reads a different set of rules than the plain directory walk. A gap
report measured only on the rglob branch says nothing about what real
projects see.
"""

import subprocess

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def git_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    try:
        _git(root, "init")
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip("git not available")
    return root


def test_ignored_files_are_not_reported_as_gaps(git_repo, db):
    """A tracked binary is excluded on purpose, not a gap in the scan."""
    (git_repo / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (git_repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (git_repo / "sprite.ttf").write_bytes(b"\x00\x01\x00\x00")
    (git_repo / "frames.dat").write_text("0 1 2\n")
    _git(git_repo, "add", "-A")

    Indexer(db, IndexConfig(root=git_repo)).index()

    assert db.get_unindexed_extensions() == {".dat": 1}


def test_a_vendored_tree_is_not_reported_as_a_gap(git_repo, db):
    (git_repo / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    vendored = git_repo / "third_party" / "zlib"
    vendored.mkdir(parents=True)
    (vendored / "notes.rst7").write_text("vendored\n")
    _git(git_repo, "add", "-A")

    Indexer(db, IndexConfig(root=git_repo)).index()

    assert db.get_unindexed_extensions() == {}


def test_a_clean_git_repo_reports_no_gap_at_all(git_repo, db):
    """The affirmative signal has to be reachable, or it says nothing."""
    (git_repo / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    (git_repo / "README.md").write_text("# shapes\n")
    _git(git_repo, "add", "-A")

    Indexer(db, IndexConfig(root=git_repo)).index()

    assert db.get_unindexed_extensions() == {}


def test_a_skipped_extension_inside_a_vendored_tree_is_not_a_gap(git_repo, db):
    (git_repo / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    vendored = git_repo / "third_party" / "zlib"
    vendored.mkdir(parents=True)
    (vendored / "table.inc").write_text("0, 1, 2\n")
    _git(git_repo, "add", "-A")

    Indexer(db, IndexConfig(root=git_repo, extension_overrides={".inc": "skip"})).index()

    assert db.get_unindexed_extensions() == {}


def test_an_unreadable_document_in_a_vendored_tree_is_not_a_gap(git_repo, db, monkeypatch):
    from srclight import extractors

    monkeypatch.delitem(extractors.DOCUMENT_EXTENSIONS, ".pdf", raising=False)
    (git_repo / "shapes.py").write_text("def draw_outline():\n    return 1\n")
    vendored = git_repo / "third_party" / "zlib"
    vendored.mkdir(parents=True)
    (vendored / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    _git(git_repo, "add", "-A")

    Indexer(db, IndexConfig(root=git_repo)).index()

    assert db.get_unindexed_extensions() == {}
