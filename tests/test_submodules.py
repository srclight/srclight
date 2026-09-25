"""Files in a git submodule are indexed.

`git ls-files` lists a submodule as its directory alone: its files were
never seen. They are listed from inside each submodule that is checked out.
"""
import subprocess

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


def _git(cwd, *args):
    subprocess.run(["git", "-c", "protocol.file.allow=always", *args], cwd=str(cwd),
                   check=True, capture_output=True)


def _repo(path, files: dict[str, str]):
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@test.com")
    _git(path, "config", "user.name", "Test User")
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")


@pytest.fixture
def with_submodule(tmp_path):
    _repo(tmp_path / "parts", {"toolkit.py": "def blend_colors(a, b):\n    return a\n"})
    main = tmp_path / "main"
    _repo(main, {"app.py": "def paint():\n    return blend_colors(1, 2)\n"})
    _git(main, "submodule", "add", "-q", str(tmp_path / "parts"), "vendor/parts")
    _git(main, "commit", "-q", "-m", "add submodule")
    return tmp_path


def _index(root):
    db = Database(root.parent / f"{root.name}.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index()
    return db


def test_a_checked_out_submodule_is_indexed(with_submodule):
    db = _index(with_submodule / "main")
    paths = {r[0].replace("\\", "/") for r in db.conn.execute("SELECT path FROM files")}
    assert {"app.py", "vendor/parts/toolkit.py"} <= paths
    callees = {r[0] for r in db.conn.execute(
        """SELECT b.name FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id WHERE a.name = 'paint'""")}
    db.close()
    assert "blend_colors" in callees


def test_a_submodule_not_checked_out_is_skipped(with_submodule):
    clone = with_submodule / "clone"
    _git(with_submodule, "clone", "-q", str(with_submodule / "main"), str(clone))
    db = _index(clone)
    paths = {r[0].replace("\\", "/") for r in db.conn.execute("SELECT path FROM files")}
    db.close()
    assert paths == {"app.py"}


def test_git_history_of_a_submodule_file_is_read_in_the_submodule(with_submodule):
    from srclight.git import blame_symbol, changes_to_file, detect_changes

    main = with_submodule / "main"
    assert len(changes_to_file(main, "vendor/parts/toolkit.py")) == 1
    assert "error" not in blame_symbol(main, "vendor/parts/toolkit.py", 1, 2)

    (main / "vendor/parts/toolkit.py").write_text(
        "def blend_colors(a, b):\n    return b\n")
    changed = {c["file"]: c["hunks"] for c in detect_changes(main)}
    assert "vendor/parts" not in changed
    assert changed["vendor/parts/toolkit.py"] == [
        {"old_start": 2, "old_count": 1, "new_start": 2, "new_count": 1}]


def test_paths_that_are_not_ascii_are_indexed(tmp_path):
    _repo(tmp_path / "parts", {"tool.py": "def polish():\n    return 1\n"})
    main = tmp_path / "main"
    _repo(main, {"naïve.py": "def trim():\n    return 1\n"})
    _git(main, "submodule", "add", "-q", str(tmp_path / "parts"), "vendor/café")
    _git(main, "commit", "-q", "-m", "add submodule")
    db = _index(main)
    paths = {r[0].replace("\\", "/") for r in db.conn.execute("SELECT path FROM files")}
    db.close()
    assert {"naïve.py", "vendor/café/tool.py"} <= paths


def test_a_file_in_a_folder_is_found_with_either_separator(with_submodule):
    db = _index(with_submodule / "main")
    for path in ("vendor/parts/toolkit.py", "vendor\\parts\\toolkit.py"):
        assert [s.name for s in db.symbols_in_file(path)] == ["blend_colors"], path
    db.close()
