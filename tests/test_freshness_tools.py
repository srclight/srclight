# tests/test_freshness_tools.py
"""Tool-level freshness: called the way other server tests call tools (directly)."""
import asyncio
import json

import pytest

import srclight.server as server
from srclight.db import Database, FileRecord, content_hash


@pytest.fixture
def wired_repo(tmp_path, monkeypatch):
    """Point the server's single-repo globals at a tiny real repo + index."""
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    f = tmp_path / "main.py"
    f.write_text("def main(): pass\n")
    raw = f.read_bytes()
    db.upsert_file(FileRecord(path="main.py", content_hash=content_hash(raw),
                              mtime=f.stat().st_mtime, language="python",
                              size=len(raw), line_count=2))
    db.commit()
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_repo_root", tmp_path)
    monkeypatch.setattr(server, "_workspace_name", None)
    yield db, tmp_path
    db.close()


def _run(coro_or_val):
    return asyncio.run(coro_or_val) if asyncio.iscoroutine(coro_or_val) else coro_or_val


def test_check_freshness_all_files_fresh(wired_repo):
    res = json.loads(_run(server.check_freshness()))
    assert res["index_freshness"] == "verified-fresh"
    assert res["checked"] == 1


def test_check_freshness_reports_the_edited_file(wired_repo):
    db, root = wired_repo
    (root / "main.py").write_text("def main(): pass  # edited\n")
    res = json.loads(_run(server.check_freshness()))
    assert res["index_freshness"]["stale"] == ["main.py"]


def test_check_freshness_specific_paths(wired_repo):
    res = json.loads(_run(server.check_freshness(paths=["main.py", "ghost.py"])))
    assert res["index_freshness"]["not_indexed"] == ["ghost.py"]


# --- Task 4: stamps in symbol-reading tools ---

def _index_symbol(db, root):
    """Give the index one symbol in main.py so symbol tools return it."""
    from srclight.db import SymbolRecord
    fid = db.get_file("main.py").id
    db.insert_symbol(SymbolRecord(
        file_id=fid, kind="function", name="main",
        start_line=1, end_line=1, content="def main(): pass",
        line_count=1,
    ), "main.py")
    db.commit()


def test_get_symbol_carries_freshness(wired_repo):
    db, root = wired_repo
    _index_symbol(db, root)
    res = json.loads(_run(server.get_symbol(name="main")))
    assert res["index_freshness"] == "verified-fresh"


def test_get_symbol_flags_stale_source_file(wired_repo):
    db, root = wired_repo
    _index_symbol(db, root)
    (root / "main.py").write_text("def main(): pass  # drifted\n")
    res = json.loads(_run(server.get_symbol(name="main")))
    assert res["index_freshness"]["stale"] == ["main.py"]


def test_search_symbols_carries_freshness(wired_repo):
    db, root = wired_repo
    _index_symbol(db, root)
    res = json.loads(_run(server.search_symbols(query="main")))
    assert "index_freshness" in res


# --- Task 5: graph tools + index_status ---

def test_find_dead_code_carries_freshness(wired_repo):
    db, root = wired_repo
    # NOT named "main": get_dead_symbols excludes entry-point names, and an
    # excluded symbol would leave the dead list empty — nothing to stamp.
    from srclight.db import SymbolRecord
    fid = db.get_file("main.py").id
    db.insert_symbol(SymbolRecord(
        file_id=fid, kind="function", name="orphan_fn",
        start_line=3, end_line=3, content="def orphan_fn(): pass",
        line_count=1,
    ), "main.py")
    db.commit()
    res = json.loads(_run(server.find_dead_code()))
    assert res["total_unreferenced"] >= 1
    assert "index_freshness" in res


def test_index_status_carries_whole_index_counts(wired_repo):
    db, root = wired_repo
    (root / "main.py").write_text("changed\n")
    res = json.loads(_run(server.index_status()))
    assert res["index_freshness"]["stale_count"] == 1
    assert res["index_freshness"]["checked"] == 1
