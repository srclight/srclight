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
