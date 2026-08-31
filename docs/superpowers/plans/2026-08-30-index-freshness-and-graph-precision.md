# Index Freshness + Graph Precision Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make srclight results carry a trustworthy freshness signal (so an agent reaches for the index instead of grepping around it), and measure the reference graph's real precision before deciding on AST work.

**Architecture:** Part A adds a small pure module `freshness.py` (stat-mtime fast path, sha256 fallback, statuses per file), a `check_freshness` MCP probe tool, and stamps an `index_freshness` block into the high-traffic symbol/graph tool results. Part B is a measurement harness (`scripts/measure_graph_precision.py`) that quantifies name-collision ambiguity, comment/string false references, and confidence distribution of the existing `calls` edges — producing a report the pack reviews before any AST-resolution investment.

**Tech Stack:** Python 3.11+, sqlite3 (existing `Database`), hashlib, pytest. No new dependencies.

**Spec:** Pack grains `canes-fideles` grain-0393 (client-seat roadmap: freshness = the unlock) and grain-0395 (correction: type edges exist; re-scope graph work as measure-first). Worklog: `~/Dropbox/personal/Vault/Projects/mcp-chassis/08-Caneslight-0.2.1-Worklog.md`.

## Global Constraints

- Repo: `~/repos/srclight/srclight`, git flow (`master`/`develop`), work on `develop`. Commit per task; never `git add -A` (untracked WIP `src/srclight/learnings.py`, `tests/test_learnings.py` must stay out).
- Run tests with: `~/repos/srclight/srclight/.venv/bin/python -m pytest` (system python lacks deps).
- `files.path` is stored **relative to repo root** (see `indexer.py` ~line 547: `rel_path = str(path.relative_to(root))`).
- `content_hash(data: bytes)` (db.py:1643) is the sha256 helper the indexer uses — freshness MUST reuse it, never reimplement.
- Token-lean outputs: when everything is fresh, the stamp is the single string `"verified-fresh"`; the verbose object appears only when something is stale/missing (absence-is-not-evidence: silence is never the fresh signal).
- Read-only discipline: freshness checks NEVER write to the db (no hash updates, no mtime touch-ups).
- No new MCP tools beyond `check_freshness` (pack ruling, grain-0393: "do not add tools" — this one probe was the explicit exception).
- No version bump / release until the final task; releases follow the estate pattern (develop → master merge --no-ff → annotated tag → push all three). No GitHub Release (that would trigger PyPI publish — Tim's call only).

## File Structure

- `src/srclight/freshness.py` — NEW. Pure freshness logic (statuses, summary, annotate). No FastMCP imports.
- `src/srclight/server.py` — MODIFY. `check_freshness` tool + stamping in 7 existing tools + `index_status` summary.
- `scripts/measure_graph_precision.py` — NEW. Part B harness (standalone script, not part of the package).
- `tests/test_freshness.py` — NEW. Unit tests for freshness.py.
- `tests/test_freshness_tools.py` — NEW. Tool-level tests (direct function calls, like existing tests).
- `tests/test_graph_precision.py` — NEW. Unit tests for the harness's classifier only.
- `README.md` — MODIFY. Freshness section.
- `docs/graph-precision-report.md` — NEW. Part B output, written by the harness run.

---

### Task 1: freshness.py core — per-file status

**Files:**
- Create: `src/srclight/freshness.py`
- Test: `tests/test_freshness.py`

**Interfaces:**
- Consumes: `Database.get_file(path) -> FileRecord | None` (db.py:388), `content_hash(data: bytes) -> str` (db.py:1643), `FileRecord.content_hash`, `FileRecord.mtime`.
- Produces: constants `FRESH = "fresh"`, `STALE = "stale"`, `MISSING = "missing_on_disk"`, `NOT_INDEXED = "not_indexed"`; function `file_freshness(db, repo_root: Path, rel_paths: Iterable[str]) -> dict[str, str]`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_freshness.py
"""Freshness = does the index still describe what is on disk.

The statuses are facts, not judgements: FRESH (index matches disk), STALE
(disk changed since indexing), MISSING (indexed file no longer on disk),
NOT_INDEXED (path was never indexed). The mtime fast path must not produce
false STALE: a touch that does not change bytes is still FRESH.
"""
import os
from pathlib import Path

import pytest

from srclight.db import Database, FileRecord, content_hash
from srclight.freshness import FRESH, MISSING, NOT_INDEXED, STALE, file_freshness


@pytest.fixture
def repo(tmp_path):
    """A tiny on-disk repo plus an index that matches it."""
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()   # open() only connects; initialize() creates the schema
    src = tmp_path / "src"
    src.mkdir()
    f = src / "a.py"
    f.write_text("def a(): pass\n")
    raw = f.read_bytes()
    db.upsert_file(FileRecord(
        path="src/a.py", content_hash=content_hash(raw),
        mtime=f.stat().st_mtime, language="python",
        size=len(raw), line_count=2,
    ))
    db.commit()
    yield db, tmp_path
    db.close()


def test_unchanged_file_is_fresh(repo):
    db, root = repo
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": FRESH}


def test_touched_but_identical_file_is_fresh_via_hash_fallback(repo):
    """mtime moved, bytes did not: the fast path misses, the hash decides."""
    db, root = repo
    f = root / "src" / "a.py"
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 100))
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": FRESH}


def test_edited_file_is_stale(repo):
    db, root = repo
    (root / "src" / "a.py").write_text("def a(): pass  # edited\n")
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": STALE}


def test_deleted_file_is_missing(repo):
    db, root = repo
    (root / "src" / "a.py").unlink()
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": MISSING}


def test_unknown_path_is_not_indexed(repo):
    db, root = repo
    assert file_freshness(db, root, ["src/ghost.py"]) == {"src/ghost.py": NOT_INDEXED}


def test_mtime_fast_path_does_not_read_the_file(repo, monkeypatch):
    """When mtimes match, the file body must not be read (cheapness is the point)."""
    db, root = repo
    def boom(self):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("read_bytes called on the fast path")
    monkeypatch.setattr(Path, "read_bytes", boom)
    assert file_freshness(db, root, ["src/a.py"]) == {"src/a.py": FRESH}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'srclight.freshness'`

- [x] **Step 3: Write the implementation**

```python
# src/srclight/freshness.py
"""Does the index still describe what is on disk?

WHY THIS EXISTS. An index that cannot say it is fresh gets grepped around: on a
repo under active edit, an agent falls back to grep because a silently stale
answer is worse than a slow one (canes-fideles grain-0393, 2026-08-30 — the
same stale-server lesson mcpkit closed for MCP daemons, applied to the index).

READ-ONLY BY DESIGN. Checking freshness never mutates the index — no hash
refresh, no mtime touch-up. A checker that writes is an indexer.

CHEAP BY DESIGN. stat() mtime equality is the fast path (no file read). Only a
moved mtime pays for a read + sha256, and a touch that changed no bytes still
reports FRESH — mtime is a hint, the content hash is the fact.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .db import content_hash

if TYPE_CHECKING:
    from .db import Database

FRESH = "fresh"
STALE = "stale"
MISSING = "missing_on_disk"
NOT_INDEXED = "not_indexed"

__all__ = ["FRESH", "STALE", "MISSING", "NOT_INDEXED", "file_freshness"]


def file_freshness(db: "Database", repo_root: Path, rel_paths: Iterable[str]) -> dict[str, str]:
    """Status per relative path: FRESH / STALE / MISSING / NOT_INDEXED."""
    out: dict[str, str] = {}
    for rel in rel_paths:
        rec = db.get_file(rel)
        if rec is None:
            out[rel] = NOT_INDEXED
            continue
        disk = repo_root / rel
        try:
            st = disk.stat()
        except OSError:
            out[rel] = MISSING
            continue
        # Fast path needs mtime AND size to match — mtime alone misses a
        # same-tick edit (found during Task 1 implementation).
        if st.st_mtime == rec.mtime and st.st_size == rec.size:
            out[rel] = FRESH          # fast path: no read
            continue
        try:
            raw = disk.read_bytes()
        except OSError:
            out[rel] = MISSING
            continue
        out[rel] = FRESH if content_hash(raw) == rec.content_hash else STALE
    return out
```

- [x] **Step 4: Run tests to verify they pass**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness.py -q`
Expected: 6 passed

- [x] **Step 5: Commit**

```bash
cd ~/repos/srclight/srclight
git add src/srclight/freshness.py tests/test_freshness.py
git commit -m "feat(freshness): per-file index freshness — mtime fast path, hash fallback, read-only"
```

---

### Task 2: summary + result annotation

**Files:**
- Modify: `src/srclight/freshness.py`
- Test: `tests/test_freshness.py` (append)

**Interfaces:**
- Consumes: Task 1's `file_freshness`, statuses.
- Produces: `freshness_summary(statuses: dict[str, str], cap: int = 10) -> dict | str` — returns the string `"verified-fresh"` when everything is FRESH, else `{"stale": [...], "missing_on_disk": [...], "not_indexed": [...], "checked": N, "note": "..."}` with each list capped at `cap` (count preserved via `checked` and per-list `"+N more"` sentinel string as final element when capped); `annotate(result: dict, db, repo_root: Path, rel_paths: Iterable[str]) -> dict` — stamps `result["index_freshness"]` and returns result.

- [x] **Step 1: Write the failing tests (append to tests/test_freshness.py)**

```python
from srclight.freshness import annotate, freshness_summary


def test_summary_all_fresh_is_one_short_string():
    assert freshness_summary({"a.py": FRESH, "b.py": FRESH}) == "verified-fresh"


def test_summary_names_stale_files_and_caps_the_list():
    statuses = {f"f{i}.py": STALE for i in range(15)}
    s = freshness_summary(statuses, cap=10)
    assert s["checked"] == 15
    assert len(s["stale"]) == 11            # 10 paths + the "+5 more" sentinel
    assert s["stale"][-1] == "+5 more"
    assert "reindex" in s["note"]


def test_summary_reports_missing_and_not_indexed_separately():
    s = freshness_summary({"gone.py": MISSING, "new.py": NOT_INDEXED})
    assert s["missing_on_disk"] == ["gone.py"]
    assert s["not_indexed"] == ["new.py"]


def test_annotate_stamps_result_in_place(repo):
    db, root = repo
    result = {"symbol": "a"}
    out = annotate(result, db, root, ["src/a.py"])
    assert out is result
    assert out["index_freshness"] == "verified-fresh"


def test_annotate_verbose_when_stale(repo):
    db, root = repo
    (root / "src" / "a.py").write_text("changed\n")
    out = annotate({}, db, root, ["src/a.py"])
    assert out["index_freshness"]["stale"] == ["src/a.py"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness.py -q`
Expected: FAIL with `ImportError: cannot import name 'annotate'`

- [x] **Step 3: Write the implementation (append to freshness.py; add names to __all__)**

```python
def freshness_summary(statuses: dict[str, str], cap: int = 10) -> dict | str:
    """One short string when all fresh; a bounded, explicit object otherwise.

    The SIZE is bounded by the server, never by how much drifted (the mcpkit
    error-cap lesson): lists cap at `cap` paths plus a "+N more" sentinel, and
    `checked` keeps the exact total.
    """
    def bucket(status: str) -> list[str]:
        hits = sorted(p for p, s in statuses.items() if s == status)
        if len(hits) > cap:
            return hits[:cap] + [f"+{len(hits) - cap} more"]
        return hits

    stale, missing, unknown = bucket(STALE), bucket(MISSING), bucket(NOT_INDEXED)
    if not stale and not missing and not unknown:
        return "verified-fresh"
    return {
        "checked": len(statuses),
        "stale": stale,
        "missing_on_disk": missing,
        "not_indexed": unknown,
        "note": (
            "Results for stale/missing files describe the code AS INDEXED, not as it "
            "is now. Reindex (`srclight index`) to refresh, or read the live file."
        ),
    }


def annotate(result: dict, db: "Database", repo_root: Path, rel_paths: Iterable[str]) -> dict:
    """Stamp result['index_freshness'] for the files this result draws on."""
    result["index_freshness"] = freshness_summary(file_freshness(db, repo_root, set(rel_paths)))
    return result
```

Update `__all__`:

```python
__all__ = [
    "FRESH", "STALE", "MISSING", "NOT_INDEXED",
    "file_freshness", "freshness_summary", "annotate",
]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness.py -q`
Expected: 11 passed

- [x] **Step 5: Commit**

```bash
cd ~/repos/srclight/srclight
git add src/srclight/freshness.py tests/test_freshness.py
git commit -m "feat(freshness): bounded summary + result annotation — verbose only when stale"
```

---

### Task 3: check_freshness MCP probe tool

**Files:**
- Modify: `src/srclight/server.py` (add the tool near `index_status`, server.py:1058)
- Test: `tests/test_freshness_tools.py`

**Interfaces:**
- Consumes: Task 2's `file_freshness`, `freshness_summary`; server helpers `_get_db()` (server.py:288), `_repo_root` module global, `_is_workspace_mode()` (server.py:213), `WorkspaceConfig.load(_workspace_name)` + `config.projects.get(project)` (pattern at server.py:2100 in `find_pattern`).
- Produces: MCP tool `check_freshness(paths: list[str] | None = None, project: str | None = None) -> str` returning JSON: `{"index_freshness": <summary>, "checked": N}` — with `paths=None` it checks EVERY file in the index (stat fast path makes this cheap); with paths, just those. Workspace mode requires `project` (same `_project_required_error` pattern as `find_pattern`).

- [x] **Step 1: Write the failing test**

```python
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
    db.initialize()   # open() only connects; initialize() creates the schema
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness_tools.py -q`
Expected: FAIL with `AttributeError: module 'srclight.server' has no attribute 'check_freshness'`

- [x] **Step 3: Implement the tool (insert in server.py directly above `def index_status`, server.py:1058)**

```python
@mcp.tool()
def check_freshness(paths: list[str] | None = None, project: str | None = None) -> str:
    """Is the index current for these files (or the whole index)?

    Compares on-disk files against the index (mtime fast path, content-hash
    fallback; never writes). Use BEFORE trusting symbol results on a repo under
    active edit, or when a result's `index_freshness` flagged staleness.

    Args:
        paths: Repo-relative paths to check. Omit to check every indexed file
               (cheap: unchanged files cost one stat each).
        project: Project name (required in workspace mode).
    """
    from .freshness import file_freshness, freshness_summary

    if _is_workspace_mode():
        if not project:
            return _project_required_error("check_freshness")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        proj_path = config.projects.get(project)
        if not proj_path:
            return _project_not_found_error(project)
        repo_root = Path(proj_path)
        db_path = repo_root / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        try:
            rels = paths if paths is not None else [
                r["path"] for r in db.conn.execute("SELECT path FROM files")
            ]
            statuses = file_freshness(db, repo_root, rels)
        finally:
            db.close()
    else:
        db = _get_db()
        repo_root = _repo_root or Path.cwd()
        rels = paths if paths is not None else [
            r["path"] for r in db.conn.execute("SELECT path FROM files")
        ]
        statuses = file_freshness(db, repo_root, rels)

    return json.dumps(
        {"index_freshness": freshness_summary(statuses), "checked": len(statuses)},
        indent=2,
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness_tools.py -q`
Expected: 3 passed

- [x] **Step 5: Run the FULL suite (the tool touches server import order)**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest -q`
Expected: 234+ passed, 0 failed (231 baseline + the new files)

- [x] **Step 6: Commit**

```bash
cd ~/repos/srclight/srclight
git add src/srclight/server.py tests/test_freshness_tools.py
git commit -m "feat(freshness): check_freshness probe tool — whole-index or per-path, both modes"
```

---

### Task 4: stamp freshness into symbol-reading tools

**Files:**
- Modify: `src/srclight/server.py` — tools `get_symbol`, `get_signature`, `symbols_in_file`, `search_symbols`, `hybrid_search`
- Test: `tests/test_freshness_tools.py` (append)

**Interfaces:**
- Consumes: Task 2's `annotate`. Each target tool already builds a `result` dict and knows the file path(s) its symbols come from (`file` / `file_path` keys in the rows it renders).
- Produces: each listed tool's JSON gains a top-level `"index_freshness"` key (string `"verified-fresh"` or the verbose object). Single-repo mode only in this task: in workspace mode, stamping requires per-project roots per result row — SKIP stamping when `_is_workspace_mode()` and the tool has no single project (stamp when a `project` param resolved one root; otherwise omit the key entirely rather than lie).

**Implementation pattern (apply to each of the five tools, immediately before its final `json.dumps(result, ...)`):**

```python
    # Freshness stamp: the paths this result draws on (single resolved root only).
    _stamp_paths = {m.get("file") or m.get("file_path") for m in matches if isinstance(m, dict)}
    _stamp_paths.discard(None)
    _root = _repo_root if not _is_workspace_mode() else _resolved_project_root  # per-tool variable
    if _root is not None and _stamp_paths:
        from .freshness import annotate
        annotate(result, db, Path(_root), _stamp_paths)
```

(The variable naming differs per tool — `matches`, `symbols`, `results`; the implementer reads each tool body and collects whatever key holds the repo-relative path. Where a tool renders a single symbol (`get_symbol`, `get_signature`), the set is that one file.)

- [x] **Step 1: Write the failing tests (append to tests/test_freshness_tools.py)**

```python
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
```

- [x] **Step 2: Run tests to verify they fail**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness_tools.py -q`
Expected: new tests FAIL with `KeyError: 'index_freshness'` (existing 3 still pass)

- [x] **Step 3: Implement the stamp in the five tools**

Read each tool body in `server.py` (`get_symbol`, `get_signature`, `symbols_in_file`, `search_symbols`, `hybrid_search`), locate its final result-dict construction, collect the repo-relative file paths its rows carry, and apply the pattern above. `symbols_in_file` takes the file path as its own argument — stamp with exactly that path.

- [x] **Step 4: Run the full suite**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest -q`
Expected: all pass, 0 failed. If an existing tool test fails on an unexpected `index_freshness` key, that test asserted full-dict equality — extend the expected dict, do not weaken the stamp.

- [x] **Step 5: Commit**

```bash
cd ~/repos/srclight/srclight
git add src/srclight/server.py tests/test_freshness_tools.py
git commit -m "feat(freshness): stamp index_freshness into symbol-reading tool results"
```

---

### Task 5: stamp freshness into graph tools + index_status summary

**Files:**
- Modify: `src/srclight/server.py` — tools `get_callers`, `get_callees`, `find_dead_code`, and `index_status`
- Test: `tests/test_freshness_tools.py` (append)

**Interfaces:**
- Consumes: same `annotate` pattern as Task 4; graph tool rows carry `file`/`file_path` for each related symbol.
- Produces: `get_callers`/`get_callees`/`find_dead_code` results gain `index_freshness` (union of the files of every symbol in the answer — a stale caller file makes the whole edge list suspect); `index_status` gains `"index_freshness": {"checked": N, "stale_count": N}` whole-index summary (counts only — index_status is the dashboard, not the detail).

- [x] **Step 1: Write the failing tests (append)**

```python
def test_find_dead_code_carries_freshness(wired_repo):
    db, root = wired_repo
    _index_symbol(db, root)
    res = json.loads(_run(server.find_dead_code()))
    assert "index_freshness" in res


def test_index_status_carries_whole_index_counts(wired_repo):
    db, root = wired_repo
    (root / "main.py").write_text("changed\n")
    res = json.loads(_run(server.index_status()))
    assert res["index_freshness"]["stale_count"] == 1
    assert res["index_freshness"]["checked"] == 1
```

- [x] **Step 2: Run tests to verify they fail**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_freshness_tools.py -q`
Expected: the two new tests FAIL (`KeyError: 'index_freshness'`)

- [x] **Step 3: Implement**

Graph tools: apply the Task 4 pattern. For `index_status`, compute counts inline:

```python
    # Whole-index freshness counts (stat fast path; counts only — the probe tool has the detail)
    if _repo_root is not None and not _is_workspace_mode():
        from .freshness import FRESH, file_freshness
        rels = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
        statuses = file_freshness(db, Path(_repo_root), rels)
        stale_n = sum(1 for s in statuses.values() if s != FRESH)
        result["index_freshness"] = {"checked": len(statuses), "stale_count": stale_n}
```

- [x] **Step 4: Run the full suite**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest -q`
Expected: all pass

- [x] **Step 5: Commit**

```bash
cd ~/repos/srclight/srclight
git add src/srclight/server.py tests/test_freshness_tools.py
git commit -m "feat(freshness): graph tools + index_status carry freshness"
```

---

### Task 6: docs — README freshness section

**Files:**
- Modify: `README.md` (insert a `## Index freshness` section directly after the existing `## MCP argument validation` section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the section**

```markdown
## Index freshness

Every symbol/graph result carries `index_freshness`: the short string
`"verified-fresh"` when the files behind the answer are byte-identical to what
was indexed, or a bounded object naming which files are `stale`, missing, or
not indexed. `check_freshness(paths?)` probes any paths — or the whole index —
on demand (unchanged files cost one `stat` each; never writes).

**AI agents:** a result stamped stale describes the code **as indexed**, not as
it is now — reindex (`srclight index`) or read the live file before acting on
line numbers or bodies from it. `"verified-fresh"` is the affirmative signal;
its absence on a workspace-mode result means freshness was not checkable for
that result, never that it is fresh.
```

- [ ] **Step 2: Commit**

```bash
cd ~/repos/srclight/srclight
git add README.md
git commit -m "docs: index freshness — the stamp, the probe, agent semantics"
```

---

### Task 7 (Part B): graph precision harness — classifier

**Files:**
- Create: `scripts/measure_graph_precision.py`
- Test: `tests/test_graph_precision.py`

**Interfaces:**
- Consumes: an existing `.srclight/index.db` (read-only), `re`, `sqlite3`, `json`, `argparse`. NOT part of the srclight package — a standalone script.
- Produces: `classify_reference(content: str, name: str) -> str` returning `"code"`, `"comment_or_string_only"`, or `"absent"`; `measure(db_path: str, sample: int = 500) -> dict` report; CLI `python scripts/measure_graph_precision.py <db_path> [--sample N] [--out report.md]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_precision.py
"""The harness's classifier: is a name occurrence real code or only comment/string?

Line-based heuristic (not AST): a reference counts as code if the name appears
on any line OUTSIDE of '#'/'//' comment tails and string literals. Conservative
by design — when in doubt it says "code" (we are measuring FALSE edges; only a
clear comment/string-only occurrence may count as false).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from measure_graph_precision import classify_reference


def test_plain_call_is_code():
    assert classify_reference("def f():\n    helper()\n", "helper") == "code"


def test_name_only_in_comment_is_flagged():
    assert classify_reference("def f():\n    # calls helper eventually\n    pass\n",
                              "helper") == "comment_or_string_only"


def test_name_only_in_string_is_flagged():
    assert classify_reference('def f():\n    return "helper"\n', "helper") == "comment_or_string_only"


def test_name_in_code_AND_comment_is_code():
    assert classify_reference("def f():\n    helper()  # helper does x\n", "helper") == "code"


def test_absent_name():
    assert classify_reference("def f(): pass\n", "helper") == "absent"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_graph_precision.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'measure_graph_precision'`

- [ ] **Step 3: Write the harness**

```python
#!/usr/bin/env python3
# scripts/measure_graph_precision.py
"""Measure the reference graph's real precision BEFORE investing in AST resolution.

Decision harness, not product code (canes-fideles grain-0395: the graph is a
name-match heuristic — measure how wrong it actually is before rebuilding it).
Reads an index.db, never writes. Reports:
  * ambiguity: fraction of edges whose target NAME maps to >1 symbol
  * comment/string-only references: sampled edges whose name occurrence in the
    source symbol's content is only in comments/strings (a false edge)
  * confidence distribution of edges (how much the <0.2 cross-file drop bites)
  * dead-code sample: symbols with no incoming edge, for human/pack spot-review
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3


def _strip_strings_and_comments(line: str) -> str:
    line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
    line = re.sub(r"'(?:[^'\\]|\\.)*'", "''", line)
    for marker in ("#", "//"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line


def classify_reference(content: str, name: str) -> str:
    """'code' | 'comment_or_string_only' | 'absent' — conservative toward 'code'."""
    word = re.compile(rf"\b{re.escape(name)}\b")
    if not word.search(content):
        return "absent"
    for line in content.split("\n"):
        if word.search(_strip_strings_and_comments(line)):
            return "code"
    return "comment_or_string_only"


def measure(db_path: str, sample: int = 500, seed: int = 42) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total_edges = conn.execute(
        "SELECT COUNT(*) c FROM symbol_edges WHERE edge_type='calls'").fetchone()["c"]

    ambiguous = conn.execute("""
        SELECT COUNT(*) c FROM symbol_edges e JOIN symbols t ON e.target_id = t.id
        WHERE e.edge_type='calls'
          AND t.name IN (SELECT name FROM symbols GROUP BY name HAVING COUNT(*) > 1)
    """).fetchone()["c"]

    conf_rows = conn.execute(
        "SELECT confidence, COUNT(*) c FROM symbol_edges WHERE edge_type='calls' "
        "GROUP BY confidence ORDER BY confidence").fetchall()

    edges = conn.execute("""
        SELECT e.id, s.content, t.name tname FROM symbol_edges e
        JOIN symbols s ON e.source_id = s.id JOIN symbols t ON e.target_id = t.id
        WHERE e.edge_type='calls' AND s.content IS NOT NULL
    """).fetchall()
    random.Random(seed).shuffle(edges)
    sampled = edges[:sample]
    counts = {"code": 0, "comment_or_string_only": 0, "absent": 0}
    false_examples = []
    for row in sampled:
        verdict = classify_reference(row["content"], row["tname"])
        counts[verdict] += 1
        if verdict != "code" and len(false_examples) < 10:
            false_examples.append({"edge_id": row["id"], "target": row["tname"], "verdict": verdict})

    dead_sample = [dict(r) for r in conn.execute("""
        SELECT s.name, s.kind, f.path FROM symbols s
        JOIN files f ON s.file_id = f.id
        LEFT JOIN symbol_edges e ON e.target_id = s.id
        WHERE e.id IS NULL AND s.kind IN ('function','method','class','struct','enum','interface')
        ORDER BY RANDOM() LIMIT 20
    """)]

    n = max(len(sampled), 1)
    return {
        "db": db_path,
        "total_calls_edges": total_edges,
        "ambiguous_target_name_rate": round(ambiguous / max(total_edges, 1), 4),
        "confidence_distribution": {str(r["confidence"]): r["c"] for r in conf_rows},
        "sampled": len(sampled),
        "sampled_verdicts": counts,
        "false_reference_rate_in_sample": round(
            (counts["comment_or_string_only"] + counts["absent"]) / n, 4),
        "false_reference_examples": false_examples,
        "dead_code_sample_for_review": dead_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db_path")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--out", help="write a markdown report here as well")
    a = ap.parse_args()
    report = measure(a.db_path, a.sample)
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            f.write("# Graph precision report\n\n```json\n"
                    + json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest tests/test_graph_precision.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd ~/repos/srclight/srclight
git add scripts/measure_graph_precision.py tests/test_graph_precision.py
git commit -m "feat(scripts): graph precision harness — measure before AST investment"
```

---

### Task 8 (Part B): run the harness, write the report

**Files:**
- Create: `docs/graph-precision-report.md` (harness output)

**Interfaces:**
- Consumes: Task 7's script; srclight's own index at `~/repos/srclight/srclight/.srclight/index.db` (reindex first so the data is current).

- [ ] **Step 1: Reindex srclight itself, then run the harness**

```bash
cd ~/repos/srclight/srclight
.venv/bin/srclight index .
.venv/bin/python scripts/measure_graph_precision.py .srclight/index.db --sample 500 --out docs/graph-precision-report.md
```

Expected: JSON report printed; `docs/graph-precision-report.md` written. Record the headline numbers (ambiguity rate, false-reference rate, confidence distribution).

- [ ] **Step 2: Commit the report**

```bash
git add docs/graph-precision-report.md
git commit -m "docs: graph precision report — the measure-first gate for AST work"
```

- [ ] **Step 3: Surface the numbers**

Print the three headline rates in the final summary for Tim and the pack — the AST-resolution decision (build/skip) is made FROM this report, not in this plan.

---

### Task 9: full-suite gate + release 0.20.4

**Files:**
- Modify: `pyproject.toml` (version `0.20.3` → `0.20.4`)

- [ ] **Step 1: Full suite**

Run: `~/repos/srclight/srclight/.venv/bin/python -m pytest -q`
Expected: 240+ passed, 0 failed

- [ ] **Step 2: Bump version, commit, release (estate pattern — no GitHub Release)**

```bash
cd ~/repos/srclight/srclight
sed -i 's/^version = "0.20.3"/version = "0.20.4"/' pyproject.toml
git add pyproject.toml
git commit -m "release: 0.20.4 — index freshness (stamps + probe) and graph precision harness"
git push origin develop
git checkout master && git merge --no-ff develop -m "Release 0.20.4: index freshness + graph precision harness"
git tag -a v0.20.4 -m "srclight v0.20.4: index_freshness stamps, check_freshness probe, precision harness"
git push origin master && git push origin v0.20.4
git checkout develop
```

- [ ] **Step 3: Land the closing grain**

Through the caneslight MCP (`pack_learn`, dog=gromit, dimension=work): freshness shipped (what the stamp/probe are, the "absence on workspace results means unchecked, never fresh" rule), harness numbers headline, AST decision handed back to Tim + pack. Include `origin_repo=srclight/srclight` and the release commit sha.

## Self-Review Notes

- Workspace-mode stamping is deliberately partial (Task 4): stamp only where one project root is resolved; omit the key otherwise — an unverifiable "fresh" would be the exact lie this plan exists to kill. README documents that absence means "not checkable", never "fresh".
- `upsert_file(FileRecord) -> int` (db.py:367), `insert_symbol(SymbolRecord, file_path) -> int` (db.py:427), `insert_edge(EdgeRecord) -> int` (db.py:826) — verified against db.py 2026-08-30; an earlier draft wrote `insert_file`, which does not exist. All server tools stamped are plain `def` (sync), so the tests' direct calls need no asyncio.
- `check_freshness` whole-index mode on srclight itself (~600 files) costs ~600 stats ≈ milliseconds; on the largest workspace project it stays under a second. No caps needed.
- Part B classifier is line-heuristic, not AST — acceptable because it powers a one-time measurement report, not a product feature; conservative toward "code" so the false-reference rate is a floor, stated as such in the report.
