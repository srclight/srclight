# tests/test_edge_resolution.py
"""Ranked edge resolution: the graph must prefer evidence over fan-out.

End-to-end through the real Indexer (the house pattern from test_indexer.py):
real files on disk, real parse, real edge build. Symbol names are chosen to
clear the edge builder's noise filters (len >= 4, not in NOISE_NAMES).
"""
import sqlite3

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    # a.py: dup_worker defined here AND in b.py; caller_one refers to it from
    # the same file -> the same_file tier must beat b.py's candidate.
    # ghost_func is defined here and referenced ONLY inside a comment in c.py.
    (src / "a.py").write_text(
        "def dup_worker():\n    return 1\n\n"
        "def ghost_func():\n    return 2\n\n"
        "def caller_one():\n    return dup_worker()\n"
    )
    # b.py: a second dup_worker (the ambiguity), plus dup_solo defined TWICE in
    # this one file -> candidates collapse to a single file (unique_file tier).
    (src / "b.py").write_text(
        "def dup_worker():\n    return 3\n\n"
        "def dup_solo(x):\n    return x\n\n"
        "def dup_solo(x, y):\n    return x + y\n"
    )
    # c.py: refers to dup_worker (two candidate FILES, no better evidence ->
    # name_only keeps the ranked fan-out) and dup_solo (unique_file).
    # ghost_func appears ONLY in a comment -> masking must prevent any edge.
    (src / "c.py").write_text(
        "# ghost_func is mentioned only in this comment\n"
        "def other_caller():\n    dup_solo(1)\n    return dup_worker()\n"
    )
    return src


def _edges(db, source_name: str, target_name: str) -> list[sqlite3.Row]:
    return db.conn.execute(
        """SELECT e.resolution, ft.path AS target_file
           FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           JOIN files ft ON t.file_id = ft.id
           WHERE e.edge_type='calls' AND s.name = ? AND t.name = ?""",
        (source_name, target_name),
    ).fetchall()


def test_same_file_candidate_wins(db, project):
    Indexer(db, IndexConfig(root=project)).index(project)
    rows = _edges(db, "caller_one", "dup_worker")
    assert rows, "caller_one -> dup_worker edge must exist"
    assert all(r["resolution"] == "same_file" for r in rows)
    assert all(r["target_file"] == "a.py" for r in rows), "b.py's dup_worker must be excluded"


def test_unique_file_candidates_collapse(db, project):
    Indexer(db, IndexConfig(root=project)).index(project)
    rows = _edges(db, "other_caller", "dup_solo")
    assert rows
    assert all(r["resolution"] == "unique_file" for r in rows)
    assert {r["target_file"] for r in rows} == {"b.py"}


def test_truly_ambiguous_keeps_ranked_fanout_as_name_only(db, project):
    Indexer(db, IndexConfig(root=project)).index(project)
    rows = _edges(db, "other_caller", "dup_worker")
    assert {r["target_file"] for r in rows} == {"a.py", "b.py"}, "the candidate LIST is kept"
    assert all(r["resolution"] == "name_only" for r in rows)


def test_comment_only_reference_creates_no_edge(db, project):
    Indexer(db, IndexConfig(root=project)).index(project)
    assert _edges(db, "other_caller", "ghost_func") == [], (
        "a name appearing only in a comment must not become an edge")


def test_get_callers_surfaces_resolution(db, project):
    """Task 4: the label must reach the tool layer, not just the table."""
    Indexer(db, IndexConfig(root=project)).index(project)
    dup = db.conn.execute(
        """SELECT s.id FROM symbols s JOIN files f ON s.file_id=f.id
           WHERE s.name='dup_worker' AND f.path='a.py'"""
    ).fetchone()
    callers = db.get_callers(dup["id"])
    assert callers
    assert all(c["resolution"] == "same_file" for c in callers
               if c["symbol"].name == "caller_one")


def test_get_callees_surfaces_resolution(db, project):
    Indexer(db, IndexConfig(root=project)).index(project)
    other = db.conn.execute(
        "SELECT id FROM symbols WHERE name='other_caller'").fetchone()
    callees = db.get_callees(other["id"])
    assert callees
    assert all("resolution" in c for c in callees)


def test_one_edge_keeps_its_strongest_route():
    from srclight.indexer import _strongest_edges

    edges = [(1, 2, 0.9, "name_only"), (1, 2, 0.9, "unique_file"), (1, 3, 0.5, "same_dir"),
             (1, 3, 0.9, "name_only")]
    assert sorted(_strongest_edges(edges)) == [(1, 2, 0.9, "unique_file"), (1, 3, 0.9, "name_only")]
    assert sorted(_strongest_edges(list(reversed(edges)))) == sorted(_strongest_edges(edges))


_SEED_PROBE = r"""
import sys
from pathlib import Path
from srclight.db import Database
from srclight.indexer import Indexer, IndexConfig
root = Path(sys.argv[1])
db = Database(root / ".srclight" / "index.db"); db.open(); db.initialize()
Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index(root)
for r in db.conn.execute(
    "SELECT s.name, t.name, f.path, e.resolution FROM symbol_edges e"
    " JOIN symbols s ON s.id = e.source_id JOIN symbols t ON t.id = e.target_id"
    " JOIN files f ON f.id = t.file_id ORDER BY 1, 2, 3"):
    print(tuple(r))
"""


def test_the_stored_resolution_does_not_follow_the_hash_seed(tmp_path):
    """Iterating a set of names made the stored route depend on PYTHONHASHSEED."""
    import os
    import subprocess
    import sys

    outputs = set()
    for seed in ("1", "2", "5"):
        root = tmp_path / f"seed{seed}"
        (root / "src").mkdir(parents=True)
        (root / "src/lamp.hpp").write_text("struct Lamp_c {\n  void glowLamp(int x);\n};\n")
        (root / "src/lamp.cpp").write_text('#include "lamp.hpp"\nvoid Lamp_c::glowLamp(int x) {}\n')
        (root / "src/user.cpp").write_text(
            '#include "lamp.hpp"\nvoid userCall(Lamp_c* lamp) {\n'
            "  Lamp_c::glowLamp(1);\n  lamp->glowLamp(2);\n}\n")
        (root / ".srclight").mkdir()
        result = subprocess.run(
            [sys.executable, "-c", _SEED_PROBE, str(root)], capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed}, check=True)
        outputs.add(result.stdout)
    assert len(outputs) == 1, outputs
