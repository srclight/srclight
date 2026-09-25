"""The index run reports the phases after the file scan.

Building the call graph and finding communities can take minutes on a large
project; a run that printed nothing in between looked stuck.
"""
from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


def test_the_phases_after_the_scan_are_reported(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return beta()\n\n\ndef beta():\n    return 1\n")
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    phases, progress = [], []
    Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index(
        root, on_progress=lambda label, cur, tot: progress.append((label, cur, tot)),
        on_phase=phases.append)
    db.close()
    assert phases[0].startswith("Building the call graph"), phases
    graph = [p for p in progress if p[0] == "call graph"]
    assert graph and graph[-1][1] == graph[-1][2], progress
