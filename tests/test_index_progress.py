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


def test_the_embedding_steps_are_reported(tmp_path, monkeypatch):
    from srclight import embeddings as embeddings_mod
    from srclight.embeddings import vector_to_bytes

    class _Stub:
        name = "stub:model"
        dimensions = 3

    monkeypatch.setattr(embeddings_mod, "get_provider", lambda spec, **kw: _Stub())
    monkeypatch.setattr(embeddings_mod, "embed_symbols", lambda provider, symbols, on_progress=None: [
        (s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols])
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return 1\n")
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    phases = []
    Indexer(db, IndexConfig(root=root, embed_model="stub:model")).index(root, on_phase=phases.append)
    db.close()
    assert "Embedding new and changed symbols" in phases
    assert any(p.startswith("Saving ") and p.endswith(" embeddings") for p in phases), phases
    assert "Rebuilding the vector cache" in phases


def test_the_cli_prints_no_blank_line_between_steps(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from srclight.cli import main

    from srclight import embeddings as embeddings_mod
    from srclight.embeddings import vector_to_bytes

    class _Stub:
        name = "stub:model"
        dimensions = 3

    monkeypatch.setattr(embeddings_mod, "get_provider", lambda spec, **kw: _Stub())
    monkeypatch.setattr(embeddings_mod, "embed_symbols", lambda provider, symbols, on_progress=None: [
        (s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols])
    (tmp_path / "a.py").write_text("def alpha():\n    return beta()\n\n\ndef beta():\n    return 1\n")
    result = CliRunner().invoke(main, ["index", str(tmp_path), "--embed", "stub:model"])
    assert result.exit_code == 0, result.output
    lines = result.output.split("\n")
    steps = [i for i, line in enumerate(lines) if line.strip().endswith("...")]
    assert steps, result.output
    for i in steps:
        assert lines[i - 1].strip(), result.output
