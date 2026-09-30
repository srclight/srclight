"""The index run reports the phases after the file scan.

Building the call graph and finding communities can take minutes on a large
project; a run that printed nothing in between looked stuck.
"""
import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer

from .test_workspace import ws_dir  # noqa: F401  (fixture re-export)


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


def test_a_log_record_ends_the_progress_line_first():
    """The call graph's summary is logged while the progress line is still
    open; printed there, it continued that line."""
    import logging

    from srclight.cli import _ProgressLine

    seen = []

    class _Probe(logging.Handler):
        def emit(self, record):
            seen.append(line.open)

    probe = _Probe()
    logging.getLogger().addHandler(probe)
    try:
        with _ProgressLine("  ", 10) as line:
            line.progress("a.c", 1, 2)
            logging.getLogger("srclight.indexer").warning("Call graph: 1 edges in 0s")
    finally:
        logging.getLogger().removeHandler(probe)
    assert seen == [False]
    assert not probe.filters


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


def test_log_lines_carry_the_time():
    """The gap between two lines of an index run says which phase was slow."""
    import logging
    import re

    from srclight.cli import LOG_DATEFMT, LOG_FORMAT

    record = logging.LogRecord("srclight.indexer", logging.INFO, __file__, 1,
                               "Call graph: %d edges in %.0fs", (12, 3.0), None)
    line = logging.Formatter(LOG_FORMAT, LOG_DATEFMT).format(record)
    assert re.fullmatch(r"\d\d:\d\d:\d\d INFO srclight\.indexer: Call graph: 12 edges in 3s",
                        line), line


def test_each_phase_is_timed_from_the_one_before(capsys):
    """Only the call graph logged its duration: the others had to be read
    off the gap between two timestamps."""
    from srclight.cli import _ProgressLine

    ticks = iter([0.0, 12.5, 30.0, 31.5, 40.0])
    with _ProgressLine("  ", 10, clock=lambda: next(ticks)) as line:
        line.progress("a.c", 1, 2)
        line.phase("Building the call graph")
        line.phase("Finding communities and execution flows")
        line.phase("Rebuilding the vector cache")
    assert line.durations() == [
        ("Indexing files", 12.5),
        ("Building the call graph", 17.5),
        ("Finding communities and execution flows", 1.5),
        ("Rebuilding the vector cache", 8.5),
    ]
    summary = line.summary()
    assert summary[0].split() == ["Indexing", "files", "12.5s"]
    assert len({len(s) for s in summary}) == 1, "durations line up on the right"


def test_phase_lines_carry_the_time(capsys):
    import re

    from srclight.cli import _ProgressLine

    with _ProgressLine("  ", 10) as line:
        line.phase("Building the call graph")
    out = capsys.readouterr().out
    # Formatted and placed like the time on a log line, so the two align.
    assert re.search(r"^\d\d:\d\d:\d\d Building the call graph\.\.\.$", out, re.M), out


def test_the_cli_summary_lists_the_phases(tmp_path):
    from click.testing import CliRunner

    from srclight.cli import main

    (tmp_path / "a.py").write_text(
        "def alpha():\n    return beta()\n\n\ndef beta():\n    return 1\n")
    result = CliRunner().invoke(main, ["index", str(tmp_path), "--no-embed"])
    assert result.exit_code == 0, result.output
    after_time = result.output.split("  Time:", 1)[1]
    assert "Indexing files" in after_time and "Building the call graph" in after_time, result.output


@pytest.mark.parametrize("embed", [False, True])
def test_the_work_after_the_last_phase_is_a_phase_of_its_own(tmp_path, monkeypatch, embed):
    """Committing and folding the WAL back into index.db can take long on a
    large first index: timed as part of the phase before, it made that one
    look slow."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return beta()\n\n\ndef beta():\n    return 1\n")
    if embed:
        from srclight import embeddings as embeddings_mod
        from srclight.embeddings import vector_to_bytes

        class _Stub:
            name = "stub:model"
            dimensions = 3

        monkeypatch.setattr(embeddings_mod, "get_provider", lambda spec, **kw: _Stub())
        monkeypatch.setattr(embeddings_mod, "embed_symbols",
                            lambda provider, symbols, on_progress=None: [
                                (s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols])
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    phases = []
    config = IndexConfig(root=root, embed_model="stub:model" if embed else None,
                         disable_embeddings=not embed)
    Indexer(db, config).index(root, on_phase=phases.append)
    db.close()
    assert phases[-1] == "Saving the index", phases


def _small_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return beta()\n\n\ndef beta():\n    return 1\n")
    return root


def test_the_file_pass_ends_on_what_it_found(tmp_path):
    """Left on the last file it showed, the progress line said nothing
    about the pass."""
    root = _small_repo(tmp_path)
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    progress = []
    Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index(
        root, on_progress=lambda label, cur, tot: progress.append((label, cur, tot)))
    db.close()
    files = [p for p in progress if p[0] != "call graph"]
    assert files[-1] == ("done: 1 indexed, 0 unchanged", 1, 1), files


@pytest.mark.parametrize("follows_phases", [False, True])
def test_the_run_summary_is_logged_where_nothing_else_prints_it(tmp_path, caplog, follows_phases):
    """The CLI prints its own summary; the MCP tool and the git hook have
    only the log."""
    import logging

    caplog.set_level(logging.DEBUG, logger="srclight.indexer")
    root = _small_repo(tmp_path)
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index(
        root, on_phase=(lambda name: None) if follows_phases else None)
    db.close()
    for opening in ("Indexed ", "Indexing "):
        [record] = [r for r in caplog.records if r.getMessage().startswith(opening)]
        assert record.levelno == (logging.DEBUG if follows_phases else logging.INFO), opening


def _stub_embeddings(monkeypatch, batches=1):
    from srclight import embeddings as embeddings_mod
    from srclight.embeddings import vector_to_bytes

    class _Stub:
        name = "stub:model"
        dimensions = 3

    def embed(provider, symbols, on_progress=None):
        for n in range(1, batches + 1):
            if on_progress:
                on_progress(n, batches)
        return [(s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols]

    monkeypatch.setattr(embeddings_mod, "get_provider", lambda spec, **kw: _Stub())
    monkeypatch.setattr(embeddings_mod, "embed_symbols", embed)


@pytest.mark.parametrize("follows_phases", [False, True])
def test_the_embedded_count_is_logged_where_nothing_else_prints_it(
        tmp_path, monkeypatch, caplog, follows_phases):
    """The CLI's summary gives it already ("N embedded now")."""
    import logging

    _stub_embeddings(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="srclight.indexer")
    root = _small_repo(tmp_path)
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, embed_model="stub:model")).index(
        root, on_phase=(lambda name: None) if follows_phases else None)
    db.close()
    [record] = [r for r in caplog.records if r.getMessage().startswith("Embedded ")]
    assert record.levelno == (logging.DEBUG if follows_phases else logging.INFO)


@pytest.mark.parametrize("batches", [1, 2])
def test_batch_progress_is_logged_only_for_several_batches(tmp_path, monkeypatch, caplog, batches):
    """"batch 1/1 (0s elapsed, ~0s remaining)" said nothing the next line did not."""
    import logging

    _stub_embeddings(monkeypatch, batches)
    caplog.set_level(logging.INFO, logger="srclight.indexer")
    root = _small_repo(tmp_path)
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, embed_model="stub:model")).index(root)
    db.close()
    logged = [r.getMessage() for r in caplog.records if "Embedding batch" in r.getMessage()]
    assert len(logged) == (0 if batches == 1 else 2), logged


def test_the_call_graph_log_gives_the_edges_the_index_holds(tmp_path, caplog):
    """The insert count also held the duplicates the table ignores, so the
    log and the summary gave two different figures for one graph."""
    import logging
    import re

    caplog.set_level(logging.INFO, logger="srclight.indexer")
    root = tmp_path / "repo"
    root.mkdir()
    # Both bases resolve to the one class named Base: two inserts, one edge.
    (root / "a.py").write_text(
        "class Base:\n    pass\n\n\nclass Child(one.Base, two.Base):\n    pass\n")
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    indexer = Indexer(db, IndexConfig(root=root, disable_embeddings=True))
    indexer.index(root)
    held = db.stats()["edges"]
    db.close()
    [logged] = re.findall(r"Call graph: (\d+) edges", caplog.text)
    assert int(logged) == held
    # The closing line, all the MCP tool and the git hook get, too.
    [closing] = re.findall(r"Indexed \d+ files \(\d+ symbols, (\d+) edges\)", caplog.text)
    assert int(closing) == held


def test_a_graph_rebuilt_empty_is_still_said_rebuilt(tmp_path):
    """Any file indexed clears the graph and rebuilds it; without the note, an
    empty result read as the old graph kept."""
    from click.testing import CliRunner

    from srclight.cli import main

    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
    result = CliRunner().invoke(main, ["index", str(tmp_path), "--no-embed"])
    assert result.exit_code == 0, result.output
    assert "  Edges:       0 in the index, call graph rebuilt this run" in result.output


def test_the_workspace_time_is_the_phases_added_up(tmp_path, ws_dir, monkeypatch):  # noqa: F811
    """The indexer's elapsed time stops before its closing checkpoint, which
    the "Saving the index" phase the workspace run prints includes."""
    from click.testing import CliRunner

    from srclight import cli
    from srclight.workspace import WorkspaceConfig

    monkeypatch.setattr(cli._ProgressLine, "total", lambda self: 123.4)
    config = WorkspaceConfig(name="time-ws")
    config.add_project("alpha", str(_small_repo(tmp_path)))
    config.save()
    result = CliRunner().invoke(cli.main, ["workspace", "index", "-w", "time-ws", "--no-embed"])
    assert result.exit_code == 0, result.output
    assert " edges in the index, 123.4s" in result.output


def test_the_file_scan_is_announced_like_the_phases_after_it(capsys):
    import re

    from srclight.cli import _ProgressLine

    with _ProgressLine("  ", 10):
        pass
    out = capsys.readouterr().out
    assert re.fullmatch(r"\d\d:\d\d:\d\d Indexing files\.\.\.\n", out), out


def test_nothing_to_embed_is_said(tmp_path, monkeypatch, caplog):
    import logging

    from srclight import embeddings as embeddings_mod
    from srclight.embeddings import vector_to_bytes

    class _Stub:
        name = "stub:model"
        dimensions = 3

    monkeypatch.setattr(embeddings_mod, "get_provider", lambda spec, **kw: _Stub())
    monkeypatch.setattr(embeddings_mod, "embed_symbols",
                        lambda provider, symbols, on_progress=None: [
                            (s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols])
    root = _small_repo(tmp_path)
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, embed_model="stub:model")).index(root)
    caplog.set_level(logging.INFO, logger="srclight.indexer")
    caplog.clear()
    Indexer(db, IndexConfig(root=root, embed_model="stub:model")).index(root)
    db.close()
    assert "No symbols to embed" in caplog.text


def test_the_summary_counts_files_and_symbols_on_one_line_each(tmp_path):
    from click.testing import CliRunner

    from srclight.cli import main

    root = _small_repo(tmp_path)
    result = CliRunner().invoke(main, ["index", str(root), "--no-embed"])
    assert result.exit_code == 0, result.output
    assert "  Files:       1 scanned, 1 indexed, 0 unchanged, 0 removed, 0 errors" in result.output
    assert "  Symbols:     2 extracted, 2 in the index" in result.output
    assert "  Edges:       1 in the index, call graph rebuilt this run" in result.output


def test_the_workspace_summary_keeps_every_figure_of_the_run(tmp_path, ws_dir):  # noqa: F811
    """The indexer's closing line is only a debug line for a caller that
    follows the phases: the workspace summary must carry what it said —
    files removed, errors and edges included."""
    from click.testing import CliRunner

    from srclight.cli import main
    from srclight.workspace import WorkspaceConfig

    project = _small_repo(tmp_path)
    (project / "gone.py").write_text("def gone():\n    return 0\n")
    config = WorkspaceConfig(name="summary-ws")
    config.add_project("alpha", str(project))
    config.save()
    assert CliRunner().invoke(main, ["workspace", "index", "-w", "summary-ws", "--no-embed"]
                              ).exit_code == 0
    (project / "gone.py").unlink()
    result = CliRunner().invoke(main, ["workspace", "index", "-w", "summary-ws", "--no-embed"])
    assert result.exit_code == 0, result.output
    assert "1 files: 0 indexed, 1 unchanged, 1 removed, 0 errors; 0 symbols, " in result.output
    assert " edges in the index, " in result.output


def test_the_file_pass_counts_the_files_that_failed(tmp_path, monkeypatch):
    """Indexed, unchanged and failed add up to the files the pass went
    through; without the failures, the line did not."""
    root = _small_repo(tmp_path)
    (root / "b.py").write_text("def other():\n    return 2\n")
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    indexer = Indexer(db, IndexConfig(root=root, disable_embeddings=True))
    real = indexer._extract_symbols

    def fails_on_b(file_id, rel_path, source, lang):
        if rel_path.endswith("b.py"):
            raise RuntimeError("unreadable")
        return real(file_id, rel_path, source, lang)

    monkeypatch.setattr(indexer, "_extract_symbols", fails_on_b)
    progress = []
    indexer.index(root, on_progress=lambda label, cur, tot: progress.append((label, cur, tot)))
    db.close()
    files = [p for p in progress if p[0] != "call graph"]
    assert files[-1] == ("done: 1 indexed, 0 unchanged, 1 failed", 2, 2), files


def test_the_total_is_the_phases_added_up():
    """The indexer's elapsed time stops before its closing checkpoint, which
    the last phase includes: printed as the total, the phases could add up
    to more than it."""
    from srclight.cli import _ProgressLine

    ticks = iter([0.0, 12.0, 30.0, 95.0])
    with _ProgressLine("  ", 10, clock=lambda: next(ticks)) as line:
        line.phase("Building the call graph")
        line.phase("Saving the index")
    assert line.total() == 95.0 == sum(s for _, s in line.durations())
