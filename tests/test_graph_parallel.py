"""The call graph scanned over several processes is the one a single scan builds."""
from collections import Counter

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer, _graph_workers


def _project(root):
    root.mkdir()
    (root / "shapes.hpp").write_text(
        "struct Circle {\n    int radius;\n    int area(int scale);\n};\n"
        "int draw_circle(Circle* c);\n")
    (root / "shapes.cpp").write_text(
        '#include "shapes.hpp"\n\n'
        "int Circle::area(int scale) { return radius * radius * scale; }\n\n"
        "int draw_circle(Circle* c) { return c->area(2) + helper_value(); }\n")
    for n in range(12):
        (root / f"unit{n}.cpp").write_text(
            '#include "shapes.hpp"\n\n'
            f"int helper_value{n}(void) {{ return {n}; }}\n\n"
            f"int unit_entry{n}(Circle* c) {{\n"
            f"    return draw_circle(c) + helper_value{n}() + c->area({n});\n}}\n")
    (root / "tools.py").write_text(
        "def parse_input(text):\n    return text.strip()\n\n\n"
        "def run_tool(text):\n    return parse_input(text)\n")


def _edges(tmp_path, monkeypatch, workers):
    root = tmp_path / f"repo{workers}"
    _project(root)
    db = Database(tmp_path / f"index{workers}.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, disable_embeddings=True,
                            graph_workers=workers)).index(root)
    rows = db.conn.execute(
        """SELECT s.qualified_name AS source, t.qualified_name AS target,
                  e.confidence, e.resolution
           FROM symbol_edges e JOIN symbols s ON s.id = e.source_id
           JOIN symbols t ON t.id = e.target_id WHERE e.edge_type = 'calls'""").fetchall()
    db.close()
    return Counter(tuple(r) for r in rows)


def test_parallel_scan_builds_the_same_graph(tmp_path, monkeypatch, caplog):
    single = _edges(tmp_path, monkeypatch, 1)
    assert single, "the project must have calls for the comparison to mean anything"
    assert _edges(tmp_path, monkeypatch, 2) == single
    # Not the fallback to a single scan, which would pass the comparison too.
    assert "in parallel" not in caplog.text


def test_a_failed_pool_falls_back_to_a_single_scan(tmp_path, monkeypatch, caplog):
    from concurrent.futures.process import BrokenProcessPool

    def broken(*args, **kwargs):
        raise BrokenProcessPool("no processes here")

    single = _edges(tmp_path, monkeypatch, 1)
    monkeypatch.setattr(Indexer, "_scan_in_processes", staticmethod(broken))
    assert _edges(tmp_path, monkeypatch, 2) == single
    assert "in parallel" in caplog.text


def test_a_library_caller_scans_in_its_own_process_by_default(tmp_path, monkeypatch):
    """Worker processes re-import the caller's `__main__`: a script without
    the guard would run again in each. Only callers that ask get them — not
    even the environment variable turns them on behind a script's back."""
    def must_not_start(*args, **kwargs):
        raise AssertionError("worker processes started for a library caller")

    monkeypatch.setenv("SRCLIGHT_GRAPH_WORKERS", "4")
    monkeypatch.setattr(Indexer, "_scan_in_processes", staticmethod(must_not_start))
    root = tmp_path / "repo"
    _project(root)
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root, disable_embeddings=True)).index(root)
    db.close()


def test_the_cli_lets_the_cpus_choose(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from srclight import cli

    seen = []

    class _Recording(Indexer):
        def __init__(self, db, config):
            seen.append(config.graph_workers)
            super().__init__(db, config)

    monkeypatch.setattr("srclight.indexer.Indexer", _Recording)
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
    result = CliRunner().invoke(cli.main, ["index", str(tmp_path), "--no-embed"])
    assert result.exit_code == 0, result.output
    assert seen == [0]


def test_the_frozen_entry_point_lets_workers_start(monkeypatch):
    """A frozen build starts graph workers as its own executable with
    `--multiprocessing-fork`; without freeze_support() first, the CLI rejects
    the flag and every pool breaks."""
    import multiprocessing
    import runpy
    from pathlib import Path

    from srclight import cli

    calls = []
    monkeypatch.setattr(multiprocessing, "freeze_support", lambda: calls.append("freeze_support"))
    monkeypatch.setattr(cli, "main", lambda: calls.append("main"))
    entry = Path(__file__).resolve().parent.parent / "packaging" / "pyinstaller" / "entry_point.py"
    runpy.run_path(str(entry), run_name="__main__")
    assert calls == ["freeze_support", "main"]


@pytest.mark.parametrize("configured,symbols,expected", [
    ("1", 10**6, 1),
    ("3", 10, 3),
    ("zero", 10, 1),
    ("", 10, 1),
])
def test_worker_count(monkeypatch, configured, symbols, expected):
    monkeypatch.setenv("SRCLIGHT_GRAPH_WORKERS", configured)
    assert _graph_workers(symbols) == expected


@pytest.mark.parametrize("cpus,expected", [(1, 1), (2, 1), (4, 2), (12, 6), (16, 8), (64, 8)])
def test_a_large_index_uses_one_worker_per_physical_core_up_to_a_cap(monkeypatch, cpus, expected):
    monkeypatch.delenv("SRCLIGHT_GRAPH_WORKERS", raising=False)
    monkeypatch.delattr("os.sched_getaffinity", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: cpus)
    assert _graph_workers(10**6) == expected


@pytest.mark.parametrize("os_name,expected", [("nt", 61), ("posix", 100)])
def test_a_configured_count_stays_within_what_the_platform_allows(monkeypatch, os_name, expected):
    monkeypatch.setenv("SRCLIGHT_GRAPH_WORKERS", "100")
    monkeypatch.setattr("os.name", os_name)
    assert _graph_workers(10**6) == expected


def test_the_cpus_the_process_may_use_count_where_known(monkeypatch):
    monkeypatch.delenv("SRCLIGHT_GRAPH_WORKERS", raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(6)), raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert _graph_workers(10**6) == 3
