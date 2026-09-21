"""Tests for include-fragment files (.inc, .inl, .ipp, .tcc).

Some C and C++ projects split an oversized translation unit into `.inc`
fragments that the main `.cpp` includes at file scope. The fragments hold
real function definitions, so an index that skips them answers from the
headers alone — declarations without bodies — and says nothing about the
gap.
"""

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer
from srclight.languages import detect_language


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def test_inc_with_cpp_method_definitions_is_cpp(tmp_path):
    path = tmp_path / "shape_draw.inc"
    path.write_text('''\
void Shape_c::drawFrame(int frame) {
    mFrame = frame;
}
''')

    assert detect_language(path) == "cpp"


def test_inc_holding_php_is_php(tmp_path):
    path = tmp_path / "config.inc"
    path.write_text('''\
<?php

function db_dsn() {
    return getenv('DSN');
}
''')

    assert detect_language(path) == "php"


def test_inc_holding_plain_c_is_c(tmp_path):
    path = tmp_path / "clamp.inc"
    path.write_text('''\
static int clamp_frame(int frame) {
    return frame < 0 ? 0 : frame;
}
''')

    assert detect_language(path) == "c"


@pytest.mark.parametrize("name", ["matrix.inl", "vector.ipp", "span.tcc"])
def test_cpp_only_fragment_extensions_are_cpp(tmp_path, name):
    path = tmp_path / name
    path.write_text("template <class T>\nT twice(T v) { return v + v; }\n")

    assert detect_language(path) == "cpp"


def test_symbols_in_an_inc_are_indexed_against_the_inc_itself(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shape.cpp").write_text('''\
#include "shape.h"

#include "shape_draw.inc"

int Shape_c::create() {
    return 1;
}
''')
    (root / "shape_draw.inc").write_text('''\
void Shape_c::drawFrame(int frame) {
    mFrame = frame;
}

void Shape_c::drawOutline(int frame) {
    drawFrame(frame);
}
''')

    indexer = Indexer(db, IndexConfig(root=root))
    indexer.index()

    rows = db.conn.execute(
        """SELECT s.name, s.start_line FROM symbols s
           JOIN files f ON s.file_id = f.id
           WHERE f.path = 'shape_draw.inc'"""
    ).fetchall()
    lines = {r["name"]: r["start_line"] for r in rows}

    assert "Shape_c::drawFrame" in lines
    assert "Shape_c::drawOutline" in lines
    # The line is the one in the fragment, not in the including .cpp.
    assert lines["Shape_c::drawFrame"] == 1

    # And the definition is what a name search reaches, not only the header
    # declaration it would otherwise answer with.
    hits = db.search_symbols("drawOutline")
    assert "shape_draw.inc" in {h["file"] for h in hits}


def test_a_caller_inside_an_inc_is_visible_in_the_graph(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shape.cpp").write_text('''\
#include "shape_draw.inc"

void reset_outline() {
    g_outline = 0;
}
''')
    (root / "shape_draw.inc").write_text('''\
void start_outline() {
    reset_outline();
}
''')

    indexer = Indexer(db, IndexConfig(root=root))
    indexer.index()

    rows = db.conn.execute(
        """SELECT sf.path AS source_file, s.name AS source_name
           FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN files sf ON s.file_id = sf.id
           JOIN symbols t ON e.target_id = t.id
           WHERE t.name = 'reset_outline'"""
    ).fetchall()

    assert "shape_draw.inc" in {r["source_file"] for r in rows}


def test_sniffing_reads_only_the_head_of_a_large_fragment(tmp_path):
    """`.inc` is also used for generated data tables, which can be huge.

    Detection runs before the indexer's size limit, so loading the file to
    read its first line has no guard in front of it.
    """
    import tracemalloc

    big = tmp_path / "table.inc"
    big.write_text("static const int table[] = {\n" + "0x00,\n" * 800_000)

    tracemalloc.start()
    try:
        assert detect_language(big) == "c"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000, f"loaded {peak} bytes to read a 4 KB head"


def test_inc_opening_with_a_php_short_echo_tag_is_php(tmp_path):
    path = tmp_path / "row.inc"
    path.write_text("<?= $row['name'] ?>\n<?= $row['size'] ?>\n")

    assert detect_language(path) == "php"
