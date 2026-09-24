"""A C/C++ function whose #if/#else branches each open a brace.

    #if PLATFORM_DESKTOP
        if (a || b) {
    #else
        if (a) {
    #endif
            ...
        }

The branches share one closing brace, which is fine for the compiler — it
only ever sees one branch. tree-sitter does not run the preprocessor: it sees
two opening braces for one closing brace, wraps the function in an ERROR node,
and the function never became a symbol. Its callers, its callees and its body
vanished from every query, with nothing to say so.
"""
import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer

GAUGE_CPP = """\
void Gauge_c::resetNeedle() {
    resetGauge();
}

void Gauge_c::updateState() {
    mFlags = 0;
#if PLATFORM_DESKTOP
    if (!isPanelVisible() || isPromptOpen()) {
#else
    if (!isPanelVisible()) {
#endif
        mFlags |= 0x1;
    } else if (isBusy()) {
        mFlags |= 0x2;
    }
    switch (readDisplayMode()) {
    case 0:
        break;
    }
}

void Gauge_c::moveNeedle() {
    resetGauge();
}
"""

HELPERS_CPP = """\
void resetGauge() {}
int isPanelVisible() { return 1; }
int isPromptOpen() { return 0; }
int isBusy() { return 0; }
int readDisplayMode() { return 1; }
"""


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    yield db
    db.close()


def _index(tmp_path, db, files: dict[str, str]):
    root = tmp_path / "repo"
    root.mkdir()
    for name, text in files.items():
        (root / name).write_text(text)
    Indexer(db, IndexConfig(root=root)).index()


def _symbols(db, path: str) -> list[tuple[str, int, int]]:
    rows = db.conn.execute(
        """SELECT s.name, s.start_line, s.end_line FROM symbols s
           JOIN files f ON s.file_id = f.id WHERE f.path = ?
           ORDER BY s.start_line, s.name""",
        (path,),
    ).fetchall()
    return [(r["name"], r["start_line"], r["end_line"]) for r in rows]


def test_the_function_is_extracted_with_its_real_lines(tmp_path, db):
    _index(tmp_path, db, {"gauge.cpp": GAUGE_CPP, "helpers.cpp": HELPERS_CPP})

    assert ("Gauge_c::updateState", 5, 20) in _symbols(db, "gauge.cpp")


def test_the_functions_around_it_are_unchanged(tmp_path, db):
    _index(tmp_path, db, {"gauge.cpp": GAUGE_CPP, "helpers.cpp": HELPERS_CPP})

    assert _symbols(db, "gauge.cpp") == [
        ("Gauge_c::resetNeedle", 1, 3),
        ("Gauge_c::updateState", 5, 20),
        ("Gauge_c::moveNeedle", 22, 24),
    ]


def test_its_calls_reach_the_graph(tmp_path, db):
    _index(tmp_path, db, {"gauge.cpp": GAUGE_CPP, "helpers.cpp": HELPERS_CPP})

    callees = {r["name"] for r in db.conn.execute(
        """SELECT t.name FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name = 'Gauge_c::updateState'"""
    )}

    assert {"readDisplayMode", "isBusy", "isPromptOpen"} <= callees


def test_its_stored_body_keeps_both_branches(tmp_path, db):
    """The reparse only exists to find the function; what get_symbol shows
    is the file as written."""
    _index(tmp_path, db, {"gauge.cpp": GAUGE_CPP, "helpers.cpp": HELPERS_CPP})

    body = db.conn.execute(
        "SELECT content FROM symbols WHERE name = 'Gauge_c::updateState'"
    ).fetchone()["content"]

    assert "#if PLATFORM_DESKTOP" in body
    assert "if (!isPanelVisible()) {" in body


def test_elif_and_nested_conditionals(tmp_path, db):
    _index(tmp_path, db, {"panel.cpp": """\
void Panel_c::updateFrame() {
#if PLATFORM_DESKTOP
    if (isPanelVisible()) {
#elif PLATFORM_CONSOLE
  #ifdef WIDE_SCREEN
    if (isPromptOpen()) {
  #else
    if (isBusy()) {
  #endif
#else
    if (readDisplayMode()) {
#endif
        refreshPanel();
    }
}

void Panel_c::refreshPanel() {
}
"""})

    assert _symbols(db, "panel.cpp") == [
        ("Panel_c::updateFrame", 1, 15),
        ("Panel_c::refreshPanel", 17, 18),
    ]


def test_what_the_first_parse_found_in_a_broken_range_is_kept(tmp_path, db):
    """The reparse sees only the first branch of each conditional. A variant
    defined in an #else, which the original parse did find inside the ERROR
    node, must not be traded for the definitions the reparse recovers."""
    _index(tmp_path, db, {"gauge.cpp": GAUGE_CPP + """
#ifdef _WIN32
void openSerialPort() {
    openWin32Handle();
}
#else
void openSerialPort() {
    openPosixDevice();
}
#endif
"""})

    assert _symbols(db, "gauge.cpp") == [
        ("Gauge_c::resetNeedle", 1, 3),
        ("Gauge_c::updateState", 5, 20),
        ("Gauge_c::moveNeedle", 22, 24),
        ("openSerialPort", 27, 29),
        ("openSerialPort", 31, 33),
    ]


def test_platform_variants_in_a_file_that_parses_are_all_kept(tmp_path, db):
    """One definition per platform is the common case and parses fine; the
    reparse must not trade those variants for the ones it recovers."""
    _index(tmp_path, db, {"port.cpp": """\
#ifdef _WIN32
void openSerialPort() {
    openWin32Handle();
}
#else
void openSerialPort() {
    openPosixDevice();
}
#endif
"""})

    assert _symbols(db, "port.cpp") == [
        ("openSerialPort", 2, 4),
        ("openSerialPort", 6, 8),
    ]
