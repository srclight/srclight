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


def test_a_header_that_differs_per_branch_gives_one_symbol(tmp_path, db):
    """Each branch opens the same body with its own header. The original
    parse starts the function at the #else header, the reparse at the first
    one: they are one definition, and must be stored once."""
    _index(tmp_path, db, {"scale.cpp": """\
#ifdef WIDE_INPUT
void scaleValue(long amount) {
#else
void scaleValue(int amount) {
#endif
    clampValue(amount);
}

void clampValue(int v) {
}
"""})

    assert _symbols(db, "scale.cpp") == [
        ("scaleValue", 2, 7),
        ("clampValue", 9, 10),
    ]


def test_a_name_that_differs_per_branch_keeps_both_names(tmp_path, db):
    """Both are real names — a caller of either one must find it. But they
    name one body, so neither calls the other."""
    _index(tmp_path, db, {"scale.cpp": """\
#ifdef WIDE_INPUT
long scaleWide(long v) {
#else
int scaleNarrow(int v) {
#endif
    return v;
}
"""})

    assert _symbols(db, "scale.cpp") == [("scaleWide", 2, 7), ("scaleNarrow", 4, 7)]
    edges = db.conn.execute(
        """SELECT count(*) AS n FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name IN ('scaleWide', 'scaleNarrow')
             AND t.name IN ('scaleWide', 'scaleNarrow')"""
    ).fetchone()["n"]
    assert edges == 0


def test_two_functions_ending_on_one_line_still_call_each_other(tmp_path, db):
    """Only names the recovery put over one body are exempt from edges. A
    file with no conditional at all keeps every call it makes."""
    _index(tmp_path, db, {"engine.cpp": """\
void startEngine() {
    warmUpEngine();
} void warmUpEngine() { }
"""})

    callees = {r["name"] for r in db.conn.execute(
        """SELECT t.name FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name = 'startEngine'"""
    )}
    assert "warmUpEngine" in callees


def test_a_comment_before_the_opening_directive_documents_the_function(tmp_path, db):
    """An opening #ifdef in between means the comment sits before the whole
    conditional, so it still applies. Only #else, #elif or #endif mean it
    belongs to another branch."""
    _index(tmp_path, db, {"device.cpp": """\
/** Opens the device. */
#ifdef PLATFORM_DESKTOP
int openDevice(int handle) {
#else
int openDevice(long handle) {
#endif
    return 0;
}
"""})

    row = db.conn.execute(
        "SELECT start_line, doc_comment FROM symbols WHERE name = 'openDevice'"
    ).fetchone()
    assert row["start_line"] == 3
    assert row["doc_comment"] == "/** Opens the device. */"


def test_an_apostrophe_does_not_hide_the_comments_after_it(tmp_path, db):
    """A digit separator is a lone apostrophe. Read as a character literal,
    it hid every comment after it, and a commented-out #else then flipped the
    live branch it sat in."""
    _index(tmp_path, db, {"gauge.cpp": """\
void Gauge_c::checkFlags() {
    int limit = 1'000;
#ifdef PLATFORM_DESKTOP
    /* was:
#else
    */
    if (isPanelVisible()) {
#else
    if (isPromptOpen()) {
#endif
        refreshGauge();
    }
}

void Gauge_c::refreshGauge() {
}
"""})

    assert _symbols(db, "gauge.cpp") == [
        ("Gauge_c::checkFlags", 1, 13),
        ("Gauge_c::refreshGauge", 15, 16),
    ]


def test_a_comment_opened_on_a_directive_line_stays_a_comment(tmp_path, db):
    """Blanking the whole directive line would cut the comment in two, and
    the reparse would read its second half as code."""
    _index(tmp_path, db, {"gauge.cpp": """\
void Gauge_c::checkFlags(int mode) {
#ifdef PLATFORM_DESKTOP   /* the PC build checks
                      both flags */
    if (mode) {
#else
    if (!mode) {
#endif
        refreshGauge();
    }
}
"""})

    assert _symbols(db, "gauge.cpp") == [("Gauge_c::checkFlags", 1, 10)]


def test_a_directive_inside_a_comment_is_not_a_directive(tmp_path, db):
    """A commented-out #else inside a first branch would flip that branch to
    inactive: the first branch's code after it would be blanked, and the
    recovered function would end at the wrong brace."""
    _index(tmp_path, db, {"gauge.cpp": """\
void Gauge_c::checkFlags() {
#ifdef PLATFORM_DESKTOP
    /* was:
#else
    */
    if (isPanelVisible()) {
#else
    if (isPromptOpen()) {
#endif
        refreshGauge();
    }
}

void Gauge_c::refreshGauge() {
}
"""})

    assert _symbols(db, "gauge.cpp") == [
        ("Gauge_c::checkFlags", 1, 12),
        ("Gauge_c::refreshGauge", 14, 15),
    ]


def test_a_split_nested_deep_enough_to_leave_no_error_node(tmp_path, db):
    """Deeper in, tree-sitter repairs the split with MISSING nodes rather
    than an ERROR node — and the function still runs on over the next one,
    and the namespace over what follows it."""
    _index(tmp_path, db, {"worker.cpp": """\
namespace app {

void Worker_c::runLoop(int count) {
    for (int i = 0; i < count; i++) {
#ifdef PLATFORM_DESKTOP
        if (count > 2) {
#else
        if (count > 3) {
#endif
            stepOnce();
        }
    }
}

void Worker_c::stepOnce() {
}

}

int afterSpace() {
    return 1;
}
"""})

    assert _symbols(db, "worker.cpp") == [
        ("app", 1, 18),
        ("Worker_c::runLoop", 3, 13),
        ("Worker_c::stepOnce", 15, 16),
        ("afterSpace", 20, 22),
    ]
    parent = db.conn.execute(
        "SELECT parent_symbol_id FROM symbols WHERE name = 'afterSpace'"
    ).fetchone()["parent_symbol_id"]
    assert parent is None


SPLIT_THEN_SPLIT_CPP = """\
void Panel_c::firstState() {
    mFlags = 0;
#if PLATFORM_DESKTOP
    Prompt_c* prompt = findPrompt();
    if (!isPanelVisible() || (prompt != NULL && prompt->isOpen())) {
#else
    if (!isPanelVisible() || findPrompt()->isOpen()) {
#endif
        mFlags |= 0x1;
    } else if (isBusy()) {
        mFlags |= 0x2;
    }
}

void Panel_c::resetNeedle() {
    clearGauge(1);
}

void Panel_c::updateState() {
    mFlags = 0;
#if PLATFORM_DESKTOP
    Prompt_c* prompt = findPrompt();
    if (!isPanelVisible() || (prompt != NULL && prompt->isOpen())) {
#else
    if (!isPanelVisible() || findPrompt()->isOpen()) {
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

void Panel_c::moveNeedle() {
    clearGauge(2);
}

void Panel_c::runLoop(int count) {
    for (int i = 0; i < count; i++) {
#if PLATFORM_DESKTOP
        if (count > 2) {
#else
        if (count > 3) {
#endif
            clearGauge(i);
        }
    }
}
"""


def test_splits_the_parse_reports_no_error_around(tmp_path, db):
    """Several splits in one file can leave tree-sitter reading a lost
    function as loose top-level fragments that carry no error flag, with the
    error surfacing somewhere else entirely. Locating the damage from the
    error flags then misses it, so the recovery must not depend on them."""
    _index(tmp_path, db, {"panel.cpp": SPLIT_THEN_SPLIT_CPP})

    assert _symbols(db, "panel.cpp") == [
        ("Panel_c::firstState", 1, 13),
        ("Panel_c::resetNeedle", 15, 17),
        ("Panel_c::updateState", 19, 35),
        ("Panel_c::moveNeedle", 37, 39),
        ("Panel_c::runLoop", 41, 51),
    ]


def test_a_recovered_function_with_an_unrelated_parse_error_inside(tmp_path, db):
    """A long function nearly always holds something tree-sitter cannot read
    — here a call through a pointer to member function. That error is local:
    the braces still close where they should, so the recovered definition
    must not be refused for it."""
    _index(tmp_path, db, {"panel.cpp": SPLIT_THEN_SPLIT_CPP.replace(
        "    switch (readDisplayMode()) {",
        "    (this->*mHandler)();\n    switch (readDisplayMode()) {",
    )})

    assert ("Panel_c::updateState", 19, 36) in _symbols(db, "panel.cpp")


def test_a_recovered_signature_has_no_blanked_gaps(tmp_path, db):
    _index(tmp_path, db, {"gauge.cpp": """\
void Gauge_c::setMode(
#ifdef PLATFORM_DESKTOP
    int mode, int extra) {
#else
    int mode) {
#endif
    applyMode(mode);
}
"""})

    sig = db.conn.execute(
        "SELECT signature FROM symbols WHERE name = 'Gauge_c::setMode'"
    ).fetchone()["signature"]

    assert "int mode, int extra" in sig
    assert "  " not in sig


def test_a_comment_across_a_directive_is_not_a_doc_comment(tmp_path, db):
    """In the reparse the directives are gone, so a comment from a preceding
    conditional block ends up right above the next definition."""
    _index(tmp_path, db, {"gauge.cpp": """\
#ifdef PLATFORM_DESKTOP
/* PC-only helpers follow */
#else
int legacyMode;
#endif
void Gauge_c::drawGauge() {
#ifdef PLATFORM_DESKTOP
    if (isPanelVisible()) {
#else
    if (isPromptOpen()) {
#endif
        refreshGauge();
    }
}
"""})

    row = db.conn.execute(
        "SELECT start_line, doc_comment FROM symbols WHERE name = 'Gauge_c::drawGauge'"
    ).fetchone()

    assert row["start_line"] == 6
    assert row["doc_comment"] is None


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
