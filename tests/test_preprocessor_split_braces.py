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


def test_a_dead_first_branch_does_not_cut_a_function_short(tmp_path, db):
    """The reparse keeps the first branch of each conditional, which is not
    always the live one: here `#if 0` is dead, closes its own brace, and the
    live `#else` opens one closed after `#endif`. The reparse then ends the
    function early — the original parse, which read it cleanly, must stand."""
    _index(tmp_path, db, {"legacy.cpp": """\
void refreshAll(int mode) {
#if 0
    if (mode) {
        clearGauge(0);
    }
#else
    if (mode > 1) {
#endif
        clearGauge(1);
    }
    clearGauge(2);
    applyMode(mode);
}

void applyMode(int mode) {
}
"""})

    assert ("refreshAll", 1, 13) in _symbols(db, "legacy.cpp")
    callees = {r["name"] for r in db.conn.execute(
        """SELECT t.name FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name = 'refreshAll'"""
    )}
    assert "applyMode" in callees


def test_complementary_conditionals_do_not_cut_a_function_short(tmp_path, db):
    """`#ifdef X` and `#ifndef X` each close a brace; the reparse keeps both
    first branches, one closing brace too many."""
    _index(tmp_path, db, {"opt.cpp": """\
void configureOptions(int flags) {
    if (flags) {
        clearGauge(1);
#ifdef OPTION_A
    }
#endif
#ifndef OPTION_A
    }
#endif
    clearGauge(2);
    applyMode(flags);
}

void applyMode(int mode) {
}
"""})

    assert ("configureOptions", 1, 12) in _symbols(db, "opt.cpp")


def test_a_raw_string_does_not_hide_the_directives_after_it(tmp_path, db):
    """A raw string can hold a quote and a `/*` without ending or opening
    anything. Read as an ordinary string, it opened a comment that hid every
    directive after it, and the recovery found nothing to repair."""
    _index(tmp_path, db, {"gauge.cpp": 'const char* kPattern = R"(a " /* b)";\n\n'
                                       + GAUGE_CPP})

    assert ("Gauge_c::updateState", 7, 22) in _symbols(db, "gauge.cpp")


def test_a_line_comment_continued_by_a_backslash_hides_the_next_line(tmp_path, db):
    """A `//` comment ending in a backslash runs on into the next line, so a
    `#else` there is commented out — not a branch switch."""
    _index(tmp_path, db, {"gauge.cpp": """\
void Gauge_c::checkFlags() {
#ifdef PLATFORM_DESKTOP
    // the old code path was here \\
#else
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
        ("Gauge_c::checkFlags", 1, 11),
        ("Gauge_c::refreshGauge", 13, 14),
    ]


def test_a_local_type_in_the_dropped_tail_is_not_a_swallowed_definition(tmp_path, db):
    """A shorter extent from the reparse is accepted only when the tail it
    drops holds a definition the original ran on over. A struct, an enum or a
    macro local to the function is part of it in both parses: no proof."""
    _index(tmp_path, db, {"walk.cpp": """\
void walkItems(int mode) {
#if 0
    if (mode) {
        legacyWalk();
    }
#else
    if (mode > 1) {
#endif
        fastWalk();
    }
    struct WalkPair_s { int key; } pairValue;
#define WALK_LIMIT 8
    finishWalk(pairValue.key);
}

void finishWalk(int key) {
}
"""})

    assert ("walkItems", 1, 14) in _symbols(db, "walk.cpp")


def test_a_function_run_on_over_globals_is_cut_back(tmp_path, db):
    """An initializer chosen by #if right after `=` breaks the original parse:
    it never sees the function close, runs on over the tables after it, and
    ends on a closing brace tree-sitter had to make up. The reparse reads the
    function cleanly, and its shorter extent is the right one even though what
    it drops holds no other function — only globals."""
    _index(tmp_path, db, {"shop.cpp": """static int createShop(Actor_c* actor) {
    prepareShop(actor);

    u32 priceTable[] =
#if PLATFORM_ALPHA
    {
        1,
        2,
    };
#else
    {
        3,
        4,
    };
#endif

    if (!openShop(actor, priceTable[0])) {
        return 0;
    }
    return 1;
}

static Method_c l_shopMethod = {
    1, 2,
};

PROFILE_TAIL;
"""})

    assert ("createShop", 1, 21) in _symbols(db, "shop.cpp")


def test_a_string_continued_across_a_crlf_line_stays_a_string():
    """An escaped line break inside a string splices the next line in. With
    CRLF endings the escape used to swallow the CR alone, end the string on
    the LF, and read what followed the next quote as a new string."""
    from srclight.indexer import _c_comment_bytes

    source = b'puts("ab\\\r\ncd"); /* note\r\n#else\r\n */\r\nlive();\r\n'
    marks = _c_comment_bytes(source)
    comment = source.index(b"/*")
    assert all(marks[comment:source.index(b"*/") + 2])
    assert not any(marks[source.index(b"live"):])


SPLIT_ELSEWHERE_CPP = """
void Gauge_c::updateState() {
#ifdef PLATFORM_DESKTOP
    if (isPanelVisible()) {
#else
    if (isPromptOpen()) {
#endif
        refreshGauge();
    }
}
"""


def _parents(db, path: str) -> dict[str, str | None]:
    rows = db.conn.execute(
        """SELECT s.name, p.name AS parent FROM symbols s
           JOIN files f ON s.file_id = f.id
           LEFT JOIN symbols p ON s.parent_symbol_id = p.id
           WHERE f.path = ?""",
        (path,),
    ).fetchall()
    return {r["name"]: r["parent"] for r in rows}


def test_a_namespace_is_not_cut_short_by_its_own_functions(tmp_path, db):
    """A container's members sit in its tail by definition: they prove no
    run-on. Only a function's dropped tail can hold such proof."""
    _index(tmp_path, db, {"app.cpp": """\
namespace app {
void refreshAll(int mode) {
#if 0
    if (mode) {
        clearGauge(0);
    }
#else
    if (mode > 1) {
#endif
        clearGauge(1);
    }
    applyMode(mode);
}

void applyMode(int mode) {
}
}
""" + SPLIT_ELSEWHERE_CPP})

    symbols = _symbols(db, "app.cpp")
    assert ("app", 1, 17) in symbols
    assert ("refreshAll", 2, 13) in symbols
    assert _parents(db, "app.cpp")["applyMode"] == "app"


def test_a_loop_macro_in_the_tail_is_not_a_swallowed_function(tmp_path, db):
    """Read at file scope, `FOR_EACH_ITEM(x) { ... }` looks like a function
    without a return type — which no real top-level definition is."""
    _index(tmp_path, db, {"walk.cpp": """\
void refreshAll(int mode) {
#if 0
    if (mode) {
        clearGauge(0);
    }
#else
    if (mode > 1) {
#endif
        clearGauge(1);
    }
    FOR_EACH_ITEM(item, mode) {
        applyMode(item);
    }
}

void applyMode(int mode) {
}
"""})

    assert ("refreshAll", 1, 14) in _symbols(db, "walk.cpp")


def test_a_class_is_not_the_return_type_of_the_function_after_it(tmp_path, db):
    """When the reparse closes a class at a stray brace, `class X {...} f() {...}`
    reads as one function returning the class. No definition returns a type
    it defines on the spot, so that extent is refused."""
    _index(tmp_path, db, {"panel.cpp": """\
class Panel_c {
    void refreshAll(int mode) {
#if 0
        if (mode) {
            clearGauge(0);
        }
#else
        if (mode > 1) {
#endif
            clearGauge(1);
        }
    }
    void applyMode(int mode) {
        clearGauge(mode);
    }
};
""" + SPLIT_ELSEWHERE_CPP})

    starts = {start for name, start, end in _symbols(db, "panel.cpp") if name == "applyMode"}
    assert 1 not in starts


BASE_UNDER_IF = """\
class Stick_c
#if TOOLING
    : public Inspectable
#endif
{
public:
    Stick_c();
    bool shiftGear(unsigned);

    bool checkMode(unsigned short flag) {
        return (flag & mMode) ? true : false;
    }

    float readHigh() { return mHigh; }

    float mHigh;
    unsigned short mMode;
};
"""


def test_a_class_with_its_base_under_if_yields_each_member_once(tmp_path, db):
    _index(tmp_path, db, {"stick.h": BASE_UNDER_IF})
    rows = db.conn.execute(
        "SELECT name, qualified_name, kind, start_line FROM symbols WHERE name IN "
        "('shiftGear', 'checkMode', 'readHigh')").fetchall()
    spans = [(r[0], r[3]) for r in rows]
    assert len(spans) == len(set(spans)) == 3
    assert ("shiftGear", "Stick_c::shiftGear") in {(r[0], r[1]) for r in rows}


# A macro the parser cannot read can make its error recovery swallow a
# closing brace and end the function early: the rest of the body is left at
# file scope. The braces still say where the function ends.
MACRO_CUT_SHORT = """\
#define PICK(a, b) a
#define BLOCK_END }

void Meter_c::draw() {
    if (ready) {
        if (level PICK(< limit + 1, == limit)) {
            level = -1;
        }
    }
    if (glow > 0.0f) {
        drawGlow(3);
    }
}

void manage() {
    if (!stopped) {
        if (!paused) {
            if (pending) {
            }
            BLOCK_END
            if (!isReady(1)) {
            } else {
                clearReady(1);
            }
            if (!stepAll()) {
            }
        }
    }
}

void after() {
    drawGlow(1);
}
"""


def test_a_macro_the_parser_cannot_read_does_not_cut_a_function_short(tmp_path, db):
    _index(tmp_path, db, {"meter.cpp": MACRO_CUT_SHORT})
    symbols = _symbols(db, "meter.cpp")
    assert ("Meter_c::draw", 4, 13) in symbols
    assert ("manage", 15, 29) in symbols
    assert ("after", 31, 33) in symbols


def test_a_function_extended_by_its_braces_calls_from_its_tail(tmp_path, db):
    _index(tmp_path, db, {"meter.cpp": MACRO_CUT_SHORT + "void drawGlow(int n) {\n}\n"})
    callers = {r[0] for r in db.conn.execute(
        """SELECT a.name FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id WHERE b.name = 'drawGlow'""")}
    assert "Meter_c::draw" in callers


def test_unbalanced_braces_never_extend_a_function_over_the_next(tmp_path, db):
    _index(tmp_path, db, {"open.cpp": """\
#define PICK(a, b) a
void first() {
    if (level PICK(< limit, == limit)) {
    }
#if NEW_PATH
    if (x) {
#endif
    step();
}

void second() {
    step();
}
"""})
    symbols = _symbols(db, "open.cpp")
    assert ("second", 11, 13) in symbols
    assert not any(n == "first" and end >= 11 for n, _s, end in symbols)


def test_a_function_template_is_extended_with_its_function(tmp_path, db):
    _index(tmp_path, db, {"tpl.cpp": """\
#define PICK(a, b) a
template <typename T>
void drawAll(T level) {
    if (ready) {
        if (level PICK(< limit + 1, == limit)) {
            level = -1;
        }
    }
    if (glow > 0.0f) {
        paint(3);
    }
}
"""})
    spans = {(n, s, e) for n, s, e in _symbols(db, "tpl.cpp") if n == "drawAll"}
    assert spans and all(e == 12 for _n, _s, e in spans), spans


def test_a_local_object_in_the_tail_does_not_block_the_extension(tmp_path, db):
    _index(tmp_path, db, {"lock.cpp": """\
#define PICK(a, b) a
void drawAll() {
    if (ready) {
        if (level PICK(< limit + 1, == limit)) {
            level = -1;
        }
    }
    Guard hold(lockObject);
    paint(3);
}
"""})
    assert ("drawAll", 2, 10) in _symbols(db, "lock.cpp")


def test_the_extension_never_covers_a_definition_the_merge_kept(tmp_path, db):
    _index(tmp_path, db, {"loop.cpp": """\
#define PICK(a, b) a
void manage() {
    if (ready) {
        if (level PICK(< limit + 1, == limit)) {
            level = -1;
        }
    }
#if FAST
    fastPath();
#else
    FOR_EACH(item) {
        slowPath(item);
    }
#endif
    finish();
}
"""})
    symbols = [s for s in _symbols(db, "loop.cpp") if s[0] != "PICK"]
    extents = {n: (s, e) for n, s, e in symbols}
    for name, (start, end) in extents.items():
        for other, (o_start, o_end) in extents.items():
            if name != other and start < o_start <= end:
                assert o_end <= end, (name, other, symbols)
                parent = db.conn.execute(
                    "SELECT parent_symbol_id FROM symbols WHERE name = ?", (other,)).fetchone()[0]
                assert parent is not None, (name, other, symbols)


def test_a_digit_separator_is_no_character_literal(tmp_path, db):
    _index(tmp_path, db, {"sep.cpp": """\
#define PICK(a, b) a
namespace outer {
void first() {
    if (v PICK(< a, == b)) {
    }
    use({1'0},'a');
}
int table_size = computeSize();
}
"""})
    assert not any(n == "first" and e > 7 for n, _s, e in _symbols(db, "sep.cpp"))


def test_a_quote_glued_to_a_word_opens_a_literal_only_after_a_keyword_or_prefix():
    from srclight.indexer import _digit_separator

    assert _digit_separator(b"n = 1'000;", 5)
    assert _digit_separator(b"v = 0x8000'0000;", 10)
    assert _digit_separator(b"TAG('longtag'), '{'", 12)
    assert not _digit_separator(b"case'{':", 4)
    assert not _digit_separator(b"u8'a'", 2)
    assert not _digit_separator(b"L'a'", 1)


def test_a_doc_comment_of_several_line_comments_is_read_whole(tmp_path, db):
    _index(tmp_path, db, {"notes.h": """\
// Kept apart by a blank line.

// The player who owns the current event. Player 0 until
// an order was seen.
int holder();
""", "notes.cpp": """\
/* A block. */
// A line after it.
int body() {
    return 0;
}
"""})
    docs = {name: (doc or "").replace("\r\n", "\n")
            for name, doc in db.conn.execute("SELECT name, doc_comment FROM symbols")}
    assert docs["holder"] == ("// The player who owns the current event. Player 0 until\n"
                              "// an order was seen.")
    assert docs["body"] == "/* A block. */\n// A line after it."


def test_a_trailing_comment_above_is_not_part_of_the_doc_comment(tmp_path, db):
    _index(tmp_path, db, {"trail.c": """\
int counter; // trailing note about counter
// doc of reset
void reset(void) {
}
"""})
    docs = dict(db.conn.execute("SELECT name, doc_comment FROM symbols"))
    assert docs["reset"] == "// doc of reset"



def test_members_parsed_alike_in_both_parses_keep_their_class(tmp_path, db):
    _index(tmp_path, db, {"holder.h": """\
template <class T>
class Holder {
public:
    T& get() {
        return v;
    }
    void set(T x) {
#if FAST
        if (x) {
#else
        if (x && ok()) {
#endif
            v = x;
        }
    }
    T v;
};
"""})
    names = {r[0]: r[1] for r in db.conn.execute(
        "SELECT name, qualified_name FROM symbols WHERE name IN ('get', 'set')")}
    assert names == {"get": "Holder::get", "set": "Holder::set"}, names
