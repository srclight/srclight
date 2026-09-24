"""C++ definitions the symbol query used to miss.

A method defined inside its class body has a field_identifier for a
declarator, where the query only knew identifier (free functions) and
qualified_identifier (methods defined outside the class). A `T&` return type
wraps the declarator in a reference_declarator, whose child carries no field
name, so no pattern reached through it. Every such definition was missing
from the index: no symbol, no body, no callers, no callees — while its calls
were credited to whatever enclosing symbol was scanned instead.
"""
import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer

PANEL_CPP = """\
class Panel_c {
public:
    int plainInline() { return 1; }
    Gauge_c* pointerInline() { return mGauge; }
    Gauge_c& referenceInline() { return *mGauge; }
    bool operator==(const Panel_c& other) const { return true; }
    ~Panel_c() { releaseGauge(); }
    Gauge_c& referenceDeclared();
};

Gauge_c& Panel_c::referenceOutOfLine() { return *mGauge; }

Gauge_c& freeReference() {
    static Gauge_c gauge;
    return gauge;
}

Gauge_c& freeReferenceDeclared();

void usePanel(Panel_c* panel) {
    panel->plainInline();
    freeReference();
}
"""


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def symbols(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "panel.cpp").write_text(PANEL_CPP)
    Indexer(db, IndexConfig(root=root)).index()
    return {(r["name"], r["kind"], r["start_line"]) for r in db.conn.execute(
        "SELECT name, kind, start_line FROM symbols"
    )}


@pytest.mark.parametrize("expected", [
    ("plainInline", "method", 3),
    ("pointerInline", "method", 4),
    ("referenceInline", "method", 5),
    ("operator==", "method", 6),
    ("~Panel_c", "method", 7),
    ("Panel_c::referenceOutOfLine", "method", 11),
    ("freeReference", "function", 13),
])
def test_the_definition_is_extracted(symbols, expected):
    assert expected in symbols


@pytest.mark.parametrize("expected", [
    ("referenceDeclared", "method", 8),
    ("freeReferenceDeclared", "prototype", 18),
])
def test_a_declaration_returning_a_reference_is_extracted(symbols, expected):
    assert expected in symbols


def test_calls_reach_an_inline_method_and_a_reference_returning_function(symbols, db):
    callees = {r["name"] for r in db.conn.execute(
        """SELECT t.name FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name = 'usePanel'"""
    )}

    assert {"plainInline", "freeReference"} <= callees
