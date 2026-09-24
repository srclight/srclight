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


OPERATORS_CPP = """\
class Vessel_c {
public:
    Vessel_c& operator=(const Vessel_c& other);
    bool operator==(const Vessel_c& other) const;
    Item_c& operator[](int index) { return mItems[index]; }
    Item_c* operator->() { return mItem; }
    operator bool() const { return mItem != 0; }
    Item_c*& itemSlot() { return mItem; }
    template <class U> U& pickAs() { return *(U*)mItem; }
};

bool operator<(const Vessel_c& a, const Vessel_c& b);
Stream_c& operator<<(Stream_c& out, const Vessel_c& v);

Stream_c& operator<<(Stream_c& out, const Vessel_c& v) { return out; }

namespace geo {
bool operator!=(const Point_c& a, const Point_c& b) { return false; }
}

template <class U> U& pickFree(U* item) { return *item; }
"""


@pytest.fixture
def operator_symbols(tmp_path, db):
    root = tmp_path / "ops"
    root.mkdir()
    (root / "vessel.cpp").write_text(OPERATORS_CPP)
    Indexer(db, IndexConfig(root=root)).index()
    return {(r["name"], r["kind"], r["start_line"]) for r in db.conn.execute(
        "SELECT name, kind, start_line FROM symbols"
    )}


@pytest.mark.parametrize("expected", [
    ("operator=", "method", 3),       # in-class declaration returning a reference
    ("operator==", "method", 4),      # in-class declaration
    ("operator[]", "method", 5),      # inline, returning a reference
    ("operator->", "method", 6),      # inline, returning a pointer
    ("operator bool", "method", 7),   # conversion operator
    ("itemSlot", "method", 8),        # returning a reference to a pointer
    ("pickAs", "method", 9),          # template method inside the class
    ("operator<", "prototype", 12),
    ("operator<<", "prototype", 13),
    ("operator<<", "function", 15),   # free operator returning a reference
    ("operator!=", "function", 18),   # free operator in a namespace
    ("pickFree", "function", 21),     # free template function
])
def test_operators_and_remaining_shapes(operator_symbols, expected):
    assert expected in operator_symbols


def test_a_template_returning_a_reference_is_named_by_its_function(operator_symbols):
    """The template symbol took its name from the declarator's own text:
    `U& pickAs()` was named "& pickAs()"."""
    templates = {name for name, kind, line in operator_symbols if kind == "template"}
    assert templates == {"pickAs", "pickFree"}


def test_a_free_operator_is_not_a_method(operator_symbols):
    kinds = {kind for name, kind, line in operator_symbols if line in (15, 18)}
    assert kinds == {"function"}


SCOPES_CPP = """\
template <typename T> struct Holder_s {
    T& holdValue() { return mValue; }
    void declaredHold();
};

union Blend_u {
    int pickPart() { return mPart; }
};
"""


def test_a_template_class_scope_is_not_doubled(tmp_path, db):
    root = tmp_path / "scopes"
    root.mkdir()
    (root / "holder.cpp").write_text(SCOPES_CPP)
    Indexer(db, IndexConfig(root=root)).index()
    qualified = {r["name"]: r["qualified_name"] for r in db.conn.execute(
        "SELECT name, qualified_name FROM symbols"
    )}

    assert qualified["holdValue"] == "Holder_s::holdValue"
    assert qualified["declaredHold"] == "Holder_s::declaredHold"
    assert qualified["pickPart"] == "Blend_u::pickPart"


def test_calls_reach_an_inline_method_and_a_reference_returning_function(symbols, db):
    callees = {r["name"] for r in db.conn.execute(
        """SELECT t.name FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name = 'usePanel'"""
    )}

    assert {"plainInline", "freeReference"} <= callees
