"""The way a C/C++ name is written narrows what it can call.

The call graph is a text scan: every symbol whose name appears in a body
becomes a callee. That links `p->scaleVec()` to a free function `scaleVec`,
a parameter called `blend` to the function `blend()`, and a class's own
`scaleVec()` to every other class's. The syntax around the name rules most
of these out, and what it cannot rule out stays labelled `name_only`.
"""
import json

import pytest

from srclight.db import Database
from srclight.indexer import (
    IndexConfig,
    Indexer,
    _declared_names,
    _parameter_names,
    _reference_forms,
)

SHAPES = {
    "vec/vec.h": """\
class Vec_c {
public:
    void scaleVec();
    void grow();
};

class Mat_c {
public:
    void scaleVec();
};
""",
    "vec/vec.cpp": """\
#include "vec.h"

void Vec_c::grow() {
    scaleVec();
}

void Vec_c::scaleVec() {
}

void Mat_c::scaleVec() {
}
""",
    "free/free.c": """\
void scaleVec() {
}

void tickAll() {
    scaleVec();
}
""",
    "use/use.cpp": """\
#include "vec.h"

void pokeVec(Vec_c* v) {
    v->scaleVec();
}

void pokeMat() {
    Mat_c::scaleVec();
}
""",
    "mix/mix.c": """\
int blendColor(int a, int b) {
    return a + b;
}

int applyMix(int blendColor, int other) {
    return blendColor * other;
}

int realMix(int x) {
    return blendColor(x, 1);
}
""",
}


@pytest.fixture
def serve(tmp_path, monkeypatch):
    from srclight import server as server_mod

    def _serve(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, text in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        db_path = root / ".srclight" / "index.db"
        db_path.parent.mkdir()
        db = Database(db_path)
        db.open()
        db.initialize()
        Indexer(db, IndexConfig(root=root)).index()
        db.close()
        monkeypatch.chdir(root)
        monkeypatch.setattr(server_mod, "_workspace_name", None)
        server_mod.configure(db_path=db_path, repo_root=root)
        return server_mod

    return _serve


def _names(raw: str, key: str) -> set[str]:
    return {entry["name"] for entry in json.loads(raw)[key]}


def test_a_parameter_named_like_a_function_is_no_call(serve):
    server = serve(SHAPES)
    callers = _names(server.get_callers("blendColor"), "callers")
    assert "realMix" in callers
    assert "applyMix" not in callers


def test_a_member_call_reaches_methods_not_the_free_function(serve):
    server = serve(SHAPES)
    assert "pokeVec" in _names(server.get_callers("Vec_c::scaleVec"), "callers")
    payload = json.loads(server.get_callees("pokeVec"))
    called = [e for e in payload["callees"] if e["name"] == "scaleVec"]
    assert called and all(e["kind"] == "method" for e in called)
    files = {loc["file"] for e in called for loc in e.get("locations", [e])}
    assert not any("free" in f for f in files)


def test_a_free_function_bare_call_reaches_no_method(serve):
    server = serve(SHAPES)
    payload = json.loads(server.get_callees("tickAll"))
    assert [e["kind"] for e in payload["callees"]] == ["function"]
    assert "tickAll" not in _names(server.get_callers("Mat_c::scaleVec"), "callers")


def test_a_method_bare_call_reaches_its_own_class(serve):
    server = serve(SHAPES)
    assert "Vec_c::grow" in _names(server.get_callers("Vec_c::scaleVec"), "callers")
    assert "Vec_c::grow" not in _names(server.get_callers("Mat_c::scaleVec"), "callers")


def test_a_qualified_call_reaches_that_class_only(serve):
    server = serve(SHAPES)
    assert "pokeMat" in _names(server.get_callers("Mat_c::scaleVec"), "callers")
    assert "pokeMat" not in _names(server.get_callers("Vec_c::scaleVec"), "callers")


def _lamps(count: int) -> dict[str, str]:
    files = {}
    for i in range(count):
        files[f"lamp{i}/lamp{i}.h"] = f"""\
class Lamp{i}_c {{
public:
    void blinkLamp();
}};
"""
    files["lamp0/lamp0.cpp"] = """\
#include "lamp0.h"

void Lamp0_c::blinkLamp() {
}

void ownLamp(Lamp0_c* lamp) {
    lamp->blinkLamp();
}
"""
    files["far/far.cpp"] = """\
void anyLamp(void* lamp) {
    lamp->blinkLamp();
}
"""
    return files


def test_a_widely_defined_name_keeps_its_decided_calls(serve):
    """Past ten definitions a call resolved by the name alone is left out,
    but one the evidence decides — the same directory here — is kept."""
    server = serve(_lamps(12))
    callers = _names(server.get_callers("Lamp0_c::blinkLamp"), "callers")
    assert "ownLamp" in callers
    assert "anyLamp" not in callers


def test_reference_forms():
    assert _reference_forms("p->scaleVec(); v.scaleVec();", "scaleVec") == ({"member"}, set())
    assert _reference_forms("this->scaleVec();", "scaleVec") == ({"this"}, set())
    assert _reference_forms("Mat_c::scaleVec();", "scaleVec") == ({"qualified"}, {"Mat_c"})
    assert _reference_forms("Box<int>::scaleVec();", "scaleVec") == ({"qualified"}, {"Box"})
    assert _reference_forms("::scaleVec(); scaleVec();", "scaleVec") == (
        {"global", "bare"}, set())
    assert _reference_forms("i_this->scaleVec();", "scaleVec") == ({"member"}, set())


def test_parameter_names():
    assert _parameter_names(
        "void paintRow(int dst, char fade, long post[4], void (*done)(int), int k = 2) {}",
        "paintRow", "function") == {"dst", "fade", "post", "done", "k"}
    assert _parameter_names(
        "#define PACK_PAIR(lo, fade) ((lo) | (fade) << 8)",
        "PACK_PAIR", "macro") == {"lo", "fade"}
    assert _parameter_names(
        "void Box_c::fill(const Vec_c& from) {}", "Box_c::fill", "method") == {"from"}


def test_declared_names_skip_statements_that_declare_nothing():
    body = """\
int walk(int n) {
    Vec_c* cursor = start();
    float scale;
    return total;
    delete handle;
    return a * factor;
}
"""
    names = _declared_names(body, "walk", "function")
    assert {"n", "cursor", "scale"} <= names
    assert "total" not in names and "handle" not in names


def test_a_caller_says_how_it_was_resolved(serve):
    server = serve(SHAPES)
    callers = json.loads(server.get_callers("Vec_c::scaleVec"))["callers"]
    by_name = {e["name"]: e["resolution"] for e in callers if "resolution" in e}
    assert by_name["Vec_c::grow"] == "same_class"


def test_a_widely_defined_name_says_callers_may_be_missing(serve):
    server = serve(_lamps(12))
    note = json.loads(server.get_callers("Lamp3_c::blinkLamp"))["graph_note"]
    assert "defined 12 times" in note and "blinkLamp" in note
    assert "graph_note" not in json.loads(server.get_callees("ownLamp"))


def test_a_common_name_says_it_is_not_in_the_graph(serve):
    server = serve({"box.cpp": """\
class Box_c {
public:
    void reset();
};

void Box_c::reset() {
}

void wipe(Box_c* box) {
    box->reset();
}
"""})
    payload = json.loads(server.get_callers("Box_c::reset"))
    assert payload["caller_count"] == 0
    assert "too short or too common" in payload["graph_note"]
    assert r"find_pattern(r'\breset\s*\(')" in payload["graph_note"]


# --- What narrowing must not lose -------------------------------------------

def _edges(files: dict[str, str], tmp_path) -> set[tuple[str, str, str]]:
    root = tmp_path / "edges"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    rows = db.conn.execute(
        """SELECT a.qualified_name, b.qualified_name, e.resolution FROM symbol_edges e
           JOIN symbols a ON a.id = e.source_id JOIN symbols b ON b.id = e.target_id""")
    edges = {tuple(r) for r in rows}
    db.close()
    return edges


def _pairs(edges) -> set[tuple[str, str]]:
    return {(a, b) for a, b, _ in edges}


def test_a_constructor_defined_in_its_class_reaches_its_methods(tmp_path):
    edges = _edges({"engine.cpp": """\
class Engine {
public:
    Engine(int v) { warmUp(); }
    void warmUp();
};

void Engine::warmUp() {
}
""", "free.cpp": """\
void warmUp() {
}
"""}, tmp_path)
    assert ("Engine::Engine", "Engine::warmUp") in _pairs(edges)
    assert ("Engine::Engine", "warmUp") not in _pairs(edges)


def test_a_global_call_is_no_call_to_the_own_class(tmp_path):
    edges = _edges({"pool.h": """\
class Pool {
public:
    void flushAll();
    void drainQueue();
};
""", "pool.cpp": """\
#include "pool.h"

void Pool::drainQueue() {
    ::flushAll();
}
""", "other/flush.cpp": """\
void flushAll() {
}
"""}, tmp_path)
    assert ("Pool::drainQueue", "flushAll") in _pairs(edges)
    assert ("Pool::drainQueue", "Pool::flushAll") not in _pairs(edges)


def test_a_c_call_through_a_pointer_field_still_reaches_the_function(tmp_path):
    edges = _edges({"irq.c": """\
struct ops { void (*handleIrq)(int); };

void handleIrq(int v) {
}

void dispatchAll(struct ops *o) {
    o->handleIrq(3);
}
"""}, tmp_path)
    assert ("dispatchAll", "handleIrq") in _pairs(edges)


def test_a_qualifier_matches_a_whole_class_name(tmp_path):
    edges = _edges({"box.h": """\
class MyBox {
public:
    void fillUp();
};
""", "use/use.cpp": """\
void pour() {
    Box::fillUp();
}
"""}, tmp_path)
    assert all(r != "qualified" for a, b, r in edges if a == "pour")


def test_a_qualified_constructor_call_reaches_the_constructor(tmp_path):
    edges = _edges({"tint.h": """\
namespace paint {
class Tint {
public:
    Tint(int r, int g) {}
};
}
""", "use/use.cpp": """\
void shade() {
    paint::Tint(1, 2);
}
"""}, tmp_path)
    assert ("shade", "paint::Tint::Tint") in _pairs(edges)


def test_a_method_naming_its_own_class_reaches_the_class(tmp_path):
    edges = _edges({"crate.h": """\
class Crate {
public:
    Crate() {}
    Crate* selfRef();
};

Crate* Crate::selfRef() {
    return (Crate*)0;
}
"""}, tmp_path)
    assert ("Crate::selfRef", "Crate") in _pairs(edges)


def test_an_unnamed_parameter_keeps_its_type_referenced(tmp_path):
    edges = _edges({"node.h": """\
struct Node {
    int weight;
};
""", "walk.cpp": """\
#include "node.h"

void visitNode(Node*, int) {
}
"""}, tmp_path)
    assert ("visitNode", "Node") in _pairs(edges)


def test_parameter_names_of_unnamed_parameters():
    assert _parameter_names("void f(Vec_c) {}", "f", "function") == set()
    assert _parameter_names("void f(const Vec_c&) {}", "f", "function") == set()
    assert _parameter_names("void f(struct Node*) {}", "f", "function") == set()
    assert _parameter_names("void f(std::vector<Item>) {}", "f", "function") == set()
    assert _parameter_names("void f(unsigned count, const int k) {}", "f", "function") == {
        "count", "k"}
    assert _parameter_names(
        "void f(bool on = lim > 3, Widget keepMe) {}", "f", "function") == {"on", "keepMe"}


def test_an_expression_is_no_declaration():
    body = """\
int probe(int n) {
    return n * helperFn;
    x = y & otherFn;
    if (mask & isReady == 0) {}
    const Widget local = make();
}
"""
    names = _declared_names(body, "probe", "function")
    assert {"helperFn", "otherFn", "isReady"}.isdisjoint(names)
    assert {"n", "local"} <= names


def test_a_common_name_with_a_qualified_definition_says_which_calls_count(serve):
    server = serve({"pool.cpp": """\
class Pool {
public:
    void init();
};

void Pool::init() {
}

void starter() {
    Pool::init();
}
"""})
    payload = json.loads(server.get_callers("Pool::init"))
    assert "starter" in {e["name"] for e in payload["callers"]}
    assert "Only calls written `Pool::init(...)`" in payload["graph_note"]


def test_a_member_call_keeps_a_cpp_function_it_may_be_a_method(tmp_path):
    """An inline method in a class body the parser lost track of is stored
    as a free C++ function; a member call may still reach it. A C function
    never is a method."""
    edges = _edges({"lost.cpp": """\
int readSpeed() {
    return 1;
}
""", "legacy.c": """\
int readSpeed() {
    return 2;
}
""", "car/car.h": """\
class Car {
public:
    int readSpeed();
};
""", "use/drive.cpp": """\
int drive(Car* car) {
    return car->readSpeed();
}
"""}, tmp_path)
    assert ("drive", "Car::readSpeed") in _pairs(edges)
    db = Database(tmp_path / "edges" / "index.db")
    db.open()
    reached = {r[0] for r in db.conn.execute(
        """SELECT f.path FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id JOIN files f ON f.id = b.file_id
           WHERE a.name = 'drive' AND b.name = 'readSpeed'""")}
    db.close()
    assert "lost.cpp" in reached and "legacy.c" not in reached
