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
int drive(void* car) {
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


def test_a_bare_call_reaches_a_free_function_template(tmp_path):
    edges = _edges({"bits.h": """\
template <typename T>
void raiseFlag(T* bits, T mask) {
    *bits |= mask;
}
""", "shop/shop.cpp": """\
void openShop(unsigned* bits) {
    raiseFlag(bits, 4u);
}
"""}, tmp_path)
    assert ("openShop", "raiseFlag") in _pairs(edges)


def test_a_class_of_the_own_namespace_keeps_its_constructor(tmp_path):
    edges = _edges({"menu.cpp": """\
namespace ui {
class Option {
public:
    Option(int v) {}
};

void pickOption() {
    Option(3);
}
}
"""}, tmp_path)
    assert ("ui::pickOption", "ui::Option::Option") in _pairs(edges)


def test_a_member_call_keeps_a_header_function(tmp_path):
    """Headers are read as C or C++; a function in one may be a lost method."""
    edges = _edges({"hub.h": """\
int fetchSlot(int i) {
    return i;
}
""", "cls/hub_c.h": """\
class Hub_c { public: int fetchSlot(int i); };
""", "use/use.cpp": """\
int peek(void* hub) {
    return hub->fetchSlot(1);
}
"""}, tmp_path)
    assert ("peek", "fetchSlot") in _pairs(edges)


def test_a_local_named_like_a_getter_hides_only_the_bare_name(tmp_path):
    edges = _edges({"layer.h": """\
class Layer {
public:
    float opacity();
    int spacing();
};
""", "use/blend.cpp": """\
void blendPanel(Layer* layer, const Layer& other) {
    float opacity = layer->opacity();
    auto spacing = other.spacing();
}
"""}, tmp_path)
    assert {("blendPanel", "Layer::opacity"), ("blendPanel", "Layer::spacing")} <= _pairs(edges)


def test_a_global_call_hides_nothing_behind_a_parameter(tmp_path):
    edges = _edges({"tally.cpp": """\
int tally(int n) {
    return n;
}

int sumUp(int tally) {
    return ::tally(tally);
}
"""}, tmp_path)
    assert ("sumUp", "tally") in _pairs(edges)


def test_a_class_template_is_its_own_scope(tmp_path):
    edges = _edges({"a/box.h": """\
namespace store {
void refill(int n);
template <typename T> class Crate {
public:
    void refill(int n);
    void topUp() { refill(3); }
};
}
""", "a/box.cpp": """\
namespace store {
void refill(int n) {
}
}
"""}, tmp_path)
    assert not any(r == "same_class" and b == "store::refill" for _, b, r in edges)


def test_a_global_call_prefers_the_global_function(tmp_path):
    edges = _edges({"gain.cpp": """\
int computeGain() {
    return 1;
}

namespace audio {
int computeGain() {
    return 2;
}

int mixTrack() {
    return ::computeGain();
}
}
"""}, tmp_path)
    reached = {b for a, b, _ in edges if a == "audio::mixTrack"}
    assert reached == {"computeGain"}


def test_a_c_member_call_ranks_no_function_first(tmp_path):
    edges = _edges({"c/ops.h": """\
static inline int flush_all(int v) {
    return v;
}
""", "c/drv.c": """\
struct ops { int (*flush_all)(int); };

static int flush_all(int v) {
    return v;
}

int run_dev(struct ops* o) {
    return o->flush_all(1);
}
"""}, tmp_path)
    assert {r for a, _, r in edges if a == "run_dev"} == {"same_file"}


def test_a_long_qualifier_chain_does_not_recurse():
    body = "void f() {" + " const" * 3000 + " Widget w;}"
    assert "w" in _declared_names(body, "f", "function")


def test_a_member_read_or_an_object_is_no_call(tmp_path):
    edges = _edges({"stream.h": """\
class Stream_c {
public:
    int flags();
};
""", "ctx/ctx.cpp": """\
int current() {
    return 0;
}
""", "actor/actor.cpp": """\
struct Pose { int angle; int pos; };
struct Attn { int flags; };

void stepActor(Pose& current, Attn& info) {
    current.angle = 3;
    if (info.flags & 4) {
    }
}
"""}, tmp_path)
    reached = {b for a, b, _ in edges if a == "stepActor"}
    assert "current" not in reached and "Stream_c::flags" not in reached


def test_a_function_pointer_field_read_is_no_call_but_its_call_is(tmp_path):
    edges = _edges({"irq.c": """\
struct ops { void (*handleIrq)(int); };

void handleIrq(int v) {
}

int hasHandler(struct ops *o) {
    return o->handleIrq != 0;
}

void fire(struct ops *o) {
    o->handleIrq(1);
}
"""}, tmp_path)
    assert ("fire", "handleIrq") in _pairs(edges)
    assert ("hasHandler", "handleIrq") not in _pairs(edges)


def test_a_local_named_like_its_method_makes_no_self_call(tmp_path):
    edges = _edges({"door.h": """\
class Door_c {
public:
    int paint();
    int visible();
};
""", "door.cpp": """\
#include "door.h"

int Door_c::paint() {
    int paint = visible();
    if (!paint) {
        return 0;
    }
    return paint;
}
"""}, tmp_path)
    assert not any(a == "Door_c::paint" and b == "Door_c::paint" for a, b, _ in edges)
    assert ("Door_c::paint", "Door_c::visible") in _pairs(edges)


def test_callees_name_the_calls_the_graph_leaves_out(serve):
    files = _lamps(12)
    files["far/far.cpp"] = """\
class Box_c {
public:
    void reset();
};

void Box_c::reset() {
}

void tickFar(void* lamp, Box_c* box) {
    lamp->blinkLamp();
    box->reset();
}
"""
    server = serve(files)
    payload = json.loads(server.get_callees("tickFar"))
    note = payload["graph_note"]
    assert "`blinkLamp`" in note and "`reset`" in note
    assert "graph_note" not in json.loads(server.get_callees("ownLamp"))


def test_a_common_name_written_with_its_class_is_a_call(tmp_path):
    edges = _edges({"stack.h": """\
class Stack_c {
public:
    static void copy(int v);
    static int get();
};
""", "use/use.cpp": """\
int useStack() {
    Stack_c::copy(1);
    return Stack_c::get();
}
"""}, tmp_path)
    assert {("useStack", "Stack_c::copy"), ("useStack", "Stack_c::get")} <= _pairs(edges)


def test_a_member_call_split_over_lines_or_through_a_pointer_is_kept(tmp_path):
    edges = _edges({"irq.c": """\
struct ops { void (*handleIrq)(int); };

void handleIrq(int v) {
}

#define CALL(f) f(1)

void viaSpace(struct ops *o) {
    o->handleIrq
            (2);
}

void viaDeref(struct ops *o) {
    (*o->handleIrq)(1);
}

void viaMacro(struct ops *o) {
    CALL(o->handleIrq);
}
"""}, tmp_path)
    for caller in ("viaSpace", "viaDeref", "viaMacro"):
        assert (caller, "handleIrq") in _pairs(edges)


def test_the_own_name_is_blanked_only_as_a_whole_name(tmp_path):
    edges = _edges({"knob.h": """\
class Knob_c {
public:
    typedef int turnMode;
    int turn();
};
""", "knob.cpp": """\
#include "knob.h"

Knob_c::turnMode Knob_c::turn() {
    int turn = 1;
    return turn;
}
"""}, tmp_path)
    assert not any(a == "Knob_c::turn" and b == "Knob_c::turn" for a, b, _ in edges)


def test_callee_note_skips_declarations_and_initializers(serve):
    files = _lamps(12)
    files["timer/timer.h"] = """\
class Timer_c {
public:
    Timer_c(int v);
};
"""
    files["far/far.cpp"] = """\
void setupFar(Timer_c* t) {
    Timer_c blinkLamp(5);
}

void resetFar(void* lamp) {
    lamp->blinkLamp();
}
"""
    server = serve(files)
    assert "graph_note" not in json.loads(server.get_callees("setupFar"))
    assert "`blinkLamp`" in json.loads(server.get_callees("resetFar"))["graph_note"]


def test_a_member_call_is_not_decided_by_the_file_among_classes(tmp_path):
    """`info.readSlot()` in a header that also defines `Play_c::readSlot`:
    the file says nothing of the type of `info`."""
    edges = _edges({"save/save.h": """\
class Info_c {
public:
    int readSlot();
};

class Save_c {
public:
    int readSlot();
};
""", "game.h": """\
class Play_c {
public:
    int readSlot(int idx);
    int readHost();
};

struct Game_c { Info_c info; Play_c play; };
extern Game_c g_game;

inline int getInfoSlot() {
    return g_game.info.readSlot();
}

inline int getPlaySlot() {
    return g_game.play.readSlot(0);
}
"""}, tmp_path)
    info = {(b, r) for a, b, r in edges if a == "getInfoSlot"}
    assert ("Play_c::readSlot", "same_file") not in info
    assert all(r == "name_only" for b, r in info if b.endswith("readSlot"))
    play = {(b, r) for a, b, r in edges if a == "getPlaySlot"}
    assert ("Play_c::readSlot", "same_file") in play
    assert not any(b.startswith(("Info_c", "Save_c")) for b, _ in play)


def test_call_arity_and_parameter_range():
    from srclight.indexer import _call_arity, _param_range

    text = "f(); g(a, h(b, c), {1, 2}); k( );"
    assert _call_arity(text, text.index("(")) == 0
    assert _call_arity(text, text.index("g(") + 1) == 3
    assert _call_arity(text, text.index("k(") + 1) == 0
    assert _param_range("int f(int a, int b = 2)", "f") == (1, 2)
    assert _param_range("void f(void)", "f") == (0, 0)
    assert _param_range("void f(std::map<int, int> m)", "f") == (1, 1)
    assert _param_range("int f(const char* fmt, ...)", "f") == (1, float("inf"))


def test_a_member_call_on_a_typed_variable_reaches_that_class(tmp_path):
    edges = _edges({"a/info.h": """\
class Info_c {
public:
    int readSlot();
};
""", "b/save.h": """\
class Save_c {
public:
    int readSlot();
};
""", "use/use.cpp": """\
int fromParam(Info_c* info) {
    return info->readSlot();
}

int fromLocal() {
    Save_c save;
    return save.readSlot();
}

int fromChain(Holder* h) {
    return h->info.readSlot();
}
"""}, tmp_path)
    assert {(b, r) for a, b, r in edges if a == "fromParam" and "readSlot" in b} == {
        ("Info_c::readSlot", "typed")}
    assert {(b, r) for a, b, r in edges if a == "fromLocal" and "readSlot" in b} == {
        ("Save_c::readSlot", "typed")}
    assert {r for a, b, r in edges if a == "fromChain" and "readSlot" in b} == {"name_only"}


def test_a_qualified_call_reaches_the_overload_its_arguments_fit(serve):
    server = serve({"stack.h": """\
class Stack_c {
public:
    static void shift(int v);
    static void shift(int x, int y, int z);
};
""", "stack.cpp": """\
#include "stack.h"

void Stack_c::shift(int v) {
}

void Stack_c::shift(int x, int y, int z) {
}
""", "use/use.cpp": """\
void moveOne() {
    Stack_c::shift(1);
}
"""})
    callees = json.loads(server.get_callees("moveOne"))["callees"]
    shifts = [e for e in callees if e["name"].endswith("shift")]
    signatures = {e.get("signature") for e in shifts} | {
        loc.get("signature") for e in shifts for loc in e.get("locations", [])}
    assert any("int v" in (s or "") for s in signatures)
    assert not any("int z" in (s or "") for s in signatures)
