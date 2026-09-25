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
    _reference_forms_all,
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
            (root / name).write_text(text, encoding="utf-8")
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
        (root / name).write_text(text, encoding="utf-8")
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


def test_default_arguments_in_the_header_prototype_count(tmp_path):
    edges = _edges({"log/log.h": """\
class Sink;
void logLine(int level, const char* text = 0);
""", "log/log.cpp": """\
void logLine(int level, const char* text) {
}
""", "other/o.cpp": """\
static void logLine(const char* s) {
}
""", "log/use.cpp": """\
void report() {
    logLine(3);
}
"""}, tmp_path)
    reached = {b for a, b, _ in edges if a == "report"}
    db = Database(tmp_path / "edges" / "index.db")
    db.open()
    files = {r[0] for r in db.conn.execute(
        """SELECT f.path FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id JOIN files f ON f.id = b.file_id
           WHERE a.name = 'report'""")}
    db.close()
    assert "logLine" in reached
    assert any(f.replace(chr(92), "/").endswith("log/log.cpp") for f in files)


def test_arity_reads_template_arguments_and_arrows():
    from srclight.indexer import _call_arity, _param_range

    text = "place(std::pair<int, int>(1, 2), 3);"
    assert _call_arity(text, text.index("(")) == 2
    assert _param_range("void fill(int a, bool b = X > 0, int c = 1, int d = 2)", "fill") == (1, 4)
    assert _param_range("void walk(Node* n = root->next, int k = 0, int j = 0)", "walk") == (0, 3)


def test_a_class_name_matches_whole(tmp_path):
    edges = _edges({"lamp.h": """\
class Lamp {
public:
    void glow();
};
class FlashLamp {
public:
    void glow();
};
""", "use/use.cpp": """\
void shine(Lamp* l) {
    l->glow();
}
"""}, tmp_path)
    assert {b for a, b, _ in edges if a == "shine" and b.endswith("glow")} == {"Lamp::glow"}


def test_a_shadowed_variable_has_no_known_type(tmp_path):
    edges = _edges({"a/lamp.h": """\
class Lamp {
public:
    void glow();
};
""", "b/mat.h": """\
class Mat {
public:
    void glow();
};
""", "use/use.cpp": """\
void shine(Lamp* x) {
    {
        Mat* x = 0;
        x->glow();
    }
}

void flash() {
    Lamp* x = 0;
    auto g = [](Mat* x) { x->glow(); };
}
"""}, tmp_path)
    for fn in ("shine", "flash"):
        assert not any(a == fn and r == "typed" for a, _, r in edges)


def test_a_class_used_as_a_type_keeps_its_constructor_arity():
    """`Angle a(v)` and `Angle* p` use the name as a type: they say nothing
    of which constructor a call reaches, so they must not switch the
    argument-count filter off as a function pointer would."""
    arities: dict = {}
    _reference_forms_all("Angle a(v);\nAngle* p = 0;\nAngle(v);\n", {"Angle"}, arities)
    assert arities["Angle"] == {1}
    arities = {}
    _reference_forms_all("Angle(v);\nrun(Angle);\n", {"Angle"}, arities)
    assert arities["Angle"] == {1, None}


def test_a_signature_is_shortened_and_never_stale():
    from srclight.server import _dedup_edges

    class S:
        def __init__(self, i, name, kind, path, line, signature):
            self.id, self.name, self.kind, self.file_path = i, name, kind, path
            self.start_line, self.signature = line, signature

    edges = [
        {"symbol": S(1, "stat", "function", "a.c", 3, "int stat(const char *path,\r\n   int x)"),
         "edge_type": "calls", "confidence": 0.5},
        {"symbol": S(2, "stat", "struct", "b.h", 9, None), "edge_type": "calls", "confidence": 0.9},
        {"symbol": S(2, "stat", "struct", "b.h", 9, None), "edge_type": "calls", "confidence": 0.9},
    ]
    (entry,) = _dedup_edges(edges)
    assert entry["kind"] == "struct" and "signature" not in entry
    assert len(entry["locations"]) == 2
    assert entry["locations"][0]["signature"] == "int stat(const char *path, int x)"


def test_a_prototype_extends_only_its_own_overload():
    from srclight.indexer import _accepts

    ctor = {"kind": "method", "signature": "Angle()",
            "other_signatures": ["Angle(short v)", "Angle(float v)", "Angle(const Angle&)"]}
    assert not _accepts(ctor, "Angle", {1})
    fn = {"kind": "function", "signature": "void log(int l, const char* t)",
          "other_signatures": ["void log(int l, const char* t = 0)", "void log(float x)"]}
    assert _accepts(fn, "log", {1}) and _accepts(fn, "log", {2})


def test_a_constructor_declared_in_its_class_is_reached(tmp_path):
    edges = _edges({"angle.h": """\
class Angle {
public:
    Angle(const Angle& other);
    Angle(short v);
    Angle(float v);
};
""", "angle.cpp": """\
#include "angle.h"

Angle::Angle(short v) {
}
""", "use/use.cpp": """\
void turn(short v) {
    Angle a(v);
    Angle(v);
}
"""}, tmp_path)
    db = Database(tmp_path / "edges" / "index.db")
    db.open()
    reached = {r[0] for r in db.conn.execute(
        """SELECT b.signature FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id WHERE a.name = 'turn'""") if r[0]}
    db.close()
    assert {"Angle(short v)", "Angle(float v)"} <= reached


def test_a_qualified_call_to_an_in_class_member_names_the_member(tmp_path):
    edges = _edges({"stack.h": """\
class Stack_c {
public:
    static void scaleBy(float f);
};
""", "use/use.cpp": """\
void grow() {
    Stack_c::scaleBy(2.0f);
}
"""}, tmp_path)
    reached = {b for a, b, _ in edges if a == "grow"}
    # The member is reached; the class it is written with stays a dependency.
    assert {"Stack_c::scaleBy", "Stack_c"} <= reached


def test_symbols_in_file_names_the_class_of_each_member(serve):
    server = serve({"flags.h": """\
class One_c {
public:
    bool checkFlag(int f);
};
class Two_c {
public:
    bool checkFlag(int f);
};
"""})
    symbols = json.loads(server.symbols_in_file("flags.h"))["symbols"]
    assert {s.get("qualified_name") for s in symbols if s["name"] == "checkFlag"} == {
        "One_c::checkFlag", "Two_c::checkFlag"}


def test_a_type_use_reaches_no_constructor_declaration(tmp_path):
    edges = _edges({"angle.h": """\
class Angle {
public:
    Angle(short v);
};
""", "use/use.cpp": """\
void keep(Angle* p) {
    Angle* q = p;
}
"""}, tmp_path)
    assert not any(a == "keep" and b == "Angle::Angle" for a, b, _ in edges)


def test_a_type_use_reaches_no_constructor_defined_in_its_class(tmp_path):
    edges = _edges({"angle.h": """\
class Angle {
public:
    Angle() {}
    Angle(const Angle& other) {}
    Angle(short v) {}
    short Val() const;
};

struct Lock_c {
    Angle mYaw;
};

inline short readAngle(const Angle& a) {
    return a.Val();
}
""", "use/use.cpp": """\
void spin(short v) {
    Angle(v);
}
"""}, tmp_path)
    ctor_callers = {a for a, b, _ in edges if b == "Angle::Angle"}
    assert "spin" in ctor_callers
    assert not ctor_callers & {"readAngle", "Lock_c"}
    assert ("readAngle", "Angle") in _pairs(edges)


def test_returning_a_class_or_destroying_it_constructs_nothing(tmp_path):
    edges = _edges({"angle.h": """\
class Angle {
public:
    Angle(short v) {}
    ~Angle() {}
};

class Cam_c {
public:
    Angle readAngle(Angle hint);
    Angle mStored;
};
""", "cam.cpp": """\
#include "angle.h"

Angle Cam_c::readAngle(Angle hint) {
    return mStored;
}

void spin(short v) {
    Angle(v);
}
"""}, tmp_path)
    ctor_callers = {a for a, b, _ in edges if b == "Angle::Angle"}
    assert "spin" in ctor_callers
    assert not ctor_callers & {"Cam_c::readAngle", "~Angle", "Angle::~Angle"}


def test_a_variable_constructed_with_arguments_calls_the_constructor(tmp_path):
    edges = _edges({"vec.h": """\
class Vec_c {
public:
    Vec_c(float x, float y, float z) {}
};
""", "use/use.cpp": """\
Vec_c handOffset() {
    Vec_c hand(0.0f, 1.0f, 2.0f);
    Vec_c braced{0.0f, 1.0f, 2.0f};
    return hand;
}

Vec_c copyOnly(const Vec_c& v) {
    return v;
}
"""}, tmp_path)
    ctor_callers = {a for a, b, _ in edges if b == "Vec_c::Vec_c"}
    assert "handOffset" in ctor_callers
    assert "copyOnly" not in ctor_callers


def test_an_inline_destructor_names_nothing(tmp_path):
    edges = _edges({"cap.cpp": """\
class Capture_c {
public:
    Capture_c() {}
    virtual ~Capture_c() {}
};
"""}, tmp_path)
    assert not any(a in ("~Capture_c", "Capture_c::~Capture_c") for a, _, _ in edges)


def test_a_macro_after_a_declarator_is_no_function_to_call(tmp_path):
    edges = _edges({"cfg.h": """\
#define WHEN_EXT(x) x
""", "heap.cpp": """\
#include "cfg.h"
void* operator new(unsigned long n) WHEN_EXT(noexcept) {
    return 0;
}

void release() {
    WHEN_EXT(freeAll());
}
"""}, tmp_path)
    assert not any(a == "release" and b == "WHEN_EXT" for a, b, _ in edges)


def test_a_macro_named_like_a_class_keeps_its_constructors(tmp_path):
    edges = _edges({"g.h": """\
#define WidgetBox(...) make_box(__VA_ARGS__)
class WidgetBox {
public:
    WidgetBox(int a, int b) : a_(a) {}
    int a_;
};
""", "g.cpp": """\
#include "g.h"
void dispatcher() {
    WidgetBox wb(1, 2);
}
"""}, tmp_path)
    assert ("dispatcher", "WidgetBox::WidgetBox") in _pairs(edges)


def test_a_name_starting_with_a_non_ascii_letter_keeps_its_calls(tmp_path):
    edges = _edges({"u.cpp": """\
int Ölstand_lesen(int a) {
    return a;
}

int caller_two(int a) {
    return Ölstand_lesen(a);
}
"""}, tmp_path)
    assert ("caller_two", "Ölstand_lesen") in _pairs(edges)


def test_constructor_declarations_are_filtered_by_argument_count(tmp_path):
    edges = _edges({"box.h": """\
class Box_c {
public:
    Box_c();
    Box_c(int w, int h);
};
""", "use/use.cpp": """\
void build() {
    Box_c b(1, 2);
}
"""}, tmp_path)
    db = Database(tmp_path / "edges" / "index.db")
    db.open()
    sigs = {r[0] for r in db.conn.execute(
        """SELECT b.signature FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id WHERE a.name = 'build' AND b.kind = 'prototype'""")}
    db.close()
    assert sigs == {"Box_c(int w, int h)"}


def test_a_namespace_qualified_type_use_reaches_no_constructor(tmp_path):
    edges = _edges({"tint.h": """\
namespace paint {
class Tint {
public:
    Tint(int r, int g) {}
};
}
""", "use/use.cpp": """\
void shade(paint::Tint* p) {
}

void make() {
    paint::Tint(1, 2);
}
"""}, tmp_path)
    ctor_callers = {a for a, b, _ in edges if b == "paint::Tint::Tint"}
    assert "make" in ctor_callers and "shade" not in ctor_callers


def test_a_class_named_like_its_namespace_keeps_its_edges(tmp_path):
    edges = _edges({"gadget.h": """\
namespace Gadget {
class Gadget {
public:
    int size;
};
}
""", "use/use.cpp": """\
void tint(Gadget* p) {
}
"""}, tmp_path)
    assert ("tint", "Gadget::Gadget") in _pairs(edges)


def test_callee_note_counts_calls_after_operators_and_labels():
    from srclight.server import _declares_or_initializes

    assert _declares_or_initializes("{\n    Timer ")
    assert _declares_or_initializes("Box(int v) : ")
    for call in ("if (a && ", "x = y * ", "return a > ", "case 1: ", "default: ",
                 "v = ok ? a : ", "return ok ? prepare(a) : ", "f(x, "):
        assert not _declares_or_initializes(call), call
    for initializer in ("Box(int v) : m(v), ", "Box(int v) noexcept : ",
                        "Box(int v) : m(v), n(w), "):
        assert _declares_or_initializes(initializer), initializer


def test_forward_declarations_do_not_make_a_defined_class_ambiguous(tmp_path):
    # Declared ahead in many headers, defined in one: the declarations name
    # the same class, and must not push it past the fan-out limit.
    files = {"math/point.h": """\
class Point_c {
public:
    Point_c() {}
    Point_c(float a, float b, float c) {}
    Point_c(const Point_c& o) {}
    float x, y, z;
};
""", "game/rig.cpp": """\
void aimRig(float fx, float fy, float fz) {
    Point_c offset(fx, fy, fz);
}
"""}
    for i in range(12):
        files[f"part{i}/fwd.h"] = "class Point_c;\nclass Other_c { public: int n; };\n"
    edges = _edges(files, tmp_path)
    assert ("aimRig", "Point_c::Point_c", "unique_file") in edges
    assert ("aimRig", "Point_c", "unique_file") in edges


def test_a_template_constructor_is_listed_once_by_its_plain_name():
    from srclight.server import _edge_name

    class S:
        def __init__(self, name, qualified, kind="method"):
            self.name, self.qualified_name, self.kind = name, qualified, kind

    assert _edge_name(S("Bag", "Bag::Bag::Bag")) == "Bag::Bag"
    assert _edge_name(S("Bag<T>::Bag", "Bag<T>::Bag")) == "Bag::Bag"
    assert _edge_name(S("Box", "ns::Box::Box")) == "ns::Box::Box"
    assert _edge_name(S("Box", "Box", "class")) == "Box"
    assert _edge_name(S("~Box", "Box::~Box")) == "~Box"


def test_forward_declarations_do_not_count_toward_the_graph_note(serve):
    files = {"math/point.h": """\
class Point_c {
public:
    float x;
};
""", "game/rig.cpp": """\
void aimRig(Point_c* p) {
}
"""}
    for i in range(12):
        files[f"part{i}/fwd.h"] = "class Point_c;\n"
    answer = json.loads(serve(files).get_callers("Point_c"))
    assert "aimRig" in {c["name"] for c in answer["callers"]}
    assert "graph_note" not in answer, answer.get("graph_note")


def test_a_forward_declaration_is_no_target_through_its_namespace(tmp_path):
    files = {"geo/shape.h": """\
namespace geo {
class Shape_c {
public:
    int sides;
};
}
""", "draw/pen.h": """\
namespace geo {
class Shape_c;
}
""", "draw/pen.cpp": """\
void outline(const geo::Shape_c* s) {
}
"""}
    edges = _edges(files, tmp_path)
    targets = {(b, r) for a, b, r in edges if a == "outline"}
    assert targets == {("geo::Shape_c", "unique_file")}, targets


def test_a_forward_declaration_in_the_same_file_still_tells_the_class(tmp_path):
    # The file declares the class it means; the name alone is defined too
    # often elsewhere to tell.
    files = {"ui/panel.h": """\
class Skin_c;

class Panel_c {
public:
    virtual void apply(Skin_c* skin) = 0;
};
""", "ui/skin/skin.h": """\
class Skin_c {
public:
    int tone;
};
"""}
    for i in range(11):
        files[f"actor{i}/a.cpp"] = f"enum Skin_c {{ Plain{i}, Other{i} }};\n"
    edges = _edges(files, tmp_path)
    assert any(a == "Panel_c::apply" and b == "Skin_c" for a, b, _ in edges), edges


def test_a_class_of_another_language_does_not_replace_a_forward_declaration(tmp_path):
    db_root = tmp_path / "edges"
    edges = _edges({"ffi/handle.h": """\
class Handle_c;
void useHandle(Handle_c* h);
""", "ffi/handle.cpp": """\
#include "handle.h"
void useHandle(Handle_c* h) {
}
""", "script/model.py": """\
class Handle_c:
    pass
"""}, tmp_path)
    db = Database(db_root / "index.db")
    db.open()
    files = {r[0].replace("\\", "/") for r in db.conn.execute(
        """SELECT f.path FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id JOIN files f ON f.id = b.file_id
           WHERE a.name = 'useHandle' AND b.name = 'Handle_c'""")}
    db.close()
    assert files == {"ffi/handle.h"}, files


def test_callees_list_constructors_apart_from_their_class(serve):
    server = serve({"angle.h": """\
class Turn_c {
public:
    Turn_c() {}
    Turn_c(short v) {}
    short mV;
};
""", "use/use.cpp": """\
void steer(Turn_c* t) {
    Turn_c(5);
}
"""})
    callees = json.loads(server.get_callees("steer"))["callees"]
    kinds = {e["kind"] for e in callees if e["name"] in ("Turn_c", "Turn_c::Turn_c")}
    assert "class" in kinds and kinds - {"class"}, callees


def _edge_signatures(tmp_path, source: str, target: str) -> set:
    db = Database(tmp_path / "edges" / "index.db")
    db.open()
    sigs = {r[0] for r in db.conn.execute(
        """SELECT b.signature FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id WHERE a.name = ? AND b.name = ?""",
        (source, target))}
    db.close()
    return sigs


def test_a_string_argument_counts_as_an_argument(tmp_path):
    _edges({"log/log.cpp": """\
void emitLine() {
}

void emitLine(const char* text) {
}
""", "use/use.cpp": """\
void driver() {
    emitLine("hello");
}
"""}, tmp_path)
    assert _edge_signatures(tmp_path, "driver", "emitLine") == {"void emitLine(const char* text)"}


def test_a_forward_declaration_in_the_calling_file_leads_to_the_definition(tmp_path):
    db_root = tmp_path / "edges"
    _edges({"heap/heap.h": """\
class HeapArena {
public:
    int used;
};
""", "use/user.cpp": """\
class HeapArena;

int useArena(HeapArena* arena) {
    return 0;
}
"""}, tmp_path)
    db = Database(db_root / "index.db")
    db.open()
    files = {r[0].replace("\\", "/") for r in db.conn.execute(
        """SELECT f.path FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id JOIN files f ON f.id = b.file_id
           WHERE a.name = 'useArena' AND b.name = 'HeapArena'""")}
    db.close()
    assert files == {"heap/heap.h"}, files


def test_a_constructor_written_with_its_full_namespace_path_is_reached(tmp_path):
    edges = _edges({"gfx/tint.h": """\
namespace app {
namespace gfx {
class Tinter {
public:
    Tinter(int r, int g) {}
};
}
}
""", "use/use.cpp": """\
void paintOne() {
    app::gfx::Tinter(1, 2);
}
"""}, tmp_path)
    assert ("paintOne", "app::gfx::Tinter::Tinter") in _pairs(edges)


def test_a_construction_with_braces_calls_the_constructor(tmp_path):
    edges = _edges({"box.h": """\
class Boxer {
public:
    Boxer(int w, int h) {}
};
""", "use/use.cpp": """\
Boxer makeOne() {
    return Boxer{1, 2};
}

void keepOne(const Boxer& b) {
}
"""}, tmp_path)
    ctor_callers = {a for a, b, _ in edges if b == "Boxer::Boxer"}
    assert "makeOne" in ctor_callers and "keepOne" not in ctor_callers


def test_a_signature_counts_no_comma_inside_a_comment_or_a_literal():
    from srclight.indexer import _param_range

    assert _param_range("void f(int a, // width, px\n int b)", "f") == (2, 2)
    assert _param_range("void f(int a, /* (x, y) */ int b)", "f") == (2, 2)
    assert _param_range('void join(const char* s = "a,b")', "join") == (0, 1)
    assert _param_range("void sep(char c = ',')", "sep") == (0, 1)


def test_a_comparison_in_a_default_argument_opens_no_template():
    from srclight.indexer import _param_range

    assert _param_range("void f(int a, bool b = x < 3, int c)", "f") == (2, 3)
    assert _param_range("void setMode(int m, int mask = 1 << 3)", "setMode") == (1, 2)
    assert _param_range("void put(std::map<int, int> m, int n)", "put") == (2, 2)
    assert _param_range("void pair(std::pair<std::vector<int>, int> p)", "pair") == (1, 1)


def test_a_constructor_is_named_by_whole_names():
    from srclight.indexer import _is_constructor

    assert _is_constructor({"kind": "method", "qualified": "Box::Box"}, "Box")
    assert _is_constructor({"kind": "method", "qualified": "ns::Box::Box"}, "Box")
    assert not _is_constructor({"kind": "method", "qualified": "MyBox::Box"}, "Box")


def test_a_literal_inside_a_comment_is_no_argument():
    from srclight.indexer import _call_arity, _literals_as_values
    from srclight.refmask import mask_noncode

    text = 'reset(/* "hard" */);'
    content = _literals_as_values(text, mask_noncode(text, "cpp"))
    assert _call_arity(content, text.index("(")) == 0
    text = 'reset("hard");'
    content = _literals_as_values(text, mask_noncode(text, "cpp"))
    assert _call_arity(content, text.index("(")) == 1



OWNER_FUNCTIONS = {
    "a/menu.cpp": "namespace game { namespace menu {\nint owner() { return 0; }\n} }\n",
    "b/event.cpp": "namespace game { namespace event {\nint owner() { return 1; }\n} }\n",
}


def test_a_field_compared_after_a_member_access_is_no_template_call(tmp_path):
    edges = _edges({**OWNER_FUNCTIONS, "c/track.cpp": """\
struct Track { int owner; };
bool claimed(Track* slot) {
    return slot->owner < 0 || slot->owner > 3;
}

int firstOwner(Holder* h) {
    return h->owner<int>();
}
"""}, tmp_path)
    targets = {b for a, b, _ in edges if a == "claimed"}
    assert not {t for t in targets if t.endswith("owner")}, targets
    assert any(a == "firstOwner" and b.endswith("owner") for a, b, _ in edges), edges


def test_a_field_with_a_namespace_qualified_type_is_a_declaration(tmp_path):
    edges = _edges({**OWNER_FUNCTIONS, "c/record.h": """\
namespace flow {
struct Record {
    mods::Module* owner = nullptr;
    int count;
};
}
"""}, tmp_path)
    assert not any(a.endswith("Record") and b.endswith("owner") for a, b, _ in edges), edges


def test_a_python_method_called_on_self_is_its_own_class_method(tmp_path):
    body = "".join(
        f"class Widget{i}:\n"
        f"    def __init__(self):\n"
        f"        self.render_item()\n\n"
        f"    def render_item(self):\n"
        f"        pass\n\n"
        for i in range(12))
    body += "class Plain(Widget0):\n    def show(self):\n        self.render_item()\n\n"
    body += "def render_item():\n    pass\n"
    edges = _edges({"ui/widgets.py": body}, tmp_path)
    from_init = {(a, b, r) for a, b, r in edges if a.endswith(".__init__")}
    assert from_init == {(f"Widget{i}.__init__", f"Widget{i}.render_item", "same_class")
                         for i in range(12)}, sorted(from_init)[:5]
    from_show = {b for a, b, _ in edges if a == "Plain.show"}
    assert "render_item" not in from_show, from_show
