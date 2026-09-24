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
    "free/free.cpp": """\
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
    assert _reference_forms("::scaleVec(); scaleVec();", "scaleVec") == ({"bare"}, set())


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
    assert "not in the graph" in payload["graph_note"]
    assert r"find_pattern(r'\breset\s*\(')" in payload["graph_note"]
