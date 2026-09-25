"""A qualified C++ method name must reach both halves of the method.

A method declared in its class and defined outside it is two symbols: the
declaration, named `moveNeedle` with qualified name `Panel_c::moveNeedle`,
and the definition, named `Panel_c::moveNeedle`. Calls written
`panel->moveNeedle()` land on the declaration, while the body scanned for
callees is the definition's. Resolving the qualified name to the definition
alone, get_callees found the method's calls and get_callers found nobody.
"""
import json

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer

PANEL = {
    "panel.h": """\
class Panel_c {
public:
    void moveNeedle();
};
""",
    "panel.cpp": """\
#include "panel.h"

void Panel_c::moveNeedle() {
    clearGauge();
}

void clearGauge() {
}

void runPanel(Panel_c* panel) {
    panel->moveNeedle();
}
""",
}

# The same class name in two namespaces. Every qualified name carries the
# namespace, while the definition's name is only class-qualified.
KNOBS = {
    "knobs.h": """\
namespace alpha {
class Knob_c {
public:
    void turnKnob();
};
}

namespace beta {
class Knob_c {
public:
    void turnKnob();
};
}
""",
    "knobs.cpp": """\
#include "knobs.h"

namespace alpha {
void Knob_c::turnKnob() {
    alphaWork();
}
}

namespace beta {
void Knob_c::turnKnob() {
    betaWork();
}
}

void alphaWork() {
}

void betaWork() {
}

void useAlpha(alpha::Knob_c* knob) {
    knob->turnKnob();
}
""",
}

GADGET = {
    "gadget.h": """\
class Gadget_c {
public:
    void spinGadget();
    void spinGadget(int turns);
};
""",
    "gadget.cpp": """\
#include "gadget.h"

void Gadget_c::spinGadget() {
    helperTick();
}

void Gadget_c::spinGadget(int turns) {
    helperTick();
}

void helperTick() {
}

void driveGadget(Gadget_c* gadget) {
    gadget->spinGadget();
}
""",
}


@pytest.fixture
def serve(tmp_path, monkeypatch):
    from srclight import server as server_mod

    def _serve(files: dict[str, str]):
        root = tmp_path / "repo"
        root.mkdir()
        for name, text in files.items():
            (root / name).write_text(text)
        db_path = root / ".srclight" / "index.db"
        db_path.parent.mkdir()
        db = Database(db_path)
        db.open()
        db.initialize()
        Indexer(db, IndexConfig(root=root)).index()
        db.close()
        # _get_db() walks up from the CWD — never let a test reach a real index.
        monkeypatch.chdir(root)
        monkeypatch.setattr(server_mod, "_workspace_name", None)
        server_mod.configure(db_path=db_path, repo_root=root)
        return server_mod

    yield _serve
    server_mod._close_databases()
    server_mod.configure(db_path=None, repo_root=None)


def _entries(payload: str, key: str) -> list[dict]:
    return json.loads(payload).get(key, [])


def _names(payload: str, key: str) -> set[str]:
    return {entry["name"] for entry in _entries(payload, key)}


def test_get_callers_finds_callers_by_the_qualified_name(serve):
    server = serve(PANEL)
    assert "runPanel" in _names(server.get_callers("Panel_c::moveNeedle"), "callers")


def test_get_callees_still_finds_callees_by_the_qualified_name(serve):
    server = serve(PANEL)
    assert "clearGauge" in _names(server.get_callees("Panel_c::moveNeedle"), "callees")


def test_the_bare_name_keeps_resolving_as_before(serve):
    server = serve(PANEL)
    assert "runPanel" in _names(server.get_callers("moveNeedle"), "callers")


def test_dependents_and_impact_agree_with_callers(serve):
    """Every graph tool must answer for the same symbols, or get_impact
    calls a method with a caller an untouched entry point."""
    server = serve(PANEL)

    assert "runPanel" in _names(server.get_dependents("Panel_c::moveNeedle"), "dependents")
    impact = json.loads(server.get_impact("Panel_c::moveNeedle"))
    assert impact["direct_dependents"] >= 1


def test_a_class_qualified_name_reaches_a_method_inside_a_namespace(serve):
    server = serve(KNOBS)
    assert "useAlpha" in _names(server.get_callers("Knob_c::turnKnob"), "callers")


def test_a_fully_qualified_name_keeps_to_its_namespace(serve):
    server = serve(KNOBS)
    callees = _names(server.get_callees("alpha::Knob_c::turnKnob"), "callees")
    assert "alphaWork" in callees
    assert "betaWork" not in callees


def test_a_name_that_matches_several_methods_says_so(serve):
    """`Knob_c::turnKnob` names a method in each namespace. Merging them is
    the honest answer to an ambiguous name — provided the answer names what
    it merged."""
    server = serve(KNOBS)
    payload = json.loads(server.get_callees("Knob_c::turnKnob"))
    assert {"alpha::Knob_c::turnKnob", "beta::Knob_c::turnKnob"} <= set(payload["matched_symbols"])


def test_overloads_do_not_list_a_location_twice(serve):
    server = serve(GADGET)
    for entry in _entries(server.get_callers("Gadget_c::spinGadget"), "callers"):
        locations = entry.get("locations", [])
        assert len(locations) == len({(loc["file"], loc["line"]) for loc in locations})


MILL = {
    "mill.h": """\
class Mill_c {
public:
    static void grindGrain();
    void firstStep();
    void secondStep();
};
""",
    "mill.cpp": """\
#include "mill.h"

void Mill_c::grindGrain() {
}

void Mill_c::firstStep() {
    grindGrain();
}

void Mill_c::secondStep() {
    grindGrain();
}

void freeOne() {
    Mill_c::grindGrain();
}

void freeTwo() {
    Mill_c::grindGrain();
}
""",
}

COG = {
    "cog.h": """\
namespace gear {
class Cog_c {
public:
    void spinCog();
};
}
""",
    "cog.cpp": """\
#include "cog.h"

using namespace gear;

void Cog_c::spinCog() {
    helperTurn();
}

void helperTurn() {
}

void runCog(gear::Cog_c* cog) {
    cog->spinCog();
}
""",
}


def test_impact_counts_the_callers_of_both_halves(serve):
    """Unqualified calls land on the declaration and `C::f()` calls on the
    definition: impact must count both, as get_callers does."""
    server = serve(MILL)
    callers = _names(server.get_callers("Mill_c::grindGrain"), "callers")
    assert {"firstStep", "secondStep", "freeOne", "freeTwo"} <= {
        name.split("::")[-1] for name in callers
    }

    impact = json.loads(server.get_impact("Mill_c::grindGrain"))
    assert impact["direct_dependents"] >= 4


def test_a_namespace_qualified_name_reaches_a_definition_written_under_using(serve):
    """Under `using namespace gear;` the definition is stored as `Cog_c::f`
    while its declaration is `gear::Cog_c::f`: one method either way."""
    server = serve(COG)
    for name in ("gear::Cog_c::spinCog", "Cog_c::spinCog"):
        payload = json.loads(server.get_callees(name))
        assert "helperTurn" in {e["name"] for e in payload["callees"]}, name
        assert "matched_symbols" not in payload, name


GATES = {
    "gate.h": """\
class Gate_c {
public:
    void openGate();
};

namespace yard {
class Gate_c {
public:
    void openGate();
};
}
""",
    "gate.cpp": """\
#include "gate.h"

void Gate_c::openGate() {
    globalWork();
}

namespace yard {
void Gate_c::openGate() {
    yardWork();
}
}

void globalWork() {
}

void yardWork() {
}

void globalCaller(Gate_c* gate) {
    gate->openGate();
}
""",
}


def test_a_namespaced_name_does_not_reach_a_global_class_of_the_same_name(serve):
    """`Gate_c::openGate` is the short form of `yard::Gate_c::openGate` only
    when no global `Gate_c` exists; here one does, and it is another class."""
    server = serve(GATES)
    callees = _names(server.get_callees("yard::Gate_c::openGate"), "callees")
    assert "yardWork" in callees
    assert "globalWork" not in callees


def test_merging_two_classes_is_reported(serve):
    server = serve(GATES)
    payload = json.loads(server.get_callees("Gate_c::openGate"))
    assert {"Gate_c::openGate", "yard::Gate_c::openGate"} <= set(payload["matched_symbols"])


def test_impact_lists_each_flow_once(serve):
    server = serve(GATES)
    flows = json.loads(server.get_impact("Gate_c::openGate"))["affected_flows"]
    assert len(flows) == len(set(flows))


def test_a_function_defined_in_a_namespace_block_is_found_by_its_qualified_name(serve):
    server = serve({"events.cpp": """\
namespace game {
namespace events {
int holder() {
    return 0;
}
}
}
"""})
    found = json.loads(server.get_symbol("game::events::holder"))
    assert "error" not in found, found
    assert "return 0" in json.dumps(found)
    signature = json.loads(server.get_signature("game::events::holder"))
    assert "error" not in signature, signature
