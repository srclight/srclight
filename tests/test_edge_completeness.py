"""The call graph must not drop calls a function really makes.

The edge builder used to stop after 30 edges per symbol, walking the
referenced names in set order. A long function kept an arbitrary 30 of its
calls, and which 30 changed from one process to the next with the string hash
seed, so two reindexes of the same tree answered get_callers differently.
"""
import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    yield db
    db.close()


def _callees(db, source_name: str) -> set[str]:
    rows = db.conn.execute(
        """SELECT t.name FROM symbol_edges e
           JOIN symbols s ON e.source_id = s.id
           JOIN symbols t ON e.target_id = t.id
           WHERE s.name = ? AND e.edge_type = 'calls'""",
        (source_name,),
    ).fetchall()
    return {r["name"] for r in rows}


def test_a_function_keeps_every_call_it_makes(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    layers = [f"paintLayer{i:02d}" for i in range(45)]
    definitions = "".join(f"void {name}() {{\n}}\n\n" for name in layers)
    calls = "".join(f"    {name}();\n" for name in layers)
    (root / "canvas.cpp").write_text(
        definitions + "void paintEverything() {\n" + calls + "}\n"
    )

    Indexer(db, IndexConfig(root=root)).index()

    assert _callees(db, "paintEverything") == set(layers)


# Without the cap, a container symbol must not flood the graph: its body
# overlaps its members' bodies, so scanning all of it re-reports every call
# they make, and names each method it declares as something it "calls".

PANEL_H = """\
class Renderer_c {
public:
    void flushQueue();
};

class Panel_c {
public:
    void refreshLayout();
    void computeMargins();
    Renderer_c* mRenderer;
};
"""

PANEL_CPP = """\
#include "panel.h"

void Panel_c::refreshLayout() {
    computeMargins();
}

void Panel_c::computeMargins() {
}
"""


@pytest.fixture
def panel_repo(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "panel.h").write_text(PANEL_H)
    (root / "panel.cpp").write_text(PANEL_CPP)
    Indexer(db, IndexConfig(root=root)).index()
    return root


def test_a_class_does_not_call_the_methods_it_declares(panel_repo, db):
    callees = _callees(db, "Panel_c")

    assert "refreshLayout" not in callees
    assert "computeMargins" not in callees


def test_a_class_still_references_the_types_of_its_members(panel_repo, db):
    assert "Renderer_c" in _callees(db, "Panel_c")


def test_a_namespace_does_not_repeat_the_calls_of_its_functions(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "outline.cpp").write_text("""\
namespace shapes {

void traceOutline() {
    fillInterior();
}

void fillInterior() {
}

}
""")

    Indexer(db, IndexConfig(root=root)).index()

    assert "fillInterior" in _callees(db, "traceOutline")
    assert _callees(db, "shapes") == set()
