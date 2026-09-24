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


def test_a_class_template_does_not_call_the_methods_it_declares(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "panel.h").write_text(
        PANEL_H.replace("class Panel_c {", "template <typename T>\nclass Panel_c {")
    )
    (root / "panel.cpp").write_text(PANEL_CPP)

    Indexer(db, IndexConfig(root=root)).index()

    callees = _callees(db, "Panel_c")
    assert "refreshLayout" not in callees
    assert "computeMargins" not in callees


def test_a_trait_does_not_repeat_the_calls_of_its_default_methods(tmp_path, db):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "shapes.rs").write_text("""\
trait Drawable {
    fn paint_all(&self) {
        self.stroke_edges();
    }

    fn stroke_edges(&self) {
        fill_region();
    }
}

fn fill_region() {
}
""")

    Indexer(db, IndexConfig(root=root)).index()

    assert "fill_region" in _callees(db, "stroke_edges")
    assert _callees(db, "Drawable") == set()


def test_no_symbol_a_swift_class_yields_calls_its_methods(tmp_path, db):
    """The Swift query reads each class declaration as a class, a struct and
    an enum at once. Every one of those twins holds the methods."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Panel.swift").write_text("""\
class Panel {
    func refreshLayout() {
        computeMargins()
    }

    func computeMargins() {
    }
}
""")

    Indexer(db, IndexConfig(root=root)).index()

    assert "computeMargins" in _callees(db, "refreshLayout")
    assert _callees(db, "Panel") == set()


def test_a_function_keeps_the_calls_of_a_function_nested_in_it(tmp_path, db):
    """Only callables keep their whole body: an outer function does call
    what the function it defines calls, and that edge must stay."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "jobs.py").write_text("""\
def schedule_jobs():
    def run_batch():
        flush_queue()
    return run_batch


def flush_queue():
    return 1
""")

    Indexer(db, IndexConfig(root=root)).index()

    assert "flush_queue" in _callees(db, "schedule_jobs")


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


def test_a_define_in_a_container_hides_only_its_own_line(tmp_path):
    """A `#define` node takes its trailing newline, so its recorded end is
    the next line; blanking that span in its container hid the line after
    it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "w.c").write_text(
        "int helper_value(void) { return 1; }\n"
        "struct Levels {\n"
        "#define IN_STRUCT_MACRO 1\n"
        "    char buf[helper_value()];\n"
        "};\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    edges = {tuple(r) for r in db.conn.execute(
        """SELECT a.name, b.name FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id""")}
    db.close()
    assert ("Levels", "helper_value") in edges


def test_a_macro_reads_its_own_body(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "log.c").write_text(
        "void report(const char* m) {\n}\n#define SHOUT(msg) report(msg)\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    edges = {tuple(r) for r in db.conn.execute(
        """SELECT a.name, b.name FROM symbol_edges e JOIN symbols a ON a.id = e.source_id
           JOIN symbols b ON b.id = e.target_id""")}
    db.close()
    assert ("SHOUT", "report") in edges
