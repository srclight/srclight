"""A C or C++ keyword is never the name of a symbol.

Error recovery can read a statement as a definition: `if (a == b) { ... }`
cut off from its chain by an #if looks like a function `if` taking
`(a == b)`, and `switch (x) { ... }` after a macro used without its
semicolon looks like a function `switch` returning that macro. Those
pseudo-symbols then became call targets for every `if` or `switch` written
anywhere else.
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


def _names(tmp_path, db, filename: str, text: str) -> list[str]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / filename).write_text(text)
    Indexer(db, IndexConfig(root=root)).index()
    return [r["name"] for r in db.conn.execute(
        "SELECT name FROM symbols ORDER BY start_line"
    )]


def test_an_else_if_cut_off_by_a_conditional_is_not_a_function(tmp_path, db):
    assert _names(tmp_path, db, "fmt.c", """\
static int formatFlag(int mode) {
    switch (mode) {
    case 'x':
        if (mode == 1) {
            emitHex(1);
        } else if (mode == 2) {
            emitHex(2);
        }
#if !COMPACT_BUILD
        else if (mode == 3) {
            emitHex(3);
        }
#endif
        break;
    }
    return 0;
}
""") == ["formatFlag"]


def test_a_switch_after_a_macro_without_its_semicolon_is_not_a_function(tmp_path, db):
    assert _names(tmp_path, db, "scene.cpp", """\
int Scene_c::drawScene() {
    TRACE_SCOPE(Scene_c::drawScene())
    getRenderer().flushQueue();
    PROFILE_MARK

    switch (mMode) {
    case 0:
        drawIdle();
        break;
    }
    return 1;
}
""") == ["Scene_c::drawScene"]


def test_a_cpp_keyword_is_an_ordinary_name_in_c(tmp_path, db):
    """`new` and `delete` are reserved in C++ only."""
    assert _names(tmp_path, db, "pool.c", """\
void *new(int size) {
    return allocBlock(size);
}

void delete(void *block) {
    freeBlock(block);
}
""") == ["new", "delete"]
