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


def test_a_macro_may_redefine_a_keyword(tmp_path, db):
    """`#define inline __inline` is legal and common in compatibility
    headers: the macro is a real definition."""
    assert _names(tmp_path, db, "compat.c", """\
#define inline __inline
#define restrict __restrict

static inline int clampLevel(int level) {
    return level < 0 ? 0 : level;
}
""") == ["inline", "restrict", "clampLevel"]


def test_a_c_header_read_as_cpp_keeps_its_c_names(tmp_path, db):
    """A `.h` file is read as C++ as soon as `::` shows up near its top, even
    in a comment. Its definitions named `new` or `delete` still parse
    cleanly at file scope, where error recovery has no part: they are real."""
    assert _names(tmp_path, db, "pool.h", """\
/* see Pool::grab */
void *new(int size) { return allocBlock(size); }
void delete(void *block) { freeBlock(block); }
struct class { int size; };
""") == ["new", "delete", "class"]


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


def test_an_error_in_the_body_does_not_disown_a_real_definition(tmp_path, db):
    """A macro used without its semicolon is common in C, and puts an error
    inside the body. Only an error in the head says the definition itself is
    error recovery's work."""
    assert _names(tmp_path, db, "pool.h", """\
/* see Pool::grab */
void *new(int size) {
    void *block = allocBlock(size);
    TRACE_ALLOC(block)
    return block;
}
""") == ["new"]


@pytest.mark.parametrize("filename, text", [
    ("opcodes.h", "VM_PROLOGUE\nswitch (opcode) {\ncase 0:\n    break;\n}\n"),
    ("spin.c", "LOCK_GUARD(mutex)\nwhile (busy) {\n    spinOnce();\n}\n"),
])
def test_a_c_keyword_never_names_a_symbol(tmp_path, db, filename, text):
    """A statement fragment meant to be included inside a function parses at
    file scope without any error, and `switch (x) { ... }` after a macro reads
    as a function `switch`. No symbol in C or C++ can be named `switch`,
    `while` or `if`, wherever it sits."""
    assert _names(tmp_path, db, filename, text) == []
