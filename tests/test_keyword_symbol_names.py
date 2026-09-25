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


@pytest.mark.parametrize("text, expected", [
    # An error inside an enum's body says nothing about its name.
    ("// namespace probe\nenum new { ALPHA = 1, BETA = FLAG(2) GAMMA };\n", ["new"]),
    # An export macro reads as a type in the C++ grammar and errs the head.
    ("// namespace probe\nAPI_EXPORT int new(int size) { return size; }\n", ["new"]),
    # A struct local to a function is legal C, whatever its name.
    ("// namespace probe\n"
     "void outerPool(void) {\n"
     "    struct new { int size; } slot;\n"
     "    slot.size = 1;\n"
     "}\n",
     ["outerPool", "new"]),
])
def test_real_c_names_that_cpp_reserves_survive(tmp_path, db, text, expected):
    """A `.h` file with `namespace ` near its top is read as C++, and C code
    may name things `new`, `delete` or `class`. Only a function read inside
    another function's body is error recovery's work."""
    assert _names(tmp_path, db, "pool.h", text) == expected


def test_a_variable_typed_by_a_macro_is_no_function(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "cfg.h").write_text("#define PICK_TYPE(a, b) a\n")
    (root / "menu.h").write_text(
        "#include \"cfg.h\"\nclass Menu_c {\npublic:\n    PICK_TYPE(float, short) mScale;\n};\n")
    (root / "rain.cpp").write_text(
        "#include \"cfg.h\"\nstatic PICK_TYPE(float, short) s_spin = 0;\n"
        "int realOne(int v) {\n    return v;\n}\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    kinds = {(r[0], r[1]) for r in db.conn.execute("SELECT name, kind FROM symbols")}
    db.close()
    assert ("PICK_TYPE", "macro") in kinds
    assert not any(n == "PICK_TYPE" and k != "macro" for n, k in kinds)
    assert ("realOne", "function") in kinds


def test_a_class_field_typed_by_a_macro_is_no_prototype(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "cfg.h").write_text("#define PICK_TYPE(a, b) a\n")
    (root / "panel.h").write_text(
        "#include \"cfg.h\"\nclass Panel_c {\npublic:\n    void redraw();\n"
        "    /* 0x10 */ PICK_TYPE(float, short) mSlots[150];\n    int mCount;\n};\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    kinds = {(r[0], r[1]) for r in db.conn.execute("SELECT name, kind FROM symbols")}
    db.close()
    assert not any(n == "PICK_TYPE" and k != "macro" for n, k in kinds)
    assert any(n == "redraw" for n, _ in kinds)


def test_a_constructor_declared_with_a_trailing_macro_is_kept(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.h").write_text(
        "#define NOEXCEPT_M\n#define DEPRECATED_M\n"
        "class Holder {\npublic:\n    Holder() NOEXCEPT_M;\n    Holder(int x) DEPRECATED_M;\n"
        "    explicit Holder(float) DEPRECATED_M;\n    int x;\n};\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    ctors = db.conn.execute(
        "SELECT COUNT(*) FROM symbols WHERE name = 'Holder' AND kind <> 'class'").fetchone()[0]
    db.close()
    assert ctors == 3
