"""Tests for the tree-sitter indexer."""

import tempfile
from pathlib import Path

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal sample project."""
    src = tmp_path / "project"
    src.mkdir()

    # Python file
    (src / "main.py").write_text('''\
def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
''')

    # Another Python file
    (src / "utils.py").write_text('''\
import os

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    with open(path) as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
''')

    return src


@pytest.fixture
def c_project(tmp_path):
    """Create a minimal C project."""
    src = tmp_path / "cproject"
    src.mkdir()

    (src / "main.c").write_text('''\
#include <stdio.h>

/* Print a greeting message. */
void greet(const char* name) {
    printf("Hello, %s!\\n", name);
}

int main(int argc, char** argv) {
    greet("World");
    return 0;
}
''')

    (src / "utils.h").write_text('''\
#ifndef UTILS_H
#define UTILS_H

typedef struct {
    int x;
    int y;
} Point;

int distance(Point a, Point b);

#endif
''')

    return src


def test_index_python(db, sample_project):
    """Indexes Python files and extracts symbols."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)
    stats = indexer.index(sample_project)

    assert stats.files_scanned == 2
    assert stats.files_indexed == 2
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    # Check symbols were created
    db_stats = db.stats()
    assert db_stats["files"] == 2
    assert db_stats["symbols"] > 0
    assert "python" in db_stats["languages"]

    # Check specific symbols
    syms = db.symbols_in_file("main.py")
    names = [s.name for s in syms]
    assert "hello" in names
    assert "Calculator" in names

    syms = db.symbols_in_file("utils.py")
    names = [s.name for s in syms]
    assert "read_file" in names
    assert "write_file" in names


def test_index_c(db, c_project):
    """Indexes C files and extracts symbols."""
    config = IndexConfig(root=c_project)
    indexer = Indexer(db, config)
    stats = indexer.index(c_project)

    assert stats.files_indexed == 2
    assert stats.symbols_extracted > 0

    syms = db.symbols_in_file("main.c")
    names = [s.name for s in syms]
    assert "greet" in names
    assert "main" in names


@pytest.fixture
def pointer_return_project(tmp_path):
    """A C and a C++ file whose functions return pointers."""
    src = tmp_path / "ptrproject"
    src.mkdir()

    (src / "alloc.c").write_text('''\
typedef struct Node {
    int value;
} Node;

Node* node_create(int value);
Node** node_table(void);

Node* node_create(int value) {
    return 0;
}

Node** node_table(void) {
    return 0;
}

Node* node_clone(Node* node) {
    return node_create(node->value);
}
''')

    (src / "alloc.cpp").write_text('''\
struct Buffer {
    int size;
};

Buffer* buffer_create(int size);

Buffer* buffer_create(int size) {
    return new Buffer{size};
}
''')

    return src


def test_index_pointer_returning_functions(db, pointer_return_project):
    """Indexes C and C++ functions whose return type is a pointer."""
    config = IndexConfig(root=pointer_return_project)
    indexer = Indexer(db, config)
    indexer.index(pointer_return_project)

    c_syms = db.symbols_in_file("alloc.c")
    assert {"node_create", "node_table", "node_clone"} <= {
        s.name for s in c_syms if s.kind == "function"
    }
    assert {"node_create", "node_table"} <= {
        s.name for s in c_syms if s.kind == "prototype"
    }

    cpp_syms = db.symbols_in_file("alloc.cpp")
    assert "buffer_create" in {s.name for s in cpp_syms if s.kind == "function"}
    assert "buffer_create" in {s.name for s in cpp_syms if s.kind == "prototype"}

    # node_clone calls node_create, and both return a pointer. The edge exists
    # only if both captures mapped to a kind: an unmapped one becomes "unknown",
    # which EDGE_TARGET_KINDS filters out.
    node_create = db.get_symbol_by_name("node_create")
    assert node_create is not None
    assert "node_clone" in [c["symbol"].name for c in db.get_callers(node_create.id)]


def test_incremental_index(db, sample_project):
    """Incremental indexing skips unchanged files."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)

    # First index
    stats1 = indexer.index(sample_project)
    assert stats1.files_indexed == 2

    # Second index — nothing changed
    stats2 = indexer.index(sample_project)
    assert stats2.files_indexed == 0
    assert stats2.files_unchanged == 2

    # Modify a file
    (sample_project / "main.py").write_text("def new_function(): pass\n")

    # Third index — only modified file re-indexed
    stats3 = indexer.index(sample_project)
    assert stats3.files_indexed == 1
    assert stats3.files_unchanged == 1


def test_search_after_index(db, sample_project):
    """Search works after indexing."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)
    indexer.index(sample_project)

    # Search by function name
    results = db.search_symbols("hello")
    assert len(results) > 0
    assert any(r["name"] == "hello" for r in results)

    # Search by class name
    results = db.search_symbols("Calculator")
    assert len(results) > 0

    # Search by doc content
    results = db.search_symbols("greet someone")
    assert len(results) > 0


@pytest.fixture
def markdown_project(tmp_path):
    """Create a minimal Markdown project."""
    src = tmp_path / "mdproject"
    src.mkdir()

    (src / "notes.md").write_text('''\
---
title: Architecture Notes
tags: [design, architecture]
---

# Architecture

Overall system design.

## Components

The main components are listed here.

### Database Layer

SQLite with FTS5 indexes.

## Deployment

Run on any Linux server.
''')

    (src / "plain.md").write_text('''\
Just a file with no headings.

Some plain text content.
''')

    (src / "single-heading.md").write_text('''\
# Quick Note

A brief note with only one heading.
''')

    return src


def test_index_markdown(db, markdown_project):
    """Indexes Markdown files and extracts heading sections as symbols."""
    config = IndexConfig(root=markdown_project)
    indexer = Indexer(db, config)
    stats = indexer.index(markdown_project)

    assert stats.files_scanned == 3
    assert stats.files_indexed == 3
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    # Check notes.md — should have 4 section symbols
    syms = db.symbols_in_file("notes.md")
    names = [s.name for s in syms]
    assert "Architecture" in names
    assert "Components" in names
    assert "Database Layer" in names
    assert "Deployment" in names

    # Check kinds are all "section"
    assert all(s.kind == "section" for s in syms)

    # Check qualified names use ">" ancestry
    arch = [s for s in syms if s.name == "Architecture"][0]
    assert arch.qualified_name == "notes > Architecture"
    db_layer = [s for s in syms if s.name == "Database Layer"][0]
    assert db_layer.qualified_name == "notes > Architecture > Components > Database Layer"

    # Check own-content: "Components" section shouldn't include "Database Layer" content
    components = [s for s in syms if s.name == "Components"][0]
    assert "The main components" in components.content
    assert "SQLite" not in components.content


def test_index_markdown_no_headings(db, markdown_project):
    """Markdown file without headings produces a single document symbol."""
    config = IndexConfig(root=markdown_project)
    indexer = Indexer(db, config)
    indexer.index(markdown_project)

    syms = db.symbols_in_file("plain.md")
    assert len(syms) == 1
    assert syms[0].kind == "document"
    assert syms[0].name == "plain"
    assert "no headings" in syms[0].content


def test_index_markdown_frontmatter(db, markdown_project):
    """YAML frontmatter is extracted as doc_comment on the first symbol."""
    config = IndexConfig(root=markdown_project)
    indexer = Indexer(db, config)
    indexer.index(markdown_project)

    syms = db.symbols_in_file("notes.md")
    # First symbol should have frontmatter in doc_comment
    first = sorted(syms, key=lambda s: s.start_line)[0]
    assert first.doc_comment is not None
    assert "title: Architecture Notes" in first.doc_comment
    assert "tags:" in first.doc_comment


def test_file_removal_detection(db, sample_project):
    """Detects and removes deleted files from index."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)

    # Index everything
    indexer.index(sample_project)
    assert db.stats()["files"] == 2

    # Delete a file
    (sample_project / "utils.py").unlink()

    # Re-index
    stats = indexer.index(sample_project)
    assert stats.files_removed == 1
    assert db.stats()["files"] == 1


@pytest.fixture
def dart_project(tmp_path):
    """Create a minimal Dart project."""
    src = tmp_path / "dartproject"
    src.mkdir()

    # Main Dart file with various constructs
    (src / "main.dart").write_text('''\
// A sample Dart file for testing.

int add(int a, int b) {
  return a + b;
}

class UserService {
  final String _name;

  UserService(this._name);

  String get name => _name;

  /// Fetches a user by ID.
  Future<User?> fetchUser(int id) async {
    return null;
  }
}

class User {
  final int id;
  final String email;

  const User({
    required this.id,
    required this.email,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] as int,
      email: json['email'] as String,
    );
  }
}

enum UserStatus {
  active,
  inactive,
}

mixin Logger {
  void log(String message) {
    print('[LOG] $message');
  }
}

class DataManager with Logger {
  Future<void> load() async {
    log('Loading data...');
  }
}

extension StringExtensions on String {
  String get capitalized {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1)}';
  }
}
''')

    return src


def test_index_dart(db, dart_project):
    """Indexes Dart files and extracts symbols."""
    config = IndexConfig(root=dart_project)
    indexer = Indexer(db, config)
    stats = indexer.index(dart_project)

    assert stats.files_scanned == 1
    assert stats.files_indexed == 1
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    # Check symbols were created
    db_stats = db.stats()
    assert db_stats["files"] == 1
    assert db_stats["symbols"] > 0
    assert "dart" in db_stats["languages"]

    # Check specific symbols
    syms = db.symbols_in_file("main.dart")
    names = [s.name for s in syms]

    # Top-level function
    assert "add" in names

    # Classes
    assert "UserService" in names
    assert "User" in names

    # Method in class - there may be duplicates due to how Dart AST works
    # Check that we have at least one fetchUser with kind=method
    fetch_methods = [s for s in syms if s.name == "fetchUser" and s.kind == "method"]
    assert len(fetch_methods) >= 1

    # Enum
    assert "UserStatus" in names

    # Mixin
    assert "Logger" in names
    # Verify mixin kind
    logger_syms = [s for s in syms if s.name == "Logger"]
    assert len(logger_syms) >= 1
    assert logger_syms[0].kind == "mixin"

    # Extension
    assert "StringExtensions" in names
    ext_syms = [s for s in syms if s.name == "StringExtensions"]
    assert len(ext_syms) >= 1
    assert ext_syms[0].kind == "extension"


@pytest.fixture
def php_project(tmp_path):
    """Create a minimal PHP project."""
    src = tmp_path / "phpproject"
    src.mkdir()

    (src / "app.php").write_text('''\
<?php
function greet($name) {
    echo "Hello, $name!";
}

class UserController {
    public function index(): void {
        echo "list users";
    }

    private function validate($input): bool {
        return true;
    }
}

interface Cacheable {
    public function cache(): void;
}

trait Loggable {
    public function log($msg): void {}
}

enum Status {
    case Active;
    case Inactive;
}
?>
''')
    return src


def test_index_php(db, php_project):
    """Indexes PHP files and extracts symbols."""
    config = IndexConfig(root=php_project)
    indexer = Indexer(db, config)
    stats = indexer.index(php_project)

    assert stats.files_scanned == 1
    assert stats.files_indexed == 1
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    db_stats = db.stats()
    assert db_stats["files"] == 1
    assert db_stats["symbols"] > 0
    assert "php" in db_stats["languages"]

    syms = db.symbols_in_file("app.php")
    names = [s.name for s in syms]

    # Top-level function
    assert "greet" in names

    # Class
    assert "UserController" in names

    # Methods
    assert "index" in names
    assert "validate" in names

    # Interface
    assert "Cacheable" in names

    # Trait
    assert "Loggable" in names

    # Enum
    assert "Status" in names


def test_index_run_leaves_the_main_db_self_contained(tmp_path, sample_project):
    """After indexing, index.db alone must carry the data (issue #16).

    While the MCP server holds the index open, the indexer's own close is not
    the last connection, so SQLite never checkpoints and every row stays in
    index.db-wal. The user then sees a 4096-byte index.db and reasonably
    concludes the reindex destroyed it.
    """
    import shutil
    import sqlite3

    db_path = tmp_path / "index.db"
    server = Database(db_path)          # long-running MCP server holds it open
    server.open()
    server.initialize()
    server.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()

    idx_db = Database(db_path)
    idx_db.open()
    Indexer(idx_db, IndexConfig(root=sample_project)).index(sample_project)

    main_only = tmp_path / "main_only.db"
    shutil.copyfile(db_path, main_only)   # no -wal, as a backup would capture
    conn = sqlite3.connect(main_only)
    try:
        count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    finally:
        conn.close()

    idx_db.close()
    server.close()
    assert count > 0, "index.db is empty on its own — the WAL was never checkpointed"


@pytest.fixture
def lua_project(tmp_path):
    """Create a minimal Lua project covering every way to define a function."""
    src = tmp_path / "luaproject"
    src.mkdir()

    (src / "stack.lua").write_text('''\
local Stack = {}

-- Pushes a value onto the stack.
function Stack.push(self, value)
    self[#self + 1] = value
end

function Stack:pop()
    return table.remove(self)
end

local function clamp(value, limit)
    return math.min(value, limit)
end

Stack.peek = function(self)
    return self[#self]
end

local wrap = function(fn)
    return fn
end

return Stack
''')
    return src


def test_index_lua(db, lua_project):
    """Indexes Lua files and extracts every definition shape."""
    config = IndexConfig(root=lua_project)
    indexer = Indexer(db, config)
    stats = indexer.index(lua_project)

    assert stats.files_scanned == 1
    assert stats.files_indexed == 1
    assert stats.errors == 0

    db_stats = db.stats()
    assert "lua" in db_stats["languages"]

    syms = db.symbols_in_file("stack.lua")
    names = [s.name for s in syms]
    kinds = {s.name: s.kind for s in syms}

    # function T.name() — a function hung off a table, kept as a dotted name
    assert "Stack.push" in names
    assert kinds["Stack.push"] == "function"

    # function T:name() — implicit self, indexed as a method
    assert "pop" in names
    assert kinds["pop"] == "method"

    # local function name()
    assert "clamp" in names
    assert kinds["clamp"] == "function"

    # T.name = function()
    assert "Stack.peek" in names
    assert kinds["Stack.peek"] == "function"

    # local name = function()
    assert "wrap" in names
    assert kinds["wrap"] == "function"


def test_lua_signature_stops_before_the_body(db, lua_project):
    """get_signature must show the parameters, not the whole function."""
    config = IndexConfig(root=lua_project)
    Indexer(db, config).index(lua_project)

    syms = {s.name: s for s in db.symbols_in_file("stack.lua")}
    sig = syms["Stack.push"].signature

    assert sig == "function Stack.push(self, value)"


def test_lua_doc_comment_is_captured(db, lua_project):
    """The `--` comment above a function is its doc comment."""
    config = IndexConfig(root=lua_project)
    Indexer(db, config).index(lua_project)

    syms = {s.name: s for s in db.symbols_in_file("stack.lua")}

    assert syms["Stack.push"].doc_comment == "-- Pushes a value onto the stack."


@pytest.fixture
def lua_assignment_project(tmp_path):
    """A Lua project built out of assignments rather than `function` statements."""
    src = tmp_path / "luaassign"
    src.mkdir()

    (src / "module.lua").write_text('''\
local M = {
    parse = function(text)
        return text
    end,
    ["decode"] = function(text)
        return text
    end,
}

local handlers = {}

handlers["upload"] = function(payload)
    return payload
end

M.mul = function(a, b)
    return a * b
end

-- Doubles a number.
local double = function(x)
    return x * 2
end

local count, makeThing = 0, function(n)
    return n
end

return M
''')
    return src


def test_lua_table_constructor_fields_are_indexed(db, lua_assignment_project):
    """A module written as a table literal still has functions in it."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    assert "parse" in syms
    assert syms["parse"].kind == "function"


def test_lua_bracketed_keys_are_indexed(db, lua_assignment_project):
    """A dispatch table keyed by string holds functions like any other."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    # `{ ["decode"] = function() }` — named by its key, like the identifier form
    assert "decode" in syms
    assert syms["decode"].kind == "function"

    # `handlers["upload"] = function()` — likewise, and the path is the
    # qualified name; test_lua_bracket_assignment_is_named_by_its_key pins that.
    assert "upload" in syms
    assert syms["upload"].kind == "function"


def test_lua_assigned_function_signature_shows_parameters(db, lua_assignment_project):
    """`T.f = function(a, b)` is a function definition like any other."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    assert syms["M.mul"].signature == "M.mul = function(a, b)"


def test_lua_local_assigned_function_keeps_local_and_doc_comment(db, lua_assignment_project):
    """The comment sits above `local`, so the symbol must start there too."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    assert syms["double"].doc_comment == "-- Doubles a number."
    assert syms["double"].content.startswith("local double = function(x)")
    assert syms["double"].signature == "local double = function(x)"


def test_lua_multiple_assignment_does_not_mint_a_function_per_name(db, lua_assignment_project):
    """`local count, makeThing = 0, function() end` — `count` is a number."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    names = [s.name for s in db.symbols_in_file("module.lua")]

    # The statement is skipped whole, so `makeThing` is missed too — the
    # deliberate trade: a miss is visible, a number typed as a function is not.
    assert "count" not in names


def test_lua_method_is_named_the_way_it_is_called(db, lua_project):
    """`function T:f()` is invoked on an instance, so `f` is the searchable name."""
    Indexer(db, IndexConfig(root=lua_project)).index(lua_project)

    syms = {s.name: s for s in db.symbols_in_file("stack.lua")}

    # `Stack:pop()` never appears at a call site — `s:pop()` does.
    assert "pop" in syms
    assert syms["pop"].qualified_name == "Stack:pop"


def test_lua_bracket_assignment_is_named_by_its_key(db, lua_assignment_project):
    """`handlers["upload"]` cannot match a call site; `upload` can."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    assert "upload" in syms
    # the path it hangs off; test_lua_bracket_key_qualifies_as_a_dotted_path
    # pins the shape that path takes
    assert syms["upload"].qualified_name == "handlers.upload"


def test_lua_computed_table_key_is_not_a_definition(db, tmp_path):
    """`[KEY] = function()` names nothing — KEY is a variable holding the key."""
    src = tmp_path / "computed"
    src.mkdir()
    (src / "d.lua").write_text('''\
local KEY_UPLOAD = "upload"

local dispatch = {
    [KEY_UPLOAD] = function(payload)
        return payload
    end,
}

return dispatch
''')
    Indexer(db, IndexConfig(root=src)).index(src)

    names = [s.name for s in db.symbols_in_file("d.lua")]

    assert "KEY_UPLOAD" not in names


def test_broken_source_yields_no_nameless_symbol(db, tmp_path):
    """Tree-sitter recovers from a syntax error with a MISSING, empty-text node."""
    src = tmp_path / "broken"
    src.mkdir()
    (src / "b.lua").write_text("a = b = function(k) return k end\n")

    Indexer(db, IndexConfig(root=src)).index(src)

    names = [s.name for s in db.symbols_in_file("b.lua")]

    assert "" not in names


def test_lua_table_field_qualified_name_carries_no_quotes(db, lua_assignment_project):
    """A `["key"]` field is qualified by its table, with the quotes off."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    assert syms["decode"].qualified_name == "M.decode"


def test_lua_attributed_local_is_indexed(db, tmp_path):
    """Lua 5.4 attributes sit beside the name — `local f <const> = function()`."""
    src = tmp_path / "attrib"
    src.mkdir()
    (src / "a.lua").write_text("local frozen <const> = function(x)\n    return x\nend\n")

    Indexer(db, IndexConfig(root=src)).index(src)

    names = [s.name for s in db.symbols_in_file("a.lua")]

    assert "frozen" in names


def test_lua_deep_path_does_not_lose_the_file(db, tmp_path):
    """A long dotted path must not exhaust the stack — the whole file is at stake."""
    src = tmp_path / "deep"
    src.mkdir()
    path = "a" + "".join(f".b{i}" for i in range(1500))
    (src / "deep.lua").write_text(f"{path} = function(x) end\nfunction Survivor() end\n")

    stats = Indexer(db, IndexConfig(root=src)).index(src)

    assert stats.errors == 0
    assert "Survivor" in [s.name for s in db.symbols_in_file("deep.lua")]


def test_lua_signature_takes_the_assigned_function_not_one_in_the_target(db, tmp_path):
    """The parameters belong to the value being assigned, wherever the target looks."""
    src = tmp_path / "target"
    src.mkdir()
    (src / "t.lua").write_text(
        "(function(NESTED) return {} end)().x = function(OUTER) end\n"
    )

    Indexer(db, IndexConfig(root=src)).index(src)

    sigs = [s.signature or "" for s in db.symbols_in_file("t.lua")]

    # The target's own function literal must not end the signature early.
    assert all(s.endswith("function(OUTER)") for s in sigs), sigs


def test_lua_bracket_key_qualifies_as_a_dotted_path(db, lua_assignment_project):
    """`M["k"]` and `M.k` are the same path — the qualified name says so once."""
    Indexer(db, IndexConfig(root=lua_assignment_project)).index(lua_assignment_project)

    syms = {s.name: s for s in db.symbols_in_file("module.lua")}

    assert syms["upload"].qualified_name == "handlers.upload"


def test_lua_table_field_is_qualified_by_its_table(db, tmp_path):
    """Two tables can hold the same key; the qualified name must tell them apart."""
    src = tmp_path / "tables"
    src.mkdir()
    (src / "v.lua").write_text('''\
local encode = {
    ["Entity"] = function(e) return e end,
}

local decode = {
    ["Entity"] = function(s) return s end,
}

return encode, decode
''')

    Indexer(db, IndexConfig(root=src)).index(src)

    quals = sorted(s.qualified_name for s in db.symbols_in_file("v.lua"))

    assert quals == ["decode.Entity", "encode.Entity"]


def test_lua_malformed_dotted_name_is_dropped(db, tmp_path):
    """Error recovery leaves a dangling `M.` — a path with no field names nothing."""
    src = tmp_path / "malformed"
    src.mkdir()
    (src / "m.lua").write_text("M. = function() end\n")

    Indexer(db, IndexConfig(root=src)).index(src)

    assert [s.name for s in db.symbols_in_file("m.lua")] == []


@pytest.fixture
def lua_nested_project(tmp_path):
    """Tables holding tables — the shape a config or a dispatch tree takes."""
    src = tmp_path / "nested"
    src.mkdir()
    (src / "n.lua").write_text('''\
local A = {
    x = { go = function(a) return a end },
    y = { go = function(b) return b end },
}

local M = {}

M["h"] = { f = function(c) return c end }

t["a"]["b"] = function(d) return d end

local P, Q = { p = function(e) return e end }, { q = function(f) return f end }

return A, M, P, Q
''')
    return src


def test_lua_nested_table_keeps_the_whole_path(db, lua_nested_project):
    """A key of a nested table is qualified by every table above it."""
    Indexer(db, IndexConfig(root=lua_nested_project)).index(lua_nested_project)

    quals = {s.qualified_name for s in db.symbols_in_file("n.lua")}

    assert {"A.x.go", "A.y.go"} <= quals




def test_lua_multi_assignment_of_tables_names_each_one(db, lua_nested_project):
    """`local P, Q = {...}, {...}` — the second table is not the first."""
    Indexer(db, IndexConfig(root=lua_nested_project)).index(lua_nested_project)

    quals = {s.qualified_name for s in db.symbols_in_file("n.lua")}

    assert {"P.p", "Q.q"} <= quals


def test_lua_bracket_path_qualifies_without_punctuation(db, lua_nested_project):
    """Every step of a path is written the way it would be read."""
    Indexer(db, IndexConfig(root=lua_nested_project)).index(lua_nested_project)

    quals = {s.qualified_name for s in db.symbols_in_file("n.lua")}

    assert "M.h.f" in quals    # table reached through ["h"]
    assert "t.a.b" in quals    # target written ["a"]["b"]


def test_lua_unwritable_target_yields_no_punctuation(db, tmp_path):
    """A target that is not a path — a call, a key with spaces — qualifies plainly."""
    src = tmp_path / "odd"
    src.mkdir()
    (src / "q.lua").write_text(
        'require("m")\n  .field = function(x) end\n'
        'local S = { ["a b"] = function(y) end }\n'
    )

    Indexer(db, IndexConfig(root=src)).index(src)

    for sym in db.symbols_in_file("q.lua"):
        qn = sym.qualified_name or ""
        assert "\n" not in qn and '"' not in qn and "[" not in qn, qn


def test_lua_key_that_is_no_identifier_still_names_its_function(db, tmp_path):
    """`t["on-connect"]` cannot join a dotted path, but it does name a function."""
    src = tmp_path / "dashed"
    src.mkdir()
    (src / "k.lua").write_text('''\
local T = {}

T["my-key"] = function(a) return a end

return T
''')

    Indexer(db, IndexConfig(root=src)).index(src)

    syms = {s.name: s for s in db.symbols_in_file("k.lua")}

    assert "my-key" in syms
    # no dotted path exists for it, so the qualified name is the bare key
    assert syms["my-key"].qualified_name == "my-key"
