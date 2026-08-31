"""Tests for new analysis tools: find_dead_code, find_pattern, find_imports."""

from pathlib import Path

import pytest

from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord
from srclight.server import _extract_imports


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def _insert_file(db, path="src/main.py", language="python", **kwargs):
    """Helper to insert a file and return its id."""
    defaults = dict(
        content_hash=path, mtime=1.0, size=100, line_count=10,
    )
    defaults.update(kwargs)
    return db.upsert_file(FileRecord(path=path, language=language, **defaults))


def _insert_symbol(db, file_id, name, file_path="src/main.py", kind="function",
                   start_line=1, end_line=5, content=None, **kwargs):
    """Helper to insert a symbol and return its id."""
    if content is None:
        content = f"def {name}(): pass"
    return db.insert_symbol(SymbolRecord(
        file_id=file_id, kind=kind, name=name,
        start_line=start_line, end_line=end_line,
        content=content, line_count=end_line - start_line + 1,
        **kwargs,
    ), file_path)


# --- find_dead_code tests ---


class TestFindDeadCode:
    def test_unreferenced_symbols_returned(self, db):
        """Symbols with no incoming edges are returned as dead code."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "used_fn", start_line=1, end_line=3)
        dead_id = _insert_symbol(db, fid, "unused_fn", start_line=5, end_line=8)
        caller_id = _insert_symbol(db, fid, "caller", start_line=10, end_line=15,
                                   content="def caller(): used_fn()")

        # used_fn has an edge from caller
        used = db.get_symbol_by_name("used_fn")
        db.insert_edge(EdgeRecord(
            source_id=caller_id, target_id=used.id, edge_type="calls",
        ))
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        # unused_fn has no callers, caller also has no callers
        assert "unused_fn" in dead_names
        assert "caller" in dead_names
        # used_fn has an incoming edge, so should NOT be in dead code
        assert "used_fn" not in dead_names

    def test_entry_points_excluded(self, db):
        """main, __init__, test_* are excluded from dead code."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "main", start_line=1, end_line=3)
        _insert_symbol(db, fid, "__init__", start_line=5, end_line=7, kind="method")
        _insert_symbol(db, fid, "test_something", start_line=9, end_line=12)
        _insert_symbol(db, fid, "TestCase", start_line=14, end_line=20, kind="class")
        _insert_symbol(db, fid, "real_function", start_line=22, end_line=25)
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        assert "main" not in dead_names
        assert "__init__" not in dead_names
        assert "test_something" not in dead_names
        assert "TestCase" not in dead_names
        # real_function has no callers and is not an entry point
        assert "real_function" in dead_names

    def test_public_visibility_excluded(self, db):
        """Symbols with 'public' or 'export' visibility are excluded."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "public_fn", start_line=1, end_line=3,
                       visibility="public")
        _insert_symbol(db, fid, "exported_fn", start_line=5, end_line=7,
                       visibility="export")
        _insert_symbol(db, fid, "private_fn", start_line=9, end_line=12,
                       visibility="private")
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        assert "public_fn" not in dead_names
        assert "exported_fn" not in dead_names
        assert "private_fn" in dead_names

    def test_types_are_eligible_and_edges_decide(self, db):
        """Types (struct/enum/interface) ARE checked for dead code — the reference
        extractor creates edges to type names appearing in symbol content, so a
        used type has incoming edges and an unused one is genuinely detectable.

        This test previously asserted the opposite ("enum not in checked kinds"),
        codifying an untested assumption that types get no edges; narrowing the
        kinds on that basis hid real dead types (verified empirically 2026-08-30).
        """
        fid = _insert_file(db)
        used_id = _insert_symbol(db, fid, "UsedStruct", kind="struct", start_line=1, end_line=4,
                                 content="struct UsedStruct { int x; }")
        _insert_symbol(db, fid, "DeadStruct", kind="struct", start_line=6, end_line=9,
                       content="struct DeadStruct { int y; }")
        user_id = _insert_symbol(db, fid, "consumer", kind="function", start_line=11, end_line=14,
                                 content="int consumer(struct UsedStruct s) { return s.x; }")
        db.insert_edge(EdgeRecord(source_id=user_id, target_id=used_id, edge_type="calls"))
        db.commit()

        dead_names = [d.name for d in db.get_dead_symbols()]
        assert "DeadStruct" in dead_names   # unreferenced type IS reported
        assert "UsedStruct" not in dead_names  # referenced type is not

    def test_limit_respected(self, db):
        """get_dead_symbols respects the limit parameter."""
        fid = _insert_file(db)
        for i in range(10):
            _insert_symbol(db, fid, f"fn_{i}", start_line=i * 5, end_line=i * 5 + 3)
        db.commit()

        dead = db.get_dead_symbols(limit=3)
        assert len(dead) == 3


# --- find_pattern tests ---


class TestFindPattern:
    def test_pattern_matches_content(self, db):
        """Regex pattern matching finds symbols with matching content."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn_with_todo", start_line=1, end_line=5,
                       content="def fn_with_todo():\n    # TODO: fix this\n    pass")
        _insert_symbol(db, fid, "clean_fn", start_line=7, end_line=10,
                       content="def clean_fn():\n    return 42")
        db.commit()

        results = db.find_pattern_in_symbols("TODO")
        assert len(results) == 1
        assert results[0]["name"] == "fn_with_todo"
        assert results[0]["match_count"] == 1

    def test_pattern_with_kind_filter(self, db):
        """Kind filter narrows results to specific symbol types."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "my_func", kind="function", start_line=1, end_line=5,
                       content="def my_func():\n    sleep(1)")
        _insert_symbol(db, fid, "MyClass", kind="class", start_line=7, end_line=12,
                       content="class MyClass:\n    def run(self):\n        sleep(2)")
        db.commit()

        # Both have sleep, but filter to function only
        results = db.find_pattern_in_symbols(r"sleep\(", kind="function")
        assert len(results) == 1
        assert results[0]["name"] == "my_func"

    def test_multiple_matches_in_symbol(self, db):
        """Multiple matches within one symbol are counted."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "many_fixmes", start_line=1, end_line=10,
                       content="def many_fixmes():\n    # FIXME: one\n    x = 1\n    # FIXME: two\n    # FIXME: three")
        db.commit()

        results = db.find_pattern_in_symbols("FIXME")
        assert len(results) == 1
        assert results[0]["match_count"] == 3

    def test_matched_lines_returned(self, db):
        """Matched lines include line offset and content."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn", start_line=1, end_line=4,
                       content="def fn():\n    x = dangerous_call()\n    return x")
        db.commit()

        results = db.find_pattern_in_symbols("dangerous_call")
        assert len(results) == 1
        matched = results[0]["matched_lines"]
        assert len(matched) == 1
        assert matched[0]["line_offset"] == 2
        assert "dangerous_call" in matched[0]["line"]
        assert matched[0]["match"] == "dangerous_call"

    def test_no_matches_returns_empty(self, db):
        """No matches returns empty list."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn", start_line=1, end_line=3,
                       content="def fn(): return 1")
        db.commit()

        results = db.find_pattern_in_symbols("NONEXISTENT_PATTERN")
        assert results == []

    def test_limit_respected(self, db):
        """find_pattern_in_symbols respects the limit parameter."""
        fid = _insert_file(db)
        for i in range(10):
            _insert_symbol(db, fid, f"fn_{i}", start_line=i * 5, end_line=i * 5 + 3,
                           content=f"def fn_{i}():\n    # TODO: item {i}")
        db.commit()

        results = db.find_pattern_in_symbols("TODO", limit=3)
        assert len(results) == 3

    def test_regex_special_characters(self, db):
        """Regex with special characters works correctly."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn", start_line=1, end_line=3,
                       content="def fn():\n    except Exception as e:")
        db.commit()

        results = db.find_pattern_in_symbols(r"except\s+\w+\s+as\s+\w+")
        assert len(results) == 1


# --- find_imports tests ---


class TestExtractImports:
    """Test the _extract_imports helper function directly."""

    def test_python_import(self):
        content = "import os\nimport sys"
        imports = _extract_imports(content, "python")
        modules = [i["module"] for i in imports]
        assert "os" in modules
        assert "sys" in modules

    def test_python_from_import(self):
        content = "from pathlib import Path\nfrom os.path import join, exists"
        imports = _extract_imports(content, "python")
        assert len(imports) == 2

        path_imp = next(i for i in imports if i["module"] == "pathlib")
        assert "Path" in path_imp["names"]

        os_imp = next(i for i in imports if i["module"] == "os.path")
        assert "join" in os_imp["names"]
        assert "exists" in os_imp["names"]

    def test_python_relative_import(self):
        content = "from .db import Database\nfrom ..utils import helper"
        imports = _extract_imports(content, "python")
        modules = [i["module"] for i in imports]
        assert ".db" in modules
        assert "..utils" in modules

    def test_javascript_import(self):
        content = "import React from 'react';\nimport { useState } from 'react';"
        imports = _extract_imports(content, "javascript")
        modules = [i["module"] for i in imports]
        assert "react" in modules

    def test_javascript_require(self):
        content = "const fs = require('fs');\nconst path = require('path');"
        imports = _extract_imports(content, "javascript")
        modules = [i["module"] for i in imports]
        assert "fs" in modules
        assert "path" in modules

    def test_typescript_import(self):
        content = "import { Component } from '@angular/core';\nimport type { Config } from './config';"
        imports = _extract_imports(content, "typescript")
        modules = [i["module"] for i in imports]
        assert "@angular/core" in modules
        assert "./config" in modules

    def test_c_include(self):
        content = '#include <stdio.h>\n#include "myheader.h"'
        imports = _extract_imports(content, "c")
        modules = [i["module"] for i in imports]
        assert "stdio.h" in modules
        assert "myheader.h" in modules

    def test_cpp_include(self):
        content = '#include <vector>\n#include <string>\n#include "utils.h"'
        imports = _extract_imports(content, "cpp")
        assert len(imports) == 3

    def test_java_import(self):
        content = "import java.util.List;\nimport com.example.MyClass;"
        imports = _extract_imports(content, "java")
        modules = [i["module"] for i in imports]
        assert "java.util.List" in modules
        assert "com.example.MyClass" in modules

    def test_go_import(self):
        content = 'import (\n    "fmt"\n    "os"\n)'
        imports = _extract_imports(content, "go")
        modules = [i["module"] for i in imports]
        assert "fmt" in modules
        assert "os" in modules

    def test_csharp_using(self):
        content = "using System;\nusing System.Collections.Generic;"
        imports = _extract_imports(content, "csharp")
        modules = [i["module"] for i in imports]
        assert "System" in modules
        assert "System.Collections.Generic" in modules

    def test_dart_import(self):
        content = "import 'dart:async';\nimport 'package:flutter/material.dart';"
        imports = _extract_imports(content, "dart")
        modules = [i["module"] for i in imports]
        assert "dart:async" in modules
        assert "package:flutter/material.dart" in modules

    def test_swift_import(self):
        content = "import Foundation\nimport UIKit"
        imports = _extract_imports(content, "swift")
        modules = [i["module"] for i in imports]
        assert "Foundation" in modules
        assert "UIKit" in modules

    def test_kotlin_import(self):
        content = "import kotlin.collections.List\nimport com.example.Utils"
        imports = _extract_imports(content, "kotlin")
        modules = [i["module"] for i in imports]
        assert "kotlin.collections.List" in modules
        assert "com.example.Utils" in modules

    def test_unsupported_language_returns_empty(self):
        imports = _extract_imports("some content", "markdown")
        assert imports == []

    def test_empty_content_returns_empty(self):
        imports = _extract_imports("", "python")
        assert imports == []

    def test_no_duplicate_statements(self):
        """Same import statement on multiple lines shouldn't duplicate."""
        content = "import os\nimport os"
        imports = _extract_imports(content, "python")
        # Both lines are "import os", so deduped
        os_imports = [i for i in imports if i["module"] == "os"]
        assert len(os_imports) == 1


class TestResolveImport:
    """Test the db.resolve_import method."""

    def test_resolve_by_symbol_name(self, db):
        """Import name matching a symbol name resolves to that symbol."""
        fid = _insert_file(db, "src/db.py")
        _insert_symbol(db, fid, "Database", file_path="src/db.py", kind="class",
                       start_line=10, end_line=50, content="class Database: pass")
        db.commit()

        result = db.resolve_import("Database")
        assert result is not None
        assert result["name"] == "Database"
        assert result["file"] == "src/db.py"
        assert result["kind"] == "class"
        assert result["match_type"] == "symbol"

    def test_resolve_by_qualified_name(self, db):
        """Import matching a qualified_name resolves."""
        fid = _insert_file(db, "src/utils.py")
        _insert_symbol(db, fid, "helper", file_path="src/utils.py",
                       qualified_name="utils.helper",
                       start_line=1, end_line=5, content="def helper(): pass")
        db.commit()

        result = db.resolve_import("utils.helper")
        assert result is not None
        assert result["name"] == "helper"
        assert result["match_type"] == "qualified_name"

    def test_resolve_by_file_path(self, db):
        """Import matching a file path resolves."""
        _insert_file(db, "src/utils.py")
        db.commit()

        result = db.resolve_import("src.utils")
        assert result is not None
        assert result["file"] == "src/utils.py"
        assert result["match_type"] == "file_path"

    def test_unresolved_returns_none(self, db):
        """Unknown import returns None."""
        db.commit()
        result = db.resolve_import("nonexistent_module")
        assert result is None

    def test_resolve_js_path(self, db):
        """JS-style import path resolves to .js file."""
        _insert_file(db, "src/utils.js", language="javascript")
        db.commit()

        result = db.resolve_import("src/utils")
        assert result is not None
        assert result["file"] == "src/utils.js"
        assert result["match_type"] == "file_path"


class TestFindImportsIntegration:
    """Integration tests for the find_imports tool using the DB + file system."""

    def test_python_imports_resolved(self, db, tmp_path):
        """Python imports are extracted and resolved against the index."""
        # Set up a project directory with files
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "main.py").write_text(
            "from db import Database\nimport json\nimport os\n"
        )
        (proj / "db.py").write_text("class Database: pass\n")

        # Index: register files and symbols
        fid_main = _insert_file(db, "main.py", language="python")
        fid_db = _insert_file(db, "db.py", language="python")
        _insert_symbol(db, fid_db, "Database", file_path="db.py", kind="class",
                       start_line=1, end_line=1, content="class Database: pass")
        db.commit()

        # Extract imports from the file content
        content = (proj / "main.py").read_text()
        raw_imports = _extract_imports(content, "python")

        assert len(raw_imports) == 3

        # Resolve each import
        resolved = []
        for imp in raw_imports:
            names_to_try = imp["names"] if imp["names"] else [imp["module"].split(".")[-1]]
            for name in names_to_try:
                result = db.resolve_import(name)
                if result:
                    resolved.append(result)
                    break

        # "Database" should be resolved
        resolved_names = [r["name"] for r in resolved]
        assert "Database" in resolved_names

    def test_javascript_imports_resolved(self, db, tmp_path):
        """JavaScript imports are extracted and resolved."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.js").write_text(
            "import { render } from './renderer';\n"
            "const utils = require('./utils');\n"
        )

        fid_app = _insert_file(db, "app.js", language="javascript")
        fid_renderer = _insert_file(db, "renderer.js", language="javascript")
        _insert_symbol(db, fid_renderer, "render", file_path="renderer.js",
                       kind="function", start_line=1, end_line=5,
                       content="function render() {}")
        db.commit()

        content = (proj / "app.js").read_text()
        raw_imports = _extract_imports(content, "javascript")

        assert len(raw_imports) == 2
        modules = [i["module"] for i in raw_imports]
        assert "./renderer" in modules
        assert "./utils" in modules

        # "render" symbol should resolve
        result = db.resolve_import("render")
        assert result is not None
        assert result["name"] == "render"

    def test_c_includes_extracted(self, db, tmp_path):
        """C #include directives are extracted."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "main.c").write_text(
            '#include <stdio.h>\n'
            '#include "mylib.h"\n'
            '\nint main() { return 0; }\n'
        )

        content = (proj / "main.c").read_text()
        raw_imports = _extract_imports(content, "c")

        assert len(raw_imports) == 2
        modules = [i["module"] for i in raw_imports]
        assert "stdio.h" in modules
        assert "mylib.h" in modules
