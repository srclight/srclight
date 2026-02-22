"""Tests for new features: search quality, edges, templates, parent-child, qualified names."""

from pathlib import Path

import pytest

from srclight.db import Database, FileRecord, SymbolRecord, split_identifier
from srclight.indexer import IndexConfig, Indexer


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


# --- 1. CamelCase / identifier splitting ---


class TestSplitIdentifier:
    def test_camel_case(self):
        tokens = split_identifier("SQLiteDictionary")
        assert "Dictionary" in tokens
        assert "dictionary" in tokens

    def test_snake_case(self):
        tokens = split_identifier("get_callers")
        assert "get" in tokens
        assert "callers" in tokens

    def test_acronym(self):
        tokens = split_identifier("OCRManager")
        assert "OCR" in tokens
        assert "Manager" in tokens

    def test_cpp_qualified(self):
        tokens = split_identifier("myapp::util::ConfigManager")
        assert "myapp" in tokens
        assert "util" in tokens
        assert "Config" in tokens
        assert "Manager" in tokens

    def test_empty(self):
        assert split_identifier("") == ""
        assert split_identifier(None) == ""


class TestCamelCaseSearch:
    def test_find_camel_case_by_suffix(self, db):
        """Searching 'Dictionary' should find 'SQLiteDictionary'."""
        fid = db.upsert_file(FileRecord(
            path="test.cpp", content_hash="abc", mtime=1.0,
            language="cpp", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid, kind="class", name="SQLiteDictionary",
            qualified_name="SQLiteDictionary",
            start_line=1, end_line=10,
            content="class SQLiteDictionary {};", line_count=10,
        ), "test.cpp")
        db.commit()

        results = db.search_symbols("Dictionary")
        assert any(r["name"] == "SQLiteDictionary" for r in results)

    def test_find_camel_case_by_prefix(self, db):
        """Searching 'Broker' should find 'BrokerClientImpl'."""
        fid = db.upsert_file(FileRecord(
            path="test.cpp", content_hash="abc", mtime=1.0,
            language="cpp", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid, kind="class", name="BrokerClientImpl",
            qualified_name="BrokerClientImpl",
            start_line=1, end_line=10,
            content="class BrokerClientImpl {};", line_count=10,
        ), "test.cpp")
        db.commit()

        results = db.search_symbols("Broker")
        assert any(r["name"] == "BrokerClientImpl" for r in results)


# --- 2. Call graph edges ---


@pytest.fixture
def python_with_calls(tmp_path):
    """Project where functions call each other."""
    src = tmp_path / "proj"
    src.mkdir()
    (src / "main.py").write_text('''\
def helper():
    return 42

def process():
    result = helper()
    return result

class Worker:
    def run(self):
        return process()
''')
    return src


class TestEdges:
    def test_edges_populated(self, db, python_with_calls):
        """Indexing creates call graph edges."""
        indexer = Indexer(db, IndexConfig(root=python_with_calls))
        stats = indexer.index(python_with_calls)
        assert stats.edges_created > 0
        assert db.stats()["edges"] > 0

    def test_callers(self, db, python_with_calls):
        """get_callers finds the calling function."""
        indexer = Indexer(db, IndexConfig(root=python_with_calls))
        indexer.index(python_with_calls)

        # helper is called by process
        helper = db.get_symbol_by_name("helper")
        assert helper is not None
        callers = db.get_callers(helper.id)
        caller_names = [c["symbol"].name for c in callers]
        assert "process" in caller_names

    def test_callees(self, db, python_with_calls):
        """get_callees finds the called function."""
        indexer = Indexer(db, IndexConfig(root=python_with_calls))
        indexer.index(python_with_calls)

        process = db.get_symbol_by_name("process")
        assert process is not None
        callees = db.get_callees(process.id)
        callee_names = [c["symbol"].name for c in callees]
        assert "helper" in callee_names


# --- 3. C++ template name extraction ---


@pytest.fixture
def cpp_templates(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "templates.cpp").write_text('''\
template <typename T>
class Container {
    T value;
};

template <typename T>
T max_value(T a, T b) {
    return a > b ? a : b;
}

template <typename K, typename V>
struct Pair {
    K key;
    V val;
};
''')
    return src


class TestTemplateNames:
    def test_template_names_extracted(self, db, cpp_templates):
        """Template symbols have their inner names extracted."""
        indexer = Indexer(db, IndexConfig(root=cpp_templates))
        indexer.index(cpp_templates)

        syms = db.symbols_in_file("templates.cpp")
        names = [s.name for s in syms if s.name is not None]
        assert "Container" in names
        assert "Pair" in names
        # max_value might appear depending on query matching
        template_syms = [s for s in syms if s.kind == "template"]
        assert len(template_syms) > 0
        # All template symbols should have names
        for s in template_syms:
            assert s.name is not None, f"Template at line {s.start_line} has no name"


# --- 4. Parent-child relationships ---


class TestParentChild:
    def test_python_methods_have_parent(self, db, tmp_path):
        """Python methods inside a class have parent_symbol_id set."""
        src = tmp_path / "proj"
        src.mkdir()
        (src / "calc.py").write_text('''\
class Calculator:
    def add(self, a, b):
        return a + b
    def multiply(self, a, b):
        return a * b
''')
        indexer = Indexer(db, IndexConfig(root=src))
        indexer.index(src)

        syms = db.symbols_in_file("calc.py")
        calc = next(s for s in syms if s.name == "Calculator")
        add = next(s for s in syms if s.name == "add")
        mul = next(s for s in syms if s.name == "multiply")

        assert add.parent_symbol_id == calc.id
        assert mul.parent_symbol_id == calc.id
        assert calc.parent_symbol_id is None


# --- 5. Multi-match get_symbol ---


class TestMultiMatch:
    def test_get_symbols_by_name_returns_all(self, db):
        """get_symbols_by_name returns all matching symbols."""
        fid1 = db.upsert_file(FileRecord(
            path="a.py", content_hash="a", mtime=1.0,
            language="python", size=100, line_count=10,
        ))
        fid2 = db.upsert_file(FileRecord(
            path="b.py", content_hash="b", mtime=1.0,
            language="python", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid1, kind="function", name="main",
            start_line=1, end_line=5, content="def main(): pass", line_count=5,
        ), "a.py")
        db.insert_symbol(SymbolRecord(
            file_id=fid2, kind="function", name="main",
            start_line=1, end_line=5, content="def main(): pass", line_count=5,
        ), "b.py")
        db.commit()

        results = db.get_symbols_by_name("main")
        assert len(results) == 2

    def test_get_symbols_fuzzy_fallback(self, db):
        """get_symbols_by_name falls back to LIKE matching."""
        fid = db.upsert_file(FileRecord(
            path="test.py", content_hash="a", mtime=1.0,
            language="python", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid, kind="function", name="calculate_total",
            start_line=1, end_line=5, content="def calculate_total(): pass",
            line_count=5,
        ), "test.py")
        db.commit()

        # Exact match returns nothing, but LIKE should find it
        results = db.get_symbols_by_name("calculate")
        assert len(results) == 1
        assert results[0].name == "calculate_total"


# --- 6. Enriched codebase_map ---


class TestCodebaseMap:
    def test_directory_summary(self, db):
        """directory_summary groups files by directory."""
        for name in ["src/a.py", "src/b.py", "lib/c.py"]:
            db.upsert_file(FileRecord(
                path=name, content_hash=name, mtime=1.0,
                language="python", size=100, line_count=10,
            ))
        db.commit()

        dirs = db.directory_summary()
        dir_paths = [d["path"] for d in dirs]
        assert "src" in dir_paths
        assert "lib" in dir_paths
        src_dir = next(d for d in dirs if d["path"] == "src")
        assert src_dir["files"] == 2

    def test_hotspot_files(self, db):
        """hotspot_files returns files with most symbols."""
        fid = db.upsert_file(FileRecord(
            path="big.py", content_hash="big", mtime=1.0,
            language="python", size=1000, line_count=100,
        ))
        for i in range(10):
            db.insert_symbol(SymbolRecord(
                file_id=fid, kind="function", name=f"fn_{i}",
                start_line=i * 10, end_line=i * 10 + 5,
                content=f"def fn_{i}(): pass", line_count=5,
            ), "big.py")
        db.commit()

        hotspots = db.hotspot_files(limit=5)
        assert len(hotspots) == 1
        assert hotspots[0]["path"] == "big.py"
        assert hotspots[0]["symbols"] == 10


# --- 7. C++ qualified names ---


class TestQualifiedNames:
    def test_cpp_namespace_class(self, db, tmp_path):
        """C++ symbols get proper namespace::class::name qualified names."""
        src = tmp_path / "proj"
        src.mkdir()
        (src / "test.cpp").write_text('''\
namespace outer {
namespace inner {

class MyClass {
};

} // namespace inner
} // namespace outer

void standalone() {}
''')
        indexer = Indexer(db, IndexConfig(root=src))
        indexer.index(src)

        syms = db.symbols_in_file("test.cpp")
        my_class = next((s for s in syms if s.name == "MyClass"), None)
        assert my_class is not None
        assert my_class.qualified_name == "outer::inner::MyClass"

        standalone = next((s for s in syms if s.name == "standalone"), None)
        assert standalone is not None
        assert standalone.qualified_name == "standalone"

    def test_python_class_method(self, db, tmp_path):
        """Python methods get Class.method qualified names."""
        src = tmp_path / "proj"
        src.mkdir()
        (src / "test.py").write_text('''\
class Foo:
    def bar(self):
        pass
''')
        indexer = Indexer(db, IndexConfig(root=src))
        indexer.index(src)

        syms = db.symbols_in_file("test.py")
        bar = next((s for s in syms if s.name == "bar"), None)
        assert bar is not None
        assert bar.qualified_name == "Foo.bar"
