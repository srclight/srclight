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
