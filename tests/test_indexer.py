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
