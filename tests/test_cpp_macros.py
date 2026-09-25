"""A C++ file's macros are symbols, as a C file's are.

The C++ symbol query had no pattern for `#define`: every macro of a file
read as C++ — a header with one class in it included — was missing from
the index, while the same macro in a file read as C was there.
"""
from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


def _symbols(tmp_path, files: dict[str, str]) -> set[tuple[str, str]]:
    root = tmp_path / "repo"
    root.mkdir()
    for name, text in files.items():
        (root / name).write_text(text)
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    rows = {(r[0], r[1]) for r in db.conn.execute("SELECT name, kind FROM symbols")}
    db.close()
    return rows


def test_cpp_macros_are_symbols(tmp_path):
    symbols = _symbols(tmp_path, {"cfg.h": """\
#ifndef CFG_H
#define CFG_H
#define SHOUT(msg) report(msg)
#if TOOLING
#define WHEN_EXT(statement) statement
#else
#define WHEN_EXT(statement)
#endif
class Cfg_c {
public:
    void apply();
};
#endif
"""})
    assert {("CFG_H", "macro"), ("SHOUT", "macro"), ("WHEN_EXT", "macro")} <= symbols
    assert ("Cfg_c", "class") in symbols


def test_a_macro_in_a_cpp_file_reaches_what_it_calls(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "log.cpp").write_text(
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
