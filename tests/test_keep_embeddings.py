"""Reparsing a file keeps the embeddings of the symbols that come back unchanged.

A file is reparsed whenever its content hash changes — an edit, or a
marker an operator sets to pick up an extractor change. Its symbols are
deleted and inserted again, and their embeddings were deleted with them:
every one had to be computed again, though most symbols were identical.
"""
from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer


def _embed_all(db: Database) -> dict[str, bytes]:
    """Give every symbol a fake embedding derived from its name."""
    blobs = {}
    for row in db.conn.execute("SELECT id, name, body_hash FROM symbols").fetchall():
        blob = row["name"].encode() * 4
        db.upsert_embedding(row["id"], "fake-model", 4, blob, row["body_hash"])
        blobs[row["name"]] = blob
    db.commit()
    return blobs


def _embeddings(db: Database) -> dict[str, bytes]:
    return {r[0]: r[1] for r in db.conn.execute(
        """SELECT s.name, e.embedding FROM symbols s
           JOIN symbol_embeddings e ON e.symbol_id = s.id""")}


def test_unchanged_symbols_keep_their_embeddings(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.c"
    source.write_text("int keep_me(int v) {\n    return v;\n}\n\n"
                      "int change_me(int v) {\n    return v + 1;\n}\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    indexer = Indexer(db, IndexConfig(root=root))
    indexer.index()
    before = _embed_all(db)

    source.write_text("#define ADDED 1\n\nint keep_me(int v) {\n    return v;\n}\n\n"
                      "int change_me(int v) {\n    return v + 2;\n}\n")
    indexer.index()
    after = _embeddings(db)
    needing = {r["name"] for r in db.get_symbols_needing_embeddings("fake-model")}
    db.close()

    assert after["keep_me"] == before["keep_me"]
    assert "change_me" not in after and "change_me" in needing
    assert "ADDED" not in after and "ADDED" in needing
    assert "keep_me" not in needing


def test_a_duplicate_symbol_takes_one_embedding_each(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "twice.c"
    body = "#if A\nint same(void) {\n    return 0;\n}\n#else\nint same(void) {\n    return 0;\n}\n#endif\n"
    source.write_text(body)
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    indexer = Indexer(db, IndexConfig(root=root))
    indexer.index()
    count = db.conn.execute("SELECT COUNT(*) FROM symbols WHERE name = 'same'").fetchone()[0]
    _embed_all(db)
    source.write_text("/* touched */\n" + body)
    indexer.index()
    kept = db.conn.execute(
        """SELECT COUNT(*) FROM symbols s JOIN symbol_embeddings e ON e.symbol_id = s.id
           WHERE s.name = 'same'""").fetchone()[0]
    db.close()
    assert kept == count


def test_a_stale_embedding_is_not_carried_over(tmp_path):
    """An embedding written for another text — a vector computed before a
    reparse and stored after it — must not be kept as current."""
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.c"
    source.write_text("int keep_me(int v) {\n    return v;\n}\n")
    db = Database(root / "index.db")
    db.open()
    db.initialize()
    indexer = Indexer(db, IndexConfig(root=root))
    indexer.index()
    sid = db.conn.execute("SELECT id FROM symbols WHERE name = 'keep_me'").fetchone()[0]
    db.upsert_embedding(sid, "fake-model", 4, b"stale!!!", "an-older-body-hash")
    db.commit()
    source.write_text("int keep_me(int v) {\n    return v;\n}\n\nint other_one(void) {\n    return 0;\n}\n")
    indexer.index()
    needing = {r["name"] for r in db.get_symbols_needing_embeddings("fake-model")}
    db.close()
    assert "keep_me" in needing
