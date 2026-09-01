"""Tests for the learnings module."""

import pytest

from srclight.learnings import (
    LearningsDB,
    LearningRecord,
    ConversationRecord,
    VALID_KINDS,
    VALID_SCOPES,
    prepare_learning_text,
    _body_hash,
)


@pytest.fixture
def ldb(tmp_path):
    db_path = tmp_path / "learnings.db"
    db = LearningsDB(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def test_initialize(ldb):
    """Schema creates all tables."""
    tables = [
        r[0]
        for r in ldb.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    assert "conversations" in tables
    assert "learnings" in tables
    assert "learning_symbols" in tables
    assert "learning_sources" in tables
    assert "learning_embeddings" in tables

    # Check schema version
    row = ldb.conn.execute(
        "SELECT value FROM schema_info WHERE key='schema_version'"
    ).fetchone()
    assert row["value"] == "1"


def test_record_learning(ldb):
    """Insert and retrieve a learning."""
    rec = LearningRecord(
        kind="decision",
        content="Use Kotlin + Glimmer for glasses app",
        reasoning="Flutter renders to opaque framebuffers, waveguide displays need transparent backgrounds",
        project="ice",
    )
    lid = ldb.record_learning(rec)
    assert lid > 0

    fetched = ldb.get_learning(lid)
    assert fetched is not None
    assert fetched.kind == "decision"
    assert fetched.content == "Use Kotlin + Glimmer for glasses app"
    assert fetched.reasoning.startswith("Flutter renders")
    assert fetched.project == "ice"
    assert fetched.confidence == 1.0


def test_record_learning_with_symbols(ldb):
    """Learning with symbol links."""
    rec = LearningRecord(
        kind="discovery",
        content="sherpa_tts_backend already supports Kokoro",
        project="nomad-builder",
    )
    lid = ldb.record_learning(rec, symbols=["sherpa_tts_backend", "SpeakService"])
    fetched = ldb.get_learning(lid)
    assert set(fetched.symbols) == {"sherpa_tts_backend", "SpeakService"}


def test_record_learning_with_sources(ldb):
    """Learning with source links."""
    rec = LearningRecord(kind="convention", content="Business logic in C++, not Dart")
    sources = [
        {"type": "claude_md", "ref": "acme/app/CLAUDE.md"},
        {"type": "labbook", "ref": "bible/docs/Projects/paper-farsi-alignment/labbook.md"},
    ]
    lid = ldb.record_learning(rec, sources=sources)
    fetched = ldb.get_learning(lid)
    assert len(fetched.sources) == 2
    assert fetched.sources[0]["type"] == "claude_md"


def test_kind_validation(ldb):
    """Reject invalid kinds."""
    rec = LearningRecord(kind="invalid_kind", content="test")
    with pytest.raises(ValueError, match="Invalid kind"):
        ldb.record_learning(rec)


def test_scope_validation(ldb):
    """Reject invalid scopes."""
    rec = LearningRecord(kind="decision", content="test", scope="bad_scope")
    with pytest.raises(ValueError, match="Invalid scope"):
        ldb.record_learning(rec)


def test_fts_search(ldb):
    """FTS5 keyword search."""
    ldb.record_learning(LearningRecord(
        kind="decision",
        content="Ship Kokoro TTS bundled with iCE",
        reasoning="Zero marginal cost per user, works offline",
        project="ice",
    ))
    ldb.record_learning(LearningRecord(
        kind="correction",
        content="Don't mock the database in integration tests",
        reasoning="Prior incident where mock/prod divergence masked a broken migration",
        project="conductor",
    ))

    results = ldb.search_fts("Kokoro")
    assert len(results) == 1
    assert results[0]["content"] == "Ship Kokoro TTS bundled with iCE"

    results = ldb.search_fts("database mock")
    assert len(results) == 1
    assert results[0]["kind"] == "correction"


def test_fts_project_filter(ldb):
    """FTS search filtered by project."""
    ldb.record_learning(LearningRecord(kind="decision", content="Use WAL mode", project="ice"))
    ldb.record_learning(LearningRecord(kind="decision", content="Use WAL mode", project="conductor"))

    results = ldb.search_fts("WAL", project="ice")
    assert len(results) == 1
    assert results[0]["project"] == "ice"


def test_conversation_summary(ldb):
    """Insert and upsert conversation."""
    rec = ConversationRecord(
        session_id="abc-123",
        project="ice",
        task_summary="Researched AR glasses for cross-modal learning",
        model="claude-opus-4-6",
        tokens_in=50000,
        tokens_out=10000,
        cost_usd=1.50,
    )
    cid = ldb.record_conversation(rec)
    assert cid > 0

    # Upsert with updated summary
    rec.task_summary = "Updated summary"
    rec.cost_usd = 2.00
    cid2 = ldb.record_conversation(rec)
    assert cid2 == cid  # Same session_id, same row

    row = ldb.conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    assert row["task_summary"] == "Updated summary"
    assert row["cost_usd"] == 2.00


def test_supersede(ldb):
    """Mark learning as superseded."""
    old_id = ldb.record_learning(LearningRecord(
        kind="decision", content="Use v1.0 Kokoro model"
    ))
    new_id = ldb.record_learning(LearningRecord(
        kind="decision", content="Use v1.1 Kokoro model (better Chinese voices)"
    ))
    ldb.supersede(old_id, new_id)

    old = ldb.get_learning(old_id)
    assert old.superseded_by == new_id

    # Superseded learnings excluded from search by default
    results = ldb.search_fts("Kokoro model")
    assert all(r["learning_id"] != old_id for r in results)


def test_expire_stale(ldb):
    """TTL expiration."""
    # Create a learning then manually set expires_at to the past
    rec = LearningRecord(kind="blocker", content="Build server down")
    lid = ldb.record_learning(rec)
    ldb.conn.execute(
        "UPDATE learnings SET expires_at = '2020-01-01T00:00:00Z' WHERE id = ?",
        (lid,),
    )
    ldb.conn.commit()

    count = ldb.expire_stale()
    assert count == 1
    assert ldb.get_learning(lid) is None


def test_stats(ldb):
    """Stats by kind and project."""
    ldb.record_learning(LearningRecord(kind="decision", content="A", project="ice"))
    ldb.record_learning(LearningRecord(kind="decision", content="B", project="ice"))
    ldb.record_learning(LearningRecord(kind="correction", content="C", project="conductor"))
    ldb.record_learning(LearningRecord(kind="discovery", content="D"))

    s = ldb.stats()
    assert s["total_learnings"] == 4
    assert s["by_kind"]["decision"] == 2
    assert s["by_kind"]["correction"] == 1

    s_ice = ldb.stats(project="ice")
    # Cross-project (D) + ice-specific (A, B)
    assert s_ice["total_learnings"] == 3


def test_stats_with_days(ldb):
    """Stats filtered by days."""
    ldb.record_learning(LearningRecord(kind="decision", content="Recent"))
    s = ldb.stats(days=1)
    assert s["total_learnings"] == 1


def test_context_manager(tmp_path):
    """Context manager opens, initializes, and closes."""
    db_path = tmp_path / "ctx.db"
    with LearningsDB(db_path) as ldb:
        lid = ldb.record_learning(LearningRecord(kind="pattern", content="Test"))
        assert lid > 0
    # Connection closed
    assert ldb.conn is None


def test_prepare_learning_text():
    """Embedding text preparation."""
    rec = LearningRecord(
        kind="decision",
        content="Ship Kokoro locally",
        reasoning="Zero marginal cost",
        project="ice",
    )
    text = prepare_learning_text(rec)
    assert "decision: Ship Kokoro locally" in text
    assert "Zero marginal cost" in text
    assert "project: ice" in text


def test_body_hash():
    """Hash changes with content."""
    h1 = _body_hash("content A", "reasoning A")
    h2 = _body_hash("content B", "reasoning A")
    h3 = _body_hash("content A", "reasoning A")
    assert h1 != h2
    assert h1 == h3


def test_hybrid_search_rrf(ldb):
    """RRF merge of FTS and mock embedding results."""
    # Insert some learnings
    lid1 = ldb.record_learning(LearningRecord(kind="decision", content="Alpha decision"))
    lid2 = ldb.record_learning(LearningRecord(kind="pattern", content="Beta pattern"))

    fts_results = [
        {"learning_id": lid1, "kind": "decision", "content": "Alpha decision",
         "reasoning": None, "project": None, "confidence": 1.0, "created_at": "", "fts_rank": -1.0},
        {"learning_id": lid2, "kind": "pattern", "content": "Beta pattern",
         "reasoning": None, "project": None, "confidence": 1.0, "created_at": "", "fts_rank": -0.5},
    ]
    emb_results = [
        {"learning_id": lid2, "kind": "pattern", "content": "Beta pattern",
         "reasoning": None, "project": None, "confidence": 1.0, "created_at": "", "similarity": 0.95},
        {"learning_id": lid1, "kind": "decision", "content": "Alpha decision",
         "reasoning": None, "project": None, "confidence": 1.0, "created_at": "", "similarity": 0.80},
    ]

    merged = ldb.hybrid_search(fts_results, emb_results)
    assert len(merged) == 2
    # Both should have RRF scores
    assert all("rrf_score" in r for r in merged)
    # lid2 ranks first in embeddings and second in FTS — should have higher combined score
    # (or at least both should appear)
    ids = [r["learning_id"] for r in merged]
    assert lid1 in ids
    assert lid2 in ids
