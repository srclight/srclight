"""Tests for embedding providers, vector search, and hybrid RRF."""

import struct
from unittest.mock import patch

import pytest

from srclight.embeddings import (
    CohereProvider,
    EmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
    VoyageProvider,
    bytes_to_vector,
    cosine_similarity,
    embed_symbols,
    get_provider,
    prepare_embedding_text,
    rrf_merge,
    vector_to_bytes,
    vectors_to_bytes,
)


# --- Fixtures ---


class MockProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dims: int = 4):
        self._dims = dims

    @property
    def name(self) -> str:
        return "mock:test-model"

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings based on text hash."""
        results = []
        for text in texts:
            h = hash(text) & 0xFFFFFFFF
            vec = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self._dims)]
            # Normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


# --- Test prepare_embedding_text ---


def test_prepare_embedding_text_full():
    sym = {
        "qualified_name": "MyClass::my_method",
        "name": "my_method",
        "signature": "void my_method(int x)",
        "doc_comment": "Does something useful.",
        "content": "void my_method(int x) { return x + 1; }",
    }
    text = prepare_embedding_text(sym)
    assert "MyClass::my_method" in text
    assert "void my_method(int x)" in text
    assert "Does something useful." in text
    assert "return x + 1" in text


def test_prepare_embedding_text_minimal():
    sym = {"name": "foo", "content": "int foo() { return 42; }"}
    text = prepare_embedding_text(sym)
    assert "foo" in text
    assert "return 42" in text


def test_prepare_embedding_text_truncation():
    sym = {"name": "big", "content": "x" * 5000}
    text = prepare_embedding_text(sym)
    # Content should be truncated to ~2000 chars
    assert len(text) < 2100


# --- Test vector math ---


def test_vector_to_bytes_roundtrip():
    vec = [1.0, 2.0, 3.0, 4.0]
    b = vector_to_bytes(vec)
    assert isinstance(b, bytes)
    assert len(b) == 16  # 4 floats * 4 bytes

    recovered = bytes_to_vector(b)
    assert recovered == pytest.approx(vec)


def test_vectors_to_bytes_batch():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    batch = vectors_to_bytes(vecs)
    assert len(batch) == 2
    for b in batch:
        assert len(b) == 8  # 2 floats * 4 bytes


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- Test RRF merge ---


def test_rrf_merge_basic():
    fts = [
        {"symbol_id": 1, "name": "foo", "rank": -20},
        {"symbol_id": 2, "name": "bar", "rank": -10},
    ]
    emb = [
        {"symbol_id": 2, "name": "bar", "similarity": 0.95},
        {"symbol_id": 3, "name": "baz", "similarity": 0.80},
    ]
    merged = rrf_merge(fts, emb)

    # bar (id=2) should be top — it appears in both lists
    assert merged[0]["symbol_id"] == 2
    assert "fts" in merged[0]["sources"]
    assert "embedding" in merged[0]["sources"]
    assert merged[0]["rrf_score"] > 0

    # All 3 symbols should appear
    ids = {r["symbol_id"] for r in merged}
    assert ids == {1, 2, 3}


def test_rrf_merge_weights():
    fts = [{"symbol_id": 1, "name": "a", "rank": -20}]
    emb = [{"symbol_id": 2, "name": "b", "similarity": 0.9}]

    # With heavy embedding weight, embedding result should rank higher
    merged = rrf_merge(fts, emb, embedding_weight=5.0, fts_weight=1.0)
    assert merged[0]["symbol_id"] == 2


def test_rrf_merge_empty():
    assert rrf_merge([], []) == []
    result = rrf_merge([{"symbol_id": 1, "name": "a"}], [])
    assert len(result) == 1


# --- Test provider factory ---


def test_get_provider_ollama():
    provider = get_provider("qwen3-embedding")
    assert isinstance(provider, OllamaProvider)
    assert "ollama" in provider.name


def test_get_provider_ollama_explicit():
    provider = get_provider("ollama:nomic-embed-text")
    assert isinstance(provider, OllamaProvider)
    assert "nomic-embed-text" in provider.name


def test_get_provider_voyage():
    with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
        provider = get_provider("voyage-code-3")
        assert isinstance(provider, VoyageProvider)
        assert "voyage" in provider.name


def test_get_provider_voyage_explicit():
    provider = get_provider("voyage:voyage-code-3", api_key="test-key")
    assert isinstance(provider, VoyageProvider)


def test_get_provider_openai():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider("openai:text-embedding-3-small")
        assert isinstance(provider, OpenAIProvider)
        assert "openai" in provider.name
        assert "text-embedding-3-small" in provider.name


def test_get_provider_openai_inferred():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider("text-embedding-3-small")
        assert isinstance(provider, OpenAIProvider)


def test_get_provider_openai_custom_base_url():
    provider = get_provider(
        "openai:my-model",
        api_key="test-key",
        base_url="https://api.together.xyz",
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider._base_url == "https://api.together.xyz"


def test_get_provider_openai_no_key():
    with patch.dict("os.environ", {}, clear=True):
        # Remove OPENAI_API_KEY if set
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="API key required"):
            get_provider("openai:text-embedding-3-small")


def test_get_provider_cohere():
    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        provider = get_provider("cohere:embed-v4.0")
        assert isinstance(provider, CohereProvider)
        assert "cohere" in provider.name


def test_get_provider_cohere_inferred():
    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        provider = get_provider("embed-v4.0")
        assert isinstance(provider, CohereProvider)


def test_get_provider_cohere_no_key():
    with patch.dict("os.environ", {}, clear=True):
        import os
        os.environ.pop("COHERE_API_KEY", None)
        with pytest.raises(ValueError, match="Cohere API key required"):
            get_provider("cohere:embed-v4.0")


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider("unknown:model")


# --- Test embed_symbols ---


def test_embed_symbols_batch():
    provider = MockProvider(dims=4)
    symbols = [
        {"id": 1, "name": "foo", "content": "int foo() {}"},
        {"id": 2, "name": "bar", "content": "int bar() {}"},
        {"id": 3, "name": "baz", "content": "int baz() {}"},
    ]
    results = embed_symbols(provider, symbols, batch_size=2)
    assert len(results) == 3
    for sym_id, emb_bytes in results:
        assert isinstance(sym_id, int)
        assert isinstance(emb_bytes, bytes)
        assert len(emb_bytes) == 16  # 4 dims * 4 bytes


# --- Test DB embedding methods ---


def test_db_embeddings(tmp_path):
    """Test embedding storage and vector search in the database."""
    from srclight.db import Database, FileRecord, SymbolRecord

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    # Insert a file and symbols
    file_id = db.upsert_file(FileRecord(
        path="test.py", content_hash="abc123", mtime=1.0,
        language="python", size=100, line_count=10,
    ))

    sym1_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="hello",
        start_line=1, end_line=5, content="def hello(): pass",
        body_hash="h1",
    ), "test.py")

    sym2_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="world",
        start_line=6, end_line=10, content="def world(): pass",
        body_hash="h2",
    ), "test.py")

    db.commit()

    # Generate mock embeddings
    provider = MockProvider(dims=4)
    vec1 = provider.embed_one("hello function")
    vec2 = provider.embed_one("world function")

    db.upsert_embedding(sym1_id, provider.name, 4, vector_to_bytes(vec1), "h1")
    db.upsert_embedding(sym2_id, provider.name, 4, vector_to_bytes(vec2), "h2")
    db.commit()

    # Check stats
    stats = db.embedding_stats()
    assert stats["total_symbols"] == 2
    assert stats["embedded_symbols"] == 2
    assert stats["coverage_pct"] == 100.0
    assert stats["model"] == provider.name
    assert stats["dimensions"] == 4

    # Vector search
    query_vec = provider.embed_one("hello function")
    query_bytes = vector_to_bytes(query_vec)
    results = db.vector_search(query_bytes, 4, limit=5)
    assert len(results) == 2
    # "hello" should be top result (identical embedding)
    assert results[0]["name"] == "hello"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=0.01)

    # Check symbols needing embeddings (none — both are done)
    needing = db.get_symbols_needing_embeddings(provider.name)
    assert len(needing) == 0

    db.close()


def test_db_embeddings_incremental(tmp_path):
    """Test that changed symbols get re-embedded."""
    from srclight.db import Database, FileRecord, SymbolRecord

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    file_id = db.upsert_file(FileRecord(
        path="test.py", content_hash="abc", mtime=1.0,
        language="python", size=50, line_count=5,
    ))

    sym_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="func",
        start_line=1, end_line=3, content="def func(): pass",
        body_hash="v1",
    ), "test.py")
    db.commit()

    # Embed it
    provider = MockProvider(dims=4)
    vec = provider.embed_one("func")
    db.upsert_embedding(sym_id, provider.name, 4, vector_to_bytes(vec), "v1")
    db.commit()

    # No symbols needing embedding
    assert len(db.get_symbols_needing_embeddings(provider.name)) == 0

    # Now "change" the symbol (update body_hash to v2 via direct SQL)
    db.conn.execute("UPDATE symbols SET body_hash = 'v2' WHERE id = ?", (sym_id,))
    db.commit()

    # Now it should need re-embedding
    needing = db.get_symbols_needing_embeddings(provider.name)
    assert len(needing) == 1
    assert needing[0]["id"] == sym_id

    db.close()
