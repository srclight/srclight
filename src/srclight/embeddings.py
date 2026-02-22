"""Embedding providers and hybrid search for Srclight Layer 7.

Supports:
- Ollama (local, default) — zero Python ML deps, just HTTP
- Voyage Code 3 (API, optional) — best code retrieval quality

Architecture:
- Providers generate embeddings via HTTP APIs (no torch/transformers needed)
- Embeddings stored as float32 BLOB in SQLite symbol_embeddings table
- Cosine similarity computed in Python (numpy fast path if available)
- Hybrid search: RRF fusion of FTS5 keyword results + embedding similarity
"""

from __future__ import annotations

import json
import logging
import struct
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("srclight.embeddings")


# --- Embedding text preparation ---


def prepare_embedding_text(symbol: dict) -> str:
    """Build the text to embed for a symbol.

    Combines name, signature, doc comment, and truncated content
    to give the embedding model rich context without exceeding limits.
    """
    parts = []

    # Qualified name or plain name
    qname = symbol.get("qualified_name") or symbol.get("name") or ""
    if qname:
        parts.append(qname)

    # Signature (if different from name)
    sig = symbol.get("signature") or ""
    if sig and sig != qname:
        parts.append(sig)

    # Doc comment (natural language context)
    doc = symbol.get("doc_comment") or ""
    if doc:
        parts.append(doc.strip())

    # Content (truncated — most embedding models handle 512-8192 tokens)
    content = symbol.get("content") or ""
    if content:
        # Truncate to ~2000 chars (~500 tokens) to stay within limits
        parts.append(content[:2000])

    return "\n".join(parts)


# --- Provider protocol ---


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier string (stored in DB)."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output embedding dimensions."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]


# --- Ollama provider ---


class OllamaProvider(EmbeddingProvider):
    """Embed via Ollama's HTTP API (local, zero Python ML deps).

    Default model: qwen3-embedding (best quality available locally).
    Fallback: nomic-embed-text (lighter, well-tested).

    Ollama endpoint: http://localhost:11434 (accessible from WSL to Windows Ollama).
    """

    def __init__(self, model: str = "qwen3-embedding", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Probe by embedding a test string
            vec = self.embed_one("test")
            self._dimensions = len(vec)
        return self._dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama /api/embed endpoint."""
        url = f"{self._base_url}/api/embed"
        payload = json.dumps({"model": self._model, "input": texts}).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        # Cache dimensions from first result
        if self._dimensions is None and embeddings:
            self._dimensions = len(embeddings[0])

        return embeddings

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self._model in models or f"{self._model}:latest" in models
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def pull_model(self) -> None:
        """Pull the model via Ollama API."""
        url = f"{self._base_url}/api/pull"
        payload = json.dumps({"name": self._model, "stream": False}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # This can take a while for large models
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()


# --- Voyage Code 3 provider ---


class VoyageProvider(EmbeddingProvider):
    """Embed via Voyage AI API (best code retrieval quality).

    Requires VOYAGE_API_KEY environment variable.
    Model: voyage-code-3 (1024 dims, 32K context).
    """

    API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str | None = None, model: str = "voyage-code-3"):
        import os
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self._model = model
        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return f"voyage:{self._model}"

    @property
    def dimensions(self) -> int:
        # voyage-code-3 outputs 1024 dimensions
        return 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Voyage API."""
        payload = json.dumps({
            "model": self._model,
            "input": texts,
            "input_type": "document",
        }).encode()

        req = urllib.request.Request(
            self.API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            raise ConnectionError(f"Voyage API error ({e.code}): {body}") from e

        results = data.get("data", [])
        return [r["embedding"] for r in sorted(results, key=lambda x: x["index"])]


# --- Vector math (pure Python, numpy fast path) ---


def vectors_to_bytes(vectors: list[list[float]]) -> list[bytes]:
    """Convert float vectors to bytes for SQLite BLOB storage."""
    return [struct.pack(f'{len(v)}f', *v) for v in vectors]


def vector_to_bytes(vector: list[float]) -> bytes:
    """Convert a single float vector to bytes."""
    return struct.pack(f'{len(vector)}f', *vector)


def bytes_to_vector(data: bytes) -> list[float]:
    """Convert bytes back to float vector."""
    n = len(data) // 4
    return list(struct.unpack(f'{n}f', data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Uses numpy/cupy when available."""
    from .vector_math import _backend, _np

    if _np is not None:
        va = _np.asarray(a, dtype=_np.float32)
        vb = _np.asarray(b, dtype=_np.float32)
        na, nb = _np.linalg.norm(va), _np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        result = _np.dot(va, vb) / (na * nb)
        return float(result.get()) if _backend == "cupy" else float(result)
    # Pure Python fallback
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Reciprocal Rank Fusion (RRF) ---


def rrf_merge(
    fts_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    k: int = 60,
    fts_weight: float = 1.0,
    embedding_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge FTS5 and embedding search results using Reciprocal Rank Fusion.

    RRF score = sum(weight / (k + rank)) for each result list.
    k=60 is the standard value from the RRF paper (Cormack et al., 2009).

    Args:
        fts_results: Results from FTS5 search (must have 'symbol_id')
        embedding_results: Results from embedding search (must have 'symbol_id')
        k: RRF parameter (default 60)
        fts_weight: Weight for FTS results (default 1.0)
        embedding_weight: Weight for embedding results (default 1.0)

    Returns:
        Merged results sorted by combined RRF score (descending).
    """
    scores: dict[int, float] = {}
    data: dict[int, dict] = {}

    # Score FTS results by rank position
    for rank, result in enumerate(fts_results):
        sid = result["symbol_id"]
        scores[sid] = scores.get(sid, 0.0) + fts_weight / (k + rank + 1)
        if sid not in data:
            data[sid] = dict(result)
            data[sid]["sources"] = []
        data[sid]["sources"].append("fts")

    # Score embedding results by rank position
    for rank, result in enumerate(embedding_results):
        sid = result["symbol_id"]
        scores[sid] = scores.get(sid, 0.0) + embedding_weight / (k + rank + 1)
        if sid not in data:
            data[sid] = dict(result)
            data[sid]["sources"] = []
        data[sid]["sources"].append("embedding")
        # Preserve similarity score
        if "similarity" in result:
            data[sid]["similarity"] = result["similarity"]

    # Build merged results sorted by RRF score
    merged = []
    for sid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = data[sid]
        entry["rrf_score"] = round(score, 6)
        merged.append(entry)

    return merged


# --- Provider factory ---


def get_provider(model: str, **kwargs) -> EmbeddingProvider:
    """Create an embedding provider from a model specifier.

    Formats:
        "ollama:qwen3-embedding" or "qwen3-embedding" -> OllamaProvider
        "voyage:voyage-code-3" or "voyage-code-3" -> VoyageProvider
        "ollama:nomic-embed-text" -> OllamaProvider with nomic model
    """
    if ":" in model:
        provider_type, model_name = model.split(":", 1)
    else:
        # Default: if it starts with "voyage", use Voyage; otherwise Ollama
        if model.startswith("voyage"):
            provider_type = "voyage"
            model_name = model
        else:
            provider_type = "ollama"
            model_name = model

    if provider_type == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaProvider(model=model_name, base_url=base_url)
    elif provider_type == "voyage":
        api_key = kwargs.get("api_key")
        return VoyageProvider(api_key=api_key, model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


# --- Batch embedding for indexing ---


def embed_symbols(
    provider: EmbeddingProvider,
    symbols: list[dict],
    batch_size: int = 32,
    on_progress: Any | None = None,
) -> list[tuple[int, bytes]]:
    """Embed a list of symbols in batches.

    Args:
        provider: Embedding provider to use
        symbols: List of symbol dicts (from db.get_symbols_needing_embeddings)
        batch_size: Number of symbols per batch
        on_progress: Optional callback(batch_num, total_batches)

    Returns:
        List of (symbol_id, embedding_bytes) tuples
    """
    results = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(symbols), batch_size):
        batch = symbols[batch_idx:batch_idx + batch_size]
        texts = [prepare_embedding_text(sym) for sym in batch]

        if on_progress:
            on_progress(batch_idx // batch_size + 1, total_batches)

        try:
            vectors = provider.embed_batch(texts)
            for sym, vec in zip(batch, vectors):
                results.append((sym["id"], vector_to_bytes(vec)))
        except Exception as e:
            logger.error("Embedding batch %d failed: %s", batch_idx // batch_size + 1, e)
            # Skip failed batch, continue with next
            continue

    return results
