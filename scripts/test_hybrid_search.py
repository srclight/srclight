#!/usr/bin/env python3
"""Test hybrid_search from Python (same code path as MCP server).

Run from repo root:
  .venv/bin/python scripts/test_hybrid_search.py

Prints timing, embedding status, and whether hybrid or keyword-only path was used.
Catches and prints any exception in the embedding path.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / ".srclight" / "index.db"


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from srclight.server import configure, hybrid_search

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        sys.exit(1)

    # Single-repo mode (same as srclight serve without --workspace)
    configure(db_path=DB_PATH, repo_root=REPO_ROOT)

    # 1) Embedding status
    from srclight.server import _get_db
    db = _get_db()
    stats = db.embedding_stats()
    print("embedding_status:", json.dumps(stats, indent=2))

    # 2) Time hybrid_search and capture mode / errors
    query = "database connection"
    limit = 5
    t0 = time.perf_counter()
    try:
        result_str = hybrid_search(query=query, limit=limit)
        elapsed = time.perf_counter() - t0
        result = json.loads(result_str)
        mode = result.get("mode", "?")
        count = result.get("result_count", 0)
        print(f"\nhybrid_search({query!r}, limit={limit})")
        print(f"  elapsed: {elapsed:.3f}s")
        print(f"  mode: {mode}")
        print(f"  result_count: {count}")
        if "model" in result and result["model"]:
            print(f"  model: {result['model']}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\nhybrid_search FAILED after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3) If we got keyword-only, optionally probe Ollama (use short timeout to avoid long hang)
    if stats.get("model") and result.get("mode") == "keyword only (no embeddings available)":
        import os
        from srclight.embeddings import get_provider, vector_to_bytes
        model_name = stats["model"]
        # Use 3s so we don't wait full SRCLIGHT_EMBED_REQUEST_TIMEOUT when Ollama is down
        os.environ["SRCLIGHT_EMBED_REQUEST_TIMEOUT"] = "3"
        print(f"\nProbing Ollama (model={model_name}, 3s timeout)...")
        t0 = time.perf_counter()
        try:
            provider = get_provider(model_name)
            vec = provider.embed_one(query)
            elapsed = time.perf_counter() - t0
            print(f"  embed_one: {elapsed:.3f}s (dims={len(vec)})")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  Ollama unreachable: {e}")
            print("  → Start Ollama (e.g. ollama serve) or hybrid_search will stay keyword-only.")


if __name__ == "__main__":
    main()
