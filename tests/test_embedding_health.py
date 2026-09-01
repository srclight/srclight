"""embedding_health must distinguish 'model pulled' from 'model resident'.

Incident 2026-08-31: hybrid_search fell back to keyword-only with
"timed out" while embedding_health reported status=ok/reachable=true.
Ollama (OLLAMA_MAX_LOADED_MODELS=1) had evicted qwen3-embedding for another
client's model; the 9.3 GB cold load took 16-18 s against a 20 s budget.
"""

import json
from unittest.mock import patch

from srclight import server


class _FakeDb:
    def embedding_stats(self, project=None):
        return {"model": "ollama:qwen3-embedding", "dimensions": 4096}


class _FakeProvider:
    name = "ollama:qwen3-embedding"

    def __init__(self, loaded):
        self._loaded = loaded

    def is_available(self):
        return True

    def is_loaded(self):
        return self._loaded


def _health(loaded):
    with patch.object(server, "_is_workspace_mode", return_value=False), \
         patch.object(server, "_get_db", return_value=_FakeDb()), \
         patch("srclight.embeddings.get_provider", return_value=_FakeProvider(loaded)):
        return json.loads(server.embedding_health())


def test_embedding_health_reports_resident_true():
    out = _health(True)
    assert out["status"] == "ok"
    assert out["resident"] is True
    assert "warning" not in out


def test_embedding_health_warns_when_pulled_but_not_resident():
    out = _health(False)
    assert out["reachable"] is True
    assert out["resident"] is False
    assert out["status"] == "ok"  # reachable, will work — but the first call is slow
    assert "cold load" in out["warning"]
    assert "SRCLIGHT_EMBED_REQUEST_TIMEOUT" in out["warning"]


def test_embedding_health_resident_unknown_when_ps_unreachable():
    out = _health(None)
    assert out["resident"] is None
