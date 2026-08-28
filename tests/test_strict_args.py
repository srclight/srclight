"""Unknown tool arguments must be REFUSED, never silently dropped.

Stock FastMCP discards unknown keys before the tool function runs. On this server that
turns a typo into a wrong answer rather than a failure — measured 2026-08-28:

    search_symbols(query="main", project="zhcorpus")   -> 20 hits, all zhcorpus
    search_symbols(query="main", projects="zhcorpus")  -> 20 hits, ZERO zhcorpus

Same shape, same count, real symbols, no error, wrong repos. These tests exist because
that failure is invisible at the call site, so nothing else would catch a regression.
"""

# pytest-asyncio is NOT installed here and these are the repo's first async tests, so the
# coroutines are driven with asyncio.run rather than adding a test dependency.

import asyncio

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from srclight.server import _StrictArgsMCP


def _server():
    srv = _StrictArgsMCP("test")

    @srv.tool()
    def scoped_search(query: str, project: str | None = None, limit: int = 10) -> dict:
        # Mirrors the real shape: the filter argument is optional, so dropping it
        # silently widens the search instead of failing.
        return {"query": query, "project": project, "limit": limit}

    return srv


def test_known_arguments_still_work():
    # The guard must not become a tax on correct calls.
    res = asyncio.run(_server().call_tool("scoped_search", {"query": "main", "project": "zhcorpus"}))
    assert "zhcorpus" in str(res)


def test_typo_on_the_filter_argument_is_refused_not_dropped():
    # The exact defect: 'projects' instead of 'project'. Stock FastMCP drops it and
    # searches everything; we must refuse instead.
    with pytest.raises(ToolError) as e:
        asyncio.run(_server().call_tool("scoped_search", {"query": "main", "projects": "zhcorpus"}))
    msg = str(e.value)
    assert "projects" in msg              # names the offending key
    assert "project" in msg               # shows what was accepted
    assert "Nothing was executed" in msg  # states no result was computed


def test_error_names_the_stale_server_diagnosis():
    # Whoever hits this has no other route to the conclusion: the call looked fine and
    # the tool exists. If this hint is ever dropped, the error stops being actionable.
    with pytest.raises(ToolError) as e:
        asyncio.run(_server().call_tool("scoped_search", {"query": "x", "bogus": 1}))
    assert "reconnect" in str(e.value).lower()


def test_every_unknown_key_is_reported_not_just_the_first():
    with pytest.raises(ToolError) as e:
        asyncio.run(_server().call_tool("scoped_search", {"query": "x", "aaa": 1, "zzz": 2}))
    assert "aaa" in str(e.value) and "zzz" in str(e.value)
