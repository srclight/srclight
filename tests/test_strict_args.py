"""Consumer smoke test: the object the DAEMON launches enforces the shared policy.

Deliberately NOT a copy of mcpkit's suite — mcpkit tests the policy (28 tests, its own repo). This
tests the only thing mcpkit cannot: that *this server* actually uses it. If server.py were reverted
to a bare FastMCP while src/srclight/_mcpkit.py sat unused in the tree, this is the test that fails
and nothing else would.

It therefore asserts against `srclight.server.mcp` itself, never a freshly constructed lookalike.
A lookalike proves the class works; it proves nothing about what the daemon serves.
"""

import asyncio

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from srclight.server import mcp


def test_the_daemons_own_object_is_the_strict_one():
    from srclight._mcpkit import StrictArgsMCP
    assert isinstance(mcp, StrictArgsMCP)


def test_a_mistyped_filter_is_refused_not_silently_widened():
    """The measured defect: `projects` for `project` returned 20 real symbols from the wrong
    repos — same hit count, same shape, no error."""
    with pytest.raises(ToolError) as e:
        asyncio.run(mcp.call_tool("search_symbols", {"query": "main", "projects": "zhcorpus"}))
    msg = str(e.value)
    assert "projects" in msg
    assert "project" in msg
    assert "Nothing was executed" in msg


def test_zero_parameter_tools_are_closed_too():
    """srclight's hand-written guard skipped these: all five zero-parameter tools accepted
    anything, silently, because an empty property set was read as 'schema unknown'."""
    with pytest.raises(ToolError) as e:
        asyncio.run(mcp.call_tool("index_status", {"zz_bogus": 1}))
    assert "zz_bogus" in str(e.value)


def test_every_tool_advertises_the_closed_contract():
    """Runtime refusal alone left the catalog lying by omission: 0 of 42 tools advertised
    additionalProperties:false, so callers kept sending extras."""
    tools = asyncio.run(mcp.list_tools())
    unstamped = [t.name for t in tools if t.inputSchema.get("additionalProperties") is not False]
    assert not unstamped, f"{len(unstamped)} tools advertise a permissive contract: {unstamped[:5]}"
