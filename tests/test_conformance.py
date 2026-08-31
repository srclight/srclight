"""ironmcp conformance: every srclight tool enforces advertisement == runtime."""
from ironmcp import assert_enforces_v2

from srclight.server import mcp


def test_all_tools_enforce_closed_contract():
    n = assert_enforces_v2(mcp)
    assert n >= 35, f"expected ~43 tools to enforce advertise==runtime, got {n}"
