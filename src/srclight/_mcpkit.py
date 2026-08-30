"""mcpkit 0.1.0 - GENERATED SINGLE-FILE BUILD. DO NOT EDIT.

Regenerate with:  python -m mcpkit.vendor --out <path>
Upstream:         github.com/srclight/mcpkit @ a1a4bb5+dirty

Hand-editing this file is the failure this package exists to prevent: six copies of one policy,
independently wrong. `mcpkit.vendor.verify()` recomputes the hash below and rejects a modified
copy, so divergence is caught mechanically rather than discovered in a wrong answer.
"""
# mcpkit-vendored-sha256: 5c740cbdbc432cb194f79f59d4c40b051b203424c1a97bc3dfab39817dac93ea
# mcpkit-policy-sha256: d0d2e9a887e83a0abac6838b8132b3aad04fa0c171a28c9d62fb8d528dfee665
from __future__ import annotations

# ---- from mcpkit/seams.py -----------------------------------------------------
"""Verify at import that the SDK seams this package depends on still exist.

WHY FAIL LOUDLY. ``StrictArgsMCP`` reaches into FastMCP internals -- ``_tool_manager``,
``Tool.parameters``, and an async ``call_tool`` to override. All three are private or unstable, and
python-sdk v2 is expected to move them.

If a seam disappears, the natural failure is the worst one available: the subclass still constructs,
still serves, and silently stops enforcing. That is precisely the family this estate has spent two
days cataloguing -- a declaration that no longer takes effect while everything reports success. A
server would advertise ``additionalProperties: false`` and accept extras.

So the seams are checked once at import and the failure is an exception naming what moved. Louder
than a wrong answer, and it happens at start-up rather than at the first mistyped argument.
"""


import inspect

LAST_KNOWN_GOOD = "mcp 1.28-1.x"

__all__ = ["verify_seams", "SeamError", "LAST_KNOWN_GOOD"]


class SeamError(RuntimeError):
    """A FastMCP internal that mcpkit depends on has moved or disappeared."""


def _sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("mcp")
    except Exception:
        return "unknown"


def verify_seams() -> None:
    from mcp.server.fastmcp import FastMCP

    missing: list[str] = []

    if not hasattr(FastMCP, "call_tool"):
        missing.append("FastMCP.call_tool (the override point)")
    elif not inspect.iscoroutinefunction(FastMCP.call_tool):
        # If it stops being async our override's signature is wrong and every call breaks --
        # noisy, but worth naming precisely rather than letting a TypeError surface at runtime.
        missing.append("FastMCP.call_tool is no longer a coroutine")

    if not hasattr(FastMCP, "list_tools"):
        missing.append("FastMCP.list_tools (schema stamping point)")

    try:
        from mcp.server.fastmcp.tools import ToolManager
        if not hasattr(ToolManager, "get_tool"):
            missing.append("ToolManager.get_tool")
    except Exception:
        missing.append("mcp.server.fastmcp.tools.ToolManager")

    try:
        from mcp.server.fastmcp.tools.base import Tool
        if "parameters" not in getattr(Tool, "model_fields", {}):
            missing.append("Tool.parameters (the advertised schema)")
    except Exception:
        missing.append("mcp.server.fastmcp.tools.base.Tool")

    if missing:
        raise SeamError(
            "mcpkit depends on FastMCP internals that have moved: "
            + "; ".join(missing)
            + f". Detected mcp version: {_sdk_version()}; last known good: {LAST_KNOWN_GOOD}. "
            "REFUSING TO IMPORT rather than silently serving without argument validation -- a "
            "server that advertises additionalProperties:false and then accepts extras is worse "
            "than one that never claimed to validate. Pin mcp<2 or update mcpkit."
        )

# ---- from mcpkit/build.py -----------------------------------------------------
"""Report the code this process is actually running.

WHY. On 2026-08-26 seven stale MCP servers ran simultaneously, each answering from the snapshot of
its source it launched with. From a client's side a stale server is indistinguishable from a current
one: same tool list, same response shape, same confident payload. A server that cannot say which
revision it is cannot be caught.

Stamped ONCE at import and never re-read. A value that changes under a running process is worse than
none -- it would report the checkout's HEAD, not the code in memory, which is exactly the lie being
guarded against. ``None`` is honest and preferable to a guess.
"""


import os
import subprocess
import time
from pathlib import Path

__all__ = ["code_sha", "started_at", "uptime_s"]

_STARTED_AT = time.time()


def _resolve_sha() -> str | None:
    root = os.environ.get("MCPKIT_CODE_ROOT") or str(Path.cwd())
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip() or None
        if sha:
            dirty = subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            # A dirty tree means the sha does not describe what is running. Say so rather than
            # implying a clean correspondence.
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "+dirty"
        return sha
    except Exception:
        return None


_CODE_SHA = _resolve_sha()


def code_sha() -> str | None:
    """The revision stamped at import. None if it could not be determined."""
    return _CODE_SHA


def started_at() -> float:
    return _STARTED_AT


def uptime_s() -> float:
    return time.time() - _STARTED_AT

# ---- from mcpkit/ops.py -------------------------------------------------------
"""Operational surface: a session-free health endpoint and fail-closed bearer auth.

BOTH ARE OPT-IN AND HTTP-ONLY. A stdio server has no port to protect and no route to serve.

WHY HEALTH MUST NOT BE A TOOL. loqu8-dart's McpServiceBase registers ``health`` via
``registerTool`` -- reachable only THROUGH the MCP session. A health check that lives inside the
session cannot report that the session is the broken thing; it goes silent exactly when it is
needed, and silence is indistinguishable from "not asked". Proven on 2026-08-28: an 8744 process
that was healthy, held a valid LISTEN socket, and was unreachable by anything on the machine. Every
process-level check reported green.

WHY NO DB PING IN THE DEFAULT. A blocked accept loop is precisely what a heavy health check cannot
report -- it blocks too. Callers may pass cheap probes explicitly; the default stays cheap.
"""


import os
from typing import Any, Callable, Mapping




def _mcpkit_version() -> str:

    return __version__

__all__ = ["attach_healthz", "bearer_middleware", "require_token_or_exit", "EX_CONFIG"]

# sysexits.h EX_CONFIG. Pairs with systemd RestartPreventExitStatus=78 so a misconfigured unit
# STOPS rather than looping a broken config into place.
EX_CONFIG = 78

HEALTH_PATH = "/healthz"


def attach_healthz(
    mcp: Any,
    *,
    name: str | None = None,
    probes: Mapping[str, Callable[[], bool]] | None = None,
    path: str = HEALTH_PATH,
) -> None:
    """Register a session-free GET endpoint reporting what this process is.

    ``probes`` are optional named callables returning bool. Keep them cheap; anything that can
    block belongs outside the health path.
    """
    from starlette.responses import JSONResponse

    @mcp.custom_route(path, methods=["GET"])
    async def _healthz(request):  # noqa: ANN001
        results: dict[str, bool] = {}
        ok = True
        for pname, probe in (probes or {}).items():
            try:
                results[pname] = bool(probe())
            except Exception:
                results[pname] = False
            ok = ok and results[pname]
        return JSONResponse(
            {
                "ok": ok,
                "name": name or getattr(mcp, "name", None),
                "pid": os.getpid(),
                "code_sha": code_sha(),
                # A chassis adds a SECOND version axis. Without it the first cross-version bug
                # is undiagnosable from outside: "which server is on which mcpkit" has no answer.
                "mcpkit_version": _mcpkit_version(),
                "started_at": started_at(),
                "uptime_s": round(uptime_s(), 3),
                **({"probes": results} if results else {}),
            },
            headers={"Cache-Control": "no-store"},  # a cached health check is not a health check
        )


def bearer_middleware(token: str, *, exempt: tuple[str, ...] = (HEALTH_PATH,)):
    """Starlette middleware requiring ``Authorization: Bearer <token>``.

    ``/healthz`` is exempt BY DESIGN: a restart script must be able to verify what came up without
    holding a credential, and the health payload carries no corpus data.
    """
    import hmac
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class _Bearer(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # noqa: ANN001
            if request.url.path in exempt:
                return await call_next(request)
            got = request.headers.get("authorization", "")
            prefix = "Bearer "
            # compare_digest, not ==: an early-exit comparison leaks the token a byte at a time.
            if not (got.startswith(prefix) and hmac.compare_digest(got[len(prefix):], token)):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    return _Bearer


def require_token_or_exit(token: str | None, *, transport: str, service: str) -> None:
    """FAIL CLOSED on the deployed path. Call this from the entry point, not the library.

    Under WSL2 with networkingMode=mirrored a 127.0.0.1 bind is reachable from any normal-privilege
    Windows process -- demonstrated 2026-08-28, when a PowerShell request read restricted corpus
    text with no credential. "Loopback" is not the boundary it sounds like here.

    The library default stays permissive so tests and ad-hoc runs are unaffected; only the
    supervised entry point refuses, which is what keeps this from becoming a test tax.
    """
    if transport == "stdio" or token:
        return
    import sys

    print(
        f"{service}: refusing to start on {transport} without a bearer token.\n"
        "This port is reachable from the Windows host under mirrored networking.\n"
        "Set the token in the unit's EnvironmentFile, or run stdio for local debugging.",
        file=sys.stderr,
    )
    raise SystemExit(EX_CONFIG)

# ---- from mcpkit/strict.py ----------------------------------------------------
"""Refuse unknown tool arguments instead of silently discarding them.

WHY THIS EXISTS. Stock FastMCP drops keys that are not in the tool signature, and it does so
BEFORE the tool function is entered, so a check inside a tool body can never fire -- the key is
already gone. The advertised ``inputSchema`` also omits ``additionalProperties: false``, so nothing
at the protocol layer flags it either. The only seam that still sees the raw argument dict is
``call_tool``.

Measured on srclight, 2026-08-28, before the fix::

    search_symbols(query="main", project="zhcorpus")   -> 20 hits, all zhcorpus
    search_symbols(query="main", projects="zhcorpus")  -> 20 hits, ZERO zhcorpus
                                                          (19 bible, 1 bank-scraper)

One added letter. No error, identical hit count, identical shape, real symbols -- from repos the
caller never asked about. That is not a lossy call, it is a WRONG one: a genuine answer to a
question nobody asked, with no way for the caller to learn their constraint was ignored.

NOT A PYTHON PROBLEM. scarlight (TypeScript SDK, low-level Server path) hit the identical bug on
2026-08-27 -- its own comment records that the low-level path "validates the REQUEST ENVELOPE only;
inputSchema is never enforced". Argument validation is something every MCP server must supply for
itself, in any runtime.

BOTH HALVES ARE REQUIRED. Refusing at runtime while still advertising a permissive schema leaves
the catalog telling agents that extras are fine, so they keep sending them. ``StrictArgsMCP`` does
runtime refusal AND stamps ``additionalProperties: false`` onto the listed schema.
"""


from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError



# Checked once, at import. A missing seam must not degrade into silent non-enforcement.
verify_seams()

__all__ = ["StrictArgsMCP"]


class StrictArgsMCP(FastMCP):
    """A FastMCP that rejects unknown tool arguments and advertises that it does."""

    async def call_tool(self, name: str, arguments: dict[str, Any]):  # type: ignore[override]
        tool = self._tool_manager.get_tool(name)
        if tool is not None and isinstance(arguments, dict):
            params = tool.parameters or {}
            # ABSENT "properties" vs PRESENT-BUT-EMPTY are different facts, and conflating them
            # leaves a hole (found 2026-08-29 by exhaustive testing against a realistic server):
            #   * key ABSENT           -> the schema could not be introspected. Say nothing;
            #                             refusing everything would brick the tool, and a guard
            #                             that becomes a wall is worse than the bug it prevents.
            #   * key PRESENT, empty   -> FastMCP generated {"properties": {}} because the tool
            #                             genuinely takes NO arguments. Extras must be refused,
            #                             or a zero-parameter tool is the one place a typo still
            #                             slips through silently.
            if "properties" in params:
                accepted = set(params.get("properties") or {})
                unknown = sorted(k for k in arguments if k not in accepted)
                if unknown:
                    accepts = ", ".join(sorted(accepted)) if accepted else "(no arguments)"
                    raise ToolError(
                        f"unknown argument(s): {', '.join(unknown)}. "
                        f"Tool {name!r} accepts: {accepts}. "
                        "Nothing was executed and no result was computed. "
                        # The stale-server hint is load-bearing: whoever hits this has no other
                        # route to the conclusion, because the call looked fine and the tool
                        # exists. A long-lived daemon serves the code it launched with.
                        "If you expected these arguments to work, this server process is probably "
                        "running older code than you think - check the server's reported revision "
                        "and reconnect the MCP."
                    )
        return await super().call_tool(name, arguments)

    async def list_tools(self):  # type: ignore[override]
        """Advertise the closed contract the runtime actually enforces."""
        tools = await super().list_tools()
        for t in tools:
            schema = getattr(t, "inputSchema", None)
            # Only stamp object schemas that declare properties. Stamping a schema with no
            # properties would advertise "accepts nothing", contradicting the call_tool rule
            # above that treats an empty property set as unknown rather than closed.
            # Stamp whenever "properties" is present -- including when empty, because an empty
            # property set is now enforced as "accepts nothing" rather than "unknown".
            if isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
                schema.setdefault("additionalProperties", False)
        return tools

# ==== mcpkit provenance - nothing below this line is policy code ====
__version__ = "0.1.0"
__mcpkit_upstream_sha__ = "a1a4bb5+dirty"
