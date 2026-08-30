"""mcpkit 0.2.1 - GENERATED SINGLE-FILE BUILD. DO NOT EDIT.

Regenerate with:  python -m mcpkit.vendor --out <path>
Upstream:         github.com/srclight/mcpkit @ 2fb7fe2

Hand-editing this file is the failure this package exists to prevent: six copies of one policy,
independently wrong. `mcpkit.vendor.verify()` recomputes the hash below and rejects a modified
copy, so divergence is caught mechanically rather than discovered in a wrong answer.
"""
# mcpkit-vendored-sha256: 22eb231ef9b05bf397508a36b74c2fd3576e5e8a939334be627d7cf03f9f82c1
# mcpkit-policy-sha256: 0346df332b8f55848b3edd94447f8e7159ab88de876b629ba978b6b325e10d30
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

ONE FastMCP, NOT TWO. These seams are the shapes of the FastMCP BUNDLED IN THE OFFICIAL ``mcp`` SDK
(``mcp.server.fastmcp``), pinned ``mcp>=1.28,<2``. They are NOT the shapes of the standalone
PrefectHQ ``fastmcp`` v3 package (which vaultlight runs) -- there ``get_tool`` is public, middleware
is first-class, and the internals mcpkit reaches into do not exist under these names. mcpkit does not
work on fastmcp v3 as-is; adopting it there is a rewrite, not a config change. Written down here so
nobody vendors this file into a v3 server and is surprised when the seam-check passes and enforcement
still does not fit.
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
                # WWW-Authenticate on the 401 is what the MCP spec's OAuth flow expects: it names
                # the scheme the client must use, so a compliant client knows HOW to retry rather
                # than only THAT it failed. The scheme is Bearer; there is no realm to leak.
                return JSONResponse(
                    {"error": "unauthorized"}, status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
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


import unicodedata
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError



# Checked once, at import. A missing seam must not degrade into silent non-enforcement.
verify_seams()

__all__ = ["StrictArgsMCP"]

# The generic stale-server diagnosis, used when a server does not name its own revision surface.
# A server that HAS one (caneslight -> pack_status, srclight -> index_status) supplies a better
# string via the reconnect_hint constructor argument. It is DATA, never a method: a consumer hands
# over a string and nothing else, so a later change to the surrounding message still reaches every
# consumer. A method hook would re-fork call_tool's neighbourhood -- the exact thing vendoring one
# shared policy exists to prevent.
_DEFAULT_RECONNECT_HINT = "check the server's reported revision and reconnect the MCP"


class StrictArgsMCP(FastMCP):
    """A FastMCP that rejects unknown tool arguments and advertises that it does."""

    def __init__(self, *args: Any, reconnect_hint: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reconnect_hint = reconnect_hint or _DEFAULT_RECONNECT_HINT

    async def call_tool(self, name: str, arguments: dict[str, Any]):  # type: ignore[override]
        tool = self._tool_manager.get_tool(name)
        if tool is not None and isinstance(arguments, dict):
            params = tool.parameters or {}
            # THREE facts about a schema, and each wants a different answer. Conflating any two of
            # them has bitten this estate once each:
            #   * "properties" ABSENT              -> schema uninstrospectable. Stay permissive;
            #                                         a guard that bricks what it cannot read is a
            #                                         wall, worse than the bug it prevents.
            #   * "properties" PRESENT, empty      -> FastMCP emits {"properties": {}} for a tool
            #                                         that genuinely takes NO arguments. Refuse
            #                                         extras, or a zero-parameter tool is the one
            #                                         place a typo still slips through silently.
            #   * additionalProperties is True     -> the author OPTED OUT: a passthrough/proxy tool
            #                                         that accepts arbitrary keys (the JSON-Schema
            #                                         standard way to say so). Honour it -- refusing
            #                                         here would advertise-open-but-refuse, the same
            #                                         catalog-lies-about-runtime bug this guard
            #                                         exists to close, pointing the other way.
            if "properties" in params and params.get("additionalProperties") is not True:
                accepted = set(params.get("properties") or {})
                unknown = sorted(k for k in arguments if k not in accepted)
                if unknown:
                    raise ToolError(_unknown_args_message(name, unknown, accepted, self._reconnect_hint))
        return await super().call_tool(name, arguments)

    async def list_tools(self):  # type: ignore[override]
        """Advertise exactly the contract the runtime enforces -- no more, no less."""
        tools = await super().list_tools()
        for t in tools:
            schema = getattr(t, "inputSchema", None)
            # setdefault, NOT force: an author who set additionalProperties:true meant it (opt-out
            # above), so leave it true and advertise open. Stamp false only when the key is absent.
            # Skip a schema with no "properties" -- stamping "accepts nothing" there would
            # contradict call_tool, which treats an absent property set as unknown, not closed.
            # The invariant this preserves is ADVERTISEMENT == RUNTIME for every tool; pin that,
            # never "true becomes false", or a proxy tool later fights its own conformance test.
            if isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
                schema.setdefault("additionalProperties", False)
        return tools


# The error message's SIZE is bounded by the server, never by its input. A caller sending 5,000
# unknown keys must not be able to reflect a 59kB error back over MCP and into the log
# (canes-fideles-d8, 2026-08-30). Values are NEVER echoed, only key NAMES — a rejected argument
# cannot be used to bounce data into logs, and that is deliberate, not incidental.
_MAX_ENUMERATED = 10


def _unknown_args_message(
    name: str, unknown: list[str], accepted: set[str],
    reconnect_hint: str = _DEFAULT_RECONNECT_HINT,
) -> str:
    shown = unknown[:_MAX_ENUMERATED]
    more = len(unknown) - len(shown)
    listed = ", ".join(shown) + (f", and {more} more" if more > 0 else "")
    accepts = ", ".join(sorted(accepted)) if accepted else "(no arguments)"

    # NFKC confusables. Python normalises identifiers at PARSE time, so a parameter written with
    # U+00B5 MICRO SIGN is advertised as U+03BC GREEK MU — two glyphs identical in nearly every
    # font. A developer copying the name from source is refused by something that looks exactly
    # like what they were told to send. Diagnose it by naming the CODEPOINT, since whoever hits
    # this has no other route to the answer. THE SCHEMA IS AUTHORITATIVE FOR ARGUMENT NAMES,
    # NEVER THE SOURCE, because normalisation happens between them.
    hints = []
    norm_accepted = {unicodedata.normalize("NFKC", a): a for a in accepted}
    for k in shown:
        canon = unicodedata.normalize("NFKC", k)
        if canon != k and canon in norm_accepted:
            cps = " ".join(f"U+{ord(c):04X}" for c in k)
            hints.append(f"{k!r} ({cps}) normalises to {norm_accepted[canon]!r}, which IS accepted")

    parts = [
        f"unknown argument(s): {listed}.",
        f"Tool {name!r} accepts: {accepts}.",
        "Nothing was executed and no result was computed.",
    ]
    if hints:
        parts.append("Note: " + "; ".join(hints) + ".")
    parts.append(
        "If you expected these arguments to work, this server process is probably running older "
        f"code than you think - {reconnect_hint}."
    )
    return " ".join(parts)

# ---- from mcpkit/conformance.py -----------------------------------------------
"""One shared conformance check, so five hand-written copies cannot drift into six.

WHY THIS EXISTS. Every consumer that adopted StrictArgsMCP also hand-wrote an "all tools are
closed" test -- srclight, conductor, model-radar, zhcorpus, caneslight: five copies of one
assertion. They ALREADY diverged (caneslight's asserted different refusal wording), which is this
package's own addition rule met exactly: three copies AND drifted. So the check moves here and each
consumer calls it in one line.

WHAT IT PINS, and why it is the RIGHT invariant. Not "additionalProperties always becomes false" --
that would cement the two-state model and fight a legitimate passthrough tool that opts out with
``additionalProperties: true``. The invariant is ADVERTISEMENT == RUNTIME: whatever the catalog
tells an agent about extra arguments, the runtime must actually do. That single property catches
BOTH failures this estate shipped -- advertised-closed-but-runtime-open (the original silent-discard
bug) and advertised-open-but-runtime-refuses (its reverse) -- and both StrictArgsMCP and any future
opt-out-aware design satisfy it.

PROVEN TO FIRE. A conformance check that never fails is theatre. ``assert_enforces`` raises against a
bare FastMCP (whose object-with-properties tools advertise NO additionalProperties -- neither closed
nor explicitly open), and ``test_conformance.py`` pins exactly that. A check you have not watched
reject a non-conforming server is a check you cannot trust.
"""


import asyncio
from typing import Any

__all__ = ["assert_enforces", "aassert_enforces"]

# A key no real tool declares. Sent as the lone argument to prove the closed contract is enforced.
_PROBE_KEY = "zz_mcpkit_conformance_probe_key"


async def aassert_enforces(mcp: Any, *, probe_key: str = _PROBE_KEY) -> int:
    """Assert ADVERTISEMENT == RUNTIME for every introspectable tool. Returns the count actually
    exercised. Raises AssertionError naming the first tool that lies. Async form; see the sync
    ``assert_enforces`` wrapper for use inside a normal test."""
    tools = await mcp.list_tools()
    if not tools:
        raise AssertionError("assert_enforces: the server advertises no tools -- nothing was checked")

    enforced = 0
    for t in tools:
        schema = getattr(t, "inputSchema", None)
        if not (isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema):
            # Uninstrospectable schema -> permissive by design; there is no closed contract to hold.
            continue

        adv = schema.get("additionalProperties")
        if adv is True:
            # The author opted the tool OPEN (a passthrough/proxy accepting arbitrary keys). That is
            # a declared, honoured contract, not a lie -- leave it be.
            continue
        if adv is not False:
            raise AssertionError(
                f"{t.name}: advertised schema is neither closed (additionalProperties:false) nor "
                "explicitly open (true). The catalog is silent, so an agent is told extras are fine "
                "while stock FastMCP would drop them -- the exact silent-discard this guard exists "
                "to close. Serve this tool through StrictArgsMCP."
            )

        # adv is False: the catalog promises extras are refused, so the runtime MUST refuse them.
        try:
            await mcp.call_tool(t.name, {probe_key: 1})
        except Exception:
            enforced += 1
            continue
        raise AssertionError(
            f"{t.name}: advertises additionalProperties:false but call_tool accepted the unknown "
            f"argument {probe_key!r} instead of refusing it. The guarantee the catalog makes to "
            "agents is not enforced at runtime -- the discarded-argument bug, back again."
        )

    if enforced == 0:
        raise AssertionError(
            "assert_enforces: no tool actually enforced a closed contract, so nothing was proven. "
            "A conformance check that verifies nothing is worse than none -- it manufactures "
            "confidence. Register at least one tool with arguments, or serve through StrictArgsMCP."
        )
    return enforced


def assert_enforces(mcp: Any, *, probe_key: str = _PROBE_KEY) -> int:
    """Synchronous wrapper for ``aassert_enforces`` -- drives the coroutine with ``asyncio.run`` so a
    plain test needs no async runner. Call from OUTSIDE an event loop (an ordinary test body)."""
    return asyncio.run(aassert_enforces(mcp, probe_key=probe_key))

# ==== mcpkit provenance - nothing below this line is policy code ====
__version__ = "0.2.1"
__mcpkit_upstream_sha__ = "2fb7fe2"
