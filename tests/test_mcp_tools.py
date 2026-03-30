"""Tests for MCP tool parameter handling.

Verifies that all srclight MCP tools accept optional parameters correctly:
- Called with only required parameters (optional params omitted)
- Called with optional parameters explicitly set to None
- Called with all parameters provided

Uses a full MCP client→server roundtrip to catch protocol-level issues
like -32602 (Invalid request parameters) that wouldn't surface in unit tests.

See: https://github.com/srclight/srclight/issues/9
"""

import asyncio
import json
import subprocess
from pathlib import Path

import anyio
import pytest
from anyio import create_memory_object_stream
from mcp.client.session import ClientSession
from mcp.shared.session import SessionMessage

from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord
from srclight.server import configure, mcp as srclight_mcp


@pytest.fixture
def indexed_repo(tmp_path):
    """Create a minimal indexed repo for tool tests."""
    # Create DB with test data
    db_path = tmp_path / ".srclight" / "index.db"
    db_path.parent.mkdir()
    db = Database(db_path)
    db.open()
    db.initialize()

    # Insert test file
    fid = db.upsert_file(FileRecord(
        path="src/main.py", language="python",
        content_hash="abc", mtime=1.0, size=100, line_count=10,
    ))

    # Insert test symbols
    sid1 = db.insert_symbol(SymbolRecord(
        file_id=fid, kind="function", name="hello",
        start_line=1, end_line=3, content="def hello(): pass",
        line_count=3,
    ), "src/main.py")

    sid2 = db.insert_symbol(SymbolRecord(
        file_id=fid, kind="function", name="world",
        start_line=5, end_line=8, content="def world(): hello()",
        line_count=4,
    ), "src/main.py")

    # Insert an edge (world calls hello)
    db.insert_edge(EdgeRecord(
        source_id=sid2, target_id=sid1, edge_type="calls",
    ))

    db.close()

    # Set up git repo so git tools don't crash
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("def hello(): pass\n\ndef world(): hello()\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

    configure(db_path=db_path, repo_root=tmp_path)
    yield tmp_path

    # Cleanup
    configure(db_path=None, repo_root=None)


async def _call_tool_via_mcp(tool_name: str, arguments: dict) -> dict:
    """Call a tool through a full MCP client→server roundtrip.

    Returns the result content (parsed JSON or raw text).
    Raises if the MCP protocol itself fails (e.g. -32602).
    """
    server_to_client_send, server_to_client_recv = create_memory_object_stream[SessionMessage](100)
    client_to_server_send, client_to_server_recv = create_memory_object_stream[SessionMessage](100)

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            srclight_mcp._mcp_server.run,
            client_to_server_recv,
            server_to_client_send,
            srclight_mcp._mcp_server.create_initialization_options(),
        )

        session = ClientSession(server_to_client_recv, client_to_server_send)
        async with session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

        tg.cancel_scope.cancel()

    # Extract text content
    text = result.content[0].text if result.content else ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": text}


# ---------------------------------------------------------------------------
# Core search tools
# ---------------------------------------------------------------------------

class TestSearchToolParams:
    """Test that search tools accept optional params in all valid forms."""

    @pytest.mark.asyncio
    async def test_search_symbols_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("search_symbols", {"query": "hello"})
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_search_symbols_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("search_symbols", {
            "query": "hello", "kind": None, "project": None,
        })
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_search_symbols_all_params(self, indexed_repo):
        result = await _call_tool_via_mcp("search_symbols", {
            "query": "hello", "kind": "function", "limit": 5,
        })
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_hybrid_search_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("hybrid_search", {"query": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_hybrid_search_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("hybrid_search", {
            "query": "hello", "kind": None, "project": None,
        })
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_semantic_search_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("semantic_search", {"query": "hello"})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Overview / status tools
# ---------------------------------------------------------------------------

class TestOverviewToolParams:
    """Test overview tools with various parameter combinations."""

    @pytest.mark.asyncio
    async def test_codebase_map_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("codebase_map", {})
        assert "repo_root" in result or "error" in result

    @pytest.mark.asyncio
    async def test_codebase_map_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("codebase_map", {"project": None})
        assert "repo_root" in result or "error" in result

    @pytest.mark.asyncio
    async def test_list_projects_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("list_projects", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_index_status_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("index_status", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_server_stats_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("server_stats", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_setup_guide_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("setup_guide", {})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Symbol lookup tools
# ---------------------------------------------------------------------------

class TestSymbolToolParams:
    """Test symbol lookup tools with optional project param."""

    @pytest.mark.asyncio
    async def test_get_symbol_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_symbol", {"name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_symbol_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("get_symbol", {"name": "hello", "project": None})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_signature_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_signature", {"name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_symbols_in_file_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("symbols_in_file", {"path": "src/main.py"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_symbols_in_file_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("symbols_in_file", {"path": "src/main.py", "project": None})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Graph tools
# ---------------------------------------------------------------------------

class TestGraphToolParams:
    """Test graph traversal tools with optional params."""

    @pytest.mark.asyncio
    async def test_get_callers_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_callers", {"symbol_name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_callers_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("get_callers", {"symbol_name": "hello", "project": None})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_callees_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_callees", {"symbol_name": "world"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_dependents_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_dependents", {"symbol_name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_dependents_all_params(self, indexed_repo):
        result = await _call_tool_via_mcp("get_dependents", {
            "symbol_name": "hello", "transitive": True, "project": None,
        })
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_implementors_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_implementors", {"interface_name": "Base"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_tests_for_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_tests_for", {"symbol_name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_type_hierarchy_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_type_hierarchy", {"name": "hello"})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------

class TestGitToolParams:
    """Test git tools with optional params."""

    @pytest.mark.asyncio
    async def test_blame_symbol_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("blame_symbol", {"symbol_name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_recent_changes_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("recent_changes", {})
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_recent_changes_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("recent_changes", {
            "n": 5, "author": None, "path_filter": None, "project": None,
        })
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_git_hotspots_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("git_hotspots", {})
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_git_hotspots_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("git_hotspots", {"n": 10, "since": None, "project": None})
        assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_whats_changed_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("whats_changed", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_whats_changed_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("whats_changed", {"project": None})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_changes_to_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("changes_to", {"symbol_name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_changes_to_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("changes_to", {"symbol_name": "hello", "n": 5, "project": None})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Build / platform tools
# ---------------------------------------------------------------------------

class TestBuildToolParams:
    """Test build and platform tools with optional params."""

    @pytest.mark.asyncio
    async def test_get_build_targets_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("get_build_targets", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_platform_variants_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("get_platform_variants", {"symbol_name": "hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_platform_conditionals_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("platform_conditionals", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_platform_conditionals_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("platform_conditionals", {"project": None, "platform": None})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Embedding tools
# ---------------------------------------------------------------------------

class TestEmbeddingToolParams:
    """Test embedding tools with optional params."""

    @pytest.mark.asyncio
    async def test_embedding_status_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("embedding_status", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_embedding_status_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("embedding_status", {"project": None})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_embedding_health_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("embedding_health", {})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

class TestAnalysisToolParams:
    """Test analysis tools with optional params."""

    @pytest.mark.asyncio
    async def test_find_imports_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("find_imports", {"path": "src/main.py"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_find_imports_null_project(self, indexed_repo):
        result = await _call_tool_via_mcp("find_imports", {"path": "src/main.py", "project": None})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_find_dead_code_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("find_dead_code", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_find_dead_code_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("find_dead_code", {"project": None, "kind": None})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_find_pattern_required_only(self, indexed_repo):
        result = await _call_tool_via_mcp("find_pattern", {"pattern": "def.*hello"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_find_pattern_null_optionals(self, indexed_repo):
        result = await _call_tool_via_mcp("find_pattern", {
            "pattern": "def.*hello", "project": None, "language": None, "kind": None,
        })
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_reindex_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("reindex", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_reindex_null_path(self, indexed_repo):
        result = await _call_tool_via_mcp("reindex", {"path": None})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# show_status tool
# ---------------------------------------------------------------------------

class TestShowStatusParams:
    """Test show_status with optional message param."""

    @pytest.mark.asyncio
    async def test_show_status_no_params(self, indexed_repo):
        result = await _call_tool_via_mcp("show_status", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_show_status_with_message(self, indexed_repo):
        result = await _call_tool_via_mcp("show_status", {"message": "test"})
        assert isinstance(result, dict)
