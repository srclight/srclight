"""Srclight CLI.

Usage:
    srclight index [PATH]       Index a codebase
    srclight search QUERY       Search indexed symbols
    srclight symbols FILE       List symbols in a file
    srclight status             Show index status
    srclight serve              Start MCP server
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from . import __version__


def _find_repo_root(start: Path) -> Path:
    """Walk up from start to find .git directory."""
    check = start.resolve()
    while check != check.parent:
        if (check / ".git").exists():
            return check
        check = check.parent
    return start.resolve()


def _migrate_legacy_dir(root: Path) -> None:
    """Migrate .codelight/ → .srclight/ if needed (project rename)."""
    legacy_dir = root / ".codelight"
    new_dir = root / ".srclight"
    if legacy_dir.exists() and not new_dir.exists():
        try:
            legacy_dir.rename(new_dir)
            click.echo(f"  Migrated {legacy_dir} -> {new_dir}")
        except OSError:
            pass  # Cross-device or permission issue — user can move manually


def _get_db_path(root: Path) -> Path:
    """Get the index database path, migrating from legacy locations if needed.

    New location: {root}/.srclight/index.db
    Legacy: {root}/.codelight/index.db (pre-rename), {root}/.srclight.db (flat-file era)
    """
    _migrate_legacy_dir(root)

    new_path = root / ".srclight" / "index.db"
    legacy_flat = root / ".srclight.db"

    # Migrate flat-file legacy if needed
    if legacy_flat.exists() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_flat.rename(new_path)
        click.echo(f"  Migrated {legacy_flat} -> {new_path}")

    return new_path


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """Srclight — Deep code indexing for AI agents."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def parse_extension_overrides(values: tuple[str, ...]) -> dict[str, str]:
    """Turn `--ext EXT=LANGUAGE` values into {extension: language}.

    The extension is normalised (`INC` and `.inc` are the same thing) and the
    language must be one srclight parses, so a typo fails here rather than
    leaving the files silently unread. The single value `none` clears a
    declaration an index already holds.
    """
    from .languages import LANGUAGES, SKIP_LANGUAGE, normalize_extension

    if len(values) == 1 and values[0].strip().lower() == "none":
        return {}

    overrides: dict[str, str] = {}
    for value in values:
        ext, sep, lang = value.partition("=")
        ext, lang = ext.strip(), lang.strip().lower()
        if not sep or not ext or not lang:
            raise ValueError(f"Malformed --ext value '{value}': expected EXT=LANGUAGE")
        if lang != SKIP_LANGUAGE and lang not in LANGUAGES:
            raise ValueError(
                f"Unknown language '{lang}' in --ext value '{value}': expected "
                f"{SKIP_LANGUAGE} or one of {', '.join(sorted(LANGUAGES))}"
            )
        normalized = normalize_extension(ext)
        # Detection looks up Path.suffix, which is only the last component:
        # a declaration on `.d.ts` would be stored, echoed back, and never
        # match a file. Better rejected than confirmed as a no-op.
        if "." in normalized[1:]:
            raise ValueError(
                f"Multi-part extension '{ext}' in --ext value '{value}': only a final "
                f"suffix can be matched (use '.ts', not '.d.ts')"
            )
        overrides[normalized] = lang
    return overrides


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--db", "db_path", type=click.Path(), help="Database path (default: .srclight/index.db)")
@click.option("--embed", "embed_model", type=str, default=None,
              help="Embedding model (e.g., qwen3-embedding, voyage-code-3). Passed once: "
                   "later runs reuse the model recorded in the index, else "
                   "$SRCLIGHT_EMBED_MODEL.")
@click.option("--no-embed", is_flag=True, default=False,
              help="Index without embeddings for this run. Changed files still lose "
                   "the embeddings of the symbols they replace.")
@click.option("--forget-embed-model", is_flag=True, default=False,
              help="Stop embedding this index for good: later runs, git hooks included, "
                   "leave embeddings alone until --embed is passed again.")
@click.option("--ext", "ext_overrides", multiple=True, metavar="EXT=LANGUAGE",
              help="Read an extra extension as the given language (e.g. --ext .inc=cpp), "
                   "or --ext .inc=skip to leave it unread. Repeatable. Recorded in the "
                   "index, so later runs and the git hooks keep the same rule; pass "
                   "--ext none to clear.")
def index(path: str, db_path: str | None, embed_model: str | None, no_embed: bool,
          forget_embed_model: bool, ext_overrides: tuple[str, ...]):
    """Index a codebase for AI-powered search."""
    from .db import Database
    from .indexer import EMBED_MODEL_ENV, IndexConfig, Indexer, resolve_embed_model

    root = Path(path).resolve()
    if not root.is_dir():
        click.echo(f"Error: {root} is not a directory", err=True)
        sys.exit(1)

    db_file = Path(db_path) if db_path else _get_db_path(root)

    # Ensure .srclight/ directory exists
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Keep .srclight/ out of git (in .git/info/exclude, never a tracked file)
    if (root / ".git").is_dir():
        _ensure_srclight_ignored(root)

    click.echo(f"Indexing {root}")
    click.echo(f"Database: {db_file}")

    db = Database(db_file)
    db.open()
    db.initialize()

    if forget_embed_model:
        db.forget_embedding_model()
        db.commit()
        note = "Embedding model: forgotten — later runs will not embed"
        if embed_model:
            # --no-embed names the flag it overrode; this one must too, or
            # the run reads as if --embed had been recorded and used.
            note += f"; --embed {embed_model} ignored"
        click.echo(note)

    try:
        declared = parse_extension_overrides(ext_overrides) if ext_overrides else None
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    config = IndexConfig(
        root=root, embed_model=embed_model,
        disable_embeddings=no_embed or forget_embed_model,
        extension_overrides=declared,
    )
    if declared:
        click.echo("Extra extensions: "
                   + ", ".join(f"{e} -> {lang}" for e, lang in sorted(declared.items())))
    elif declared == {}:
        click.echo("Extra extensions: cleared")
    # Resolve once and pin the result: resolving again inside the indexer, after
    # the file pass, can disagree with what we printed here — a checkout that
    # drops every embedded file cascade-deletes its embeddings mid-run.
    resolved_model = resolve_embed_model(db, config)
    config.embed_model = resolved_model
    if no_embed:
        if embed_model:
            click.echo(f"Embeddings: skipped (--no-embed); --embed {embed_model} ignored")
        else:
            click.echo("Embeddings: skipped (--no-embed)")
    elif resolved_model and embed_model:
        click.echo(f"Embedding model: {resolved_model}")
    elif resolved_model:
        origin = ("from the existing index"
                  if resolved_model == db.detect_embedding_model()
                  else f"from ${EMBED_MODEL_ENV}")
        click.echo(f"Embedding model: {resolved_model} ({origin})")

    indexer = Indexer(db, config)

    line_open = [False]  # a progress line waits for its newline

    def on_progress(file: str, current: int, total: int):
        pct = (current / total * 100) if total > 0 else 0
        click.echo(f"\r  [{current}/{total}] {pct:5.1f}% {file[:60]:<60}", nl=False)
        line_open[0] = True

    def on_phase(name: str):
        if line_open[0]:
            click.echo()
            line_open[0] = False
        click.echo(f"  {name}...")

    stats = indexer.index(root, on_progress=on_progress, on_phase=on_phase)
    if line_open[0]:
        click.echo()

    click.echo()
    click.echo(f"  Files scanned:   {stats.files_scanned}")
    click.echo(f"  Files indexed:   {stats.files_indexed}")
    click.echo(f"  Files unchanged: {stats.files_unchanged}")
    click.echo(f"  Files removed:   {stats.files_removed}")
    click.echo(f"  Symbols found:   {stats.symbols_extracted}")
    click.echo(f"  Errors:          {stats.errors}")
    click.echo(f"  Time:            {stats.elapsed_seconds:.2f}s")

    db_stats = db.stats()
    click.echo(f"  Database size:   {db_stats['db_size_mb']} MB")

    if resolved_model:
        emb_stats = db.embedding_stats()
        click.echo(f"  Embedded now:    {stats.symbols_embedded}")
        click.echo(f"  Embeddings:      {emb_stats['embedded_symbols']}/{emb_stats['total_symbols']}"
                    f" ({emb_stats['coverage_pct']}%)")

    db.close()


@main.command()
@click.argument("query")
@click.option("--kind", "-k", help="Filter by symbol kind (function, class, method, ...)")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def search(query: str, kind: str | None, limit: int, db_path: str | None, json_output: bool):
    """Search indexed code symbols."""
    from .db import Database

    root = _find_repo_root(Path.cwd())
    db_file = Path(db_path) if db_path else _get_db_path(root)

    if not db_file.exists():
        click.echo(f"No index found at {db_file}. Run 'srclight index' first.", err=True)
        sys.exit(1)

    db = Database(db_file)
    db.open()

    results = db.search_symbols(query, kind=kind, limit=limit)

    if json_output:
        click.echo(json.dumps(results, indent=2))
    else:
        if not results:
            click.echo(f"No results for '{query}'")
        else:
            click.echo(f"Found {len(results)} results for '{query}':\n")
            for r in results:
                source_tag = f"[{r['source']}]" if r.get('source') else ""
                click.echo(f"  {r['kind']:<12} {r['name']:<30} {r['file']}  {source_tag}")
                if r.get("snippet"):
                    snippet = r["snippet"].replace("\n", " ")[:100]
                    click.echo(f"               {snippet}")
                click.echo()

    db.close()


@main.command()
@click.argument("file_path")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
def symbols(file_path: str, db_path: str | None):
    """List all symbols in a file."""
    from .db import Database

    root = _find_repo_root(Path.cwd())
    db_file = Path(db_path) if db_path else _get_db_path(root)

    if not db_file.exists():
        click.echo(f"No index found at {db_file}. Run 'srclight index' first.", err=True)
        sys.exit(1)

    db = Database(db_file)
    db.open()

    syms = db.symbols_in_file(file_path)
    if not syms:
        click.echo(f"No symbols found in '{file_path}'")
    else:
        click.echo(f"Symbols in {file_path}:\n")
        for s in syms:
            sig = s.signature or s.name or "(anonymous)"
            doc = f"  -- {s.doc_comment[:60]}" if s.doc_comment else ""
            click.echo(f"  L{s.start_line:<5} {s.kind:<12} {sig}{doc}")

    db.close()


@main.command()
@click.option("--db", "db_path", type=click.Path(), help="Database path")
def status(db_path: str | None):
    """Show index status and statistics."""
    from .db import Database

    root = _find_repo_root(Path.cwd())
    db_file = Path(db_path) if db_path else _get_db_path(root)

    if not db_file.exists():
        click.echo(f"No index found at {db_file}. Run 'srclight index' first.", err=True)
        sys.exit(1)

    db = Database(db_file)
    db.open()

    stats = db.stats()
    state = db.get_index_state(str(root))

    click.echo(f"Srclight Index Status")
    click.echo(f"  Database:    {db_file}")
    click.echo(f"  Repo root:   {root}")
    click.echo(f"  DB size:     {stats['db_size_mb']} MB")
    click.echo()
    click.echo(f"  Files:       {stats['files']}")
    click.echo(f"  Symbols:     {stats['symbols']}")
    click.echo(f"  Edges:       {stats['edges']}")

    if stats["languages"]:
        click.echo(f"\n  Languages:")
        for lang, count in stats["languages"].items():
            click.echo(f"    {lang:<15} {count} files")

    if stats["symbol_kinds"]:
        click.echo(f"\n  Symbol kinds:")
        for kind, count in stats["symbol_kinds"].items():
            click.echo(f"    {kind:<15} {count}")

    if state:
        click.echo(f"\n  Last commit: {state.get('last_commit', 'unknown')}")
        click.echo(f"  Indexed at:  {state.get('indexed_at', 'unknown')}")

    db.close()


# uvicorn's Server.shutdown() drains open connections BEFORE running the ASGI
# lifespan that closes the MCP session manager, so an open `GET /mcp` or `/sse`
# stream is a circular wait. Left unset, timeout_graceful_shutdown is None and
# asyncio.wait_for() waits forever: 9 of srclight.service's last 18 stops hung
# for the full 90s TimeoutStopSec and ended in SIGKILL (2026-09-02).
GRACEFUL_SHUTDOWN_SECONDS = 10


def _uvicorn_config(app, port: int, log_level: str):
    """Build the uvicorn config for `serve --web`, with a bounded shutdown."""
    import uvicorn

    return uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )


@main.command()
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--workspace", "-w", "workspace_name", help="Workspace name (multi-repo mode)")
@click.option("--transport", "-t", type=click.Choice(["stdio", "sse"]), default="sse", help="Transport (stdio or sse, default: sse)")
@click.option("--port", "-p", default=8742, help="Port for SSE transport (default: 8742)")
@click.option("--web", is_flag=True, help="Serve dashboard and REST API at / and /api/* (SSE only)")
def serve(db_path: str | None, workspace_name: str | None, transport: str, port: int, web: bool):
    """Start the MCP server."""
    from .server import configure, configure_workspace, run_server

    if workspace_name:
        configure_workspace(workspace_name)
    elif db_path:
        db_file = Path(db_path).resolve()
        root = _find_repo_root(db_file.parent)
        configure(db_path=db_file, repo_root=root)
    else:
        root = _find_repo_root(Path.cwd())
        db_file = _get_db_path(root)
        configure(db_path=db_file, repo_root=root)

    if transport == "sse" and web:
        import anyio
        import time
        from . import server as server_mod
        if server_mod._server_start_time is None:
            server_mod._server_start_time = time.time()
        from .server import make_sse_and_streamable_http_app
        from .web import add_web_routes
        app = make_sse_and_streamable_http_app(mount_path="/")
        add_web_routes(app)
        import uvicorn
        log_level = getattr(server_mod.mcp.settings, "log_level", "info")
        if isinstance(log_level, str):
            log_level = log_level.lower()
        config = _uvicorn_config(app, port=port, log_level=log_level)

        async def _run():
            srv = uvicorn.Server(config)
            await srv.serve()

        anyio.run(_run)
        return

    if web and transport != "sse":
        click.echo("--web requires --transport sse; ignoring --web.", err=True)

    run_server(transport=transport, port=port)


@main.command(
    "tool",
    add_help_option=False,          # so `tool <name> --help` reaches the tool
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("tool_name", required=False)
@click.option("--list", "list_tools_flag", is_flag=True, help="List every available tool")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--workspace", "-w", "workspace_name", help="Workspace name (multi-repo mode)")
@click.pass_context
def tool(ctx: click.Context, tool_name: str | None, list_tools_flag: bool,
         db_path: str | None, workspace_name: str | None):
    """Run any MCP tool from the shell.

    The tools, their arguments and their help text come from the running
    server's own registry, so this command tracks the MCP surface exactly —
    including across upgrades, which means a tool renamed there is renamed
    here too.

    Output is the tool's JSON on stdout and nothing else. Exit codes: 0 on
    success, 1 when the tool reports an error, 2 on a usage error — and a
    usage error, having no tool result to report, leaves stdout empty and
    says why on stderr.
    """
    import asyncio

    from .server import configure, configure_workspace, mcp
    from .tool_dispatch import (
        ToolArgumentError,
        coerce_arguments,
        format_tool_help,
        parse_cli_pairs,
    )

    if tool_name in ("--help", "-h"):
        click.echo(ctx.get_help())
        ctx.exit(0)

    tools = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in tools}

    if list_tools_flag or not tool_name:
        for name in sorted(by_name):
            summary = (by_name[name].description or "").strip().split("\n")[0]
            click.echo(f"{name}  {summary}")
        return

    spec = by_name.get(tool_name)
    if spec is None:
        close = [n for n in sorted(by_name) if n.startswith(tool_name[:4])]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        click.echo(f"Error: unknown tool '{tool_name}'.{hint} "
                   f"Run 'srclight tool --list' to see all tools.", err=True)
        sys.exit(2)

    if tool_name == "restart_server":
        click.echo(
            "Error: 'restart_server' only makes sense against a long-lived SSE "
            "server process; a one-shot CLI invocation has nothing left to "
            "restart into. Run it over MCP against a running 'srclight serve' "
            "instead.", err=True,
        )
        sys.exit(2)

    if "--help" in ctx.args or "-h" in ctx.args:
        click.echo(format_tool_help(spec.name, spec.description or "", spec.input_schema))
        return

    try:
        properties = spec.input_schema.get("properties", {})
        arguments = coerce_arguments(
            spec.input_schema, parse_cli_pairs(list(ctx.args), properties)
        )
    except ToolArgumentError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if workspace_name:
        configure_workspace(workspace_name)
    elif db_path:
        db_file = Path(db_path).resolve()
        configure(db_path=db_file, repo_root=_find_repo_root(db_file.parent))
    else:
        root = _find_repo_root(Path.cwd())
        configure(db_path=_get_db_path(root), repo_root=root)

    try:
        result = asyncio.run(mcp.call_tool(spec.name, arguments))
    except Exception as e:
        # mcp.call_tool() raises rather than returning an isError result, so
        # this is the only path most tool failures take (missing index, bad
        # --db, unknown workspace, ...). Keep stdout as JSON here too, so a
        # caller that reached the tool at all still parses one shape. A usage
        # error is the other case and does not: it exits 2 with stdout empty
        # and the reason on stderr, because there is no tool result to speak of.
        click.echo(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        click.echo(f"Error: tool '{spec.name}' failed: {e}", err=True)
        sys.exit(1)

    text = "\n".join(
        block.text for block in result.content if getattr(block, "text", None) is not None
    )
    click.echo(text)

    if getattr(result, "isError", False):
        sys.exit(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "error" in parsed:
        sys.exit(1)


# --- Workspace commands ---


@main.group()
def workspace():
    """Manage multi-repo workspaces."""
    pass


@workspace.command("init")
@click.argument("name")
def workspace_init(name: str):
    """Create a new workspace."""
    from .workspace import WorkspaceConfig

    config = WorkspaceConfig(name=name)
    config.save()
    click.echo(f"Created workspace '{name}' at {config.config_path}")


@workspace.command("add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Project name (default: directory name)")
@click.option("--workspace", "-w", "ws_name", required=True, help="Workspace to add to")
def workspace_add(path: str, name: str | None, ws_name: str):
    """Add a project to a workspace."""
    from .workspace import WorkspaceConfig

    config = WorkspaceConfig.load(ws_name)
    resolved = Path(path).resolve()
    project_name = name or resolved.name
    config.add_project(project_name, str(resolved))
    click.echo(f"Added '{project_name}' ({resolved}) to workspace '{ws_name}'")


@workspace.command("remove")
@click.argument("project_name")
@click.option("--workspace", "-w", "ws_name", required=True, help="Workspace to remove from")
def workspace_remove(project_name: str, ws_name: str):
    """Remove a project from a workspace."""
    from .workspace import WorkspaceConfig

    config = WorkspaceConfig.load(ws_name)
    config.remove_project(project_name)
    click.echo(f"Removed '{project_name}' from workspace '{ws_name}'")


@workspace.command("index")
@click.option("--workspace", "-w", "ws_name", required=True, help="Workspace to index")
@click.option("--project", "-p", help="Index only this project (default: all)")
@click.option("--embed", "embed_model", type=str, default=None,
              help="Embedding model (e.g., qwen3-embedding, voyage-code-3). Passed once: "
                   "later runs reuse the model recorded in each index, else "
                   "$SRCLIGHT_EMBED_MODEL.")
@click.option("--no-embed", is_flag=True, default=False,
              help="Index without embeddings for this run. Changed files still lose "
                   "the embeddings of the symbols they replace.")
@click.option("--forget-embed-model", is_flag=True, default=False,
              help="Stop embedding every index in the workspace for good: later runs, "
                   "git hooks included, leave embeddings alone until --embed is passed again.")
def workspace_index(ws_name: str, project: str | None, embed_model: str | None,
                    no_embed: bool, forget_embed_model: bool):
    """Index all (or one) project in a workspace."""
    from .db import Database
    from .indexer import EMBED_MODEL_ENV, IndexConfig, Indexer, resolve_embed_model
    from .workspace import WorkspaceConfig

    config = WorkspaceConfig.load(ws_name)
    entries = config.get_entries()

    if project:
        entries = [e for e in entries if e.name == project]
        if not entries:
            click.echo(f"Project '{project}' not found in workspace '{ws_name}'", err=True)
            sys.exit(1)

    if no_embed or forget_embed_model:
        # Forgetting disables embeddings for the run too, so announcing
        # --embed here would name a model that goes nowhere.
        why = "--no-embed" if no_embed else "--forget-embed-model"
        if embed_model:
            click.echo(f"Embeddings: skipped ({why}); --embed {embed_model} ignored")
        else:
            click.echo(f"Embeddings: skipped ({why})")
    elif embed_model:
        click.echo(f"Embedding model: {embed_model}")

    for entry in entries:
        root = Path(entry.path)
        if not root.exists():
            click.echo(f"  SKIP {entry.name}: {root} not found")
            continue

        db_file = root / ".srclight" / "index.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)

        click.echo(f"\n  Indexing {entry.name} ({root})")

        db = Database(db_file)
        try:
            db.open()
            db.initialize()

            if forget_embed_model:
                db.forget_embedding_model()
                db.commit()
                click.echo("    Embedding model: forgotten — later runs will not embed")

            indexer_config = IndexConfig(
                root=root, embed_model=embed_model,
                disable_embeddings=no_embed or forget_embed_model,
            )
            resolved_model = resolve_embed_model(db, indexer_config)
            indexer_config.embed_model = resolved_model  # pin it, see index()
            if resolved_model and not embed_model:
                origin = ("from the existing index"
                          if resolved_model == db.detect_embedding_model()
                          else f"from ${EMBED_MODEL_ENV}")
                click.echo(f"    Embedding model: {resolved_model} ({origin})")
            indexer = Indexer(db, indexer_config)

            line_open = [False]  # a progress line waits for its newline

            def on_progress(file: str, current: int, total: int):
                pct = (current / total * 100) if total > 0 else 0
                click.echo(f"\r    [{current}/{total}] {pct:5.1f}% {file[:55]:<55}", nl=False)
                line_open[0] = True

            def on_phase(name: str):
                if line_open[0]:
                    click.echo()
                    line_open[0] = False
                click.echo(f"    {name}...")

            stats = indexer.index(root, on_progress=on_progress, on_phase=on_phase)
            if line_open[0]:
                click.echo()

            click.echo(f"    {stats.files_scanned} files, {stats.symbols_extracted} symbols, "
                        f"{stats.files_unchanged} unchanged, {stats.elapsed_seconds:.1f}s")

            db_stats = db.stats()
            click.echo(f"    DB: {db_stats['db_size_mb']} MB")
        except Exception as e:
            click.echo(f"\n    ERROR: {e}", err=True)
        finally:
            db.close()


@workspace.command("status")
@click.option("--workspace", "-w", "ws_name", required=True, help="Workspace to check")
def workspace_status(ws_name: str):
    """Show workspace status and statistics."""
    from .workspace import WorkspaceConfig, WorkspaceDB

    config = WorkspaceConfig.load(ws_name)
    click.echo(f"Workspace: {ws_name}")
    click.echo(f"Config:    {config.config_path}")
    click.echo()

    with WorkspaceDB(config) as wdb:
        projects = wdb.list_projects()
        for p in projects:
            if p.get("indexed") is False:
                click.echo(f"  {p['project']:<20} (not indexed)")
                continue
            if "error" in p:
                click.echo(f"  {p['project']:<20} ERROR: {p['error']}")
                continue
            langs = ", ".join(f"{l}:{n}" for l, n in p.get("languages", {}).items())
            click.echo(f"  {p['project']:<20} {p['files']:>5} files  {p['symbols']:>6} symbols  "
                        f"{p['db_size_mb']:>5.1f} MB  [{langs}]")

        stats = wdb.codebase_map()
        click.echo(f"\n  Total: {stats['totals']['files']} files, "
                    f"{stats['totals']['symbols']} symbols, "
                    f"{stats['totals']['edges']} edges across "
                    f"{stats['projects_attached']} projects")


@workspace.command("list")
def workspace_list():
    """List all workspaces."""
    from .workspace import WorkspaceConfig

    names = WorkspaceConfig.list_all()
    if not names:
        click.echo("No workspaces configured. Run 'srclight workspace init NAME'.")
    else:
        for name in names:
            config = WorkspaceConfig.load(name)
            click.echo(f"  {name:<20} {len(config.projects)} projects")


@workspace.command("search")
@click.argument("query")
@click.option("--workspace", "-w", "ws_name", required=True, help="Workspace to search")
@click.option("--kind", "-k", help="Filter by symbol kind")
@click.option("--project", "-p", help="Filter by project name")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def workspace_search(query: str, ws_name: str, kind: str | None, project: str | None,
                     limit: int, json_output: bool):
    """Search across all projects in a workspace."""
    from .workspace import WorkspaceConfig, WorkspaceDB

    config = WorkspaceConfig.load(ws_name)
    with WorkspaceDB(config) as wdb:
        results = wdb.search_symbols(query, kind=kind, project=project, limit=limit)

        if json_output:
            click.echo(json.dumps(results, indent=2))
        else:
            if not results:
                click.echo(f"No results for '{query}'")
            else:
                click.echo(f"Found {len(results)} results for '{query}':\n")
                for r in results:
                    proj = f"[{r.get('project', '?')}]"
                    click.echo(f"  {proj:<20} {r['kind']:<12} {r['name']:<30} {r['file']}")
                    if r.get("snippet"):
                        snippet = r["snippet"].replace("\n", " ")[:100]
                        click.echo(f"  {'':20} {snippet}")
                    click.echo()


# --- Hook commands ---


@main.group()
def hook():
    """Manage git hooks for auto-reindexing."""
    pass


_HOOK_MARKER_START = "# --- srclight auto-reindex start ---"
_HOOK_MARKER_END = "# --- srclight auto-reindex end ---"

# Legacy markers (pre-rename) — detected for uninstall/status but never written
_LEGACY_MARKER_START = "# --- codelight auto-reindex start ---"
_LEGACY_MARKER_END = "# --- codelight auto-reindex end ---"

# Hooks we install into
_HOOK_NAMES = ["post-commit", "post-checkout"]


def _srclight_bin() -> str:
    """Find the srclight binary path."""
    import shutil
    # Prefer the bin next to sys.executable (same venv)
    venv_bin = Path(sys.executable).parent / "srclight"
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which("srclight")
    if found:
        return found
    # Fallback: invoke via python -m
    return f"{sys.executable} -m srclight.cli"


def _missing_bin_trace(srclight_path: str, indent: str) -> str:
    """Shell lines that record a missing binary, for the hook snippets.

    The log directory is created first (a fresh clone or a worktree has no
    .srclight/), and the same line goes to stderr, so a person committing sees
    it even when the log cannot be written.
    """
    msg = (f"srclight hook: {srclight_path} not executable; "
           "auto-reindex skipped (run: srclight hook install)")
    return (
        f'{indent}mkdir -p "$_srclight_top/.srclight" 2>/dev/null\n'
        f'{indent}echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) {msg}" 2>/dev/null >> "$_srclight_top/.srclight/reindex.log"\n'
        f'{indent}echo "{msg}" >&2'
    )


def _post_commit_snippet(srclight_path: str) -> str:
    """Hook snippet for post-commit: reindex after every commit.

    One conditional with no `exit`, so hook lines after the block still run,
    closed by `:` so the block leaves the hook's status at 0.
    """
    trace = _missing_bin_trace(srclight_path, " " * 12)
    return f"""{_HOOK_MARKER_START}
# Auto-reindex after commit (installed by srclight hook install)
# Git for Windows also runs these hooks (e.g. WSL clones under /mnt/c), but
# the srclight binary is a Linux path, so do nothing there.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *)
        _srclight_top="$(git rev-parse --show-toplevel 2>/dev/null)"
        if [ -x "{srclight_path}" ]; then
            (
                cd "$_srclight_top" && \\
                mkdir -p .srclight && \\
                flock -n .srclight/reindex.lock \\
                    "{srclight_path}" index . \\
                    >> .srclight/reindex.log 2>&1
            ) &
            disown 2>/dev/null
        elif [ -n "$_srclight_top" ]; then
{trace}
        fi
        ;;
esac
:
{_HOOK_MARKER_END}"""


def _post_checkout_snippet(srclight_path: str) -> str:
    """Hook snippet for post-checkout: reindex on branch switch.

    post-checkout receives: $1=prev_HEAD $2=new_HEAD $3=is_branch_checkout
    Only acts when $3=1 (branch checkout, not file checkout) and HEAD changed.
    git takes post-checkout's exit status as its own, so the block must end at 0.
    """
    trace = _missing_bin_trace(srclight_path, " " * 16)
    return f"""{_HOOK_MARKER_START}
# Auto-reindex on branch switch (installed by srclight hook install)
# $1=prev_HEAD $2=new_HEAD $3=1 if branch checkout
# Git for Windows also runs these hooks (e.g. WSL clones under /mnt/c), but
# the srclight binary is a Linux path, so do nothing there.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *)
        if [ "$3" = "1" ] && [ "$1" != "$2" ]; then
            _srclight_top="$(git rev-parse --show-toplevel 2>/dev/null)"
            if [ -x "{srclight_path}" ]; then
                (
                    cd "$_srclight_top" && \\
                    mkdir -p .srclight && \\
                    flock -n .srclight/reindex.lock \\
                        "{srclight_path}" index . \\
                        >> .srclight/reindex.log 2>&1
                ) &
                disown 2>/dev/null
            elif [ -n "$_srclight_top" ]; then
{trace}
            fi
        fi
        ;;
esac
:
{_HOOK_MARKER_END}"""


_HOOK_SNIPPETS = {
    "post-commit": _post_commit_snippet,
    "post-checkout": _post_checkout_snippet,
}


def _hook_blocks(text: str) -> list:
    """Every complete srclight block (START..END) in a hook file, as re.Match objects."""
    import re as _re
    pattern = _re.compile(
        _re.escape(_HOOK_MARKER_START) + r".*?" + _re.escape(_HOOK_MARKER_END),
        _re.DOTALL,
    )
    return list(pattern.finditer(text))


def _hook_target(block: str) -> str | None:
    """The binary a srclight block guards on with `[ -x "..." ]`, read from the block only."""
    import re as _re
    m = _re.search(r'\[ -x "([^"]+)" \]', block)
    return m.group(1) if m else None


def _is_executable(path: str | None) -> bool:
    import os
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _git(repo_path: Path, *args: str):
    """Run git in repo_path; None when git cannot be run at all."""
    import subprocess
    try:
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_path(repo_path: Path, name: str) -> Path:
    """`git rev-parse --git-path name`, falling back to .git/<name> outside a real repo."""
    out = _git(repo_path, "rev-parse", "--git-path", name)
    if out is None or out.returncode != 0 or not out.stdout.strip():
        return repo_path / ".git" / name
    p = Path(out.stdout.strip())
    return p if p.is_absolute() else repo_path / p


def _hooks_dir(repo_path: Path) -> Path:
    """The directory git actually runs hooks from, honouring core.hooksPath."""
    return _git_path(repo_path, "hooks")


def _hooks_disabled(repo_path: Path) -> bool:
    """True when the repo opted out with `git config srclight.hooks false`."""
    out = _git(repo_path, "config", "--bool", "--get", "srclight.hooks")
    return out is not None and out.returncode == 0 and out.stdout.strip() == "false"


def _committable_hooks_dir(repo_path: Path, hooks_dir: Path) -> str | None:
    """Why srclight must not write into hooks_dir, or None when it may.

    The hooks name an absolute local binary path. Written into a directory the
    repository tracks (core.hooksPath=.husky, say) they would be committed.
    """
    try:
        rel = hooks_dir.resolve().relative_to(repo_path.resolve())
    except ValueError:
        return None  # outside the work tree
    if rel.parts and rel.parts[0] == ".git":
        return None
    out = _git(repo_path, "check-ignore", "-q", (rel / "post-commit").as_posix())
    if out is not None and out.returncode == 0:
        return None
    return (f"core.hooksPath {hooks_dir} is inside the work tree and not ignored; "
            "srclight's hooks name a local binary path and would be committed")


def _write_hook_file(hook_file: Path, make_snippet, srclight_path: str,
                     force: bool = False) -> bool:
    """Install or update srclight's block in a hook file. True if the file changed.

    - A block whose binary no longer runs is rewritten for srclight_path.
    - A block whose binary runs keeps that binary, but its text is upgraded to
      the current snippet (the OUTDATED case), so snippet changes reach working
      hooks without --force and installs from different environments cannot
      re-point each other's hooks. --force rewrites for srclight_path.
    - A hook file that lost its execute bit gets it back: git skips such hooks.
    """
    import os
    if hook_file.exists():
        existing = hook_file.read_text()
        if _HOOK_MARKER_START in existing:
            blocks = _hook_blocks(existing)
            if len(blocks) != 1:
                # No end marker, or duplicates: hook status reports these as
                # BROKEN; guessing at a repair could eat the user's own lines.
                return False
            block = blocks[0]
            current = _hook_target(block.group(0))
            if force or not _is_executable(current):
                wanted = make_snippet(srclight_path)
            else:
                wanted = make_snippet(current)
            changed = False
            if block.group(0) != wanted:
                hook_file.write_text(existing[:block.start()] + wanted + existing[block.end():])
                changed = True
            if not os.access(hook_file, os.X_OK):
                hook_file.chmod(hook_file.stat().st_mode | 0o111)
                changed = True
            return changed
        # Remove legacy (codelight) hook if present, then install new one
        if _LEGACY_MARKER_START in existing:
            import re as _re
            pattern = _re.compile(
                _re.escape(_LEGACY_MARKER_START) + r".*?" + _re.escape(_LEGACY_MARKER_END),
                _re.DOTALL,
            )
            existing = pattern.sub("", existing).strip()
            if existing == "#!/bin/sh" or not existing:
                existing = "#!/bin/sh"
        hook_file.write_text(existing.rstrip() + "\n\n" + make_snippet(srclight_path) + "\n")
    else:
        hook_file.write_text("#!/bin/sh\n\n" + make_snippet(srclight_path) + "\n")
    hook_file.chmod(0o755)
    return True


def _remove_hook_snippet(hook_file: Path) -> bool:
    """Remove srclight snippet from a hook file. Also removes legacy codelight markers.

    Returns True if any markers were removed.
    """
    if not hook_file.exists():
        return False
    content = hook_file.read_text()

    has_new = _HOOK_MARKER_START in content
    has_legacy = _LEGACY_MARKER_START in content
    if not has_new and not has_legacy:
        return False

    import re as _re

    cleaned = content
    if has_new:
        pattern = _re.compile(
            _re.escape(_HOOK_MARKER_START) + r".*?" + _re.escape(_HOOK_MARKER_END),
            _re.DOTALL,
        )
        cleaned = pattern.sub("", cleaned)
    if has_legacy:
        pattern = _re.compile(
            _re.escape(_LEGACY_MARKER_START) + r".*?" + _re.escape(_LEGACY_MARKER_END),
            _re.DOTALL,
        )
        cleaned = pattern.sub("", cleaned)

    cleaned = cleaned.strip()
    if cleaned == "#!/bin/sh" or not cleaned:
        hook_file.unlink()
    else:
        hook_file.write_text(cleaned + "\n")
    return True


def _ensure_srclight_ignored(repo_path: Path) -> None:
    """Keep .srclight/ out of git without editing a tracked file.

    Appends to .git/info/exclude, which is local and never committed, unless
    git already ignores .srclight/ (a committed .gitignore, say). Writing the
    tracked .gitignore dirtied third-party clones and rewrote CRLF files on
    every hook run.
    """
    # Probe a name no generic pattern matches: a repo that ignores `*.db` would
    # report .srclight/index.db ignored while embeddings.npy stays untracked.
    out = _git(repo_path, "check-ignore", "-q", ".srclight/.srclight-ignore-probe")
    if out is not None and out.returncode == 0:
        return
    exclude = _git_path(repo_path, "info/exclude")
    content = exclude.read_bytes() if exclude.exists() else b""
    if b".srclight/" in (line.strip() for line in content.splitlines()):
        return
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("ab") as f:
        if content and not content.endswith(b"\n"):
            f.write(b"\n")
        f.write(b".srclight/\n")


def _repo_hook_health(repo_path: Path) -> tuple[bool, str]:
    """(healthy, summary) for srclight's auto-reindex hooks in one repo.

    Unhealthy means auto-reindex does not run as installed: a hook missing,
    BROKEN, STALE (binary gone), NOT EXECUTABLE (git skips it), OUTDATED (an
    older snippet), or a core.hooksPath at a missing directory. hook status
    exits 1 on any of these and /healthz lists them, so a dead hook is no
    longer something only a reader of a log could find.
    """
    import os
    if not repo_path.exists():
        return False, "path not found"
    if not (repo_path / ".git").is_dir():
        return True, "not a git repo"
    if _hooks_disabled(repo_path):
        return True, "disabled (srclight.hooks=false)"
    hooks_dir = _hooks_dir(repo_path)
    if not hooks_dir.is_dir() and hooks_dir != repo_path / ".git" / "hooks":
        return False, f"STALE: core.hooksPath points at missing {hooks_dir}"
    statuses = []
    healthy = True
    for hook_name, make_snippet in _HOOK_SNIPPETS.items():
        hf = hooks_dir / hook_name
        if not hf.exists():
            statuses.append(f"{hook_name} (MISSING)")
            healthy = False
            continue
        text = hf.read_text()
        if _HOOK_MARKER_START not in text:
            label = "legacy codelight block" if _LEGACY_MARKER_START in text else "MISSING: no srclight block"
            statuses.append(f"{hook_name} ({label})")
            healthy = False
            continue
        blocks = _hook_blocks(text)
        target = _hook_target(blocks[0].group(0)) if len(blocks) == 1 else None
        if not blocks:
            problem = "BROKEN: no end marker"
        elif len(blocks) > 1:
            problem = f"BROKEN: {len(blocks)} srclight blocks"
        elif not _is_executable(target):
            problem = f"STALE: {target} not executable"
        elif not os.access(hf, os.X_OK):
            problem = "NOT EXECUTABLE: git skips this hook"
        elif blocks[0].group(0) != make_snippet(target):
            problem = "OUTDATED: run srclight hook install"
        else:
            statuses.append(hook_name)
            continue
        statuses.append(f"{hook_name} ({problem})")
        healthy = False
    return healthy, ", ".join(statuses)


def _install_hooks_in_repo(repo_path: Path, srclight_path: str, force: bool = False) -> str:
    """Install post-commit + post-checkout hooks. Returns status message."""
    git_dir = repo_path / ".git"
    if not git_dir.is_dir():
        return f"  SKIP {repo_path.name}: not a git repo"
    if _hooks_disabled(repo_path):
        return f"  SKIP {repo_path.name}: disabled (srclight.hooks=false)"

    hooks_dir = _hooks_dir(repo_path)
    if hooks_dir == git_dir / "hooks":
        hooks_dir.mkdir(exist_ok=True)
    elif not hooks_dir.is_dir():
        # A core.hooksPath left pointing at a moved directory: git runs no
        # hooks at all, so writing .git/hooks would only look like a repair.
        return f"  FAIL {repo_path.name}: core.hooksPath points at missing {hooks_dir}"
    else:
        problem = _committable_hooks_dir(repo_path, hooks_dir)
        if problem:
            return f"  FAIL {repo_path.name}: {problem}"

    installed = [
        hook_name for hook_name, make_snippet in _HOOK_SNIPPETS.items()
        if _write_hook_file(hooks_dir / hook_name, make_snippet, srclight_path, force=force)
    ]

    # Ensure .srclight dir exists for log file
    (repo_path / ".srclight").mkdir(exist_ok=True)

    # Indexes + embeddings should never be committed
    _ensure_srclight_ignored(repo_path)

    if not installed:
        return f"  SKIP {repo_path.name}: hooks already installed"
    return f"  OK   {repo_path.name}: {', '.join(installed)}"


def _uninstall_hooks_in_repo(repo_path: Path) -> str:
    """Remove srclight hooks from a repo. Returns status message."""
    hooks_dir = _hooks_dir(repo_path)
    removed = []
    for hook_name in _HOOK_NAMES:
        if _remove_hook_snippet(hooks_dir / hook_name):
            removed.append(hook_name)

    if not removed:
        return f"  SKIP {repo_path.name}: no srclight hooks found"
    return f"  OK   {repo_path.name}: removed {', '.join(removed)}"


@hook.command("install")
@click.option("--workspace", "-w", "ws_name", help="Install across all repos in a workspace")
@click.option("--force", is_flag=True,
              help="Also rewrite srclight blocks whose binary still runs")
def hook_install(ws_name: str | None, force: bool):
    """Install git hooks for auto-reindexing (post-commit + post-checkout).

    Installs two hooks:
    - post-commit: reindex after every commit
    - post-checkout: reindex when switching branches

    Both run in the background and only re-parse changed files (incremental).
    On an index that has a recorded embedding model they also refresh
    embeddings, which calls that model — `srclight index --forget-embed-model`
    turns that off for a repo.

    Without --workspace, installs in the current repo.
    With --workspace, installs across all repos in the workspace.
    """
    srclight_path = _srclight_bin()
    click.echo(f"Using srclight: {srclight_path}")
    if not _is_executable(srclight_path):
        # The hooks guard on `[ -x path ]`; a path that fails it (the
        # `python -m` fallback, a missing file) installs hooks that never run.
        click.echo(
            f"Error: {srclight_path} is not an executable file, so hooks using it "
            "would never run. Run this from the srclight venv's bin/srclight.",
            err=True,
        )
        sys.exit(1)

    results = []
    if ws_name:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(ws_name)
        for entry in config.get_entries():
            repo_path = Path(entry.path)
            if not repo_path.exists():
                results.append(f"  FAIL {entry.name}: path not found ({entry.path})")
            else:
                results.append(_install_hooks_in_repo(repo_path, srclight_path, force=force))
            click.echo(results[-1])
    else:
        root = _find_repo_root(Path.cwd())
        results.append(_install_hooks_in_repo(root, srclight_path, force=force))
        click.echo(results[-1])
    failed = sum(1 for r in results if r.lstrip().startswith("FAIL"))
    if failed:
        click.echo(f"Error: hooks could not be installed in {failed} repo(s).", err=True)
        sys.exit(1)


@hook.command("uninstall")
@click.option("--workspace", "-w", "ws_name", help="Uninstall from all repos in a workspace")
def hook_uninstall(ws_name: str | None):
    """Remove srclight git hooks.

    Without --workspace, removes from the current repo.
    With --workspace, removes from all repos in the workspace.
    """
    if ws_name:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(ws_name)
        for entry in config.get_entries():
            repo_path = Path(entry.path)
            if not repo_path.exists():
                click.echo(f"  SKIP {entry.name}: path not found")
                continue
            click.echo(_uninstall_hooks_in_repo(repo_path))
    else:
        root = _find_repo_root(Path.cwd())
        click.echo(_uninstall_hooks_in_repo(root))


@hook.command("install-agent")
@click.option("--port", "-p", default=8742, help="Srclight server port (default: 8742)")
@click.option("--settings-path", type=click.Path(),
              default="~/.claude/settings.json",
              help="Path to Claude Code settings.json")
def hook_install_agent(port: int, settings_path: str):
    """Install Claude Code hooks to redirect Grep/Glob to srclight.

    Writes a hook script to ~/.srclight/hooks/agent-redirect.sh and
    registers it as a PreToolUse hook in Claude Code settings.json.

    When srclight is running, Grep/Glob calls are denied with a message
    suggesting srclight tools instead. When srclight is not running,
    calls are allowed (graceful fallback).
    """
    hook_dir = Path.home() / ".srclight" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook_script = hook_dir / "agent-redirect.sh"

    # Write the hook script
    script_content = f"""#!/usr/bin/env bash
# Srclight agent redirect hook for Claude Code.
# Denies Grep/Glob when srclight is running, allows otherwise.
# Installed by: srclight hook install-agent

if curl -sf --max-time 1 http://127.0.0.1:{port}/sse >/dev/null 2>&1; then
    cat <<'DENY_EOF'
{{"decision": "deny", "reason": "srclight is running. Use hybrid_search(query) for search, symbols_in_file(path) for file contents, or search_symbols(query) for exact names. These give structured, ranked results with relationship data instead of raw text."}}
DENY_EOF
else
    cat <<'ALLOW_EOF'
{{"decision": "approve"}}
ALLOW_EOF
fi
"""
    hook_script.write_text(script_content)
    hook_script.chmod(0o755)
    click.echo(f"Wrote hook script: {hook_script}")

    # Update Claude Code settings.json
    settings_file = Path(settings_path).expanduser()
    settings_file.parent.mkdir(parents=True, exist_ok=True)

    settings: dict = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            click.echo(f"Error reading {settings_file}: {e}", err=True)
            sys.exit(1)

    hook_entry = {
        "matcher": "Grep|Glob",
        "hooks": [
            {
                "type": "command",
                "command": str(hook_script),
            }
        ],
    }

    # Ensure hooks.PreToolUse exists
    hooks = settings.setdefault("hooks", {})
    pre_tool_use = hooks.setdefault("PreToolUse", [])

    # Idempotent: replace existing srclight hook or add new one
    replaced = False
    for i, entry in enumerate(pre_tool_use):
        hook_cmds = entry.get("hooks", [])
        if any("agent-redirect.sh" in h.get("command", "") for h in hook_cmds):
            pre_tool_use[i] = hook_entry
            replaced = True
            break

    if not replaced:
        pre_tool_use.append(hook_entry)

    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    click.echo(f"Updated {settings_file}")
    click.echo("Claude Code will now prefer srclight tools over Grep/Glob when srclight is running.")


@hook.command("uninstall-agent")
@click.option("--settings-path", type=click.Path(),
              default="~/.claude/settings.json",
              help="Path to Claude Code settings.json")
def hook_uninstall_agent(settings_path: str):
    """Remove Claude Code agent redirect hooks.

    Removes the PreToolUse hook from Claude Code settings.json and
    deletes the hook script from ~/.srclight/hooks/.
    """
    # Remove from settings.json
    settings_file = Path(settings_path).expanduser()
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            click.echo(f"Error reading {settings_file}: {e}", err=True)
            sys.exit(1)

        pre_tool_use = settings.get("hooks", {}).get("PreToolUse", [])
        original_len = len(pre_tool_use)
        pre_tool_use[:] = [
            entry for entry in pre_tool_use
            if not any("agent-redirect.sh" in h.get("command", "")
                       for h in entry.get("hooks", []))
        ]

        if len(pre_tool_use) < original_len:
            # Clean up empty structures
            if not pre_tool_use:
                settings.get("hooks", {}).pop("PreToolUse", None)
            if not settings.get("hooks"):
                settings.pop("hooks", None)
            settings_file.write_text(json.dumps(settings, indent=2) + "\n")
            click.echo(f"Removed hook from {settings_file}")
        else:
            click.echo(f"No srclight hook found in {settings_file}")
    else:
        click.echo(f"Settings file not found: {settings_file}")

    # Remove hook script
    hook_script = Path.home() / ".srclight" / "hooks" / "agent-redirect.sh"
    if hook_script.exists():
        hook_script.unlink()
        click.echo(f"Removed {hook_script}")
    else:
        click.echo(f"Hook script not found: {hook_script}")


@hook.command("status")
@click.option("--workspace", "-w", "ws_name", help="Check all repos in a workspace")
def hook_status(ws_name: str | None):
    """Check that auto-reindex hooks are installed and will run.

    Exits 1 when any repo's hooks are missing, BROKEN, STALE, NOT EXECUTABLE
    or OUTDATED, so cron and scripts can act on it.
    """
    repos = []
    if ws_name:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(ws_name)
        for entry in config.get_entries():
            repos.append((entry.name, Path(entry.path)))
    else:
        root = _find_repo_root(Path.cwd())
        repos.append((root.name, root))

    unhealthy = 0
    for name, repo_path in repos:
        healthy, summary = _repo_hook_health(repo_path)
        click.echo(f"  {name:<20} {summary}")
        if not healthy:
            unhealthy += 1
    if unhealthy:
        click.echo(
            f"{unhealthy} repo(s) will not auto-reindex as installed. "
            "Run `srclight hook install` there (or with --workspace).",
            err=True,
        )
        sys.exit(1)


# --- Config commands ---


@main.group()
def config():
    """Generate IDE MCP configuration snippets."""
    pass


@config.command("claude-code")
@click.option("--port", "-p", default=8742, help="Srclight server port (default: 8742)")
@click.option("--workspace", "-w", "workspace_name", help="Workspace name (for display only)")
def config_claude_code(port: int, workspace_name: str | None):
    """Print MCP config for Claude Code (~/.claude/settings.json under mcpServers)."""
    snippet = {
        "srclight": {
            "type": "sse",
            "url": f"http://127.0.0.1:{port}/sse",
        }
    }
    click.echo("Add this to ~/.claude/settings.json under \"mcpServers\":\n")
    click.echo(json.dumps(snippet, indent=2))


@config.command("cursor")
@click.option("--port", "-p", default=8742, help="Srclight server port (default: 8742)")
@click.option("--workspace", "-w", "workspace_name", help="Workspace name (for display only)")
def config_cursor(port: int, workspace_name: str | None):
    """Print MCP config for Cursor (.cursor/mcp.json)."""
    snippet = {
        "mcpServers": {
            "srclight": {
                "url": f"http://127.0.0.1:{port}/sse",
            }
        }
    }
    click.echo("Add this to .cursor/mcp.json:\n")
    click.echo(json.dumps(snippet, indent=2))


@config.command("vscode")
@click.option("--port", "-p", default=8742, help="Srclight server port (default: 8742)")
@click.option("--workspace", "-w", "workspace_name", help="Workspace name (for display only)")
def config_vscode(port: int, workspace_name: str | None):
    """Print MCP config for VS Code (.vscode/settings.json)."""
    snippet = {
        "mcp": {
            "servers": {
                "srclight": {
                    "type": "sse",
                    "url": f"http://127.0.0.1:{port}/sse",
                }
            }
        }
    }
    click.echo("Add this to .vscode/settings.json:\n")
    click.echo(json.dumps(snippet, indent=2))


if __name__ == "__main__":
    main()
