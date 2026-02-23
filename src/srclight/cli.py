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


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--db", "db_path", type=click.Path(), help="Database path (default: .srclight/index.db)")
@click.option("--embed", "embed_model", type=str, default=None,
              help="Embedding model (e.g., qwen3-embedding, voyage-code-3)")
def index(path: str, db_path: str | None, embed_model: str | None):
    """Index a codebase for AI-powered search."""
    from .db import Database
    from .indexer import IndexConfig, Indexer

    root = Path(path).resolve()
    if not root.is_dir():
        click.echo(f"Error: {root} is not a directory", err=True)
        sys.exit(1)

    db_file = Path(db_path) if db_path else _get_db_path(root)

    # Ensure .srclight/ directory exists
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .srclight/ is in .gitignore
    if (root / ".git").is_dir():
        _ensure_gitignore(root)

    click.echo(f"Indexing {root}")
    click.echo(f"Database: {db_file}")
    if embed_model:
        click.echo(f"Embedding model: {embed_model}")

    db = Database(db_file)
    db.open()
    db.initialize()

    config = IndexConfig(root=root, embed_model=embed_model)
    indexer = Indexer(db, config)

    def on_progress(file: str, current: int, total: int):
        pct = (current / total * 100) if total > 0 else 0
        click.echo(f"\r  [{current}/{total}] {pct:5.1f}% {file[:60]:<60}", nl=False)

    stats = indexer.index(root, on_progress=on_progress)
    click.echo()  # newline after progress

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

    if embed_model:
        emb_stats = db.embedding_stats()
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


@main.command()
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--workspace", "-w", "workspace_name", help="Workspace name (multi-repo mode)")
@click.option("--transport", "-t", type=click.Choice(["stdio", "sse"]), default="sse", help="Transport (stdio or sse, default: sse)")
@click.option("--port", "-p", default=8742, help="Port for SSE transport (default: 8742)")
def serve(db_path: str | None, workspace_name: str | None, transport: str, port: int):
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

    run_server(transport=transport, port=port)


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
              help="Embedding model (e.g., qwen3-embedding, voyage-code-3)")
def workspace_index(ws_name: str, project: str | None, embed_model: str | None):
    """Index all (or one) project in a workspace."""
    from .db import Database
    from .indexer import IndexConfig, Indexer
    from .workspace import WorkspaceConfig

    config = WorkspaceConfig.load(ws_name)
    entries = config.get_entries()

    if project:
        entries = [e for e in entries if e.name == project]
        if not entries:
            click.echo(f"Project '{project}' not found in workspace '{ws_name}'", err=True)
            sys.exit(1)

    if embed_model:
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

            indexer_config = IndexConfig(root=root, embed_model=embed_model)
            indexer = Indexer(db, indexer_config)

            def on_progress(file: str, current: int, total: int):
                pct = (current / total * 100) if total > 0 else 0
                click.echo(f"\r    [{current}/{total}] {pct:5.1f}% {file[:55]:<55}", nl=False)

            stats = indexer.index(root, on_progress=on_progress)
            click.echo()  # newline after progress

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


def _post_commit_snippet(srclight_path: str) -> str:
    """Hook snippet for post-commit: reindex after every commit."""
    return f"""{_HOOK_MARKER_START}
# Auto-reindex after commit (installed by srclight hook install)
if [ -x "{srclight_path}" ]; then
    (
        cd "$(git rev-parse --show-toplevel)" && \\
        mkdir -p .srclight && \\
        flock -n .srclight/reindex.lock \\
            "{srclight_path}" index . \\
            >> .srclight/reindex.log 2>&1
    ) &
    disown 2>/dev/null
fi
exit 0
{_HOOK_MARKER_END}"""


def _post_checkout_snippet(srclight_path: str) -> str:
    """Hook snippet for post-checkout: reindex on branch switch.

    post-checkout receives: $1=prev_HEAD $2=new_HEAD $3=is_branch_checkout
    Only reindex when $3=1 (branch checkout, not file checkout) and HEAD changed.
    """
    return f"""{_HOOK_MARKER_START}
# Auto-reindex on branch switch (installed by srclight hook install)
# $1=prev_HEAD $2=new_HEAD $3=1 if branch checkout
if [ "$3" = "1" ] && [ "$1" != "$2" ] && [ -x "{srclight_path}" ]; then
    (
        cd "$(git rev-parse --show-toplevel)" && \\
        mkdir -p .srclight && \\
        flock -n .srclight/reindex.lock \\
            "{srclight_path}" index . \\
            >> .srclight/reindex.log 2>&1
    ) &
    disown 2>/dev/null
fi
exit 0
{_HOOK_MARKER_END}"""


def _write_hook_file(hook_file: Path, snippet: str) -> bool:
    """Write snippet into a hook file. Returns True if newly installed, False if already present."""
    if hook_file.exists():
        existing = hook_file.read_text()
        if _HOOK_MARKER_START in existing:
            return False
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
        hook_file.write_text(existing.rstrip() + "\n\n" + snippet + "\n")
    else:
        hook_file.write_text("#!/bin/sh\n\n" + snippet + "\n")
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


def _ensure_gitignore(repo_path: Path) -> None:
    """Ensure .srclight/ is listed in the repo's .gitignore."""
    gitignore = repo_path / ".gitignore"
    pattern = ".srclight/"
    if gitignore.exists():
        content = gitignore.read_text()
        if pattern in content:
            return
        # Append with a newline guard
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"{pattern}\n"
        gitignore.write_text(content)
    else:
        gitignore.write_text(f"{pattern}\n")


def _install_hooks_in_repo(repo_path: Path, srclight_path: str) -> str:
    """Install post-commit + post-checkout hooks. Returns status message."""
    git_dir = repo_path / ".git"
    if not git_dir.is_dir():
        return f"  SKIP {repo_path.name}: not a git repo"

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    snippets = {
        "post-commit": _post_commit_snippet(srclight_path),
        "post-checkout": _post_checkout_snippet(srclight_path),
    }

    installed = []
    skipped = []
    for hook_name, snippet in snippets.items():
        if _write_hook_file(hooks_dir / hook_name, snippet):
            installed.append(hook_name)
        else:
            skipped.append(hook_name)

    # Ensure .srclight dir exists for log file
    (repo_path / ".srclight").mkdir(exist_ok=True)

    # Ensure .srclight/ is in .gitignore (indexes + embeddings should never be committed)
    _ensure_gitignore(repo_path)

    if not installed:
        return f"  SKIP {repo_path.name}: hooks already installed"
    return f"  OK   {repo_path.name}: {', '.join(installed)}"


def _uninstall_hooks_in_repo(repo_path: Path) -> str:
    """Remove srclight hooks from a repo. Returns status message."""
    hooks_dir = repo_path / ".git" / "hooks"
    removed = []
    for hook_name in _HOOK_NAMES:
        if _remove_hook_snippet(hooks_dir / hook_name):
            removed.append(hook_name)

    if not removed:
        return f"  SKIP {repo_path.name}: no srclight hooks found"
    return f"  OK   {repo_path.name}: removed {', '.join(removed)}"


@hook.command("install")
@click.option("--workspace", "-w", "ws_name", help="Install across all repos in a workspace")
def hook_install(ws_name: str | None):
    """Install git hooks for auto-reindexing (post-commit + post-checkout).

    Installs two hooks:
    - post-commit: reindex after every commit
    - post-checkout: reindex when switching branches

    Both run in the background and only re-parse changed files (incremental).

    Without --workspace, installs in the current repo.
    With --workspace, installs across all repos in the workspace.
    """
    srclight_path = _srclight_bin()
    click.echo(f"Using srclight: {srclight_path}")

    if ws_name:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(ws_name)
        for entry in config.get_entries():
            repo_path = Path(entry.path)
            if not repo_path.exists():
                click.echo(f"  SKIP {entry.name}: path not found ({entry.path})")
                continue
            click.echo(_install_hooks_in_repo(repo_path, srclight_path))
    else:
        root = _find_repo_root(Path.cwd())
        click.echo(_install_hooks_in_repo(root, srclight_path))


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


@hook.command("status")
@click.option("--workspace", "-w", "ws_name", help="Check all repos in a workspace")
def hook_status(ws_name: str | None):
    """Check if auto-reindex hooks are installed."""
    repos = []
    if ws_name:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(ws_name)
        for entry in config.get_entries():
            repos.append((entry.name, Path(entry.path)))
    else:
        root = _find_repo_root(Path.cwd())
        repos.append((root.name, root))

    for name, repo_path in repos:
        if not repo_path.exists():
            click.echo(f"  {name:<20} path not found")
            continue
        hooks_dir = repo_path / ".git" / "hooks"
        statuses = []
        for hook_name in _HOOK_NAMES:
            hf = hooks_dir / hook_name
            if hf.exists() and (_HOOK_MARKER_START in hf.read_text()
                                or _LEGACY_MARKER_START in hf.read_text()):
                statuses.append(hook_name)
        if statuses:
            click.echo(f"  {name:<20} {', '.join(statuses)}")
        else:
            click.echo(f"  {name:<20} no hooks")


if __name__ == "__main__":
    main()
