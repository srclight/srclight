"""Tree-sitter based code indexer.

Walks a directory, parses files, extracts symbols, populates the database.
Incremental: only re-indexes files whose content hash has changed.
"""

from __future__ import annotations

import bisect
import fnmatch
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tree_sitter import Language, Node, Parser, Query, QueryCursor

from . import __version__
from .db import Database, EdgeRecord, FileRecord, SymbolRecord, content_hash
from .extractors import (
    DOCUMENT_EXTENSIONS,
    detect_document_language,
    get_registry,
    unreadable_document_extensions,
)
from .languages import (
    LANGUAGES,
    SKIP_LANGUAGE,
    LanguageConfig,
    code_extensions,
    detect_language,
    detect_language_by_filename,
    get_language,
    normalize_extension,
)

logger = logging.getLogger("srclight.indexer")


# Default ignore patterns
DEFAULT_IGNORE = [
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.a",
    "*.lib",
    "*.exe",
    "*.bin",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.rar",
    "*.7z",
    "*.pdf",
    "*.sqlite",
    "*.db",
    "*.sqlite3",
    # Build artifacts
    "CMakeFiles",
    "__cmake_systeminformation",
    "*.cmake",
    "CMakeCache.txt",
    # C# / .NET artifacts
    "bin",
    "obj",
    "packages",
    ".vs",
    "*.Designer.cs",
    "*.g.cs",
    "*.g.i.cs",
    "*.AssemblyInfo.cs",
    # Vendored / third-party
    "vendor",
    "third_party",
    "third-party",
    "ext",
    "depends",
    # Srclight index
    ".srclight",
    ".codelight",
    # Obsidian
    ".obsidian",
    ".trash",
]

# Max file size to index (1 MB)
MAX_FILE_SIZE = 1_000_000

# Embedding model for indexes that have none recorded — see resolve_embed_model
EMBED_MODEL_ENV = "SRCLIGHT_EMBED_MODEL"


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    symbols_extracted: int = 0
    edges_created: int = 0
    errors: int = 0
    symbols_embedded: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IndexConfig:
    root: Path = field(default_factory=Path)
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    max_file_size: int = MAX_FILE_SIZE
    max_doc_file_size: int = 50_000_000  # 50 MB for documents (PDF, DOCX, etc.)
    languages: list[str] | None = None  # None = all supported
    # Extra extensions to read, as {extension: language}. None leaves the
    # index's stored declaration alone (the flag-less reindex the git hooks
    # run); an empty dict clears it.
    extension_overrides: dict[str, str] | None = None
    embed_model: str | None = None  # e.g. "qwen3-embedding", "voyage-code-3"
    disable_embeddings: bool = False  # --no-embed: index without touching embeddings


def resolve_embed_model(db: Database, config: IndexConfig) -> str | None:
    """Pick the embedding model for a run.

    Priority: the explicit model (--embed) > the model the index already
    holds > SRCLIGHT_EMBED_MODEL. The middle one is what keeps embeddings
    alive across the flag-less reindexes run by the git hooks and the MCP
    server — without it, every symbol added after the first `--embed` run
    stays unembedded.

    The environment variable comes LAST on purpose: it is a default for
    indexes that have no model yet, not an override. Ahead of the recorded
    model, exporting it once — the natural way to set a default — would make
    the next commit in an unrelated repo re-embed every symbol it holds,
    silently, from a detached background hook, against a metered API in the
    paid case.
    Switching an existing index stays an explicit `--embed`.

    `disable_embeddings` opts out of all three.
    """
    if config.disable_embeddings:
        return None

    explicit = (config.embed_model or "").strip()
    if explicit:
        return explicit

    recorded = db.detect_embedding_model()
    if recorded:
        return recorded

    # An index told to forget stays off, whatever the environment says. The
    # docs send users to export the variable and the hooks inherit it, so
    # letting it win here would leave the one person who most needs the off
    # switch — the one paying a metered provider on every commit — without one.
    if db.embedding_model_forgotten():
        return None

    return os.environ.get(EMBED_MODEL_ENV, "").strip() or None


# Suffixes that never hold code srclight could index — configuration, data
# and manifests. They are skipped like any unknown extension, but they are
# not a GAP: every repo carries some, so counting them would leave the tally
# non-empty everywhere, put the "not a whole-tree answer" warning on every
# result, and bury the extensions that genuinely hold unread code. Files with
# no suffix at all (LICENSE, Dockerfile, Makefile) are the same case.
INERT_EXTENSIONS = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".lock",
    ".txt", ".log", ".xml", ".properties", ".plist", ".editorconfig",
})


def _count_unindexed(counts: dict[str, int], path: Path) -> None:
    """Tally one file the walk is about to skip, by extension.

    Inert suffixes and suffixless files are not tallied — see
    INERT_EXTENSIONS for why an over-eager tally is worse than none.
    """
    ext = path.suffix.lower()
    if not ext or ext in INERT_EXTENSIONS:
        return
    counts[ext] = counts.get(ext, 0) + 1


def _should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern."""
    rel = str(path.relative_to(root))

    # Check each component of the relative path against directory patterns
    parts = path.relative_to(root).parts
    for part in parts:
        for pattern in patterns:
            if fnmatch.fnmatch(part, pattern):
                return True

    # Check full relative path
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return True

    return False


def _git_tracked_files(root: Path) -> set[str] | None:
    """Get the set of git-tracked files (respects .gitignore).

    Returns None if not a git repo or git is unavailable.
    Returns relative paths as strings.
    """
    try:
        # -z: paths come raw, not quoted and escaped as `"pi\303\250ce.py"`.
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            errors="surrogateescape", timeout=30,
        )
        if result.returncode != 0:
            return None
        files = {path for path in result.stdout.split("\0") if path}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    # A submodule is listed as its directory alone; the files it holds are
    # listed from inside it, when it is checked out. `--recurse-submodules`
    # cannot be combined with `--others`.
    for sub in _git_submodule_paths(root):
        if not (root / sub / ".git").exists():
            continue
        inner = _git_tracked_files(root / sub)
        if inner is not None:
            files.discard(sub)
            files.update(f"{sub}/{rel}" for rel in inner)
    return files


def _git_submodule_paths(root: Path) -> list[str]:
    """The paths of the submodules a repository records (gitlinks)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--stage"],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            errors="surrogateescape", timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []
    # `<mode> <object> <stage>\t<path>`; a gitlink's mode is 160000.
    return [entry.split("\t", 1)[1] for entry in result.stdout.split("\0")
            if entry.startswith("160000 ") and "\t" in entry]


def _get_git_head(root: Path) -> str | None:
    """Get current git HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _shared_body(metadata: str | None) -> int | None:
    """The shared-body mark the #if/#else recovery leaves in a symbol's metadata."""
    if not metadata:
        return None
    try:
        return json.loads(metadata).get("shared_body")
    except (ValueError, AttributeError):
        return None


def _doc_comment_text(source_bytes: bytes, comment: Node, node: Node) -> str | None:
    """The text of `comment` as `node`'s doc comment, or None if it is not one.

    Read from the file rather than from the node: a node from the #if/#else
    reparse would give the reparse's text. An #else, #elif or #endif in
    between means the comment belongs to another branch than the definition;
    only the reparse, with the directives blanked, can make such a comment
    look adjacent. An opening #if in between is fine — the comment then
    documents the whole conditional, the definition included.
    """
    if _BRANCH_END_RE.search(source_bytes, comment.end_byte, node.start_byte):
        return None
    # A doc comment of several line comments is one node per line: the
    # comments right above, with no blank line between, are read with it —
    # each on a line of its own, not trailing a statement (`int x; // x`).
    first = comment
    while (above := first.prev_sibling) is not None and above.type == "comment":
        gap = source_bytes[above.end_byte:first.start_byte]
        line_start = source_bytes.rfind(b"\n", 0, above.start_byte) + 1
        if (gap.strip() or gap.count(b"\n") > 1
                or source_bytes[line_start:above.start_byte].strip()):
            break
        first = above
    return source_bytes[first.start_byte:comment.end_byte].decode(
        "utf-8", errors="replace").strip()


def _extract_doc_comment(source_bytes: bytes, node: Node) -> str | None:
    """Extract doc comment preceding a symbol node."""
    # Look at the previous sibling for a comment
    prev = node.prev_named_sibling
    if prev is None:
        # Check for comment as first child or preceding line
        # Look at previous unnamed siblings too
        prev_sib = node.prev_sibling
        if prev_sib and prev_sib.type == "comment":
            return _doc_comment_text(source_bytes, prev_sib, node)
        return None

    if prev.type == "comment":
        return _doc_comment_text(source_bytes, prev, node)

    # Python: check for docstring (first child expression_statement with string)
    if node.type in ("function_definition", "class_definition"):
        body = node.child_by_field_name("body")
        if body and body.named_child_count > 0:
            first_stmt = body.named_children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.named_children[0] if first_stmt.named_child_count > 0 else None
                if expr and expr.type == "string":
                    return expr.text.decode("utf-8", errors="replace").strip().strip('"""').strip("'''").strip()

    return None


def _extract_signature(source_bytes: bytes, node: Node, lang: str) -> str | None:
    """Extract function/method signature (without body)."""
    if lang == "python":
        # Everything up to the colon before the body
        params = node.child_by_field_name("parameters")
        ret = node.child_by_field_name("return_type")
        name_node = node.child_by_field_name("name")
        if name_node:
            sig_end = (ret.end_byte if ret else
                       params.end_byte if params else
                       name_node.end_byte)
            return source_bytes[node.start_byte:sig_end].decode("utf-8", errors="replace").strip()

    elif lang in ("c", "cpp"):
        # For function definitions, get the declarator
        declarator = node.child_by_field_name("declarator")
        ret_type = node.child_by_field_name("type")
        if declarator:
            parts = []
            if ret_type:
                parts.append(ret_type.text.decode("utf-8", errors="replace"))
            parts.append(declarator.text.decode("utf-8", errors="replace"))
            return " ".join(parts)

    elif lang in ("javascript", "typescript"):
        name_node = node.child_by_field_name("name")
        params = node.child_by_field_name("parameters")
        ret = node.child_by_field_name("return_type")
        if name_node:
            sig_end = (ret.end_byte if ret else
                       params.end_byte if params else
                       name_node.end_byte)
            prefix = source_bytes[node.start_byte:sig_end].decode("utf-8", errors="replace")
            # Trim off decorators/export
            lines = prefix.split("\n")
            for i, line in enumerate(lines):
                if "function " in line or "class " in line or "(" in line:
                    return "\n".join(lines[i:]).strip()
            return prefix.strip()

    elif lang == "rust":
        name_node = node.child_by_field_name("name")
        params = node.child_by_field_name("parameters")
        ret = node.child_by_field_name("return_type")
        if name_node:
            sig_end = (ret.end_byte if ret else
                       params.end_byte if params else
                       name_node.end_byte)
            return source_bytes[node.start_byte:sig_end].decode("utf-8", errors="replace").strip()

    elif lang == "lua":
        # Lua declares no return type, so the parameter list ends the signature.
        # It may hang off the node itself (`function f(a)`) or off the value it
        # is assigned (`f = function(a)`), which is the same definition.
        params = _lua_parameters(node)
        if params is not None:
            return source_bytes[node.start_byte:params.end_byte].decode(
                "utf-8", errors="replace").strip()

    return None


def _lua_nameless_definition(node: Node) -> bool:
    """True when a definition has no name a caller could write.

    A computed key — `{ [k] = function() end }` — is one: `k` holds the key
    rather than being it. A string key does name the function, and the query
    captures it from inside the string.
    """
    if node.type == "field":
        if node.child_count == 0 or node.child(0).type != "[":
            return False
        key = node.child_by_field_name("name")
        return key is None or key.type != "string"

    # An assignment target that is not a path names nothing either: a call, a
    # parenthesised expression. Its source text would carry newlines and
    # punctuation into the name index in place of a name. This asks the shape
    # of the target, not whether a dotted path can be spelled from it —
    # `t["my-key"]` has no dotted form and still names its function.
    if node.type == "variable_declaration":
        inner = node.named_children[0] if node.named_children else None
        return inner is None or _lua_nameless_definition(inner)

    if node.type == "assignment_statement":
        return not _lua_is_path(node.child_by_field_name("name"))

    return False


def _lua_is_path(node: Node | None) -> bool:
    """Whether a node is a name or a chain of index expressions ending in one."""
    while node is not None and node.type != "identifier":
        if node.type not in (
            "dot_index_expression", "bracket_index_expression", "method_index_expression",
        ):
            return False
        node = node.child_by_field_name("table")
    return node is not None


def _lua_definition_path(node: Node) -> str | None:
    """The full path a Lua definition hangs off, e.g. `Stack:pop` or `von.Entity`.

    The name a call site writes is not always the whole path, so the path is
    kept as the qualified name. `t["k"]` is written as the dotted path it is
    equivalent to: a qualified name carrying quotes and brackets matches
    nothing a reader or a caller would write.
    """
    if node.type == "field":
        name = _lua_key_name(node.child_by_field_name("name"))
        if name is None:
            return None
        table = _lua_enclosing_table(node)
        return f"{table}.{name}" if table else name

    if node.type == "variable_declaration":
        inner = node.named_children[0] if node.named_children else None
        return _lua_definition_path(inner) if inner is not None else None

    return _lua_target_path(node.child_by_field_name("name"))


# A key only joins a dotted path if it could have been written as one.
_LUA_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _lua_key_name(node: Node | None) -> str | None:
    """A table key as a path segment, or None if it cannot be written as one."""
    if node is None:
        return None
    if node.type == "string":
        content = node.child_by_field_name("content")
        node_text = content.text if content is not None else b""
    elif node.type == "identifier":
        node_text = node.text
    else:
        return None
    name = node_text.decode("utf-8", errors="replace")
    return name if _LUA_IDENTIFIER.match(name) else None


def _lua_target_path(node: Node | None) -> str | None:
    """The dotted path of an assignment target, e.g. `t.a.b` for `t["a"]["b"]`.

    Walked iteratively: a generated file can carry a path thousands of segments
    long, and recursing over one exhausts the stack.

    None for anything that is not a path — a call, a parenthesised expression, a
    computed key. Such a target has no name a reader would write, and its source
    text can carry newlines and punctuation into the name index.
    """
    segments: list[tuple[str, str]] = []
    while node is not None and node.type != "identifier":
        if node.type == "method_index_expression":
            separator, key = ":", node.child_by_field_name("method")
        elif node.type in ("dot_index_expression", "bracket_index_expression"):
            separator, key = ".", node.child_by_field_name("field")
        else:
            return None
        name = _lua_key_name(key)
        if name is None:
            return None
        segments.append((separator, name))
        node = node.child_by_field_name("table")

    if node is None:
        return None
    path = node.text.decode("utf-8", errors="replace")
    for separator, name in reversed(segments):
        path += separator + name
    return path


def _lua_enclosing_table(field: Node, depth: int = 0) -> str | None:
    """The path of the table a literal's field belongs to, when it has one.

    `local encode = { ["Entity"] = function() end }` gives `encode`, so that two
    tables holding the same key do not collapse onto one qualified name. A table
    nested in another field answers with that field's own path, which is where
    the depth limit comes in — a path deeper than this says nothing useful.
    """
    if depth > 16:
        return None
    table = field.parent
    if table is None or table.type != "table_constructor":
        return None

    holder = table.parent
    if holder is None:
        return None

    if holder.type == "field":                       # a table inside a table
        name = _lua_key_name(holder.child_by_field_name("name"))
        if name is None:
            return None
        outer = _lua_enclosing_table(holder, depth + 1)
        return f"{outer}.{name}" if outer else name

    if holder.type != "expression_list":
        return None
    assignment = holder.parent
    if assignment is None or assignment.type != "assignment_statement":
        return None

    # `local P, Q = {…}, {…}` — the table's own position picks its name.
    values = list(holder.named_children)
    variables = next(
        (c for c in assignment.named_children if c.type == "variable_list"), None,
    )
    if variables is None or table not in values:
        return None
    position = values.index(table)
    targets = list(variables.named_children)
    if position >= len(targets):
        return None
    return _lua_target_path(targets[position])


def _lua_parameters(node: Node) -> Node | None:
    """The parameter list of the function a Lua definition node defines.

    The route is spelled out per shape rather than searched: a general descent
    finds whichever function comes first in the tree, which for an assignment
    is the one in the *target* if any (`(function(a) end)().x = function(b) end`
    gives `a`), and on a long dotted path it recurses deep enough to exhaust
    the stack — losing not just the symbol but every symbol in the file.
    """
    params = node.child_by_field_name("parameters")
    if params is not None:                       # function f(a) / function(a)
        return params

    if node.type == "variable_declaration":      # local f = function(a)
        inner = node.named_children[0] if node.named_children else None
        return _lua_parameters(inner) if inner is not None else None

    if node.type == "assignment_statement":      # T.f = function(a)
        for child in node.named_children:
            if child.type == "expression_list":
                value = child.child_by_field_name("value")
                return _lua_parameters(value) if value is not None else None
        return None

    if node.type == "field":                     # { f = function(a) }
        value = node.child_by_field_name("value")
        return _lua_parameters(value) if value is not None else None

    return None


# `\b` is Unicode-aware, so a boundary depends on characters this module must
# not assume are ASCII: `caféhandler` and `123handler` contain no boundary before
# `handler`, and a scanner that simply looked for identifier runs would report a
# name that is really part of a larger word.
_IS_WORD_CHAR = re.compile(r"\w").match
# Identifier runs, as candidate starting points. The classes are Unicode: Python
# and C# both allow `émetteur` as an identifier, and an ASCII-only head would
# push every such name off the grouped path and back onto a scan of the whole
# body, which is the cost this grouping exists to remove. `[^\W\d]` is a word
# character that is not a digit.
#
# The lookbehind IS the guard for these positions, not a filter: the walk does
# not re-check the leading boundary. A run starts on a word character and the
# lookbehind refuses a word character before it, so exactly one side is a word
# character — `\b`, by construction. Relax it and names inside larger words
# (`123handler`, `caféhandler`) start matching.
_IDENT_RUN_RE = re.compile(r"(?<!\w)[^\W\d]\w*")
_LEADING_RUN_RE = re.compile(r"[^\W\d]\w*")


def _on_boundary(content: str, index: int) -> bool:
    """`\b` at `index`: exactly one side is a word character."""
    before = index > 0 and _IS_WORD_CHAR(content[index - 1]) is not None
    after = index < len(content) and _IS_WORD_CHAR(content[index]) is not None
    return before != after


def build_name_matcher(names: set[str]) -> Callable[[str], set[str]]:
    """Return a function mapping a symbol body to the known names it references.

    Matching is leftmost, longest-at-that-position and non-overlapping: where
    several names match at the same spot the longest one wins, and the shorter
    names inside it are not reported. `Widget::~Widget` in a destructor body
    therefore yields the destructor, never a bare `Widget`.

    An alternation of every name expresses that directly, but Python's re
    engine walks alternatives one at a time at each position, so the cost grows
    with the size of the name set rather than with the body being scanned. On a
    codebase with tens of thousands of symbols it dominates indexing entirely.

    So group the names by their leading identifier instead. A name can only
    begin where an identifier run begins on a boundary, so the run under the
    cursor selects a handful of candidates by dictionary lookup, and the
    longest one that the body actually starts with -- and that ends on a
    boundary -- wins. `Vec<T>::push_back` and `Foo::operator+=` need no special
    handling: they group under `Vec` and `Foo` like everything else, and the
    punctuation is just part of the string being compared.

    One scan, one cursor. Splitting the work over several passes and merging
    the results afterwards is NOT equivalent, however carefully the merge is
    written: once a match is accepted, the search has to RESUME inside what the
    other passes had already scanned. `Registry<T>::Lookup::Inner::Leaf` is the
    case that proves it -- accepting `Registry<T>::Lookup` must leave
    `Inner::Leaf` still findable.
    """
    # Names that do not begin with an identifier character (extraction can
    # produce a few). They cannot be reached from an identifier run, so they
    # are located directly.
    unanchored: list[str] = []
    grouping: dict[str, list[str]] = {}
    for name in names:
        head = _LEADING_RUN_RE.match(name)
        if head is None:
            unanchored.append(name)
        else:
            grouping.setdefault(head.group(0), []).append(name)
    # Longest first, so the first candidate that matches at a position is the
    # one the alternation would have chosen. Frozen into tuples: the scan hands
    # these lists straight to the caller's walk, and a shared list that anything
    # could append to is a trap waiting for the next change.
    buckets = {
        head: tuple(sorted(candidates, key=len, reverse=True))
        for head, candidates in grouping.items()
    }

    def anchored_candidates(content: str):
        """Identifier runs, in order, with the names that could start there.

        The lookbehind in _IDENT_RUN_RE has already established the leading
        boundary — the run begins on an identifier character and the character
        before it is not a word character — so the walk need only check where
        each candidate ENDS.
        """
        for run in _IDENT_RUN_RE.finditer(content):
            names_here = buckets.get(run.group(0))
            if names_here is not None:
                yield run.start(), names_here

    def all_candidates(content: str):
        """The same, plus the names that no identifier run can reach.

        Only used when such names exist. They carry no boundary guarantee, so
        they are filtered here rather than in the walk.

        Several of them can start at the SAME position, and the walk takes the
        first candidate that matches — so they have to be grouped per position
        and ordered longest first, exactly as the buckets are. Emitting them one
        by one left the order to however the name set happened to iterate, and
        `émetteur` would beat `émetteur.envoyer` about half the time.
        """
        by_start: dict[int, list[str]] = {}
        for name in unanchored:
            at = content.find(name)
            while at != -1:
                if _on_boundary(content, at):
                    by_start.setdefault(at, []).append(name)
                at = content.find(name, at + 1)

        found_at = list(anchored_candidates(content))
        found_at.extend(
            (start, tuple(sorted(names_here, key=len, reverse=True)))
            for start, names_here in by_start.items()
        )
        found_at.sort(key=lambda candidate: candidate[0])
        return found_at

    def match(content: str) -> set[str]:
        candidates = all_candidates(content) if unanchored else anchored_candidates(content)

        found: set[str] = set()
        cursor = 0
        for start, names_here in candidates:
            if start < cursor:
                continue
            for name in names_here:
                end = start + len(name)
                if content.startswith(name, start) and _on_boundary(content, end):
                    found.add(name)
                    cursor = end
                    break
        return found

    return match


# Languages whose parse can break on a conditional that splits a brace across
# its branches (see _first_branch_only).
_PREPROCESSED_LANGS = frozenset({"c", "cpp"})

_CONDITIONAL_RE = re.compile(
    rb"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|elifdef|elifndef|else|endif)\b", re.MULTILINE
)

# The directives that end a branch: whatever came before one of them belongs
# to another branch than whatever comes after.
_BRANCH_END_RE = re.compile(
    rb"^[ \t]*#[ \t]*(elif|elifdef|elifndef|else|endif)\b", re.MULTILINE
)

# A character literal: one character or escape, or a short multi-character
# constant ('RIFF'), closed on the same line.
_CHAR_LITERAL_RE = re.compile(rb"'(?:\\[^\n]{1,8}?|[^'\\\n]{1,4})'")


# What follows the quote of a C++ raw string: its delimiter, then `(`.
_RAW_DELIMITER_RE = re.compile(rb'([^()\\\s]{0,16})\(')


def _raw_string_prefix(source: bytes, quote: int) -> bool:
    """Whether the quote at `quote` opens a C++ raw string: R, u8R, uR, UR or
    LR right before it, not ending a longer identifier."""
    if quote == 0 or source[quote - 1] != 0x52:  # R
        return False
    start = quote - 1
    if source[start - 2:start] == b"u8":
        start -= 2
    elif start > 0 and source[start - 1] in (0x75, 0x55, 0x4C):  # u U L
        start -= 1
    before = source[start - 1] if start > 0 else 0x20
    return not (chr(before).isalnum() or before == 0x5F)


def _ends_with_backslash(source: bytes, newline: int) -> bool:
    """Whether the line ending at `newline` ends in a backslash, CRLF too."""
    j = newline - 1
    if j >= 0 and source[j] == 0x0D:
        j -= 1
    return j >= 0 and source[j] == 0x5C


def _digit_separator(source: bytes, i: int) -> bool:
    """Whether the quote at `i` opens nothing because a word or a number is
    glued to it: a digit separator, `1'000`, or the closing quote of a
    multi-character constant too long to read as a literal, `'longtag'`.
    After a keyword or an encoding prefix it opens a character literal:
    `case'{':`, `L'x'`, `u8'x'`."""
    j = i
    while j > 0 and (chr(source[j - 1]).isalnum() or source[j - 1] == 0x5F):
        j -= 1
    word = source[j:i].decode("latin-1")
    return bool(word) and word not in _BEFORE_A_CHARACTER


# Words a character literal can be glued to: encoding prefixes, and the
# keywords a value follows.
_BEFORE_A_CHARACTER = frozenset({
    "L", "u", "U", "u8", "case", "return", "else", "do", "throw", "sizeof",
    "co_return", "co_yield", "and", "or", "not", "xor", "bitand", "bitor",
    "compl", "not_eq", "and_eq", "or_eq", "xor_eq",
})


def _c_comment_bytes(source: bytes, literals: bool = False) -> bytearray:
    """Mark the bytes of C/C++ comments with 1, everything else with 0 — and
    with `literals`, the bytes of strings and character literals too.

    Strings are stepped over, so a `/*` inside one opens nothing — raw
    strings included, which may hold quotes and newlines. A quote opens a
    character literal only when one closes it shortly after on the same
    line: a digit separator (1'000) or the apostrophe of an #error message
    is a lone quote, and taking it for an opening one would hide every
    comment after it. A `//` comment ending in a backslash runs on into the
    next line, as the preprocessor splices the two.
    """
    marks = bytearray(len(source))
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == 0x2F and i + 1 < n and source[i + 1] in (0x2A, 0x2F):  # /* or //
            if source[i + 1] == 0x2A:
                end = source.find(b"*/", i + 2)
                end = n if end == -1 else end + 2
            else:
                end = source.find(b"\n", i + 2)
                while end != -1 and _ends_with_backslash(source, end):
                    end = source.find(b"\n", end + 1)
                end = n if end == -1 else end
            marks[i:end] = b"\x01" * (end - i)
            i = end
        elif c == 0x22 and _raw_string_prefix(source, i) and (
                opening := _RAW_DELIMITER_RE.match(source, i + 1)):  # R"delim( ... )delim"
            closing = b")" + opening.group(1) + b'"'
            end = source.find(closing, opening.end())
            end = n if end == -1 else end + len(closing)
            if literals:
                marks[i:end] = b"\x01" * (end - i)
            i = end
        elif c == 0x22:  # "
            start = i
            i += 1
            while i < n and source[i] not in (0x22, 0x0A):
                if source[i] == 0x5C:  # an escape; before a CRLF it splices both bytes
                    i += 3 if source[i + 1:i + 3] == b"\r\n" else 2
                else:
                    i += 1
            i = min(i + 1, n)
            if literals:
                marks[start:i] = b"\x01" * (i - start)
        elif c == 0x27 and _digit_separator(source, i):  # 1'000
            i += 1
        elif c == 0x27:  # '
            literal = _CHAR_LITERAL_RE.match(source, i)
            if literal and literals:
                marks[i:literal.end()] = b"\x01" * (literal.end() - i)
            i = literal.end() if literal else i + 1
        else:
            i += 1
    return marks


def _first_branch_only(source: bytes) -> bytes:
    """Keep the first branch of every #if chain and blank everything else.

    tree-sitter does not run the preprocessor, so a function whose #if and
    #else branches each open a brace — closed once, after #endif — reads as
    two opening braces for one closing brace, and the parse gives up on it.
    With the directives and the later branches blanked, the braces balance
    again.

    Blanked bytes become spaces and newlines stay, so offsets and line
    numbers are those of the original: a node found in this text points at
    the same bytes in the file.

    Comments are never blanked. One that opens on a directive line, or in a
    dropped branch, and closes elsewhere keeps both of its ends — blanking
    one would leave the other half to be read as code. And a directive
    written inside a comment is not a directive.
    """
    in_comment = _c_comment_bytes(source)

    out = bytearray(source)
    first_branch: list[bool] = []  # per open conditional: still in its first branch
    continued = False  # the previous directive line ended with a backslash
    pos = 0
    for line in source.splitlines(keepends=True):
        end = pos + len(line)
        body = line.rstrip(b"\r\n")
        directive = None if continued else _CONDITIONAL_RE.match(body)
        if directive and in_comment[pos + len(body) - len(body.lstrip(b" \t"))]:
            directive = None
        if directive:
            word = directive.group(1)
            if word in (b"if", b"ifdef", b"ifndef"):
                first_branch.append(True)
            elif word == b"endif":
                if first_branch:
                    first_branch.pop()
            elif first_branch:
                first_branch[-1] = False
        if continued or directive or not all(first_branch):
            for i in range(pos, end):
                if out[i] not in (0x0A, 0x0D) and not in_comment[i]:
                    out[i] = 0x20
        continued = bool(continued or directive) and body.endswith(b"\\")
        pos = end
    return bytes(out)


_CALLABLE_KINDS_CPP = frozenset({"function", "method"})

# Where a C/C++ definition sits at top level: file scope, a namespace or an
# `extern "C"` block.
_TOP_LEVEL_PARENTS = frozenset({"translation_unit", "declaration_list"})


def _under_error(node: Node) -> bool:
    """Whether error recovery put a node inside an ERROR node."""
    parent = node.parent
    while parent is not None:
        if parent.type == "ERROR":
            return True
        parent = parent.parent
    return False


def _at_top_level(node: Node) -> bool:
    parent = node.parent
    if parent is not None and parent.type == "template_declaration":
        parent = parent.parent
    return parent is not None and parent.type in _TOP_LEVEL_PARENTS


def _looks_like_a_function(node: Node) -> bool:
    """Whether a top-level function definition has the shape of a real one: a
    return type, or a qualified name (a constructor or destructor defined
    outside its class). A loop macro read at file scope —
    `FOR_EACH_ITEM(x) { ... }` — has neither."""
    if node.type == "template_declaration":
        node = next((c for c in node.named_children if c.type == "function_definition"), node)
    if node.child_by_field_name("type") is not None:
        return True
    declarator = node.child_by_field_name("declarator")
    while declarator is not None and declarator.type not in (
            "identifier", "field_identifier", "qualified_identifier",
            "destructor_name", "operator_name"):
        inner = declarator.child_by_field_name("declarator")
        if inner is None:
            inner = next((c for c in declarator.named_children
                          if c.type.endswith("declarator") or c.type.endswith("identifier")), None)
        declarator = inner
    return declarator is not None and declarator.type == "qualified_identifier"


def _returns_a_type_it_defines(node: Node) -> bool:
    """Whether a function definition's return type is a class, struct, union
    or enum defined on the spot — `class X {...} f() {...}`, which is what a
    reparse makes of a class closed at a stray brace and the function after
    it."""
    ret = node.child_by_field_name("type")
    return ret is not None and ret.type.endswith("_specifier") and ret.child_by_field_name(
        "body") is not None


def _closes_for_real(node: Node) -> bool:
    """Whether a definition ends on a closing token that is in the source,
    rather than one tree-sitter made up because the braces never closed."""
    last = node
    while last.child_count:
        last = last.children[-1]
    return not last.is_missing


def _brace_view(source: bytes) -> bytes:
    """The source as its braces read: the first branch of each conditional,
    with comments, strings, character literals and preprocessor lines
    blanked — a brace in any of them is no block. Offsets are the source's."""
    view = bytearray(_first_branch_only(source))
    for run in re.finditer(rb"\x01+", _c_comment_bytes(source, literals=True)):
        view[run.start():run.end()] = re.sub(
            rb"[^\r\n]", b" ", bytes(view[run.start():run.end()]))
    pos = 0
    continued = False
    for line in bytes(view).splitlines(keepends=True):
        end = pos + len(line)
        body = line.rstrip(b"\r\n")
        if continued or body.lstrip(b" \t").startswith(b"#"):
            for i in range(pos, pos + len(body)):
                view[i] = 0x20
            continued = body.endswith(b"\\")
        pos = end
    return bytes(view)


def _brace_close(view: bytes, open_at: int) -> int | None:
    """The end of the block whose `{` is at `open_at`, or None when it never
    closes."""
    depth = 0
    for i in range(open_at, len(view)):
        if view[i] == 0x7B:  # {
            depth += 1
        elif view[i] == 0x7D:  # }
            depth -= 1
            if depth == 0:
                return i + 1
    return None


class _ExtendedNode:
    """A definition node whose end its braces put further than the parse."""

    def __init__(self, node: Node, end_byte: int, end_point: tuple[int, int]):
        self._node = node
        self.end_byte = end_byte
        self.end_point = end_point

    def __getattr__(self, name):
        return getattr(self._node, name)


def _extend_to_braces(symbols: list, source: bytes, view_of) -> list:
    """Let a function end where its braces close.

    A macro the parser cannot read — `PICK(< a, == b)`, or a bare `BLOCK_END`
    standing for a brace — can make error recovery swallow a closing brace:
    the function ends early and the rest of its body is left at file scope.
    Only a function holding errors is checked — with the template around
    it, if any — and it is extended only when no other definition starts in
    the part it gains: an unbalanced brace in the text must never merge two
    definitions. A declaration there proves nothing: the tail's statements
    read at file scope, `Guard hold(lock);`, give prototypes.
    """
    view = None
    blockers: list[int] = []
    newlines: list[int] = []
    extended = []
    for sym in symbols:
        node, kind, name = sym
        function = node
        if node.type == "template_declaration":
            function = next((c for c in node.named_children
                             if c.type == "function_definition"), node)
        body = (function.child_by_field_name("body")
                if function.type == "function_definition" and function.has_error else None)
        if body is not None and body.type == "compound_statement":
            if view is None:
                view = view_of()
                blockers = sorted(n.start_byte for n, k, _ in symbols
                                  if k not in ("macro", "prototype"))
                newlines = [m.start() for m in re.finditer(b"\n", source)]
            close = (_brace_close(view, body.start_byte)
                     if view[body.start_byte:body.start_byte + 1] == b"{" else None)
            if close is not None and close > node.end_byte:
                first = bisect.bisect_left(blockers, node.end_byte)
                if first == len(blockers) or blockers[first] >= close:
                    row = bisect.bisect_left(newlines, close)
                    column = close - (newlines[row - 1] + 1 if row else 0)
                    sym = (_ExtendedNode(node, close, (row, column)), kind, name)
        extended.append(sym)
    return extended


def _extent_is_sound(node: Node) -> bool:
    """Whether a definition from the #if/#else reparse can be trusted for its
    extent.

    A node without errors can. A long function nearly always holds something
    tree-sitter cannot read — a call through a pointer to member function, a
    macro — and such an error is local: it does not move the braces. So a
    node with errors is trusted too, as long as it ends on a real closing
    token, no closing brace inside it had to be made up, and no definition is
    nested in its errors — the signs of braces that do not balance.
    """
    if node.type == "function_definition" and (
            _returns_a_type_it_defines(node)
            or (_at_top_level(node) and not _looks_like_a_function(node))):
        return False
    if not node.has_error:
        return True
    last = node
    while last.child_count:
        last = last.children[-1]
    if last.is_missing or last.type not in ("}", ";"):
        return False
    stack = [child for child in node.children if child.has_error]
    while stack:
        child = stack.pop()
        if child.type == "function_definition" or (child.is_missing and child.type == "}"):
            return False
        stack.extend(grandchild for grandchild in child.children
                     if grandchild.has_error or grandchild.is_missing
                     or grandchild.type == "function_definition")
    return True


# Words a C or C++ symbol can never be named. Error recovery can still hand
# one to the extractor: `if (a == b) { ... }` cut off from its chain reads as
# a function `if` taking `(a == b)`. C++ reserves more than C, where `new`,
# `delete` or `class` are ordinary identifiers.
_C_KEYWORDS = frozenset({
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
    "void", "volatile", "while", "_Alignas", "_Alignof", "_Atomic", "_Bool",
    "_Complex", "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
    "_Thread_local",
})
_CPP_KEYWORDS = _C_KEYWORDS | frozenset({
    "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor", "bool",
    "catch", "char8_t", "char16_t", "char32_t", "class", "co_await",
    "co_return", "co_yield", "compl", "concept", "const_cast", "consteval",
    "constexpr", "constinit", "decltype", "delete", "dynamic_cast", "explicit",
    "export", "false", "friend", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
    "protected", "public", "reinterpret_cast", "requires", "static_assert",
    "static_cast", "template", "this", "thread_local", "throw", "true", "try",
    "typeid", "typename", "using", "virtual", "wchar_t", "xor", "xor_eq",
})
_RESERVED_NAMES = {"c": _C_KEYWORDS, "cpp": _CPP_KEYWORDS}


_AFTER_PARAMS_WORDS = frozenset({
    "const", "volatile", "override", "final", "noexcept", "throw", "try", "requires",
    "mutable", "__attribute__",
})


def _macro_typed_declaration(def_node: Node, name: str) -> bool:
    """Whether a "function" is a variable declared with a macro as its type:
    no return type before the name, and after the parentheses another name
    closed by `;`, `=`, `,` or `[` — `MACRO(f32, s16) mField;`."""
    if def_node.child_by_field_name("type") is not None:
        return False
    # A constructor has no return type either: `Box() NOEXCEPT_M;` is one,
    # its trailing macro an attribute.
    parent = def_node.parent
    while parent is not None and parent.type not in (
            "class_specifier", "struct_specifier", "union_specifier", "translation_unit"):
        parent = parent.parent
    if parent is not None and parent.type != "translation_unit":
        owner = parent.child_by_field_name("name")
        if owner is not None and owner.text.decode("utf-8", errors="replace").rsplit(
                "::", 1)[-1].split("<", 1)[0] == name:
            return False
    text = def_node.text.decode("utf-8", errors="replace")
    m = re.match(
        rf"\s*(?:(?:static|extern|inline|const|volatile)\s+)*{re.escape(name)}\s*\(",
        text)
    if m is None:
        return False
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    rest = text[i:]
    if not rest.strip() and def_node.parent is not None:
        # In a class body the parser may end the node at the parenthesis
        # and leave the field's name to what follows.
        parent = def_node.parent
        rest = parent.text[def_node.end_byte - parent.start_byte:][:200].decode(
            "utf-8", errors="replace")
    after = re.match(r"\s*([A-Za-z_]\w*)\s*[;=,\[]", rest)
    return after is not None and after.group(1) not in _AFTER_PARAMS_WORDS


def _function_inside_a_function(node: Node, kind: str) -> bool:
    """Whether a definition is a function read inside another function's
    body, where C and C++ allow none: error recovery's work.

    Only functions: a struct or an enum local to a function is legal C,
    whatever its name. And nothing about errors: an export macro before a
    real C function, or a stray token in an enum, puts an error in the C++
    parse of perfectly real code.
    """
    if kind not in ("function", "method"):
        return False
    parent = node.parent
    while parent is not None:
        if parent.type == "compound_statement":
            return True
        parent = parent.parent
    return False


def _kind_from_capture(capture_name: str) -> str:
    """Map tree-sitter capture names to symbol kinds."""
    prefix = capture_name.split(".")[0]
    mapping = {
        "fn": "function",
        "dec_fn": "function",
        "export_fn": "function",
        "cls": "class",
        "dec_cls": "class",
        "export_cls": "class",
        "method": "method",
        "struct": "struct",
        "enum": "enum",
        "iface": "interface",
        "type": "type_alias",
        "typedef": "type_alias",
        "ns": "namespace",
        "mod": "module",
        "macro": "macro",
        "define": "macro",
        "proto": "prototype",
        "qproto": "prototype",
        "ptrfn": "function",   # C/C++ pointer return types
        "ptrfn2": "function",
        "ptrproto": "prototype",
        "ptrproto2": "prototype",
        "ptrmethod": "method",
        "ptrmethod2": "method",
        "ptrfield_fn": "method",
        "ptrfield_fn2": "method",
        "trait": "trait",
        "impl": "impl",
        "template": "template",
        "field_fn": "method",  # method declarations in class bodies (headers)
        # C++ methods defined inside their class body
        "inline_method": "method",
        "ptrinline": "method",
        "ptrinline2": "method",
        # An operator is a method or a free function depending on where it
        # sits; _in_class_body turns these into methods inside a class.
        "inline_op": "function",
        "refop": "function",
        "ptrop": "function",
        "field_op": "method",
        "reffield_op": "method",
        "ptrfield_op": "method",
        "opproto": "prototype",
        "refopproto": "prototype",
        "conv": "method",
        "convdecl": "method",
        "qconv": "method",
        "ptrop2": "function",
        "ptropproto": "prototype",
        "ptropproto2": "prototype",
        "ptrrefproto": "prototype",
        "ptrreffield": "method",
        "ptrreffn": "function",
        "ptrrefinline": "method",
        "ptrrefmethod": "method",
        "inline_dtor": "method",
        # C++ definitions and declarations returning a reference
        "reffn": "function",
        "refmethod": "method",
        "refinline": "method",
        "refproto": "prototype",
        "reffield_fn": "method",
        "var": "function",     # arrow functions
        "var2": "function",
        "ctor": "function",    # C# constructors
        "prop": "property",    # C# properties
        # Dart-specific
        "ext": "extension",
        "mixin": "mixin",
        "getter": "method",    # Dart getters
        # Bash-specific
        "export_var": "variable",
        # SQL-specific
        "table": "table",
        "view": "view",
    }
    return mapping.get(prefix, "unknown")


def _get_enclosing_scope(node: Node) -> list[str]:
    """Walk up the AST to collect enclosing namespace/class/struct names.

    Returns a list like ["myapp", "util", "ConfigManager"] for a method
    defined inside namespace myapp { namespace util { class ConfigManager { ... } } }
    """
    scopes: list[str] = []
    previous = node
    current = node.parent
    while current is not None:
        if current.type in (
            "namespace_definition", "class_specifier", "struct_specifier",
            "union_specifier",
            "class_definition",  # Python
            "class_declaration", "namespace_declaration",  # C#
        ):
            name_node = current.child_by_field_name("name")
            if name_node:
                scopes.append(name_node.text.decode("utf-8", errors="replace"))
        elif current.type == "template_declaration":
            # Look for named child inside the template — unless the walk just
            # came up through it: that class is already in the scopes, and
            # adding it again named a template's members `Holder::Holder::f`.
            for child in current.children:
                if child.type in ("class_specifier", "struct_specifier", "union_specifier"):
                    name_node = child.child_by_field_name("name")
                    if name_node and child != previous:
                        scopes.append(name_node.text.decode("utf-8", errors="replace"))
                    break
        previous = current
        current = current.parent
    scopes.reverse()
    return scopes


def _build_qualified_name(symbol_name: str | None, node: Node, lang: str) -> str | None:
    """Build a proper qualified name using enclosing scope context.

    For C++: "myapp::util::ConfigManager::process"
    For Python: "Calculator.add"
    For other languages: "Module.Class.method"
    """
    if symbol_name is None:
        return None

    if lang in ("c", "cpp"):
        scopes = _get_enclosing_scope(node)
        if scopes:
            return "::".join(scopes + [symbol_name])
        # If the name already has :: (from qualified_identifier), keep it
        if "::" in symbol_name:
            return symbol_name
        return symbol_name
    elif lang == "python":
        scopes = _get_enclosing_scope(node)
        if scopes:
            return ".".join(scopes + [symbol_name])
        return symbol_name
    elif lang == "lua":
        # The table a function hangs off is written on the definition itself,
        # not in an enclosing scope node the way a class body is.
        return _lua_definition_path(node) or symbol_name
    else:
        scopes = _get_enclosing_scope(node)
        if scopes:
            return ".".join(scopes + [symbol_name])
        return symbol_name


_DECLARATOR_NAME_TYPES = frozenset({
    "identifier", "field_identifier", "qualified_identifier", "operator_name",
    "destructor_name", "type_identifier",
})


def _conversion_of(node: Node) -> Node | None:
    """The operator_cast a name node stands for, through any number of
    qualification levels (`ns::Box::operator int`), or None."""
    while node is not None and node.type == "qualified_identifier":
        node = node.child_by_field_name("name")
    return node if node is not None and node.type == "operator_cast" else None


def _names_a_conversion(node: Node) -> bool:
    """Whether a name node is a conversion operator, qualified or not."""
    return _conversion_of(node) is not None


def _operator_cast_name(node: Node) -> str:
    """`operator const char*() const` -> "operator const char*".

    The name is everything before the conversion's own parameter list — found
    in the tree, not by the first `(`, which may belong to the target type
    (`operator Callback<void(int)>`) or to the scope. The target type alone
    would drop its pointer, reference and const. A qualified definition keeps
    its scope: `ns::Box::operator bool`.
    """
    cast = _conversion_of(node)
    declarator = cast.child_by_field_name("declarator") if cast is not None else None
    while declarator is not None and declarator.type != "abstract_function_declarator":
        inner = declarator.child_by_field_name("declarator")
        if inner is None:
            inner = next((c for c in declarator.named_children
                          if c.type.endswith("declarator")), None)
        declarator = inner
    parameters = (declarator.child_by_field_name("parameters")
                  if declarator is not None else None)
    end = parameters.start_byte if parameters is not None else node.end_byte
    text = node.text[:end - node.start_byte]
    return " ".join(text.decode("utf-8", errors="replace").split())


def _declarator_name(node: Node | None) -> str | None:
    """The name a C/C++ declarator declares, through any pointer, reference or
    function declarator around it.

    A reference_declarator holds its inner declarator without a field name,
    so the walk falls back on its first named declarator child.
    """
    while node is not None:
        if _names_a_conversion(node):
            return _operator_cast_name(node)
        if node.type in _DECLARATOR_NAME_TYPES:
            return node.text.decode("utf-8", errors="replace")
        inner = node.child_by_field_name("declarator")
        if inner is None:
            inner = next((child for child in node.named_children
                          if child.type.endswith("declarator")
                          or child.type in _DECLARATOR_NAME_TYPES
                          or child.type == "operator_cast"), None)
        node = inner
    return None


def _in_class_body(node: Node) -> bool:
    """Whether a C++ definition sits in a class body — through a template, or
    through the #if blocks a class body may hold — and so is a method,
    whatever pattern matched it."""
    parent = node.parent
    while parent is not None and (parent.type == "template_declaration"
                                  or parent.type.startswith("preproc_")):
        parent = parent.parent
    return parent is not None and parent.type == "field_declaration_list"


def _extract_template_name(node: Node) -> str | None:
    """Extract the name from a template_declaration's inner declaration.

    template<T> class Container -> "Container"
    template<T> T max_value(T a) -> "max_value"
    template<T> struct Pair -> "Pair"
    """
    for child in node.children:
        if child.type in ("class_specifier", "struct_specifier", "enum_specifier"):
            name_node = child.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
        elif child.type in ("function_definition", "declaration"):
            # A function, a template variable or a forward declaration. The
            # name can sit under pointer and reference declarators — reading
            # the declarator's own text named `T& pick()` "& pick()".
            declarator = child.child_by_field_name("declarator")
            if declarator:
                return (_declarator_name(declarator)
                        or declarator.text.decode("utf-8", errors="replace"))
        elif child.type == "alias_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
    return None


def _active_doc_extensions() -> set[str]:
    """Return file extensions (e.g. '*.pdf') that have active extractors."""
    return {f"*{ext}" for ext in DOCUMENT_EXTENSIONS}


def _doc_languages() -> set[str]:
    """Return the set of document language names with active extractors."""
    return set(get_registry().keys()) | {"markdown"}


class Indexer:
    """Indexes a codebase into a Srclight database."""

    def __init__(self, db: Database, config: IndexConfig | None = None):
        self.db = db
        self.config = config or IndexConfig()
        self._parsers: dict[str, Parser] = {}
        self._queries: dict[str, Query] = {}
        self._ext_overrides: dict[str, str] = {}

        # Remove ignore patterns for extensions that have active extractors
        active_exts = _active_doc_extensions()
        self.config.ignore_patterns = [
            p for p in self.config.ignore_patterns if p not in active_exts
        ]

    def _resolve_extension_overrides(self) -> dict[str, str]:
        """Pick the extra-extension map for this run and keep it with the index.

        An explicit map — including an empty one, which clears — wins and is
        recorded; otherwise the recorded one is used, so a reindex that
        passes no flags reads the same files as the run that declared them.
        """
        declared = self.config.extension_overrides
        if declared is None:
            return self.db.get_extension_overrides()
        overrides = {normalize_extension(str(k)): str(v).lower() for k, v in declared.items()}
        # Validated here, not only in the CLI: a language that does not exist
        # is written into every FileRecord it touches, produces no symbols,
        # and shows up in index_status as a language of the codebase.
        unknown = {
            lang for lang in overrides.values()
            if lang != SKIP_LANGUAGE and lang not in LANGUAGES
        }
        if unknown:
            raise ValueError(
                f"Unknown language(s) in extension_overrides: {', '.join(sorted(unknown))}"
            )
        self.db.set_extension_overrides(overrides)
        return overrides

    def _ignored_only_by_its_own_extension(self, path: Path, root: Path,
                                           patterns: list[str]) -> bool:
        """True when a readable format is hidden by its own extension pattern alone.

        Two cases reach here: a document format whose extractor is missing,
        and a source extension that carries an ignore pattern anyway —
        `*.cmake` sits in the default list next to the build artefacts, while
        cmake is a language srclight parses. Both are unread by accident
        rather than by intent.

        It holds only if nothing ELSE excludes the file: a PDF or a generated
        `.cmake` inside `build/`, `node_modules` or a vendored tree is
        excluded on purpose whatever srclight can read, and reporting it
        would put the warning back on every result.
        """
        ext = path.suffix.lower()
        if ext not in code_extensions() and ext not in unreadable_document_extensions():
            return False
        return not _should_ignore(path, root, [p for p in patterns if p != f"*{ext}"])

    def _grammar_missing(self, lang: str) -> bool:
        """True when a source language has no usable tree-sitter grammar here.

        Indexing such a file writes a row and extracts nothing, so it would
        be reported as read while holding no searchable symbol — the same
        false completeness in another guise. Document languages are parsed
        by extractors, not grammars, so they never answer True.
        """
        if lang not in LANGUAGES:
            return False
        return self._get_parser(lang) is None

    def _effective_ignore_patterns(self) -> list[str]:
        """Ignore patterns minus those a declaration overrides.

        `--ext .cmake=cmake` has to reach files that `*.cmake` would hide,
        the way an installed extractor makes `*.pdf` stop applying —
        otherwise the declaration is a silent no-op in one walk and works in
        the other.
        """
        declared = {
            f"*{ext}" for ext, lang in self._ext_overrides.items() if lang != SKIP_LANGUAGE
        }
        if not declared:
            return self.config.ignore_patterns
        return [p for p in self.config.ignore_patterns if p not in declared]

    def _declared_unreadable(self, path: Path) -> bool:
        """True when this index was told to leave the extension unread.

        Scoped to the extension, so it never outranks a whole-filename rule:
        `--ext .txt=skip` must not drop every CMakeLists.txt from the index.
        """
        if detect_language_by_filename(path):
            return False
        return self._ext_overrides.get(path.suffix.lower()) == SKIP_LANGUAGE

    def _detect_language(self, path: Path) -> str | None:
        """Detect a file's language, honouring this index's declared extensions.

        A declaration is scoped to an extension, so it never outranks a rule
        keyed on the whole filename: `--ext .txt=markdown` must not turn
        every CMakeLists.txt in the tree into markdown.
        """
        by_name = detect_language_by_filename(path)
        if by_name:
            return by_name
        override = self._ext_overrides.get(path.suffix.lower())
        if override:
            return override
        return detect_language(path)

    def _get_parser(self, lang_name: str) -> Parser | None:
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        language = get_language(lang_name)
        if language is None:
            return None

        parser = Parser(language)
        self._parsers[lang_name] = parser
        return parser

    def _get_query(self, lang_name: str) -> Query | None:
        if lang_name in self._queries:
            return self._queries[lang_name]

        language = get_language(lang_name)
        config = LANGUAGES.get(lang_name)
        if language is None or config is None:
            return None

        try:
            query = Query(language, config.symbol_query)
            self._queries[lang_name] = query
            return query
        except Exception as e:
            logger.warning("Failed to compile query for %s: %s", lang_name, e)
            return None

    def index(
        self,
        root: Path | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> IndexStats:
        """Index a codebase. Returns statistics.

        `on_progress(label, current, total)` follows the file scan, then the
        call graph under the label "call graph". `on_phase(name)` announces
        each step after the scan, which can take minutes on a large project.
        """
        root = root or self.config.root
        root = root.resolve()
        # Read by _build_embeddings, whose signature stays that of the hook.
        self._on_phase = on_phase
        stats = IndexStats()
        start = time.monotonic()

        logger.info("Indexing %s", root)

        self._ext_overrides = self._resolve_extension_overrides()
        ignore_patterns = self._effective_ignore_patterns()

        # Try to use git ls-files for .gitignore-aware file listing
        git_files = _git_tracked_files(root)
        use_git = git_files is not None
        if use_git:
            logger.info("Using git ls-files (%d tracked files)", len(git_files))

        # Collect files to process
        # (path, language): detected once, during collection. Detecting again
        # in the processing loop reopened `.h` and `.inc` files and read them
        # as they were THEN, not as the bytes that were hashed.
        files_to_index: list[tuple[Path, str]] = []
        # Extensions walked past, so the index can say what it never read
        # instead of letting every answer imply it read everything.
        unindexed_exts: dict[str, int] = {}
        # Files srclight could read but refused on size — its own limit, not
        # the project's choice, so it belongs in the gap report too.
        oversize_skipped = 0
        failed_files = 0
        if use_git:
            for rel in sorted(git_files):
                path = root / rel
                if not path.is_file():
                    continue

                # git ls-files has already applied .gitignore, so this branch
                # indexes what git tracks. The ignore patterns still decide
                # what counts as a GAP: a tracked font or a vendored tree is
                # excluded on purpose, and reporting it would leave the tally
                # non-empty on every real repo — burying the extensions that
                # are genuinely missing.
                skipped = self._declared_unreadable(path)
                if not skipped:
                    lang = self._detect_language(path)
                    is_doc = False
                    if lang is None:
                        lang = detect_document_language(path.suffix)
                        is_doc = True
                    skipped = lang is None
                if skipped:
                    if (not _should_ignore(path, root, ignore_patterns)
                            or self._ignored_only_by_its_own_extension(
                                path, root, ignore_patterns)):
                        _count_unindexed(unindexed_exts, path)
                    continue

                size_limit = self.config.max_doc_file_size if is_doc else self.config.max_file_size
                try:
                    if path.stat().st_size > size_limit:
                        stats.files_skipped += 1
                        oversize_skipped += 1
                        continue
                except OSError:
                    continue

                if self.config.languages and lang not in self.config.languages:
                    continue

                if self._grammar_missing(lang):
                    _count_unindexed(unindexed_exts, path)
                    continue

                files_to_index.append((path, lang))
                stats.files_scanned += 1
        else:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if _should_ignore(path, root, ignore_patterns):
                    # Excluded on purpose — a gap only if the sole reason is
                    # a pattern that is itself conditional on a missing
                    # extractor.
                    if self._ignored_only_by_its_own_extension(path, root, ignore_patterns):
                        _count_unindexed(unindexed_exts, path)
                    continue

                if self._declared_unreadable(path):
                    _count_unindexed(unindexed_exts, path)
                    continue

                lang = self._detect_language(path)
                is_doc = False
                if lang is None:
                    lang = detect_document_language(path.suffix)
                    is_doc = True
                if lang is None:
                    _count_unindexed(unindexed_exts, path)
                    continue

                size_limit = self.config.max_doc_file_size if is_doc else self.config.max_file_size
                try:
                    if path.stat().st_size > size_limit:
                        stats.files_skipped += 1
                        oversize_skipped += 1
                        continue
                except OSError:
                    # Build output and editor temp files vanish mid-walk. The
                    # git branch has always tolerated it; here it aborted the
                    # run, and the coverage record with it.
                    continue

                if self.config.languages and lang not in self.config.languages:
                    continue

                if self._grammar_missing(lang):
                    _count_unindexed(unindexed_exts, path)
                    continue

                files_to_index.append((path, lang))
                stats.files_scanned += 1

        # Track existing files for removal detection
        existing_paths = self.db.all_file_paths()
        indexed_paths: set[str] = set()

        # Process each file
        for i, (path, lang) in enumerate(files_to_index):
            rel_path = str(path.relative_to(root))
            indexed_paths.add(rel_path)

            if on_progress:
                on_progress(rel_path, i + 1, len(files_to_index))

            try:
                raw = path.read_bytes()
                file_hash = content_hash(raw)

                # Skip if unchanged
                if not self.db.file_needs_reindex(rel_path, file_hash):
                    stats.files_unchanged += 1
                    continue

                line_count = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)

                # Upsert file record
                file_rec = FileRecord(
                    path=rel_path,
                    content_hash=file_hash,
                    mtime=path.stat().st_mtime,
                    language=lang,
                    size=len(raw),
                    line_count=line_count,
                )
                file_id = self.db.upsert_file(file_rec)

                # Clear old symbols for this file, keeping the embeddings of
                # those that come back unchanged
                kept_embeddings = self.db.take_embeddings_for_file(file_id)
                self.db.delete_symbols_for_file(file_id)

                # Parse and extract symbols
                n_symbols = self._extract_symbols(file_id, rel_path, raw, lang)
                self.db.restore_embeddings_for_file(file_id, kept_embeddings)
                stats.symbols_extracted += n_symbols
                stats.files_indexed += 1

            except Exception as e:
                # Unread is unread, whatever the reason: a file that raised
                # here is absent from the index, and a coverage report that
                # omitted it would call the scan complete without it.
                logger.error("Error indexing %s: %s", path, e)
                stats.errors += 1
                failed_files += 1

        # Remove files that no longer exist
        for old_path in existing_paths - indexed_paths:
            file_rec = self.db.get_file(old_path)
            if file_rec and file_rec.id is not None:
                self.db.delete_file(file_rec.id)
                stats.files_removed += 1

        # Build call graph and inheritance edges (second pass)
        if stats.files_indexed > 0:
            if on_phase:
                on_phase("Building the call graph")
            phase_start = time.monotonic()
            stats.edges_created = self._build_edges(on_progress=on_progress)
            stats.edges_created += self._build_inheritance_edges()
            logger.info("Call graph: %d edges in %.0fs",
                        stats.edges_created, time.monotonic() - phase_start)

        # Community detection and execution flow tracing (post-edge phase)
        # Run if new edges were created OR if communities table is empty (first run after v5 migration)
        needs_communities = stats.edges_created > 0
        if not needs_communities:
            try:
                count = self.db.conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
                has_edges = self.db.conn.execute(
                    "SELECT 1 FROM symbol_edges WHERE edge_type = 'calls' LIMIT 1"
                ).fetchone()
                needs_communities = count == 0 and has_edges is not None
            except Exception:
                pass
        if needs_communities:
            try:
                from .community import detect_communities, trace_execution_flows
                if on_phase:
                    on_phase("Finding communities and execution flows")
                communities = detect_communities(self.db)
                if communities:
                    sym_to_comm = {}
                    for c in communities:
                        for m in c["members"]:
                            sym_to_comm[m["id"]] = c["id"]
                    flows = trace_execution_flows(self.db, sym_to_comm)
                    self.db.store_communities(communities)
                    self.db.store_execution_flows(flows)
                    logger.info(
                        "Detected %d communities, %d execution flows",
                        len(communities), len(flows),
                    )
            except ImportError:
                logger.debug("networkx not available — skipping community detection")
            except Exception:
                logger.warning("Community detection failed", exc_info=True)

        # Make the file pass durable BEFORE embedding. index() otherwise runs
        # as one transaction opened at the first file upsert, so the embedding
        # pass — minutes of HTTP calls — held the write lock the whole time and
        # took the parse work down with it if it failed. A second writer (the
        # git hook firing while the MCP server embeds) got 'database is locked'
        # and lost its own run: no busy_timeout is set.
        # The coverage record goes with it, for the same reason: an index
        # whose gaps changed would otherwise keep serving the old tally.
        # Every run walks the whole tree — the content-hash skip happens
        # later — so this replaces the previous record rather than adding to
        # it, and a gap that has been closed disappears.
        self.db.set_unindexed_extensions(unindexed_exts)
        self.db.set_oversize_skipped(oversize_skipped)
        self.db.set_failed_files(failed_files)

        self.db.commit()

        # Build embeddings (optional, only if a model is configured or known)
        embed_model = resolve_embed_model(self.db, self.config)
        if embed_model:
            if on_phase:
                on_phase("Embedding new and changed symbols")
            stats.symbols_embedded = self._build_embeddings(embed_model)
            if stats.symbols_embedded > 0:
                logger.info("Embedded %d symbols with %s", stats.symbols_embedded, embed_model)

        # Symbols moved and nothing was embedded: the sidecar now describes a
        # database that has changed, and symbols.id is a rowid reused after
        # deletion, so leaving it valid serves one symbol's score under
        # another symbol's identity. What matters is that the pass wrote
        # nothing — not why. A configured model whose provider is down, and a
        # reindex that only removed files, both land here.
        if (stats.files_indexed or stats.files_removed) and not stats.symbols_embedded:
            self._invalidate_sidecar()

        # Update index state
        git_head = _get_git_head(root)
        self.db.update_index_state(
            repo_root=str(root),
            last_commit=git_head,
            files_indexed=stats.files_scanned,
            symbols_indexed=stats.symbols_extracted,
            indexer_version=__version__,
        )

        self.db.commit()
        stats.elapsed_seconds = time.monotonic() - start

        logger.info(
            "Indexed %d files (%d symbols, %d edges) in %.2fs. %d unchanged, %d removed, %d errors.",
            stats.files_indexed, stats.symbols_extracted, stats.edges_created,
            stats.elapsed_seconds, stats.files_unchanged, stats.files_removed, stats.errors,
        )

        # Signal index completion via timestamp file
        try:
            signal_file = root / ".srclight" / "last-indexed"
            signal_file.parent.mkdir(parents=True, exist_ok=True)
            signal_file.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": stats.files_scanned,
                "symbols": stats.symbols_extracted,
                "commit": git_head,
                "elapsed_seconds": round(stats.elapsed_seconds, 2),
            }))
        except Exception:
            logger.debug("Failed to write index signal file", exc_info=True)

        # Fold the WAL into index.db. SQLite only checkpoints when the LAST
        # connection closes, and a running MCP server means ours never is — so
        # without this the whole index stays in index.db-wal and the main file
        # keeps a 4096-byte header. An index.db copied or backed up on its own
        # would then be an empty database (issue #16).
        self.db.checkpoint()

        return stats

    def _extract_symbols(
        self, file_id: int, rel_path: str, source: bytes, lang: str,
    ) -> int:
        """Parse a file and extract symbols. Returns count of symbols extracted."""
        if lang == "markdown":
            return self._extract_markdown_symbols(file_id, rel_path, source)

        # Document extractors
        doc_registry = get_registry()
        if lang in doc_registry:
            return doc_registry[lang].extract(file_id, rel_path, source, self.db)

        parser = self._get_parser(lang)
        query = self._get_query(lang)
        if parser is None or query is None:
            return 0

        tree = parser.parse(source)
        root = tree.root_node

        # First pass: collect all symbol info
        def collect(node: Node) -> list[tuple[Node, str, str | None]]:
            found: list[tuple[Node, str, str | None]] = []  # (def_node, kind, name)
            for _pattern_idx, match_captures in QueryCursor(query).matches(node):
                def_node = None
                symbol_name = None
                kind = "unknown"

                for capture_name, nodes in match_captures.items():
                    if capture_name.endswith(".def") and nodes:
                        def_node = nodes[0]
                        kind = _kind_from_capture(capture_name)
                    elif capture_name.endswith(".name") and nodes:
                        if _names_a_conversion(nodes[0]):
                            symbol_name = _operator_cast_name(nodes[0])
                        else:
                            symbol_name = nodes[0].text.decode("utf-8", errors="replace")

                if def_node is None:
                    continue

                # In C++ the same shape defines a free function or a method; where
                # it sits decides — a class body, a template one included.
                if lang == "cpp" and kind == "function" and _in_class_body(def_node):
                    kind = "method"

                # For templates without a name, extract from the inner declaration
                if symbol_name is None and kind == "template":
                    symbol_name = _extract_template_name(def_node)

                # Error recovery inserts MISSING nodes whose text is empty, and a
                # path left dangling by one — `M. = function() end` — ends on its
                # separator. An empty name is not NULL, so it would slip past every
                # IS NOT NULL filter and reach the name index.
                if symbol_name == "" or (symbol_name or "").endswith((".", ":")):
                    continue

                if lang == "lua" and _lua_nameless_definition(def_node):
                    continue

                # A keyword-named definition is error recovery's work — a statement
                # read as a definition — except for a macro, which may legally
                # redefine a keyword. A C keyword names nothing in either language
                # and is always dropped. A word only C++ reserves (`new`, `class`)
                # is a valid C name, and C headers are often read as C++: it is
                # dropped only as a function read inside another function's body
                # (the `catch` of a try chain cut by #if).
                if kind != "macro" and symbol_name in _RESERVED_NAMES.get(lang, ()) and (
                        symbol_name in _C_KEYWORDS
                        or _function_inside_a_function(def_node, kind)):
                    continue

                # `MACRO_TYPE(f32, s16) mField;` declares a variable whose type a
                # macro spells; the parser reads the macro as a function.
                if (lang in ("c", "cpp") and kind in ("function", "prototype")
                        and symbol_name and _macro_typed_declaration(def_node, symbol_name)):
                    continue

                found.append((def_node, kind, symbol_name))
            return found

        raw_symbols = collect(root)

        # A conditional that splits a brace across its branches breaks the
        # parse: the definitions it catches are lost, cut short, or run on
        # over the ones after them. Where the damage shows up is not reliable
        # — an ERROR node, MISSING nodes, or loose top-level fragments that
        # carry no error flag at all while the error surfaces elsewhere — so
        # it is not located. The file is parsed a second time with only the
        # first branch of each conditional, and a definition whose extent
        # that second parse reads soundly is trusted (see _extent_is_sound).
        #
        # It completes the original parse and never replaces it wholesale:
        # the reparse sees one branch, so a variant in an #else — one
        # definition per platform, an alternative macro — exists only in the
        # original. A definition both parses find keeps the reparse's extent
        # when the two differ, since that is the one the braces give. They
        # are the same definition when they share kind, name and start — or
        # kind, name and end: a header written once per branch over one
        # shared body starts the two symbols on different lines. A different
        # name is never the same definition. When the name itself differs per
        # branch, both names are real and both are kept; and an original that
        # ran on to the end of another definition must not be taken for it.
        recovered_nodes: set[int] = set()
        shared_bodies: dict[int, int] = {}  # id(def_node) -> end byte of the shared body
        if lang in _PREPROCESSED_LANGS and root.has_error and not _CONDITIONAL_RE.search(source):
            raw_symbols = _extend_to_braces(raw_symbols, source, lambda: _brace_view(source))
        elif (lang in _PREPROCESSED_LANGS and root.has_error
                and _CONDITIONAL_RE.search(source)):
            recovery_tree = parser.parse(_first_branch_only(source))
            recovered = [sym for sym in collect(recovery_tree.root_node)
                         if _extent_is_sound(sym[0])]
            by_start = {(k, n, node.start_byte): i for i, (node, k, n) in enumerate(recovered)}
            by_end = {(k, n, node.end_byte): i for i, (node, k, n) in enumerate(recovered)}
            by_span = {(n, node.start_byte, node.end_byte): i
                       for i, (node, _k, n) in enumerate(recovered)}
            # Proof that an original ran on over something: a function the reparse
            # reads at top level. A struct, an enum or a macro local to a function
            # sits in its tail in both parses and proves nothing.
            recovered_starts = sorted(node.start_byte for node, kind, _name in recovered
                                      if kind in _CALLABLE_KINDS_CPP and _at_top_level(node)
                                      and _looks_like_a_function(node))
            used: set[int] = set()
            merged = []
            for sym in raw_symbols:
                node, kind, name = sym
                for key, index in (((kind, name, node.start_byte), by_start),
                                   ((kind, name, node.end_byte), by_end)):
                    i = index.get(key)
                    if i is not None and i not in used:
                        used.add(i)
                        twin = recovered[i][0]
                        # The reparse keeps each conditional's FIRST branch,
                        # which is not always the live one (`#if 0`), and
                        # can close one brace more than the real code. So it
                        # may extend a definition the original parse cut
                        # short, but it shortens one only when the original
                        # demonstrably ran on: it never closed (tree-sitter
                        # made its closing brace up), or the part the reparse
                        # drops holds another function. A tail of mere
                        # statements after a real closing brace means the
                        # reparse ended too early.
                        if (twin.start_byte, twin.end_byte) != (node.start_byte, node.end_byte):
                            grows = (twin.start_byte <= node.start_byte
                                     and twin.end_byte >= node.end_byte)
                            first = bisect.bisect_left(recovered_starts, twin.end_byte)
                            swallowed = (first < len(recovered_starts)
                                         and recovered_starts[first] < node.end_byte)
                            if (grows or not _closes_for_real(node)
                                    or (swallowed and kind in _CALLABLE_KINDS_CPP)):
                                sym = recovered[i]
                                recovered_nodes.add(id(twin))
                        elif _under_error(node) and not _under_error(twin):
                            # Same extent, but the split elsewhere broke the
                            # class around it in the original parse: only
                            # the reparse knows the class it belongs to.
                            sym = recovered[i]
                            recovered_nodes.add(id(twin))
                        break
                else:
                    # One definition read as two kinds: a class head broken
                    # by a conditional (`class C #if X : public B #endif {`)
                    # leaves its members at file scope in the original parse
                    # — functions and prototypes — where the reparse reads
                    # the class and its methods. Same name, same extent: the
                    # same definition, named by the parse that read the class.
                    i = by_span.get((name, node.start_byte, node.end_byte))
                    if i is not None and i not in used:
                        used.add(i)
                        sym = recovered[i]
                        recovered_nodes.add(id(sym[0]))
                merged.append(sym)
            added = [sym for i, sym in enumerate(recovered) if i not in used]
            recovered_nodes.update(id(sym[0]) for sym in added)
            raw_symbols = merged + added
            # Extended once both parses are merged, so that no definition
            # either of them kept is covered.
            extended = _extend_to_braces(raw_symbols, source, lambda: _brace_view(source))
            recovered_nodes.update(id(new[0]) for old, new in zip(raw_symbols, extended)
                                   if new[0] is not old[0] and id(old[0]) in recovered_nodes)
            raw_symbols = extended

            # Symbols of one kind that end on the same byte under different
            # names, one of them from the reparse, name one body: the name
            # differs per branch, or an original ran on to the end of another
            # definition. Each one's text holds the other's name, which the
            # edge builder must not read as a call — so mark them here, where
            # it is known, rather than guess later.
            bodies: dict[tuple[str, int], list[tuple[Node, str, str | None]]] = {}
            for sym in raw_symbols:
                bodies.setdefault((sym[1], sym[0].end_byte), []).append(sym)
            shared_bodies = {
                id(sym[0]): end for (_kind, end), group in bodies.items()
                if len({s[2] for s in group}) > 1
                and any(id(s[0]) in recovered_nodes for s in group)
                for sym in group
            }
            # Containers before what they contain: the second pass finds a
            # parent among the symbols already inserted.
            raw_symbols.sort(key=lambda sym: (sym[0].start_byte, -sym[0].end_byte))

        # Second pass: insert symbols and track parent-child relationships
        # Track container symbols (classes, structs, namespaces) by their byte ranges
        container_kinds = {"class", "struct", "namespace", "impl", "module"}
        # Map (start_byte, end_byte) -> symbol_id for containers
        inserted: list[tuple[int, int, int, str | None]] = []  # (start, end, sym_id, kind)
        count = 0

        for def_node, kind, symbol_name in raw_symbols:
            # From the file, not from the node: a node recovered from the
            # reparse would otherwise store its body with the later branches
            # blanked out.
            body_bytes = source[def_node.start_byte:def_node.end_byte]
            content_text = body_bytes.decode("utf-8", errors="replace")
            doc = _extract_doc_comment(source, def_node)
            sig = _extract_signature(source, def_node, lang)
            if sig and id(def_node) in recovered_nodes:
                # Read from the reparse, where the directives and the other
                # branches left runs of blanks.
                sig = " ".join(sig.split())

            body_h = hashlib.sha256(body_bytes).hexdigest()[:16]

            # Find parent: look for the tightest container that encloses this symbol
            parent_id = None
            best_span = float("inf")
            for c_start, c_end, c_id, c_kind in inserted:
                if c_kind not in container_kinds:
                    continue
                if c_start < def_node.start_byte and def_node.end_byte <= c_end:
                    span = c_end - c_start
                    if span < best_span:
                        best_span = span
                        parent_id = c_id

            qualified = _build_qualified_name(symbol_name, def_node, lang)

            sym = SymbolRecord(
                file_id=file_id,
                kind=kind,
                name=symbol_name,
                qualified_name=qualified,
                signature=sig,
                start_line=def_node.start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                content=content_text,
                doc_comment=doc,
                body_hash=body_h,
                line_count=def_node.end_point[0] - def_node.start_point[0] + 1,
                parent_symbol_id=parent_id,
                metadata=({"shared_body": shared_bodies[id(def_node)]}
                          if id(def_node) in shared_bodies else None),
            )

            sym_id = self.db.insert_symbol(sym, rel_path)
            inserted.append((def_node.start_byte, def_node.end_byte, sym_id, kind))
            count += 1

        return count

    def _extract_markdown_symbols(
        self, file_id: int, rel_path: str, source: bytes,
    ) -> int:
        """Extract symbols from a Markdown file using heading-based sections.

        Each heading section becomes a symbol (kind='section'). A file with
        no headings becomes a single 'document' symbol. YAML frontmatter is
        stored as doc_comment on the first symbol.
        """
        parser = self._get_parser("markdown")
        if parser is None:
            return 0

        tree = parser.parse(source)
        root = tree.root_node
        file_stem = Path(rel_path).stem

        # Extract frontmatter if present
        frontmatter: str | None = None
        for child in root.children:
            if child.type == "minus_metadata":
                frontmatter = child.text.decode("utf-8", errors="replace").strip()
                break

        count = 0
        # Track inserted symbols for parent lookup: (start, end, sym_id)
        inserted: list[tuple[int, int, int]] = []

        def _get_own_content(section_node: Node) -> str:
            """Get text of all children except nested sections."""
            parts = []
            for child in section_node.children:
                if child.type != "section":
                    parts.append(source[child.start_byte:child.end_byte])
            return b"".join(parts).decode("utf-8", errors="replace").strip()

        def _get_heading_info(section_node: Node) -> tuple[str | None, str | None, int]:
            """Extract heading name, markdown signature, and level from a section.

            Returns (name, signature, level). Level is 0 if no heading found.
            """
            for child in section_node.children:
                if child.type == "atx_heading":
                    # Get inline text as name
                    inlines = [c for c in child.children if c.type == "inline"]
                    name = inlines[0].text.decode("utf-8", errors="replace").strip() if inlines else None
                    sig = child.text.decode("utf-8", errors="replace").strip()
                    # Determine level from marker (atx_h1_marker, atx_h2_marker, etc.)
                    markers = [c for c in child.children if c.type.startswith("atx_h")]
                    level = int(markers[0].type[5]) if markers else 0  # "atx_h2_marker" -> 2
                    return name, sig, level
            return None, None, 0

        def _walk_sections(
            node: Node, ancestry: list[str],
        ) -> None:
            nonlocal count

            for child in node.children:
                if child.type != "section":
                    continue

                name, sig, level = _get_heading_info(child)
                if name is None:
                    # Section without heading (rare) — skip
                    _walk_sections(child, ancestry)
                    continue

                own_content = _get_own_content(child)
                current_ancestry = ancestry + [name]
                qualified = file_stem + " > " + " > ".join(current_ancestry)

                # First paragraph (after heading) as doc_comment
                doc = None
                for sc in child.children:
                    if sc.type == "paragraph":
                        doc = sc.text.decode("utf-8", errors="replace").strip()
                        break

                # Attach frontmatter to the first symbol in the file
                is_first = count == 0
                if is_first and frontmatter:
                    doc = frontmatter + ("\n\n" + doc if doc else "")

                body_h = hashlib.sha256(own_content.encode("utf-8")).hexdigest()[:16]

                # Find parent: tightest enclosing section we've inserted
                parent_id = None
                best_span = float("inf")
                for c_start, c_end, c_id in inserted:
                    if c_start < child.start_byte and child.end_byte <= c_end:
                        span = c_end - c_start
                        if span < best_span:
                            best_span = span
                            parent_id = c_id

                sym = SymbolRecord(
                    file_id=file_id,
                    kind="section",
                    name=name,
                    qualified_name=qualified,
                    signature=sig,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    content=own_content,
                    doc_comment=doc,
                    body_hash=body_h,
                    line_count=child.end_point[0] - child.start_point[0] + 1,
                    parent_symbol_id=parent_id,
                )
                sym_id = self.db.insert_symbol(sym, rel_path)
                inserted.append((child.start_byte, child.end_byte, sym_id))
                count += 1

                # Recurse into child sections
                _walk_sections(child, current_ancestry)

        _walk_sections(root, [])

        # If no sections found, create a single document symbol for the whole file
        if count == 0:
            content_text = source.decode("utf-8", errors="replace").strip()
            body_h = hashlib.sha256(source).hexdigest()[:16]
            sym = SymbolRecord(
                file_id=file_id,
                kind="document",
                name=file_stem,
                qualified_name=file_stem,
                signature=None,
                start_line=1,
                end_line=root.end_point[0] + 1,
                content=content_text,
                doc_comment=frontmatter,
                body_hash=body_h,
                line_count=root.end_point[0] + 1,
                parent_symbol_id=None,
            )
            self.db.insert_symbol(sym, rel_path)
            count = 1

        return count

    def _build_edges(self, on_progress: Callable[[str, int, int], None] | None = None) -> int:
        """Build call graph edges by scanning symbol content for references.

        For each symbol, scan its body for references to other known symbol names.
        Creates "calls" edges with confidence scoring based on proximity.
        Returns the number of edges created.
        """
        assert self.db.conn is not None
        from .db import is_vendored_path

        # Clear all existing edges (full rebuild)
        self.db.conn.execute("DELETE FROM symbol_edges")

        # Build name -> [(symbol_id, file_path, kind)] lookup
        # Exclude markdown and document types — sections don't "call" anything
        # and scanning their prose would create noise with zero useful edges.
        excluded = _doc_languages()
        placeholders = ",".join("?" * len(excluded))
        rows = self.db.conn.execute(
            f"""SELECT s.id, s.name, s.kind, s.metadata, f.path as file_path
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL AND f.language NOT IN ({placeholders})""",
            list(excluded),
        ).fetchall()

        name_to_symbols: dict[str, list[dict]] = {}
        symbol_info: dict[int, dict] = {}
        for row in rows:
            name = row["name"]
            info = {"id": row["id"], "file": row["file_path"], "kind": row["kind"],
                    "body": _shared_body(row["metadata"])}
            symbol_info[row["id"]] = info
            if name not in name_to_symbols:
                name_to_symbols[name] = []
            name_to_symbols[name].append(info)

        # Filter out short/common names that would create noise
        MIN_NAME_LEN = 4
        NOISE_NAMES = {
            # Common short identifiers
            "get", "set", "run", "new", "end", "add", "put", "pop", "top",
            "map", "key", "val", "len", "str", "int", "err", "log", "max",
            "min", "abs", "all", "any", "for", "not", "and", "the",
            "def", "var", "let", "con", "ret", "gen", "ptr", "pos",
            # Common C/C++ names
            "init", "main", "next", "prev", "data", "size", "type", "name",
            "node", "list", "info", "item", "test", "self", "this", "true",
            "false", "none", "null", "void", "char", "bool", "auto",
            "file", "path", "text", "line", "args", "argv", "argc",
            "read", "open", "send", "recv", "copy", "move", "swap",
            "push", "find", "sort", "hash", "lock", "call", "bind",
            "from", "into", "with", "each", "then", "done", "fail",
            "pass", "skip", "stop", "wait", "save", "load",
            "value", "begin", "close", "clear", "reset", "write",
            "check", "parse", "print", "state", "count", "index",
            "start", "empty", "erase", "front", "apply",
            # Common variable names that create cross-file noise
            "result", "output", "input", "buffer", "config", "params",
            "status", "error", "offset", "length", "width", "height",
            "tensor", "image", "model", "layer", "batch", "channel",
            # Catch2/test framework internals
            "Clara", "Detail", "Catch", "Matchers",
        }

        # Only create edges TO meaningful symbol kinds (not prototypes/namespaces)
        EDGE_TARGET_KINDS = {"function", "method", "class", "struct", "enum", "interface", "template"}

        filtered_names = {
            name: syms for name, syms in name_to_symbols.items()
            if len(name) >= MIN_NAME_LEN and name not in NOISE_NAMES
        }

        # Skip names with too many symbols (ambiguous)
        MAX_SYMBOL_FANOUT = 10
        filtered_names = {
            name: syms for name, syms in filtered_names.items()
            if len(syms) <= MAX_SYMBOL_FANOUT
        }

        if not filtered_names:
            return 0
        match_names = build_name_matcher(set(filtered_names))

        def _dir_of(path: str) -> str:
            """Get directory component of a path."""
            idx = path.rfind("/")
            return path[:idx] if idx >= 0 else ""

        def _compute_confidence(source_file: str, target_file: str) -> float:
            """Score edge confidence by proximity."""
            if source_file == target_file:
                return 1.0
            s_vendored = is_vendored_path(source_file)
            t_vendored = is_vendored_path(target_file)
            # Cross vendored/project boundary = low confidence
            if s_vendored != t_vendored:
                return 0.2
            # Both vendored = skip entirely
            if s_vendored and t_vendored:
                return 0.1
            # Same directory
            if _dir_of(source_file) == _dir_of(target_file):
                return 0.9
            # Same top-level module (e.g., both under src/libcapture/)
            s_parts = source_file.split("/")[:3]
            t_parts = target_file.split("/")[:3]
            if s_parts == t_parts:
                return 0.7
            return 0.5

        # Scan each symbol's content for references
        edge_count = 0
        MAX_REFS_PER_SYMBOL = 30

        content_rows = self.db.conn.execute(
            f"""SELECT s.id, s.name, s.content, s.metadata, f.path as file_path, f.language
               FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL AND f.language NOT IN ({placeholders})""",
            list(excluded),
        ).fetchall()

        from .imports import extract_imports
        from .refmask import mask_noncode

        # Per-file import evidence (a boost/filter for the import tier, never a
        # resolver). Reads the file head from disk at index time; a file gone
        # missing simply never hits the import tier.
        file_imports: dict[str, set[str]] = {}

        def _imports_for(file_path: str, language: str | None) -> set[str]:
            if file_path not in file_imports:
                names: set[str] = set()
                try:
                    head = "\n".join(
                        (self.config.root / file_path).read_text(errors="ignore")
                        .splitlines()[:100]
                    )
                    for imp in extract_imports(head, language or ""):
                        mod = imp.get("module") or ""
                        if mod:
                            stem = mod.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
                            names.add(stem.rsplit(".", 1)[0] or stem)
                        for nm in imp.get("names") or []:
                            names.add(nm)
                except OSError:
                    pass
                file_imports[file_path] = names
            return file_imports[file_path]

        def _select_targets(targets: list[dict], source_file: str,
                            imported: set[str], ref_name: str) -> tuple[list[dict], str]:
            """Ranked, field-standard selection: prefer evidence,
            and when none discriminates, keep the ranked LIST as name_only —
            a labeled candidate list beats a fabricated winner."""
            same_file = [t for t in targets if t["file"] == source_file]
            if same_file:
                return same_file, "same_file"
            files = {t["file"] for t in targets}
            if len(files) == 1:
                return targets, "unique_file"
            if imported:
                imp = [t for t in targets
                       if t["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0] in imported]
                if imp and len({t["file"] for t in imp}) == 1:
                    return imp, "import"
            sdir = _dir_of(source_file)
            sd = [t for t in targets if _dir_of(t["file"]) == sdir]
            if sd and len({t["file"] for t in sd}) == 1:
                return sd, "same_dir"
            return targets, "name_only"

        for done, row in enumerate(content_rows, 1):
            if on_progress and (done % 500 == 0 or done == len(content_rows)):
                on_progress("call graph", done, len(content_rows))
            source_id = row["id"]
            source_name = row["name"]
            source_file = row["file_path"]
            # Mask comments/strings BEFORE scanning: a name that appears only in
            # prose is not a reference (12.8% of sampled edges were this class).
            content = mask_noncode(row["content"], row["language"] or "")

            referenced_names = match_names(content)
            referenced_names.discard(source_name)

            imported = _imports_for(source_file, row["language"])
            # Names the #if/#else recovery put over one shared body each hold
            # the other's name; the extractor marked them, and between them
            # that is not a call.
            body = _shared_body(row["metadata"])
            refs_for_this = 0
            for ref_name in referenced_names:
                if refs_for_this >= MAX_REFS_PER_SYMBOL:
                    break
                targets = [t for t in filtered_names.get(ref_name, [])
                           if t["id"] != source_id and t["kind"] in EDGE_TARGET_KINDS
                           and not (body is not None and t["body"] == body
                                    and t["file"] == source_file)]
                if not targets:
                    continue
                chosen, resolution = _select_targets(targets, source_file, imported, ref_name)
                for target in chosen:
                    confidence = _compute_confidence(source_file, target["file"])
                    # Skip very low confidence edges
                    if confidence < 0.2:
                        continue
                    self.db.insert_edge(EdgeRecord(
                        source_id=source_id,
                        target_id=target["id"],
                        edge_type="calls",
                        confidence=confidence,
                        resolution=resolution,
                    ))
                    edge_count += 1
                    refs_for_this += 1

        return edge_count

    def _build_inheritance_edges(self) -> int:
        """Build 'inherits' edges by parsing base class specifiers.

        Scans class/struct symbols for base class references in their content.
        For C++: "class Foo : public Bar" → Foo inherits Bar
        For Python: "class Foo(Bar)" → Foo inherits Bar
        Returns the number of edges created.
        """
        assert self.db.conn is not None

        # Get all class/struct symbols
        class_rows = self.db.conn.execute(
            """SELECT s.id, s.name, s.kind, s.content, f.language, f.path as file_path
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.kind IN ('class', 'struct') AND s.name IS NOT NULL"""
        ).fetchall()

        # Build name → symbol_id mapping for classes/structs only
        class_name_to_ids: dict[str, list[int]] = {}
        for row in class_rows:
            name = row["name"]
            if name not in class_name_to_ids:
                class_name_to_ids[name] = []
            class_name_to_ids[name].append(row["id"])

        # C++ base class pattern: "class Foo : public Bar, private Baz"
        # Also handles struct: "struct Foo : Bar"
        cpp_base_pattern = re.compile(
            r'(?:class|struct)\s+\w+\s*(?:<[^>]*>)?\s*:\s*'
            r'((?:(?:public|protected|private)\s+)?[\w:]+(?:\s*<[^>]*>)?'
            r'(?:\s*,\s*(?:(?:public|protected|private)\s+)?[\w:]+(?:\s*<[^>]*>)?)*)'
        )
        # Extract individual base class names
        cpp_base_name_pattern = re.compile(
            r'(?:public|protected|private)?\s*([\w]+)(?:::\w+)*(?:\s*<[^>]*>)?'
        )

        # Python base class pattern: "class Foo(Bar, Baz):"
        py_base_pattern = re.compile(r'class\s+\w+\s*\(([^)]+)\)')

        edge_count = 0
        for row in class_rows:
            symbol_id = row["id"]
            content = row["content"]
            lang = row["language"]

            base_names: list[str] = []

            if lang in ("cpp", "c"):
                match = cpp_base_pattern.search(content)
                if match:
                    bases_str = match.group(1)
                    for base_match in cpp_base_name_pattern.finditer(bases_str):
                        base_name = base_match.group(1)
                        if base_name and base_name not in ("public", "protected", "private"):
                            base_names.append(base_name)

            elif lang == "python":
                match = py_base_pattern.search(content)
                if match:
                    bases_str = match.group(1)
                    for base in bases_str.split(","):
                        base = base.strip()
                        # Remove keyword args like metaclass=...
                        if "=" in base:
                            continue
                        # Get just the name (strip module prefix)
                        parts = base.split(".")
                        base_names.append(parts[-1])

            # Create edges
            for base_name in base_names:
                target_ids = class_name_to_ids.get(base_name, [])
                for target_id in target_ids:
                    if target_id == symbol_id:
                        continue
                    self.db.insert_edge(EdgeRecord(
                        source_id=symbol_id,
                        target_id=target_id,
                        edge_type="inherits",
                    ))
                    edge_count += 1

        return edge_count

    def _invalidate_sidecar(self) -> None:
        """Mark the .npy sidecar stale, if this index has one to invalidate.

        Keyed on the sidecar's existence rather than on rows in
        symbol_embeddings: a reindex that removes every embedded file leaves
        that table empty while the sidecar still lists the deleted symbols.
        """
        from .vector_cache import VectorCache

        try:
            if VectorCache(self.config.root / ".srclight").sidecar_exists():
                self.db.bump_embedding_cache_version()
        except Exception:
            # Not a detail: a sidecar left valid over a changed index serves
            # one symbol's score under another symbol's identity, because
            # rowids are reused. Say so at a level people see.
            logger.warning("Could not invalidate the embedding sidecar", exc_info=True)

    def _build_embeddings(self, model_spec: str) -> int:
        """Generate embeddings for symbols that need them.

        Only embeds symbols missing embeddings or with changed body_hash.
        Uses the configured embedding provider (Ollama or Voyage).

        Returns the number of symbols embedded.
        """
        from .embeddings import embed_symbols, get_provider

        on_phase = getattr(self, "_on_phase", None)

        try:
            provider = get_provider(model_spec)
        except (ValueError, ConnectionError) as e:
            logger.warning("Cannot initialize embedding provider '%s': %s", model_spec, e)
            return 0

        # Get symbols needing embeddings
        symbols = self.db.get_symbols_needing_embeddings(provider.name)
        if not symbols:
            logger.debug("All symbols already embedded with %s", provider.name)
            return 0

        logger.info("Embedding %d symbols with %s...", len(symbols), provider.name)

        embed_start = time.monotonic()

        def _on_progress(batch_num: int, total: int) -> None:
            elapsed = time.monotonic() - embed_start
            rate = batch_num / elapsed if elapsed > 0 else 0
            remaining = (total - batch_num) / rate if rate > 0 else 0
            logger.info("  Embedding batch %d/%d (%.0fs elapsed, ~%.0fs remaining)",
                        batch_num, total, elapsed, remaining)

        # Nothing below may escape this method. Embedding is a best-effort
        # extra: the index itself is already committed, and every caller —
        # the CLI, the MCP reindex tool, and above all the git hooks, which
        # run flag-less on every commit — must survive a provider that is
        # down, slow, or serving a model that does not exist.
        try:
            results = embed_symbols(provider, symbols, on_progress=_on_progress)

            if not results:
                # embed_symbols swallows per-batch failures and returns [], so
                # an empty list means the provider is unreachable as often as
                # it means there was nothing to do. Stop here either way:
                # provider.dimensions would re-probe the network and raise.
                logger.warning("Embedded no symbols with %s — provider unreachable?",
                               provider.name)
                return 0

            # Remember what actually embedded, so the next flag-less run
            # continues with it. Recorded only on success: a typo'd model must
            # not become the index's choice, and a switch that failed part way
            # through must not be reverted by the old model's row count.
            self.db.remember_embedding_model(provider.name)

            # Store embeddings
            if on_phase:
                on_phase(f"Saving {len(results)} embeddings")
            dims = provider.dimensions
            body_hashes = {s["id"]: s["body_hash"] for s in symbols}
            for symbol_id, emb_bytes in results:
                self.db.upsert_embedding(symbol_id, provider.name, dims, emb_bytes,
                                         body_hashes.get(symbol_id))

            self.db.commit()
        except Exception as e:
            # exc_info: this catch also covers upsert/commit, so a programming
            # error must not be reported as one line reading like an outage.
            logger.error("Embedding failed, index left intact: %s", e, exc_info=True)
            try:
                self.db.rollback()
            except Exception:
                logger.debug("Rollback after embedding failure failed", exc_info=True)
            return 0

        # Build .npy sidecar for GPU-resident vector cache
        if results:
            try:
                from .vector_cache import VectorCache
                if on_phase:
                    on_phase("Rebuilding the vector cache")
                srclight_dir = self.config.root / ".srclight"
                cache = VectorCache(srclight_dir)
                cache.build_from_db(self.db.conn)
                logger.info("Embedding sidecar built: %d vectors", len(results))
            except Exception as e:
                logger.warning("Failed to build embedding sidecar: %s", e)

        return len(results)
