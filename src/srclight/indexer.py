"""Tree-sitter based code indexer.

Walks a directory, parses files, extracts symbols, populates the database.
Incremental: only re-indexes files whose content hash has changed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tree_sitter import Language, Node, Parser, Query, QueryCursor

from . import __version__
from .db import Database, EdgeRecord, FileRecord, SymbolRecord, content_hash
from .extractors import DOCUMENT_EXTENSIONS, detect_document_language, get_registry
from .languages import (
    LANGUAGES,
    LanguageConfig,
    detect_language,
    get_language,
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
    elapsed_seconds: float = 0.0


@dataclass
class IndexConfig:
    root: Path = field(default_factory=Path)
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    max_file_size: int = MAX_FILE_SIZE
    max_doc_file_size: int = 50_000_000  # 50 MB for documents (PDF, DOCX, etc.)
    languages: list[str] | None = None  # None = all supported
    embed_model: str | None = None  # e.g. "qwen3-embedding", "voyage-code-3"


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
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return {line for line in result.stdout.splitlines() if line}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


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


def _extract_doc_comment(source_bytes: bytes, node: Node) -> str | None:
    """Extract doc comment preceding a symbol node."""
    # Look at the previous sibling for a comment
    prev = node.prev_named_sibling
    if prev is None:
        # Check for comment as first child or preceding line
        # Look at previous unnamed siblings too
        prev_sib = node.prev_sibling
        if prev_sib and prev_sib.type == "comment":
            return prev_sib.text.decode("utf-8", errors="replace").strip()
        return None

    if prev.type == "comment":
        return prev.text.decode("utf-8", errors="replace").strip()

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

    return None


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
        "trait": "trait",
        "impl": "impl",
        "template": "template",
        "field_fn": "method",  # method declarations in class bodies (headers)
        "var": "function",     # arrow functions
        "var2": "function",
        "ctor": "function",    # C# constructors
        "prop": "property",    # C# properties
        # Dart-specific
        "ext": "extension",
        "mixin": "mixin",
        "getter": "method",    # Dart getters
    }
    return mapping.get(prefix, "unknown")


def _get_enclosing_scope(node: Node) -> list[str]:
    """Walk up the AST to collect enclosing namespace/class/struct names.

    Returns a list like ["myapp", "util", "ConfigManager"] for a method
    defined inside namespace myapp { namespace util { class ConfigManager { ... } } }
    """
    scopes: list[str] = []
    current = node.parent
    while current is not None:
        if current.type in (
            "namespace_definition", "class_specifier", "struct_specifier",
            "class_definition",  # Python
            "class_declaration", "namespace_declaration",  # C#
        ):
            name_node = current.child_by_field_name("name")
            if name_node:
                scopes.append(name_node.text.decode("utf-8", errors="replace"))
        elif current.type == "template_declaration":
            # Look for named child inside the template
            for child in current.children:
                if child.type in ("class_specifier", "struct_specifier"):
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        scopes.append(name_node.text.decode("utf-8", errors="replace"))
                    break
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
    else:
        scopes = _get_enclosing_scope(node)
        if scopes:
            return ".".join(scopes + [symbol_name])
        return symbol_name


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
        elif child.type == "function_definition":
            declarator = child.child_by_field_name("declarator")
            if declarator:
                # Could be function_declarator -> identifier or qualified_identifier
                inner = declarator.child_by_field_name("declarator")
                if inner:
                    return inner.text.decode("utf-8", errors="replace")
                return declarator.text.decode("utf-8", errors="replace")
        elif child.type == "declaration":
            # Template variable or forward declaration
            declarator = child.child_by_field_name("declarator")
            if declarator:
                inner = declarator.child_by_field_name("declarator")
                if inner:
                    return inner.text.decode("utf-8", errors="replace")
                return declarator.text.decode("utf-8", errors="replace")
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

        # Remove ignore patterns for extensions that have active extractors
        active_exts = _active_doc_extensions()
        self.config.ignore_patterns = [
            p for p in self.config.ignore_patterns if p not in active_exts
        ]

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
    ) -> IndexStats:
        """Index a codebase. Returns statistics."""
        root = root or self.config.root
        root = root.resolve()
        stats = IndexStats()
        start = time.monotonic()

        logger.info("Indexing %s", root)

        # Try to use git ls-files for .gitignore-aware file listing
        git_files = _git_tracked_files(root)
        use_git = git_files is not None
        if use_git:
            logger.info("Using git ls-files (%d tracked files)", len(git_files))

        # Collect files to process
        files_to_index: list[Path] = []
        if use_git:
            for rel in sorted(git_files):
                path = root / rel
                if not path.is_file():
                    continue

                lang = detect_language(path)
                is_doc = False
                if lang is None:
                    lang = detect_document_language(path.suffix)
                    is_doc = True
                if lang is None:
                    continue

                size_limit = self.config.max_doc_file_size if is_doc else self.config.max_file_size
                try:
                    if path.stat().st_size > size_limit:
                        stats.files_skipped += 1
                        continue
                except OSError:
                    continue

                if self.config.languages and lang not in self.config.languages:
                    continue

                files_to_index.append(path)
                stats.files_scanned += 1
        else:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if _should_ignore(path, root, self.config.ignore_patterns):
                    continue

                lang = detect_language(path)
                is_doc = False
                if lang is None:
                    lang = detect_document_language(path.suffix)
                    is_doc = True
                if lang is None:
                    continue

                size_limit = self.config.max_doc_file_size if is_doc else self.config.max_file_size
                if path.stat().st_size > size_limit:
                    stats.files_skipped += 1
                    continue

                if self.config.languages and lang not in self.config.languages:
                    continue

                files_to_index.append(path)
                stats.files_scanned += 1

        # Track existing files for removal detection
        existing_paths = self.db.all_file_paths()
        indexed_paths: set[str] = set()

        # Process each file
        for i, path in enumerate(files_to_index):
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

                lang = detect_language(path)
                if lang is None:
                    lang = detect_document_language(path.suffix)
                if lang is None:
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

                # Clear old symbols for this file
                self.db.delete_symbols_for_file(file_id)

                # Parse and extract symbols
                n_symbols = self._extract_symbols(file_id, rel_path, raw, lang)
                stats.symbols_extracted += n_symbols
                stats.files_indexed += 1

            except Exception as e:
                logger.error("Error indexing %s: %s", path, e)
                stats.errors += 1

        # Remove files that no longer exist
        for old_path in existing_paths - indexed_paths:
            file_rec = self.db.get_file(old_path)
            if file_rec and file_rec.id is not None:
                self.db.delete_file(file_rec.id)
                stats.files_removed += 1

        # Build call graph and inheritance edges (second pass)
        if stats.files_indexed > 0:
            stats.edges_created = self._build_edges()
            stats.edges_created += self._build_inheritance_edges()

        # Build embeddings (optional, only if embed_model configured)
        if self.config.embed_model:
            n_embedded = self._build_embeddings(self.config.embed_model)
            if n_embedded > 0:
                logger.info("Embedded %d symbols with %s", n_embedded, self.config.embed_model)

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

        cursor = QueryCursor(query)
        matches = cursor.matches(root)

        # First pass: collect all symbol info
        raw_symbols: list[tuple[Node, str, str | None]] = []  # (def_node, kind, name)

        for _pattern_idx, match_captures in matches:
            def_node = None
            symbol_name = None
            kind = "unknown"

            for capture_name, nodes in match_captures.items():
                if capture_name.endswith(".def") and nodes:
                    def_node = nodes[0]
                    kind = _kind_from_capture(capture_name)
                elif capture_name.endswith(".name") and nodes:
                    symbol_name = nodes[0].text.decode("utf-8", errors="replace")

            if def_node is None:
                continue

            # For templates without a name, extract from the inner declaration
            if symbol_name is None and kind == "template":
                symbol_name = _extract_template_name(def_node)

            raw_symbols.append((def_node, kind, symbol_name))

        # Second pass: insert symbols and track parent-child relationships
        # Track container symbols (classes, structs, namespaces) by their byte ranges
        container_kinds = {"class", "struct", "namespace", "impl", "module"}
        # Map (start_byte, end_byte) -> symbol_id for containers
        inserted: list[tuple[int, int, int, str | None]] = []  # (start, end, sym_id, kind)
        count = 0

        for def_node, kind, symbol_name in raw_symbols:
            content_text = def_node.text.decode("utf-8", errors="replace")
            doc = _extract_doc_comment(source, def_node)
            sig = _extract_signature(source, def_node, lang)

            body_bytes = def_node.text
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

    def _build_edges(self) -> int:
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
            f"""SELECT s.id, s.name, s.kind, f.path as file_path
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL AND f.language NOT IN ({placeholders})""",
            list(excluded),
        ).fetchall()

        name_to_symbols: dict[str, list[dict]] = {}
        symbol_info: dict[int, dict] = {}
        for row in rows:
            name = row["name"]
            info = {"id": row["id"], "file": row["file_path"], "kind": row["kind"]}
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

        # Pre-compile regex
        sorted_names = sorted(filtered_names.keys(), key=len, reverse=True)
        if not sorted_names:
            return 0

        import re
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in sorted_names) + r")\b"
        )

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
            f"""SELECT s.id, s.name, s.content FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL AND f.language NOT IN ({placeholders})""",
            list(excluded),
        ).fetchall()

        for row in content_rows:
            source_id = row["id"]
            source_name = row["name"]
            content = row["content"]
            source_info = symbol_info.get(source_id)
            if not source_info:
                continue
            source_file = source_info["file"]

            referenced_names = set(pattern.findall(content))
            referenced_names.discard(source_name)

            refs_for_this = 0
            for ref_name in referenced_names:
                if refs_for_this >= MAX_REFS_PER_SYMBOL:
                    break
                targets = filtered_names.get(ref_name, [])
                for target in targets:
                    if target["id"] == source_id:
                        continue
                    # Only link to meaningful symbol kinds
                    if target["kind"] not in EDGE_TARGET_KINDS:
                        continue
                    confidence = _compute_confidence(source_file, target["file"])
                    # Skip very low confidence edges
                    if confidence < 0.2:
                        continue
                    self.db.insert_edge(EdgeRecord(
                        source_id=source_id,
                        target_id=target["id"],
                        edge_type="calls",
                        confidence=confidence,
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
        import re

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

    def _build_embeddings(self, model_spec: str) -> int:
        """Generate embeddings for symbols that need them.

        Only embeds symbols missing embeddings or with changed body_hash.
        Uses the configured embedding provider (Ollama or Voyage).

        Returns the number of symbols embedded.
        """
        from .embeddings import embed_symbols, get_provider

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

        try:
            results = embed_symbols(provider, symbols, on_progress=_on_progress)
        except ConnectionError as e:
            logger.error("Embedding failed: %s", e)
            return 0

        # Store embeddings
        dims = provider.dimensions
        for symbol_id, emb_bytes in results:
            # Find body_hash from the symbols list
            sym = next((s for s in symbols if s["id"] == symbol_id), None)
            body_hash = sym["body_hash"] if sym else None
            self.db.upsert_embedding(symbol_id, provider.name, dims, emb_bytes, body_hash)

        self.db.commit()

        # Build .npy sidecar for GPU-resident vector cache
        if results:
            try:
                from .vector_cache import VectorCache
                srclight_dir = self.config.root / ".srclight"
                cache = VectorCache(srclight_dir)
                cache.build_from_db(self.db.conn)
                logger.info("Embedding sidecar built: %d vectors", len(results))
            except Exception as e:
                logger.warning("Failed to build embedding sidecar: %s", e)

        return len(results)
