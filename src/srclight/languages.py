"""Language detection and tree-sitter grammar loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tree_sitter import Language

# Lazy-loaded language modules
_LANGUAGES: dict[str, Language] = {}


@dataclass
class LanguageConfig:
    """Configuration for a supported language."""
    name: str
    extensions: tuple[str, ...]
    loader: str  # module path for tree-sitter grammar
    # tree-sitter query patterns for symbol extraction
    symbol_query: str


# Symbol extraction queries per language
# These use tree-sitter's S-expression query syntax

_PYTHON_QUERY = """
(function_definition
    name: (identifier) @fn.name) @fn.def

(class_definition
    name: (identifier) @cls.name) @cls.def

(decorated_definition
    definition: (function_definition
        name: (identifier) @dec_fn.name)) @dec_fn.def

(decorated_definition
    definition: (class_definition
        name: (identifier) @dec_cls.name)) @dec_cls.def
"""

_C_QUERY = """
(function_definition
    declarator: (function_declarator
        declarator: (identifier) @fn.name)) @fn.def

(struct_specifier
    name: (type_identifier) @struct.name) @struct.def

(enum_specifier
    name: (type_identifier) @enum.name) @enum.def

(declaration
    declarator: (function_declarator
        declarator: (identifier) @proto.name)) @proto.def

(type_definition
    declarator: (type_identifier) @typedef.name) @typedef.def

(preproc_function_def
    name: (identifier) @macro.name) @macro.def

(preproc_def
    name: (identifier) @define.name) @define.def
"""

_CPP_QUERY = """
(function_definition
    declarator: (function_declarator
        declarator: (identifier) @fn.name)) @fn.def

(function_definition
    declarator: (function_declarator
        declarator: (qualified_identifier) @method.name)) @method.def

(class_specifier
    name: (type_identifier) @cls.name) @cls.def

(struct_specifier
    name: (type_identifier) @struct.name) @struct.def

(enum_specifier
    name: (type_identifier) @enum.name) @enum.def

(namespace_definition
    name: (namespace_identifier) @ns.name) @ns.def

(template_declaration) @template.def

(declaration
    declarator: (function_declarator
        declarator: (identifier) @proto.name)) @proto.def

(declaration
    declarator: (function_declarator
        declarator: (qualified_identifier) @qproto.name)) @qproto.def

(field_declaration
    declarator: (function_declarator
        declarator: (field_identifier) @field_fn.name)) @field_fn.def
"""

_JS_QUERY = """
(function_declaration
    name: (identifier) @fn.name) @fn.def

(class_declaration
    name: (identifier) @cls.name) @cls.def

(method_definition
    name: (property_identifier) @method.name) @method.def

(lexical_declaration
    (variable_declarator
        name: (identifier) @var.name
        value: (arrow_function))) @var.def

(variable_declaration
    (variable_declarator
        name: (identifier) @var2.name
        value: (arrow_function))) @var2.def

(export_statement
    declaration: (function_declaration
        name: (identifier) @export_fn.name)) @export_fn.def

(export_statement
    declaration: (class_declaration
        name: (identifier) @export_cls.name)) @export_cls.def
"""

_TS_QUERY = """
(function_declaration
    name: (identifier) @fn.name) @fn.def

(class_declaration
    name: (type_identifier) @cls.name) @cls.def

(method_definition
    name: (property_identifier) @method.name) @method.def

(interface_declaration
    name: (type_identifier) @iface.name) @iface.def

(type_alias_declaration
    name: (type_identifier) @type.name) @type.def

(enum_declaration
    name: (identifier) @enum.name) @enum.def

(lexical_declaration
    (variable_declarator
        name: (identifier) @var.name
        value: (arrow_function))) @var.def

(export_statement
    declaration: (function_declaration
        name: (identifier) @export_fn.name)) @export_fn.def
"""

_RUST_QUERY = """
(function_item
    name: (identifier) @fn.name) @fn.def

(struct_item
    name: (type_identifier) @struct.name) @struct.def

(enum_item
    name: (type_identifier) @enum.name) @enum.def

(trait_item
    name: (type_identifier) @trait.name) @trait.def

(impl_item
    trait: (type_identifier)? @impl.trait
    type: (type_identifier) @impl.name) @impl.def

(mod_item
    name: (identifier) @mod.name) @mod.def

(type_item
    name: (type_identifier) @type.name) @type.def

(macro_definition
    name: (identifier) @macro.name) @macro.def
"""

_CSHARP_QUERY = """
(class_declaration
    name: (identifier) @cls.name) @cls.def

(interface_declaration
    name: (identifier) @iface.name) @iface.def

(struct_declaration
    name: (identifier) @struct.name) @struct.def

(enum_declaration
    name: (identifier) @enum.name) @enum.def

(method_declaration
    name: (identifier) @method.name) @method.def

(constructor_declaration
    name: (identifier) @ctor.name) @ctor.def

(namespace_declaration
    name: [(identifier) (qualified_name)] @ns.name) @ns.def

(property_declaration
    name: (identifier) @prop.name) @prop.def
"""


LANGUAGES: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        name="python",
        extensions=(".py", ".pyi"),
        loader="tree_sitter_python",
        symbol_query=_PYTHON_QUERY,
    ),
    "c": LanguageConfig(
        name="c",
        extensions=(".c", ".h"),
        loader="tree_sitter_c",
        symbol_query=_C_QUERY,
    ),
    "cpp": LanguageConfig(
        name="cpp",
        extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h", ".mm"),
        loader="tree_sitter_cpp",
        symbol_query=_CPP_QUERY,
    ),
    "javascript": LanguageConfig(
        name="javascript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        loader="tree_sitter_javascript",
        symbol_query=_JS_QUERY,
    ),
    "typescript": LanguageConfig(
        name="typescript",
        extensions=(".ts", ".tsx"),
        loader="tree_sitter_typescript",
        symbol_query=_TS_QUERY,
    ),
    "rust": LanguageConfig(
        name="rust",
        extensions=(".rs",),
        loader="tree_sitter_rust",
        symbol_query=_RUST_QUERY,
    ),
    "csharp": LanguageConfig(
        name="csharp",
        extensions=(".cs",),
        loader="tree_sitter_c_sharp",
        symbol_query=_CSHARP_QUERY,
    ),
}

# Extension to language mapping (handle .h ambiguity: default to C, override if cpp detected)
_EXT_TO_LANG: dict[str, str] = {}
for lang_name, config in LANGUAGES.items():
    for ext in config.extensions:
        # C++ gets priority for .h only if explicitly chosen; default to C
        if ext == ".h" and lang_name == "cpp":
            continue
        _EXT_TO_LANG.setdefault(ext, lang_name)


def detect_language(path: Path) -> str | None:
    """Detect language from file extension."""
    suffix = path.suffix.lower()
    lang = _EXT_TO_LANG.get(suffix)

    # Heuristic: .h files — check for C++ indicators
    if suffix == ".h" and lang == "c":
        try:
            content = path.read_text(errors="replace")[:4096]
            cpp_indicators = ("class ", "namespace ", "template", "::", "std::")
            if any(ind in content for ind in cpp_indicators):
                return "cpp"
        except OSError:
            pass

    return lang


def get_language(name: str) -> Language | None:
    """Get a tree-sitter Language object by name. Lazy-loads the grammar."""
    if name in _LANGUAGES:
        return _LANGUAGES[name]

    config = LANGUAGES.get(name)
    if config is None:
        return None

    try:
        import importlib
        mod = importlib.import_module(config.loader)
        if name == "typescript":
            lang = Language(mod.language_typescript())
        elif hasattr(mod, "language"):
            lang = Language(mod.language())
        else:
            return None
        _LANGUAGES[name] = lang
        return lang
    except (ImportError, AttributeError):
        return None


def get_tsx_language() -> Language | None:
    """Get TSX language (separate from TypeScript)."""
    if "tsx" in _LANGUAGES:
        return _LANGUAGES["tsx"]
    try:
        import tree_sitter_typescript as tsts
        lang = Language(tsts.language_tsx())
        _LANGUAGES["tsx"] = lang
        return lang
    except (ImportError, AttributeError):
        return None
