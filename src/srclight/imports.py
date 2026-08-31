# src/srclight/imports.py
"""Regex-based import extraction, shared by the server tool and the indexer.

Lifted verbatim from server.py (2026-08-31) so the edge builder can use import
evidence without importing the server module — the indexer sits below the MCP
layer and must stay importable on its own.
"""

from __future__ import annotations

import re

__all__ = ["IMPORT_PATTERNS", "extract_imports"]

# Import extraction patterns by language (regex-based, not tree-sitter)
IMPORT_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)",
    ],
    "javascript": [
        r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "typescript": [
        r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "c": [r'#include\s*[<"]([^>"]+)[>"]'],
    "cpp": [r'#include\s*[<"]([^>"]+)[>"]'],
    "go": [r'"([^"]+)"'],
    "java": [r"^import\s+([\w.]+);"],
    "kotlin": [r"^import\s+([\w.]+)"],
    "dart": [r"import\s+['\"]([^'\"]+)['\"]"],
    "swift": [r"^import\s+(\w+)"],
    "csharp": [r"^using\s+([\w.]+);"],
    "php": [
        r"^use\s+([\w\\]+)",
        r"(?:require|include)(?:_once)?\s*['\"]([^'\"]+)['\"]",
    ],
}


def extract_imports(content: str, language: str) -> list[dict]:
    """Extract import statements from file content using regex patterns."""
    patterns = IMPORT_PATTERNS.get(language, [])
    if not patterns:
        return []

    imports = []
    seen_statements = set()

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") and language != "c" and language != "cpp":
            if not stripped.startswith("#include"):
                continue

        for pat in patterns:
            for m in re.finditer(pat, stripped):
                statement = stripped
                if statement in seen_statements:
                    continue
                seen_statements.add(statement)

                groups = [g for g in m.groups() if g is not None]
                if not groups:
                    continue

                if language == "python":
                    from_module = m.group(1)
                    import_names = m.group(2)
                    if from_module:
                        names = [n.strip() for n in import_names.split(",") if n.strip()]
                        imports.append({
                            "statement": statement,
                            "module": from_module,
                            "names": names,
                        })
                    else:
                        for name in import_names.split(","):
                            name = name.strip().split(" as ")[0].strip()
                            if name:
                                imports.append({
                                    "statement": statement,
                                    "module": name,
                                    "names": [],
                                })
                else:
                    module = groups[0]
                    imports.append({
                        "statement": statement,
                        "module": module,
                        "names": [],
                    })

    return imports
