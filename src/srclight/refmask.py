# src/srclight/refmask.py
"""Mask comment/string spans so the edge builder only sees code.

WHY. 12.8% of sampled reference edges existed only because a symbol NAME sat in
a comment or string (measured, srclight self-index, 2026-08-30). ctags-lineage
tools never had this class — they tag AST nodes; Sourcegraph rejects
isString/isComment tokens at query time. Masking at BUILD time is the cheapest
point (grain-0399). Offsets are preserved (spaces, newlines kept) so any line
math downstream stays valid. Heuristic by design — a character scanner, not a
parser; multi-line raw-string exotica in non-python languages may over- or
under-mask a line, which the measure gate will show if it matters.
"""

from __future__ import annotations

__all__ = ["mask_noncode"]

_HASH_LANGS = {"python", "shell", "bash", "ruby", "yaml", "toml", "perl"}
_SLASH_LANGS = {"c", "cpp", "js", "javascript", "ts", "typescript", "java", "go",
                "rust", "dart", "swift", "csharp", "c_sharp", "kotlin", "scala", "php"}


def mask_noncode(content: str, language: str) -> str:
    lang = (language or "").lower()
    use_hash = lang in _HASH_LANGS or lang not in _SLASH_LANGS   # generic: allow # too
    use_slash = lang in _SLASH_LANGS or lang not in _HASH_LANGS  # generic: allow // too
    hash_is_directive = lang in ("c", "cpp")                     # keep #include lines

    out = list(content)
    i, n = 0, len(content)

    def blank(a: int, b: int) -> None:
        for j in range(a, b):
            if out[j] != "\n":
                out[j] = " "

    while i < n:
        ch = content[i]
        two = content[i:i + 2]
        # python triple-quoted strings
        if lang == "python" and content[i:i + 3] in ('"""', "'''"):
            q = content[i:i + 3]
            end = content.find(q, i + 3)
            end = n if end == -1 else end + 3
            blank(i, end)
            i = end
            continue
        if ch in ("'", '"'):
            j = i + 1
            while j < n and content[j] != ch:
                j += 2 if content[j] == "\\" else 1
            j = min(j + 1, n)
            blank(i, j)
            i = j
            continue
        if use_slash and two == "//":
            j = content.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if use_slash and two == "/*":
            j = content.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
            continue
        if use_hash and ch == "#" and not (hash_is_directive and content[i:i + 8] == "#include"):
            if hash_is_directive:
                i += 1              # other preprocessor lines: leave them alone
                continue
            j = content.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        i += 1
    return "".join(out)
