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
# `--` comments. Lua needs its own tier because `#` measures a table there and
# `//` divides, so the generic one — which allows both — blanks live code.
_DASH_LANGS = {"lua"}


def _long_bracket_end(content: str, i: int) -> int | None:
    """End offset of the Lua long bracket opening at `i` (`[[`, `[=[`, ...).

    Returns None when no bracket opens there — `t[1]` must stay code.
    """
    if i >= len(content) or content[i] != "[":
        return None
    j = i + 1
    while j < len(content) and content[j] == "=":
        j += 1
    if j >= len(content) or content[j] != "[":
        return None
    close = "]" + "=" * (j - i - 1) + "]"
    end = content.find(close, j + 1)
    return len(content) if end == -1 else end + len(close)


def mask_noncode(content: str, language: str, *, mask_strings: bool = True) -> str:
    """Blank comments and strings, keeping every offset.

    Pass mask_strings=False to blank comments only. Strings are still scanned —
    a `--` inside one is not a comment — but their content survives, which is
    what a caller needs when the thing it looks for lives in a string. In that
    mode an unbalanced quote leaves the rest of the input unscanned, so it suits
    a language where one cannot occur outside a string, not prose.
    """
    lang = (language or "").lower()
    generic = lang not in _SLASH_LANGS and lang not in _HASH_LANGS and lang not in _DASH_LANGS
    use_hash = lang in _HASH_LANGS or generic                    # generic: allow # too
    use_slash = lang in _SLASH_LANGS or generic                  # generic: allow // too
    use_dash = lang in _DASH_LANGS
    # Long brackets are Lua's alone, and stay tied to the language rather than
    # to the `--` tier: elsewhere `[[` is a nested index, and reading it as a
    # string opener blanks the rest of the body when no `]]` ever follows.
    use_long_brackets = lang == "lua"
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
            if mask_strings:
                blank(i, end)
            i = end
            continue
        # `-- to end of line`, or in lua `--[[ ... ]]` spanning lines
        if use_dash and two == "--":
            end = _long_bracket_end(content, i + 2) if use_long_brackets else None
            if end is None:
                end = content.find("\n", i)
                end = n if end == -1 else end
            blank(i, end)
            i = end
            continue
        # lua long strings: [[ ... ]], [=[ ... ]=]
        if use_long_brackets and ch == "[":
            end = _long_bracket_end(content, i)
            if end is not None:
                if mask_strings:
                    blank(i, end)
                i = end
                continue
        if ch in ("'", '"'):
            j = i + 1
            while j < n and content[j] != ch:
                j += 2 if content[j] == "\\" else 1
            j = min(j + 1, n)
            if mask_strings:
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
