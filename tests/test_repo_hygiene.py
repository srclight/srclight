"""Guards against committed artifacts that would break clones or leak local paths."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_language_count_matches_the_registry():
    """The count drifted twice, and a stale one undersells the tool.

    Both the feature bullet and the comparison table quote a number that
    only LANGUAGES knows.
    """
    from srclight.languages import LANGUAGES

    readme = _readme()
    expected = len(LANGUAGES)

    bullet = re.search(r"\*\*(\d+) languages\*\*", readme)
    assert bullet and int(bullet.group(1)) == expected, (
        f"README feature bullet says {bullet.group(1) if bullet else '?'} "
        f"languages, LANGUAGES has {expected}"
    )
    row = re.search(r"^\| Languages \| (\d+) \|", readme, re.MULTILINE)
    assert row and int(row.group(1)) == expected, (
        f"README comparison table says {row.group(1) if row else '?'} "
        f"languages, LANGUAGES has {expected}"
    )


def test_readme_lists_every_registered_tool():
    """A tool absent from the table is a tool nobody knows to call.

    Nine were missing when this was written, which is also how the count
    above it drifted: nothing tied either to the registry.
    """
    from srclight.server import mcp

    readme = _readme()
    start = readme.index("## MCP Tools (")
    section = readme[start:readme.index("\n## ", start + 10)]

    registered = {t.name for t in asyncio.run(mcp.list_tools())}
    listed = set(re.findall(r"`([a-z_]+)\(", section))

    assert not registered - listed, (
        "tools missing from the README table: "
        + ", ".join(sorted(registered - listed))
    )


def test_readme_tool_count_matches_the_registry():
    """Same drift, on the number an agent reads to know what it can call."""
    from srclight.server import mcp

    readme = _readme()
    expected = len(asyncio.run(mcp.list_tools()))

    heading = re.search(r"^## MCP Tools \((\d+)\)", readme, re.MULTILINE)
    assert heading and int(heading.group(1)) == expected, (
        f"README heading says {heading.group(1) if heading else '?'} tools, "
        f"the registry has {expected}"
    )
    prose = re.search(r"exposes (\d+) MCP tools", readme)
    assert prose and int(prose.group(1)) == expected, (
        f"README prose says {prose.group(1) if prose else '?'} tools, "
        f"the registry has {expected}"
    )
    row = re.search(r"^\| MCP tools \| (\d+) \|", readme, re.MULTILINE)
    assert row and int(row.group(1)) == expected, (
        f"README comparison table says {row.group(1) if row else '?'} tools, "
        f"the registry has {expected}"
    )


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="Not a git checkout (e.g. installed from an sdist tarball)",
)
def test_no_tracked_absolute_path_symlinks():
    """No tracked entry may be a symlink whose target is an absolute path.

    An absolute-path symlink dangles on every clone and in every .zip/.tar.gz
    source archive, and typically leaks the author's local directory layout.
    See issue #21 (CLAUDE.md was committed as a symlink to /mnt/c/Users/...).
    """
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    offenders = []
    for line in result.stdout.splitlines():
        mode, sha, _stage_and_path = line.split(" ", 2)
        if mode != "120000":
            continue
        path = _stage_and_path.split("\t", 1)[1]
        target = subprocess.run(
            ["git", "cat-file", "-p", sha],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if target.startswith("/") or (len(target) > 1 and target[1] == ":"):
            offenders.append((path, target))

    assert not offenders, (
        "Tracked absolute-path symlinks found (will dangle on every clone):\n"
        + "\n".join(f"  {p} -> {t}" for p, t in offenders)
    )


# SHA-256 of private project names that must not appear in public files. Stored
# as digests so this public test does not itself publish the names it guards.
_PRIVATE_NAME_DIGESTS = frozenset({
    "673a605c07170e3658d3a0ad65e6906ab69d29a32a298bc4948c45b2bb9fafa0",
    "de3b36a9bec2cba05045f708b07fb4f3df42b2b251148ef6e3a550dbee38afcd",
})
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _word_candidates(word: str):
    """The word, and every run of its `-`/`_`-separated parts.

    `-` and `_` are word characters here, so a guarded name glued to a suffix
    (`name-mcp`, `name_db`, `two-part-name-x`) is a different word and would
    hash differently. Each contiguous run is tried with either separator.
    """
    yield word
    parts = [p for p in re.split(r"[-_]", word) if p]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts) + 1):
            run = parts[i:j]
            yield "-".join(run)
            yield "_".join(run)


def _names_private_project(line: str) -> bool:
    import hashlib
    return any(
        hashlib.sha256(candidate.encode()).hexdigest() in _PRIVATE_NAME_DIGESTS
        for word in _WORD.findall(line.lower())
        for candidate in _word_candidates(word)
    )


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="Not a git checkout (e.g. installed from an sdist tarball)",
)
def test_no_private_workstation_references_in_tracked_text():
    """This repository and its PyPI packages are public.

    Home-directory paths and private review bookkeeping (review-log entry ids
    and review-session names) reached shipped code comments before this guard
    existed. The rationale belongs in the comment; where it was decided does not.
    """
    patterns = [
        re.compile(r"/home/(?!you/|user/|nobody/|runner/)[a-z][a-z0-9_-]*/"),
        re.compile(r"/Users/(?!you/|user/)[A-Za-z][A-Za-z0-9_-]*/"),
        re.compile(r"\bgrain-\d{3,4}\b"),
        re.compile(r"\bcouncil [0-9a-f]{8}\b"),
        re.compile(r"\bpack review\b", re.IGNORECASE),
        # user@host on a private LAN (a build machine's address and login)
        re.compile(r"\b[a-z][a-z0-9_-]*@(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    ]
    files = subprocess.run(
        ["git", "ls-files", "--", "src", "tests", "docs", "scripts", "packaging",
         "README.md", "pyproject.toml", "glama.json", "server.json"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.split()
    offenders = []
    for rel in files:
        if rel == "tests/test_repo_hygiene.py" or rel.startswith("tests/fixtures/"):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if any(p.search(line) for p in patterns) or _names_private_project(line):
                offenders.append(f"  {rel}:{n}: {line.strip()[:120]}")
    assert not offenders, "Private references in tracked public files:\n" + "\n".join(offenders)


def test_private_name_matcher_sees_names_glued_to_suffixes(monkeypatch):
    """`-` and `_` are word characters, so a glued suffix changed the hash."""
    import hashlib

    digests = {hashlib.sha256(n.encode()).hexdigest() for n in ("lumen", "dim-star")}
    monkeypatch.setattr(sys.modules[__name__], "_PRIVATE_NAME_DIGESTS", frozenset(digests))

    for line in ("see lumen here", "lumen-mcp server", "the lumen_db file",
                 "a dim-star-pack", "x_dim_star_y", "Dim-Star"):
        assert _names_private_project(line), line
    for line in ("illumination", "dimstar", "dim star", "lumens"):
        assert not _names_private_project(line), line
