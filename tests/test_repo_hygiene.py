"""Guards against committed artifacts that would break clones or leak local paths."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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
