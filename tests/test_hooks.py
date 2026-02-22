"""Tests for git hook install/uninstall."""

import os
import stat
from pathlib import Path

import pytest

from srclight.cli import (
    _HOOK_MARKER_END,
    _HOOK_MARKER_START,
    _install_hooks_in_repo,
    _uninstall_hooks_in_repo,
)


@pytest.fixture
def fake_repo(tmp_path):
    """Create a fake git repo with .git/hooks dir."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir()
    return tmp_path


def test_install_creates_both_hooks(fake_repo):
    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "OK" in result
    assert "post-commit" in result
    assert "post-checkout" in result

    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pco = fake_repo / ".git" / "hooks" / "post-checkout"
    assert pc.exists()
    assert pco.exists()

    # Both should be executable
    assert pc.stat().st_mode & stat.S_IXUSR
    assert pco.stat().st_mode & stat.S_IXUSR

    # Both should have shebang
    assert pc.read_text().startswith("#!/bin/sh")
    assert pco.read_text().startswith("#!/bin/sh")

    # Both should have markers
    assert _HOOK_MARKER_START in pc.read_text()
    assert _HOOK_MARKER_START in pco.read_text()

    # post-checkout should check $3 for branch checkout
    assert '"$3" = "1"' in pco.read_text()


def test_install_idempotent(fake_repo):
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "SKIP" in result
    assert "already installed" in result


def test_install_preserves_existing_hooks(fake_repo):
    # Write existing post-commit hook
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text("#!/bin/sh\necho 'existing hook'\n")
    pc.chmod(0o755)

    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "OK" in result

    content = pc.read_text()
    assert "existing hook" in content
    assert _HOOK_MARKER_START in content


def test_uninstall_removes_both_hooks(fake_repo):
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    result = _uninstall_hooks_in_repo(fake_repo)
    assert "OK" in result
    assert "post-commit" in result
    assert "post-checkout" in result

    # Files should be gone (no other content)
    assert not (fake_repo / ".git" / "hooks" / "post-commit").exists()
    assert not (fake_repo / ".git" / "hooks" / "post-checkout").exists()


def test_uninstall_preserves_existing_hooks(fake_repo):
    # Write existing post-commit hook
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text("#!/bin/sh\necho 'existing hook'\n")
    pc.chmod(0o755)

    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    _uninstall_hooks_in_repo(fake_repo)

    # post-commit should still exist with original content
    assert pc.exists()
    content = pc.read_text()
    assert "existing hook" in content
    assert _HOOK_MARKER_START not in content

    # post-checkout had no prior content, should be deleted
    assert not (fake_repo / ".git" / "hooks" / "post-checkout").exists()


def test_uninstall_noop_when_no_hooks(fake_repo):
    result = _uninstall_hooks_in_repo(fake_repo)
    assert "SKIP" in result
    assert "no srclight hooks" in result


def test_skip_non_git_dir(tmp_path):
    result = _install_hooks_in_repo(tmp_path, "/usr/bin/srclight")
    assert "SKIP" in result
    assert "not a git repo" in result


def test_srclight_dir_created(fake_repo):
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert (fake_repo / ".srclight").is_dir()


def test_post_checkout_only_on_branch_switch(fake_repo):
    """post-checkout hook should guard on $3=1 and $1!=$2."""
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    content = (fake_repo / ".git" / "hooks" / "post-checkout").read_text()
    assert '"$3" = "1"' in content
    assert '"$1" != "$2"' in content


def test_hooks_exit_zero(fake_repo):
    """Hooks must exit 0 to avoid breaking git-flow and other tools."""
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    for name in ("post-commit", "post-checkout"):
        content = (fake_repo / ".git" / "hooks" / name).read_text()
        assert "exit 0" in content, f"{name} hook missing 'exit 0'"


def test_hooks_use_flock(fake_repo):
    """Hooks must use flock to prevent concurrent reindex processes."""
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    for name in ("post-commit", "post-checkout"):
        content = (fake_repo / ".git" / "hooks" / name).read_text()
        assert "flock -n" in content, f"{name} hook missing 'flock -n'"
