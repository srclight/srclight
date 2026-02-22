"""Tests for git intelligence layer."""

import subprocess
from pathlib import Path

import pytest

from srclight.git import (
    blame_lines,
    changes_to_file,
    hotspots,
    recent_changes,
    whats_changed,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with a few commits."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path), capture_output=True,
    )

    # First commit
    (tmp_path / "main.py").write_text("def hello():\n    print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add hello function"],
        cwd=str(tmp_path), capture_output=True,
    )

    # Second commit
    (tmp_path / "main.py").write_text(
        "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('bye')\n"
    )
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Update hello, add goodbye"],
        cwd=str(tmp_path), capture_output=True,
    )

    # Third commit — new file
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add utils"],
        cwd=str(tmp_path), capture_output=True,
    )

    return tmp_path


def test_recent_changes(git_repo):
    commits = recent_changes(git_repo, n=10)
    assert len(commits) == 3
    assert commits[0]["message"] == "Add utils"
    assert commits[0]["author"] == "Test User"
    assert "utils.py" in commits[0]["files"]


def test_recent_changes_with_author_filter(git_repo):
    commits = recent_changes(git_repo, n=10, author="Test User")
    assert len(commits) == 3
    commits = recent_changes(git_repo, n=10, author="Nobody")
    assert len(commits) == 0


def test_recent_changes_with_path_filter(git_repo):
    commits = recent_changes(git_repo, n=10, path_filter="utils.py")
    assert len(commits) == 1
    assert commits[0]["message"] == "Add utils"


def test_hotspots(git_repo):
    spots = hotspots(git_repo, n=5)
    assert len(spots) >= 1
    # main.py was changed in 2 commits, utils.py in 1
    assert spots[0]["file"] == "main.py"
    assert spots[0]["changes"] == 2


def test_whats_changed_clean(git_repo):
    result = whats_changed(git_repo)
    assert result["total_changes"] == 0


def test_whats_changed_with_modifications(git_repo):
    (git_repo / "main.py").write_text("def hello():\n    print('modified')\n")
    result = whats_changed(git_repo)
    assert result["total_changes"] >= 1
    files = [c["file"] for c in result["changes"]]
    assert "main.py" in files


def test_whats_changed_with_untracked(git_repo):
    (git_repo / "new_file.py").write_text("# new\n")
    result = whats_changed(git_repo)
    assert any(c["file"] == "new_file.py" and c["type"] == "untracked"
               for c in result["changes"])


def test_changes_to_file(git_repo):
    commits = changes_to_file(git_repo, "main.py", n=10)
    assert len(commits) == 2  # main.py was touched in commits 1 and 2
    assert commits[0]["message"] == "Update hello, add goodbye"


def test_blame_lines(git_repo):
    blames = blame_lines(git_repo, "main.py", 1, 5)
    assert len(blames) >= 1
    # At least one blame entry should have author info
    assert any(b.author for b in blames)
