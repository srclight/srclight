"""Git intelligence layer for Srclight.

Provides change-aware context by parsing git blame, log, and diff output,
then correlating with indexed symbols.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("srclight.git")


def _run_git(repo_root: Path, *args: str, timeout: int = 30) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _run_git_lines(repo_root: Path, *args: str, timeout: int = 30) -> list[str]:
    """Run a git command and return non-empty stdout lines."""
    out = _run_git(repo_root, *args, timeout=timeout)
    return [l for l in out.splitlines() if l.strip()]


@dataclass
class BlameInfo:
    """Blame info for a line range."""
    commit: str
    author: str
    date: str
    message: str


@dataclass
class CommitInfo:
    """A git commit with parsed fields."""
    sha: str
    author: str
    date: str
    message: str
    files_changed: list[str]
    insertions: int = 0
    deletions: int = 0


def blame_lines(repo_root: Path, file_path: str, start_line: int, end_line: int) -> list[BlameInfo]:
    """Get blame info for a range of lines in a file.

    Returns one BlameInfo per unique commit touching those lines.
    """
    lines = _run_git_lines(
        repo_root, "blame", "--porcelain",
        f"-L{start_line},{end_line}",
        "--", file_path,
    )

    commits: dict[str, BlameInfo] = {}
    current_sha = None

    for line in lines:
        if line and len(line) >= 40 and line[0] != '\t' and ' ' in line:
            parts = line.split()
            if len(parts[0]) == 40:
                current_sha = parts[0]
                if current_sha not in commits:
                    commits[current_sha] = BlameInfo(
                        commit=current_sha, author="", date="", message=""
                    )
        if current_sha and current_sha in commits:
            if line.startswith("author "):
                commits[current_sha].author = line[7:]
            elif line.startswith("author-time "):
                # Unix timestamp — convert to ISO
                try:
                    import datetime
                    ts = int(line[12:])
                    commits[current_sha].date = datetime.datetime.fromtimestamp(
                        ts, tz=datetime.timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, OSError):
                    pass
            elif line.startswith("summary "):
                commits[current_sha].message = line[8:]

    return list(commits.values())


def blame_symbol(repo_root: Path, file_path: str, start_line: int, end_line: int) -> dict[str, Any]:
    """Get blame summary for a symbol's line range.

    Returns: last modifier, total unique commits, total unique authors,
    age in days, and the list of commits.
    """
    blames = blame_lines(repo_root, file_path, start_line, end_line)
    if not blames:
        return {"error": "No blame data available"}

    # Sort by date descending
    blames.sort(key=lambda b: b.date, reverse=True)
    last = blames[0]

    # Age in days
    age_days = None
    if last.date:
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(last.date.replace("Z", "+00:00"))
            age_days = (datetime.datetime.now(datetime.timezone.utc) - dt).days
        except (ValueError, OSError):
            pass

    unique_authors = list({b.author for b in blames if b.author})

    return {
        "last_modified": {
            "commit": last.commit[:12],
            "author": last.author,
            "date": last.date,
            "message": last.message,
        },
        "total_commits": len(blames),
        "total_authors": len(unique_authors),
        "authors": unique_authors,
        "age_days": age_days,
        "commits": [
            {
                "commit": b.commit[:12],
                "author": b.author,
                "date": b.date,
                "message": b.message,
            }
            for b in blames
        ],
    }


def recent_changes(
    repo_root: Path, n: int = 20, author: str | None = None,
    path_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent commits with files changed.

    Args:
        repo_root: Repository root path
        n: Number of commits to return
        author: Filter by author name (substring match)
        path_filter: Filter by file path (prefix match)
    """
    args = [
        "log", f"-{n}", "--format=%H|%an|%aI|%s",
        "--name-only",
    ]
    if author:
        args.extend(["--author", author])
    if path_filter:
        args.extend(["--", path_filter])

    output = _run_git(repo_root, *args)
    if not output:
        return []

    commits = []
    current: dict | None = None

    for line in output.splitlines():
        if "|" in line and len(line.split("|")[0]) == 40:
            if current:
                commits.append(current)
            parts = line.split("|", 3)
            current = {
                "commit": parts[0][:12],
                "author": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else "",
                "message": parts[3] if len(parts) > 3 else "",
                "files": [],
            }
        elif current and line.strip():
            current["files"].append(line.strip())

    if current:
        commits.append(current)

    return commits


def hotspots(
    repo_root: Path, n: int = 20, since: str | None = None,
) -> list[dict[str, Any]]:
    """Find most frequently changed files (churn hotspots).

    Args:
        repo_root: Repository root path
        n: Number of files to return
        since: Time period (e.g., '30.days', '3.months', '1.year')
    """
    args = ["log", "--format=", "--name-only"]
    if since:
        args.append(f"--since={since}")

    output = _run_git(repo_root, *args, timeout=60)
    if not output:
        return []

    file_counts: dict[str, int] = {}
    for line in output.splitlines():
        line = line.strip()
        if line:
            file_counts[line] = file_counts.get(line, 0) + 1

    # Sort by frequency
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)

    return [
        {"file": path, "changes": count}
        for path, count in sorted_files[:n]
    ]


def whats_changed(repo_root: Path) -> dict[str, Any]:
    """Get uncommitted changes (staged + unstaged).

    Returns modified, added, deleted files with change type.
    """
    # Staged changes
    staged = _run_git_lines(repo_root, "diff", "--cached", "--name-status")
    # Unstaged changes
    unstaged = _run_git_lines(repo_root, "diff", "--name-status")
    # Untracked files
    untracked = _run_git_lines(repo_root, "ls-files", "--others", "--exclude-standard")

    changes = []
    seen = set()

    for line in staged:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            status, path = parts
            change_type = {"M": "modified", "A": "added", "D": "deleted"}.get(
                status[0], "modified"
            )
            changes.append({"file": path, "type": change_type, "staged": True})
            seen.add(path)

    for line in unstaged:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            status, path = parts
            if path not in seen:
                change_type = {"M": "modified", "A": "added", "D": "deleted"}.get(
                    status[0], "modified"
                )
                changes.append({"file": path, "type": change_type, "staged": False})
                seen.add(path)

    for path in untracked:
        if path not in seen:
            changes.append({"file": path, "type": "untracked", "staged": False})

    return {
        "total_changes": len(changes),
        "changes": changes,
    }


def changes_to_file(
    repo_root: Path, file_path: str, n: int = 20,
) -> list[dict[str, Any]]:
    """Get commit history for a specific file.

    Args:
        repo_root: Repository root path
        file_path: Relative file path
        n: Number of commits to return
    """
    args = [
        "log", f"-{n}", "--format=%H|%an|%aI|%s",
        "--", file_path,
    ]

    output = _run_git(repo_root, *args)
    if not output:
        return []

    commits = []
    for line in output.splitlines():
        if "|" in line and len(line.split("|")[0]) == 40:
            parts = line.split("|", 3)
            commits.append({
                "commit": parts[0][:12],
                "author": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else "",
                "message": parts[3] if len(parts) > 3 else "",
            })

    return commits
