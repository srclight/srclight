#!/usr/bin/env python3
# scripts/measure_graph_precision.py
"""Measure the reference graph's real precision BEFORE investing in AST resolution.

Decision harness, not product code (canes-fideles grain-0395: the graph is a
name-match heuristic — measure how wrong it actually is before rebuilding it).
Reads an index.db, never writes. Reports:
  * ambiguity: fraction of edges whose target NAME maps to >1 symbol
  * comment/string-only references: sampled edges whose name occurrence in the
    source symbol's content is only in comments/strings (a false edge)
  * confidence distribution of edges (how much the <0.2 cross-file drop bites)
  * dead-code sample: symbols with no incoming edge, for human/pack spot-review
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3


def _strip_strings_and_comments(line: str) -> str:
    line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
    line = re.sub(r"'(?:[^'\\]|\\.)*'", "''", line)
    for marker in ("#", "//"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line


def classify_reference(content: str, name: str) -> str:
    """'code' | 'comment_or_string_only' | 'absent' — conservative toward 'code'."""
    word = re.compile(rf"\b{re.escape(name)}\b")
    if not word.search(content):
        return "absent"
    for line in content.split("\n"):
        if word.search(_strip_strings_and_comments(line)):
            return "code"
    return "comment_or_string_only"


def measure(db_path: str, sample: int = 500, seed: int = 42) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total_edges = conn.execute(
        "SELECT COUNT(*) c FROM symbol_edges WHERE edge_type='calls'").fetchone()["c"]

    ambiguous = conn.execute("""
        SELECT COUNT(*) c FROM symbol_edges e JOIN symbols t ON e.target_id = t.id
        WHERE e.edge_type='calls'
          AND t.name IN (SELECT name FROM symbols GROUP BY name HAVING COUNT(*) > 1)
    """).fetchone()["c"]

    conf_rows = conn.execute(
        "SELECT confidence, COUNT(*) c FROM symbol_edges WHERE edge_type='calls' "
        "GROUP BY confidence ORDER BY confidence").fetchall()

    edges = conn.execute("""
        SELECT e.id, s.content, t.name tname FROM symbol_edges e
        JOIN symbols s ON e.source_id = s.id JOIN symbols t ON e.target_id = t.id
        WHERE e.edge_type='calls' AND s.content IS NOT NULL
    """).fetchall()
    random.Random(seed).shuffle(edges)
    sampled = edges[:sample]
    counts = {"code": 0, "comment_or_string_only": 0, "absent": 0}
    false_examples = []
    for row in sampled:
        verdict = classify_reference(row["content"], row["tname"])
        counts[verdict] += 1
        if verdict != "code" and len(false_examples) < 10:
            false_examples.append({"edge_id": row["id"], "target": row["tname"], "verdict": verdict})

    dead_sample = [dict(r) for r in conn.execute("""
        SELECT s.name, s.kind, f.path FROM symbols s
        JOIN files f ON s.file_id = f.id
        LEFT JOIN symbol_edges e ON e.target_id = s.id
        WHERE e.id IS NULL AND s.kind IN ('function','method','class','struct','enum','interface')
        ORDER BY RANDOM() LIMIT 20
    """)]

    # How the edge builder resolved targets (0.20.5+; absent/None = pre-resolution DB)
    resolution_rows = conn.execute(
        "SELECT COALESCE(resolution,'(none)') r, COUNT(*) c FROM symbol_edges "
        "WHERE edge_type='calls' GROUP BY resolution ORDER BY c DESC").fetchall()

    n = max(len(sampled), 1)
    return {
        "db": db_path,
        "total_calls_edges": total_edges,
        "resolution_distribution": {r["r"]: r["c"] for r in resolution_rows},
        "ambiguous_target_name_rate": round(ambiguous / max(total_edges, 1), 4),
        "confidence_distribution": {str(r["confidence"]): r["c"] for r in conf_rows},
        "sampled": len(sampled),
        "sampled_verdicts": counts,
        "false_reference_rate_in_sample": round(
            (counts["comment_or_string_only"] + counts["absent"]) / n, 4),
        "false_reference_examples": false_examples,
        "dead_code_sample_for_review": dead_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db_path")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--out", help="write a markdown report here as well")
    a = ap.parse_args()
    report = measure(a.db_path, a.sample)
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            f.write("# Graph precision report\n\n```json\n"
                    + json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
