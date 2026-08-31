# Graph Disambiguation Implementation Plan (srclight 0.20.5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the reference graph's measured imprecision (62.5% ambiguous-name edges, 12.8% comment/string false edges) with the field-standard heuristic tier — build-time comment/string masking, ranked single-target selection with per-edge `resolution` labels — then re-measure and release only if the numbers actually moved.

**Architecture:** Two small pure modules (`refmask.py` for comment/string masking, `imports.py` lifted from server.py's `_extract_imports`) feed a rewritten candidate-selection block inside `Indexer._build_edges` (indexer.py:918). `symbol_edges` gains a `resolution` column (`same_file | unique_file | import | same_dir | name_only`) surfaced in `get_callers`/`get_callees`. The Part-B harness is extended to report tier distribution and gates the release.

**Tech Stack:** Python 3.11+, sqlite3, re, pytest. No new dependencies.

**Spec:** canes-fideles grain-0399 (pack verdict: field survey — Sourcegraph ships this exact playbook; GitHub archived stack-graphs and kept the heuristic tier) and the probe: 318 ambiguous pairs = 30.5% same-file + 39.0% unique-file + 30.5% truly ambiguous. Report baseline: `docs/graph-precision-report.md` (v0.20.4).

## Global Constraints

- Repo `~/repos/srclight/srclight`, branch `develop`. Tests: `~/repos/srclight/srclight/.venv/bin/python -m pytest`. Never `git add -A` (untracked WIP `src/srclight/learnings.py`, `tests/test_learnings.py` stay out). Commit per task, full suite green before each commit.
- Baseline: 255 passed, 1 skipped.
- MEASURE GATE (Task 5): after reindex, ambiguity must drop materially (expect ≈62% → ≤30%) and the comment/string false-reference rate to ≈0 in the harness sample. If not, STOP — commit the report, land a grain saying so, and do NOT release. Numbers gate the tag.
- Never resolve by coin flip: when candidates stay ambiguous, keep edges to all of them labeled `name_only` — a labeled candidate list beats a fabricated winner.
- Release (Task 6) follows the estate pattern: develop → master `merge --no-ff` → annotated tag `v0.20.5` → push all three. No GitHub Release (PyPI is Tim's).

## File Structure

- `src/srclight/refmask.py` — NEW. Pure: mask comment/string spans in code, preserving offsets.
- `src/srclight/imports.py` — NEW. `extract_imports(content, language)` lifted verbatim from server.py's `_extract_imports` + `IMPORT_PATTERNS` (server.py re-imports from here; indexer must not import server).
- `src/srclight/indexer.py` — MODIFY `_build_edges` (line ~918): mask content, ranked selection, resolution labels.
- `src/srclight/db.py` — MODIFY: `resolution` column on `symbol_edges` (+ migration), `EdgeRecord.resolution`, `insert_edge`, and include `resolution` in `get_callers`/`get_callees` row dicts.
- `scripts/measure_graph_precision.py` — MODIFY: add resolution-tier distribution to the report.
- `tests/test_refmask.py`, `tests/test_edge_resolution.py` — NEW.
- `README.md` — MODIFY: one paragraph in the graph/tools area about `resolution` labels.

---

### Task 1: refmask.py — comment/string masking

**Files:** Create `src/srclight/refmask.py`; Test `tests/test_refmask.py`.

**Interfaces:**
- Produces: `mask_noncode(content: str, language: str) -> str` — same length as input, newlines preserved, every character inside a comment or string literal replaced with a space. Handles: `#` tails (python/shell/ruby), `//` tails and `/* ... */` blocks (c/cpp/js/ts/java/go/rust/dart/swift), `'...'`/`"..."` single-line strings with `\\` escapes (all languages), `'''...'''`/`\"\"\"...\"\"\"` multi-line strings (python). Unknown language → apply the generic set (`#`, `//`, block, quotes): over-masking a comment-like tail in an exotic language is cheaper than keeping the measured false-edge class.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_refmask.py
"""Masking must remove names in comments/strings and NOTHING else, offset-stable."""
from srclight.refmask import mask_noncode


def test_python_hash_comment_masked():
    out = mask_noncode("x = helper()  # calls helper\n", "python")
    assert "helper()" in out                  # the code call survives
    assert out.count("helper") == 1           # the comment copy is gone
    assert len(out) == len("x = helper()  # calls helper\n")


def test_python_string_masked():
    out = mask_noncode('name = "helper"\n', "python")
    assert "helper" not in out


def test_python_docstring_masked_across_lines():
    src = 'def f():\n    """uses helper\n    a lot"""\n    helper()\n'
    out = mask_noncode(src, "python")
    assert out.count("helper") == 1           # only the real call remains
    assert out.count("\n") == src.count("\n")  # newlines preserved


def test_c_block_comment_masked():
    src = "int f() {\n/* helper does x */\nreturn helper();\n}\n"
    out = mask_noncode(src, "cpp")
    assert out.count("helper") == 1


def test_cpp_line_comment_masked():
    out = mask_noncode("int y = helper(); // helper again\n", "cpp")
    assert out.count("helper") == 1


def test_escaped_quote_does_not_derail():
    out = mask_noncode('s = "a \\" quote"; helper();\n', "cpp")
    assert "helper()" in out


def test_python_hash_not_treated_as_comment_in_c_include():
    out = mask_noncode('#include "helper.h"\nhelper();\n', "cpp")
    assert "#include" in out                  # preprocessor line survives masking
    assert out.count("helper") >= 1           # the call survives
```

- [x] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest tests/test_refmask.py -q` → `ModuleNotFoundError: srclight.refmask`

- [x] **Step 3: Implement**

```python
# src/srclight/refmask.py
"""Mask comment/string spans so the edge builder only sees code.

WHY. 12.8% of sampled reference edges existed only because a symbol NAME sat in
a comment or string (measured, srclight self-index, 2026-08-30). ctags-lineage
tools never had this class — they tag AST nodes; Sourcegraph rejects
isString/isComment tokens at query time. Masking at BUILD time is the cheapest
point (grain-0399). Offsets are preserved (spaces, newlines kept) so any line
math downstream stays valid. Heuristic by design — a character scanner, not a
parser; multi-line raw-string exotica in non-python languages may over- or
under-mask a line, which the measure gate (Task 5) will show if it matters.
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
```

- [x] **Step 4: Verify pass** — `.venv/bin/python -m pytest tests/test_refmask.py -q` → 7 passed
- [x] **Step 5: Full suite, then commit** — `git add src/srclight/refmask.py tests/test_refmask.py && git commit -m "feat(graph): comment/string masking for the edge builder"`

---

### Task 2: imports.py — lift _extract_imports out of server.py

**Files:** Create `src/srclight/imports.py`; Modify `src/srclight/server.py`.

**Interfaces:**
- Produces: `extract_imports(content: str, language: str) -> list[dict]` and `IMPORT_PATTERNS` — MOVED verbatim from server.py:1829's `_extract_imports` (plus its `IMPORT_PATTERNS` constant, wherever it is defined in server.py; find with `grep -n "IMPORT_PATTERNS" src/srclight/server.py`). server.py replaces its definitions with `from .imports import IMPORT_PATTERNS, extract_imports as _extract_imports` so `find_imports` and its tests keep working unchanged. The indexer may then import it without touching server.

- [ ] **Step 1: Move the code** (no behavior change, so the existing `TestExtractImports` suite in tests/test_new_tools.py IS the test — it already passes and must still pass, now exercising the moved module through server's re-import).
- [ ] **Step 2: Full suite** — `.venv/bin/python -m pytest -q` → all pass (existing import tests prove the move).
- [ ] **Step 3: Commit** — `git add src/srclight/imports.py src/srclight/server.py && git commit -m "refactor: lift import extraction into srclight.imports (indexer needs it without importing server)"`

---

### Task 3: resolution column + ranked selection in _build_edges

**Files:** Modify `src/srclight/db.py`, `src/srclight/indexer.py`; Test `tests/test_edge_resolution.py`.

**Interfaces:**
- `EdgeRecord` gains `resolution: str | None = None`. `symbol_edges` gains `resolution TEXT` (add to CREATE TABLE **and** a migration in `initialize()`: `ALTER TABLE symbol_edges ADD COLUMN resolution TEXT` wrapped in try/except `sqlite3.OperationalError` for existing DBs — match how earlier migrations in `initialize()` are done). `insert_edge` writes it.
- `_build_edges` (indexer.py:918): (a) content_rows query also selects `f.path as file_path, f.language`; (b) content is masked: `content = mask_noncode(row["content"], row["language"])`; (c) per-file import names cached: `imported = {last-dotted-segment of each import module/name}` via `extract_imports` on the FILE's... symbol content is all we have per row — use the source symbol's file: cache per file_id the union of imports extracted from that file's symbols' contents is wrong; instead read the file's import set ONCE per file from the first 100 lines of the file on disk IF present, else skip the import tier (the tier is a boost, not a requirement — files missing on disk simply never hit the import tier). Cache: `dict[file_path, set[str]]`.
- (d) THE SELECTION (replaces `for target in targets:` create-edge-to-every-candidate):

```python
def _select_targets(targets: list[dict], source_file: str,
                    imported: set[str], ref_name: str) -> tuple[list[dict], str]:
    """Ranked, field-standard selection (grain-0399). Returns (chosen, resolution)."""
    same_file = [t for t in targets if t["file"] == source_file]
    if same_file:
        return same_file, "same_file"
    files = {t["file"] for t in targets}
    if len(files) == 1:
        return targets, "unique_file"
    if imported:
        imp = [t for t in targets
               if t["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0] in imported
               or ref_name in imported]
        if imp and len({t["file"] for t in imp}) == 1:
            return imp, "import"
    sdir = source_file.rsplit("/", 1)[0] if "/" in source_file else ""
    sd = [t for t in targets if (t["file"].rsplit("/", 1)[0] if "/" in t["file"] else "") == sdir]
    if sd and len({t["file"] for t in sd}) == 1:
        return sd, "same_dir"
    return targets, "name_only"   # keep the ranked LIST — never a coin flip
```

Chosen targets get edges with the existing `_compute_confidence` (unchanged) plus `resolution=<tier>`. The `refs_for_this >= MAX_REFS_PER_SYMBOL` cap counts CHOSEN edges as before.

**Test sketch (tests/test_edge_resolution.py):** build a Database in tmp_path (open+initialize), upsert 3 files (`a.py`, `b.py`, `c.py`), insert symbols: `caller` in a.py whose content references `dup`, plus `dup` defined in a.py AND b.py (same-file tier wins → all edges from caller→dup have `resolution='same_file'` and target only a.py's dup); a second caller in c.py referencing `dup` (no same-file, two files → `name_only`, edges to BOTH); a symbol `solo` defined twice in b.py only (unique_file). Drive `Indexer` via its public index/edge-build path the way tests/test_indexer.py does (copy its fixture pattern — read that file first), or instantiate the indexer on a real tmp dir with three small .py files and let it index end-to-end, then assert on `symbol_edges` rows (join names) — end-to-end is PREFERRED (it also proves masking: put one reference in a comment and assert no edge).

- [ ] **Step 1: Write failing tests** (end-to-end fixture per tests/test_indexer.py's pattern; assertions above + a comment-reference producing NO edge)
- [ ] **Step 2: Verify fail** (no `resolution` column / all-candidates edges present)
- [ ] **Step 3: Implement** (db.py column+record+insert+migration; indexer masking+selection)
- [ ] **Step 4: Verify pass, full suite** (`get_dead_symbols` tests must still pass — selection only REDUCES spurious edges)
- [ ] **Step 5: Commit** — `git add src/srclight/db.py src/srclight/indexer.py tests/test_edge_resolution.py && git commit -m "feat(graph): ranked single-target selection with per-edge resolution labels"`

---

### Task 4: surface resolution in graph tools + README

**Files:** Modify `src/srclight/db.py` (`get_callers`/`get_callees` row dicts include `"resolution": r["resolution"]`), `README.md`.

- [ ] **Step 1: Add `resolution` to the two row dicts** (db.py:837/858 region — they already select `s.*`-adjacent fields; add the column to the SELECT and dict).
- [ ] **Step 2: Test** (append to tests/test_edge_resolution.py): call `db.get_callers(dup_id)` on the Task 3 fixture and assert rows carry `resolution`.
- [ ] **Step 3: README** — extend the *Index freshness* area with a short **Graph resolution labels** paragraph: every caller/callee edge carries `resolution` (`same_file | unique_file | import | same_dir | name_only`); `name_only` means a ranked candidate list across same-named symbols — treat it as "one of these", not a confirmed link.
- [ ] **Step 4: Full suite, commit** — `git add src/srclight/db.py tests/test_edge_resolution.py README.md && git commit -m "feat(graph): resolution labels surfaced in get_callers/get_callees; docs"`

---

### Task 5: extend harness, reindex, MEASURE GATE

**Files:** Modify `scripts/measure_graph_precision.py`; regenerate `docs/graph-precision-report.md`.

- [ ] **Step 1: Add tier distribution to `measure()`** — `"resolution_distribution": {tier: count}` via `SELECT resolution, COUNT(*) c FROM symbol_edges WHERE edge_type='calls' GROUP BY resolution`, and keep every existing metric so before/after compares.
- [ ] **Step 2: Reindex + run** — `.venv/bin/srclight index . && .venv/bin/python scripts/measure_graph_precision.py .srclight/index.db --sample 500 --out docs/graph-precision-report.md`
- [ ] **Step 3: THE GATE** — compare to baseline (62.5% ambiguity / 12.8% false-ref): ambiguity ≤30% and false-ref ≈0 → proceed. Otherwise STOP: commit the report with a `## Gate: FAILED` note, land a grain, report to Tim — no release.
- [ ] **Step 4: Commit** — `git add scripts/measure_graph_precision.py docs/graph-precision-report.md && git commit -m "docs: graph precision re-measured after disambiguation (before/after)"`

---

### Task 6: release 0.20.5 + closing grain

- [ ] **Step 1: Full suite green**, bump `version = "0.20.4"` → `"0.20.5"` in pyproject.toml, commit `release: 0.20.5 — graph disambiguation (masking + ranked resolution labels)`.
- [ ] **Step 2: Release chain** — push develop; `git checkout master && git merge --no-ff develop -m "Release 0.20.5: graph disambiguation"`; `git tag -a v0.20.5 -m "srclight v0.20.5: comment/string masking, ranked edge resolution"`; push master + tag; back to develop.
- [ ] **Step 3: Closing grain** (pack_learn, dog=gromit, dimension=work, origin_repo=srclight/srclight, origin_commit=<release sha>): before/after numbers, tier distribution, what stayed `name_only`, and the SCIP-ingestion note as the only sanctioned path to more precision.

## Self-Review Notes

- The import tier reads the file from disk (first ~100 lines) — acceptable: edge build already runs at index time when files are present; a missing file skips the tier, never fails the build.
- `_select_targets` returning `name_only` keeps the old fan-out for genuinely ambiguous names — so recall cannot DROP below the current graph for those; the change strictly removes edges that had better-tier evidence against them, plus comment/string ghosts.
- Migration must handle existing `.srclight/index.db` files (ALTER TABLE guarded) — `_build_edges` does `DELETE FROM symbol_edges` + full rebuild anyway, so old rows never linger with NULL resolution after a reindex; NULL only appears in never-reindexed DBs and readers must tolerate it.
- Baseline suite count 255; expect ≈270 after.
