# Design System Auditing with Srclight: Gaps and Proposed Improvements

## Context

Srclight is increasingly used by AI design review agents (e.g., UI reviewers in Claude Code) to audit Flutter/Dart codebases for design system compliance. This document captures gaps discovered during real-world UI reviews of the Copyworks project (Feb 2026) and proposes improvements.

## Current Capabilities vs. Design Review Needs

| Need | Srclight Tool | Works? | Notes |
|------|--------------|--------|-------|
| File structure overview | `symbols_in_file` | Yes | Gives class/method outlines without reading full files |
| Function caller graph | `get_callers` | Yes | Traces who calls `paperShadow()`, etc. |
| Class consumer graph | `get_callers` | Yes | Found all 9 consumers of `CopyworksColors` |
| Static constant usage | `get_callers` | **No** | `Spacing.md`, `AppColors.tone1` — static field access not tracked |
| Raw color literal scan | — | **No** | `Color(0xFF...)` literals are code patterns, not symbols |
| Spacing violation scan | — | **No** | `EdgeInsets.all(10)` vs `Spacing.xs` — pattern matching, not symbol search |
| Font string literal scan | — | **No** | `'Noto Sans'` repeated 8x — string literals are invisible to symbol index |
| Magic number detection | — | **No** | `height: 52` repeated 3x — numeric literals in widget code |

**Bottom line:** Srclight excels at symbol-level navigation (callers, callees, type hierarchy) but cannot audit design token adoption because tokens are typically consumed as static constant references and raw literals, neither of which appear in the call graph.

## Gap 1: Static Field Access Not Tracked

### Problem

`get_callers("Spacing")` returns 0 results despite 60+ usages across the codebase. This is because `Spacing.md`, `Spacing.xs`, `Spacing.controlBarHeight` are static field accesses, not function/method calls. The ctags-based indexer doesn't generate edges for field access patterns.

### Impact

Design tokens in Dart/Flutter are almost always static constants:
```dart
abstract final class Spacing {
  static const double xs = 8;
  static const double md = 16;
}

// Usage — invisible to get_callers:
padding: const EdgeInsets.all(Spacing.md),
```

This means srclight cannot answer: "Where is `Spacing.md` used?" or "Is `AppColors.cellBorder` adopted everywhere it should be?"

### Proposed Solution: `get_references(symbol)` Tool

Add a new tool that combines call graph edges with text-based references:

```python
def get_references(symbol_name: str, project: str) -> list[Reference]:
    """Find all references to a symbol — calls, field access, and text matches.

    Combines:
    1. Call graph edges (existing get_callers behavior)
    2. Regex search for `ClassName.symbol_name` patterns in indexed files
    3. Import-aware filtering (only files that import the defining module)
    """
```

This would surface:
- `Spacing.md` in `EdgeInsets.all(Spacing.md)` — field access
- `AppColors.cellBorder` in `color: AppColors.cellBorder` — field access
- `CjkFonts.family` in `fontFamily: CjkFonts.family` — field access

**Implementation complexity:** Low-medium. This is essentially a scoped grep with symbol context. The index already knows which file defines `Spacing`, so it can construct the right regex patterns (`Spacing\.\w+`) and search only Dart files.

## Gap 2: Raw Literal Detection (Design Token Violations)

### Problem

The #1 task in every UI design review is finding raw literals that should be named tokens:
- `Color(0xFF999999)` should be `AppColors.cellBorder`
- `'Noto Sans'` should be `CjkFonts.annotationFamily`
- `height: 52` should be `Spacing.controlBarHeight`
- `EdgeInsets.all(10)` should use `Spacing.*`

These are syntactic patterns in source code, not symbols in the index. Srclight's FTS5 and embedding search operate on symbol names, signatures, and doc comments — not on raw source text.

### Proposed Solution: `audit_tokens(project, config)` Tool

A new design-system-aware audit tool:

```python
def audit_tokens(
    project: str,
    token_files: list[str],  # e.g., ["lib/theme/app_theme.dart"]
    rules: list[AuditRule],
) -> list[Violation]:
    """Scan source files for raw literals that should use design tokens.

    Process:
    1. Parse token_files to extract all named constants
       (e.g., Spacing.md = 16, AppColors.cellBorder = Color(0xFF999999))
    2. Scan all other project files for matching raw literals
    3. Report violations with suggested token replacements
    """
```

#### Built-in Rule Types

```python
class AuditRule:
    # Color literals: Color(0x...) or Colors.xxx not in token_files
    COLOR_LITERALS = "color_literals"

    # Spacing violations: EdgeInsets/SizedBox/height:/width: with bare numbers
    # not matching any Spacing.* constant
    SPACING_VIOLATIONS = "spacing_violations"

    # Font string literals: fontFamily: 'xxx' not using CjkFonts.*
    FONT_LITERALS = "font_literals"

    # Magic numbers: bare numeric literals in widget build methods
    # (height:, width:, fontSize:, borderRadius:, elevation:)
    MAGIC_NUMBERS = "magic_numbers"
```

#### Example Output

```json
{
  "violations": [
    {
      "file": "lib/widgets/preview_panel.dart",
      "line": 468,
      "literal": "'Noto Sans'",
      "context": "fontFamily: 'Noto Sans'",
      "suggested_token": "CjkFonts.annotationFamily (not yet defined)",
      "rule": "font_literals"
    },
    {
      "file": "lib/services/image_export_service.dart",
      "line": 148,
      "literal": "Color(0xFF999999)",
      "context": "color: Color(0xFF999999)",
      "suggested_token": "AppColors.cellBorder",
      "rule": "color_literals"
    }
  ]
}
```

**Implementation complexity:** Medium. Requires:
1. A simple Dart constant parser (regex-based is sufficient for `static const` declarations)
2. A pattern scanner for each rule type
3. Cross-referencing found literals against the known token set

This does NOT require tree-sitter parsing — regex patterns for `Color(0x`, `EdgeInsets.all(`, `fontFamily:`, `height:` are sufficient and fast.

### Configuration

Projects could provide a `.srclight/audit.toml` config:

```toml
[design_tokens]
token_files = ["lib/theme/app_theme.dart"]

[rules.color_literals]
enabled = true
# Files where raw colors are acceptable (e.g., PDF rendering context)
exclude = ["lib/services/pdf_service.dart"]

[rules.spacing_violations]
enabled = true
grid_size = 4
spacing_class = "Spacing"
exclude = ["lib/services/pdf_service.dart"]

[rules.font_literals]
enabled = true
font_class = "CjkFonts"

[rules.magic_numbers]
enabled = true
properties = ["height", "width", "fontSize", "borderRadius", "elevation"]
```

## Gap 3: Duplicate Code Detection Across Files

### Problem

The UI review found identical helper methods (`_sectionHeader`, `_menuRow`, `_toggleRow`) duplicated between `editor_panel.dart` and `format_panel.dart`. Srclight's `hybrid_search` does not flag duplicates because each method is a distinct symbol with a distinct name in a distinct file.

### Proposed Solution: `find_duplicates(project)` Tool

```python
def find_duplicates(
    project: str,
    min_similarity: float = 0.9,  # body text similarity threshold
    min_lines: int = 5,           # ignore trivial duplicates
) -> list[DuplicatePair]:
    """Find symbols with near-identical bodies across different files.

    Uses embedding similarity on symbol bodies (already computed for
    semantic_search) to find duplicate implementations.
    """
```

**Implementation complexity:** Low. The embeddings already exist. This is a nearest-neighbor search on the existing embedding matrix with a high similarity threshold, filtered to exclude same-file pairs.

## Gap 4: ctags Dart Parser Limitations

### Problem

`symbols_in_file` returns duplicate entries — every method appears as both `kind: "method"` and `kind: "function"`. This is a ctags artifact specific to the Dart parser. It clutters output and confuses calorie counts.

Additionally, `get_signature` returned `null` for all Dart symbols, meaning the agent couldn't assess API shape without reading source.

### Proposed Fix

1. **Dedup:** In the Dart indexer, when a symbol appears as both `method` and `function` at the same line, keep only `method` (more specific).
2. **Signatures:** The ctags Dart parser may not emit `signature:` fields. Consider extracting signatures from the first line of the symbol body as a fallback: `static const double xs = 8;` → signature `double xs = 8`.

## Gap 5: Widget-Specific Metrics (Future)

For Flutter-specific UI reviews, these metrics would be valuable but are lower priority:

| Metric | Description | Value |
|--------|-------------|-------|
| Widget depth | Max nesting depth of `build()` method | Identifies over-nested widgets |
| `RepaintBoundary` coverage | % of `CustomPaint` widgets wrapped | Performance audit |
| `Semantics` coverage | % of `CustomPaint` widgets with `Semantics` | Accessibility audit |
| Theme token adoption rate | % of color/spacing values using tokens vs. raw literals | Design system health |

These require Dart-specific AST analysis beyond what ctags provides.

## Summary: Recommended Roadmap

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| **P1** | `get_references()` — static field access tracking | Low-Medium | Unlocks design token usage queries |
| **P1** | `audit_tokens()` — raw literal violation scanner | Medium | The #1 request from UI review agents |
| **P2** | Dart ctags dedup (method vs function) | Low | Cleaner `symbols_in_file` output |
| **P2** | Dart signature extraction fallback | Low | Enables `get_signature` for Dart |
| **P3** | `find_duplicates()` — embedding-based duplicate detection | Low | Catches copy-paste across files |
| **P3** | Widget-specific metrics | High | Flutter-specific, requires AST |
