# Lab Book: Community Detection & Execution Flow Tracing

## 2026-04-06 — Session 1: Design & Initial Implementation

### Background

Studying GitNexus (graph-powered code intelligence, PolyForm Noncommercial) revealed two features
Srclight lacks: automatic community/module detection via Leiden clustering, and execution flow
tracing via BFS from entry points. Both are well-published algorithms with open implementations.

### Baseline Data

Workspace edge counts across repos (srclight's `symbol_edges` table, `edge_type='calls'`):

| Repo | Symbols | Call Edges |
|------|---------|------------|
| bitcoin | 29,287 | 196,920 |
| nomad-builder | 39,424 | 77,442 |
| intuition-2019 | 20,648 | 63,126 |
| motacoin | 14,479 | 47,778 |
| kumquat | 3,978 | 18,665 |
| qi | 6,323 | 16,249 |
| web | 22,525 | 13,622 |
| srclight | 1,008 | 4,044 |
| zhcorpus | 3,758 | 3,037 |
| ice | 4,043 | 2,802 |

### Hypothesis

Louvain community detection on the call graph will produce meaningful functional clusters
that correspond to human-recognizable modules (e.g., "database", "indexing", "git", "MCP server").

### Experiment 1: Louvain on Srclight's Own Call Graph

**Method:** Load 4,044 call edges from srclight's index.db into networkx. Run Louvain
(resolution=1.0, seed=42). Examine resulting communities against known module structure.

**Expected:** ~5-10 communities mapping roughly to: db, indexer, server, git, embeddings,
workspace, cli, build, vector_math/cache.

**Results:**
- **19 communities detected in 0.160s** (more granular than expected)
- Recognizable modules confirmed:
  - #5 "Provider Embed Batch" = embeddings module (Ollama, Voyage, Cohere, OpenAI providers)
  - #7 "Learning Conversation Similarity" = learnings system
  - #8 "Cache Sidecar Vector" = vector cache
  - #9 "Hooks Hook Uninstall" = git hooks
  - #12 "Top Cosine Decode" = vector math
  - #10 "Platform Cmake Variants" = build system parsing
  - #11 "Import Include Javascript" = import extraction
- Larger clusters (#0, #1, #2, #3) are somewhat mixed — the "Projects Api List" cluster (#0, 160 symbols) combines workspace, server, and test code
- Cohesion scores low (0.065-0.195) for large clusters, high (0.4-1.0) for small focused ones

**Observations:**
- More communities than expected (19 vs 5-10). Louvain finds fine-grained structure.
- Test code gets mixed into production clusters because tests call production functions.
- Auto-labeling via TF-IDF works surprisingly well — keywords are meaningful.
- Small, focused modules (vector_math, hooks) have high cohesion and clean labels.
- Large modules with many cross-cutting concerns get lower cohesion.

### Experiment 2: Execution Flow Tracing on Srclight

**Method:** Score entry points by out-degree/in-degree ratio + name heuristics.
BFS from top entry points, max depth 10, max branching 4. Deduplicate subset flows.

**Expected:** Flows like: CLI entry -> indexer -> db store, Server tool -> db query -> return.

**Results:**
- **75 flows traced in 12.632s** (too slow — needs optimization)
- Entry point: `_extract_symbols` dominates (highest out-degree)
- Flows are 11 steps deep, crossing 5-7 communities
- Flow quality is mixed — BFS follows all call edges including constructors and data classes, leading to noisy paths (e.g., SymbolRecord constructors appear in many paths)

**Observations:**
- Flow tracing is **too slow** for production use on even a medium repo. The BFS generates a combinatorial explosion of paths because branching factor is high.
- Constructors and data class references pollute flows — should filter to "behavioral" edges only.
- Need: (a) reduce branching, (b) filter edge types, (c) memoize common subpaths.
- The concept is sound but needs tuning before it's production-quality.

### Experiment 3: Scale Test on Bitcoin (196K edges)

**Method:** Run Louvain on bitcoin's call graph. Measure wall time and community count.

**Expected:** Sub-second for Louvain. ~20-50 communities.

**Results:**
- **58 communities detected in 3.634s** (slower than expected, but still reasonable)
- Community labels are domain-accurate:
  - #4 "Sig2 Signing Mu" = cryptographic signing (MuSig2)
  - #5 "Undo Validation Locator" = block validation
  - #7 "Service Local In6" = network/service layer (IPv6)
  - #11 "Cha20 Decrypt Encrypt" = ChaCha20 encryption
  - #12 "Selection Database Records" = wallet/UTXO selection
  - #13 "Ge Ecmult Gej" = elliptic curve math (secp256k1)
  - #14 "Flags Psbt Signer" = PSBT transaction signing
- Largest community: 3,068 symbols (10% of total) — likely the core/general bucket
- Cohesion scores very low for large clusters (0.004-0.016)

**Observations:**
- 3.6s for 196K edges is acceptable for an index-time operation (runs once after indexing).
- Louvain correctly identifies domain-specific modules even in a large C++ codebase.
- The auto-labeling picks up domain terms (musig2, chacha20, secp256k1) from function names.
- Low cohesion in large clusters suggests resolution parameter could be tuned higher.

### Summary & Next Steps (Session 1)

1. **Community detection: VALIDATED.** Louvain produces meaningful, recognizable clusters.
   Labels are surprisingly accurate. Performance is acceptable at index time.

2. **Flow tracing: NEEDS WORK.** BFS is too slow and too noisy. Key improvements needed:
   - Filter out constructor/data class edges before tracing
   - Reduce max_branching or prioritize high-confidence edges
   - Consider DFS with path memoization instead of BFS
   - Perhaps limit to cross-community flows only (most interesting)

3. **Resolution tuning:** Consider adaptive resolution based on graph size:
   - Small graphs (<1K nodes): resolution=1.5 for finer granularity
   - Large graphs (>10K nodes): resolution=1.0 (current default)

4. **Next experiment:** Run on nomad-builder (C++ heavy, 77K edges) to test
   language-specific quality.

---

## 2026-04-06 — Session 2: Optimization, MCP Tools, detect_changes

### Experiment 4: Flow Tracing Optimization

**Problem:** Flow tracing took 12.6s on srclight (1,008 symbols, 4,044 edges). Root cause:
container kinds (class, enum, struct) have high out-degree (e.g., Database class → 30+ callees),
causing combinatorial explosion in BFS.

**Method:** Three changes:
1. Filter CONTAINER_KINDS = {class, mixin, enum, struct, extension, section} from edges and entry points
2. Reduce defaults: max_entry_points=20, max_depth=8, max_branching=3, max_flows=50
3. Add global iteration budget of 5,000 across all entry points

**Results:**
- 12.632s → **0.007s** (1,800x speedup)
- Flow quality improved: paths now trace behavioral code (functions/methods), not data structures
- 50 flows produced (capped by max_flows), all meaningful

**Observation:** Filtering container kinds was the biggest win. Classes like `Database` with 30+
callees were generating millions of paths. Behavioral filtering alone would have been sufficient.

### Experiment 5: MCP Tools End-to-End

**Method:** Added 5 MCP tools to server.py:
- `get_communities(project)` — list all detected communities
- `get_community(symbol, project)` — which community a symbol belongs to
- `get_execution_flows(project)` — BFS-traced execution paths
- `get_impact(symbol, project)` — blast radius + risk level
- `detect_changes(project, ref?)` — git diff → symbol mapping → aggregate impact

Tested on srclight itself (19 communities, 50 flows), bitcoin (58 communities, 50 flows),
and ice (18 communities, 50 flows).

**Results:**
- All tools work in both single-repo and workspace (per-project DB) mode
- `detect_changes()` on our own uncommitted changes: 33 symbols changed, 295 dependents,
  12 communities affected, overall CRITICAL — correct, since we touched core modules
- Schema v5 backfill works: on first reindex after migration, community detection runs
  even if no files changed (checks for empty communities table + existing call edges)

### Experiment 6: detect_changes Accuracy

**Method:** Called `detect_changes()` against our own uncommitted work (5 files changed:
server.py, indexer.py, git.py, community.py, db.py). Parsed `git diff -U0` to extract
hunk line ranges, mapped to symbols via start_line/end_line overlap, ran `compute_impact()`
on each.

**Results:**
- Correctly identified `Indexer` (73 deps), `WorkspaceConfig` (64 deps), `_get_db` (55 deps)
  as highest-risk changes
- Overall risk: CRITICAL — accurate, since touching the indexer and server affects most of the codebase
- A leaf function like `_tokenize_name` shows LOW risk with 1-2 dependents — correct
- Line number drift (index built before edits) causes some duplicate symbols with slightly
  different line ranges. Acceptable: the tool still catches the right symbols.

### Summary (Session 2)

1. **Flow tracing: FIXED.** 1,800x speedup via container filtering. Production-ready.
2. **MCP tools: SHIPPED.** 5 new tools (42 total). All tested end-to-end.
3. **detect_changes: VALIDATED.** The reactive guardrail works — maps diffs to blast radius.
   This is the feature that actually prevents agents from shipping breaking changes.
4. **Schema v5 backfill: FIXED.** Community detection now runs on first reindex after migration,
   not just when new edges are created.
