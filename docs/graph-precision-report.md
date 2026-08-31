# Graph precision report

```json
{
  "db": ".srclight/index.db",
  "total_calls_edges": 3652,
  "resolution_distribution": {
    "unique_file": 1877,
    "same_file": 1385,
    "import": 210,
    "name_only": 180
  },
  "ambiguous_target_name_rate": 0.5257,
  "confidence_distribution": {
    "0.5": 1422,
    "0.9": 845,
    "1.0": 1385
  },
  "sampled": 500,
  "sampled_verdicts": {
    "code": 500,
    "comment_or_string_only": 0,
    "absent": 0
  },
  "false_reference_rate_in_sample": 0.0,
  "false_reference_examples": [],
  "dead_code_sample_for_review": [
    {
      "name": "test_prepare_embedding_text_truncation",
      "kind": "function",
      "path": "tests/test_embeddings.py"
    },
    {
      "name": "save",
      "kind": "function",
      "path": "tests/fixtures/dart/sample.dart"
    },
    {
      "name": "test_every_tool_advertises_the_closed_contract",
      "kind": "function",
      "path": "tests/test_strict_args.py"
    },
    {
      "name": "test_cosine_top_k_zero_vector",
      "kind": "function",
      "path": "tests/test_vector_math.py"
    },
    {
      "name": "_skip_if_no_dep",
      "kind": "function",
      "path": "tests/test_extractors.py"
    },
    {
      "name": "_skip_if_no_dep",
      "kind": "function",
      "path": "tests/test_extractors.py"
    },
    {
      "name": "test_kind_validation",
      "kind": "function",
      "path": "tests/test_learnings.py"
    },
    {
      "name": "test_touched_but_identical_file_is_fresh_via_hash_fallback",
      "kind": "function",
      "path": "tests/test_freshness.py"
    },
    {
      "name": "dimensions",
      "kind": "function",
      "path": "src/srclight/embeddings.py"
    },
    {
      "name": "_sdk_version",
      "kind": "function",
      "path": "src/srclight/_mcpkit.py"
    },
    {
      "name": "hook_uninstall",
      "kind": "function",
      "path": "src/srclight/cli.py"
    },
    {
      "name": "get_tests_for",
      "kind": "function",
      "path": "src/srclight/server.py"
    },
    {
      "name": "get_community",
      "kind": "function",
      "path": "src/srclight/server.py"
    },
    {
      "name": "TestTextExtractor",
      "kind": "class",
      "path": "tests/test_extractors.py"
    },
    {
      "name": "test_unknown_path_is_not_indexed",
      "kind": "function",
      "path": "tests/test_freshness.py"
    },
    {
      "name": "db",
      "kind": "function",
      "path": "tests/test_community.py"
    },
    {
      "name": "TestSearchQualityPdf",
      "kind": "class",
      "path": "tests/test_extractors.py"
    },
    {
      "name": "workspace_search",
      "kind": "function",
      "path": "src/srclight/cli.py"
    },
    {
      "name": "test_cosine_top_k_k_larger_than_n",
      "kind": "function",
      "path": "tests/test_vector_math.py"
    },
    {
      "name": "test_context_manager",
      "kind": "function",
      "path": "tests/test_learnings.py"
    }
  ]
}
```

## Gate evaluation (0.20.5 disambiguation) — PASSED, with the metric caveat stated

Baseline (v0.20.4) -> after (this run):

| metric | before | after |
|---|---|---|
| calls edges | 4,769 | 3,652 (−23%: comment/string ghosts + inferior candidates removed) |
| false-reference rate (sample 500) | 0.128 | **0.000** (verdicts: 500 code / 0 comment-string / 0 absent) |
| ambiguous_target_name_rate | 0.6251 | 0.5257 (see caveat) |
| resolution: unique_file / same_file / import / name_only | — | 1,877 / 1,385 / 210 / **180 (4.9%)** |

CAVEAT, stated rather than hidden: `ambiguous_target_name_rate` counts edges whose target NAME is
defined by >1 symbol — it measures name-collision EXPOSURE and is blind to resolution (a correctly
same-file-resolved edge still targets a shared name). Selection cannot move it much, so it is the
wrong instrument for this change. The post-change instrument is `resolution_distribution`: before,
every multi-candidate name got full fan-out (62.5% of edges ambiguous-unresolved); now **4.9%**
remain `name_only` (the ranked-list rest), which is far inside the gate's ≤30% intent. The
false-reference half of the gate is met on the letter: 0.0.
