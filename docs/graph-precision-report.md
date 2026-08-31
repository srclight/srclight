# Graph precision report

```json
{
  "db": ".srclight/index.db",
  "total_calls_edges": 4769,
  "ambiguous_target_name_rate": 0.6251,
  "confidence_distribution": {
    "0.5": 1878,
    "0.9": 1440,
    "1.0": 1451
  },
  "sampled": 500,
  "sampled_verdicts": {
    "code": 436,
    "comment_or_string_only": 64,
    "absent": 0
  },
  "false_reference_rate_in_sample": 0.128,
  "false_reference_examples": [
    {
      "edge_id": 1945,
      "target": "embedding_status",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 2537,
      "target": "search",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 4753,
      "target": "find_imports",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 761,
      "target": "symbols",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 4204,
      "target": "codebase_map",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 352,
      "target": "symbols",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 1553,
      "target": "symbols",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 643,
      "target": "workspace",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 3691,
      "target": "check_freshness",
      "verdict": "comment_or_string_only"
    },
    {
      "edge_id": 3012,
      "target": "symbols",
      "verdict": "comment_or_string_only"
    }
  ],
  "dead_code_sample_for_review": [
    {
      "name": "test_db_embeddings_incremental",
      "kind": "function",
      "path": "tests/test_embeddings.py"
    },
    {
      "name": "dimensions",
      "kind": "function",
      "path": "src/srclight/embeddings.py"
    },
    {
      "name": "test_the_daemons_own_object_is_the_strict_one",
      "kind": "function",
      "path": "tests/test_strict_args.py"
    },
    {
      "name": "test_cmake_system_name_conditionals",
      "kind": "function",
      "path": "tests/test_build.py"
    },
    {
      "name": "test_get_provider_unknown_prefix_falls_through_to_ollama",
      "kind": "function",
      "path": "tests/test_embeddings.py"
    },
    {
      "name": "test_community_cohesion_range",
      "kind": "function",
      "path": "tests/test_community.py"
    },
    {
      "name": "test_stats",
      "kind": "function",
      "path": "tests/test_db.py"
    },
    {
      "name": "test_scope_validation",
      "kind": "function",
      "path": "tests/test_learnings.py"
    },
    {
      "name": "test_load_sidecar",
      "kind": "function",
      "path": "tests/test_vector_cache.py"
    },
    {
      "name": "__init__",
      "kind": "function",
      "path": "src/srclight/workspace.py"
    },
    {
      "name": "get_impact",
      "kind": "function",
      "path": "src/srclight/server.py"
    },
    {
      "name": "test_record_learning_with_sources",
      "kind": "function",
      "path": "tests/test_learnings.py"
    },
    {
      "name": "_git_tracked_files",
      "kind": "function",
      "path": "src/srclight/indexer.py"
    },
    {
      "name": "test_detect_changes_with_ref",
      "kind": "function",
      "path": "tests/test_git.py"
    },
    {
      "name": "config",
      "kind": "function",
      "path": "src/srclight/cli.py"
    },
    {
      "name": "get_community",
      "kind": "function",
      "path": "src/srclight/server.py"
    },
    {
      "name": "get_type_hierarchy",
      "kind": "function",
      "path": "src/srclight/server.py"
    },
    {
      "name": "test_workspace_db_attach_and_search",
      "kind": "function",
      "path": "tests/test_workspace.py"
    },
    {
      "name": "main",
      "kind": "function",
      "path": "scripts/test_hybrid_search.py"
    },
    {
      "name": "test_index_status_carries_whole_index_counts",
      "kind": "function",
      "path": "tests/test_freshness_tools.py"
    }
  ]
}
```
