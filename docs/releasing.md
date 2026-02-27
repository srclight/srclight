# Releasing Srclight

## Version Locations

Version must be updated in **two files** — both must match:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/srclight/__init__.py` | `__version__ = "X.Y.Z"` |

`server.json` is **not** bumped manually — CI patches it from the git tag automatically.

## Semantic Versioning

- **Patch** (0.8.x): bug fixes, docs, metadata
- **Minor** (0.x.0): new features, new MCP tools, new language support
- **Major** (x.0.0): breaking changes to MCP tool signatures or DB schema

## Release Checklist

```bash
# 1. Commit feature work on develop
git checkout develop
# ... work, commit ...

# 2. Create release branch and bump version
git checkout -b release/X.Y.Z
# Edit pyproject.toml and src/srclight/__init__.py
git commit -am "chore: Bump version to X.Y.Z"

# 3. Merge release to master
git checkout master
git merge release/X.Y.Z --no-ff

# 4. Tag the release
git tag -a vX.Y.Z -m "vX.Y.Z description"

# 5. Merge release back to develop
git checkout develop
git merge release/X.Y.Z --no-ff
git branch -d release/X.Y.Z

# 6. Push everything
git push origin master develop --tags

# 7. Create GitHub release (triggers PyPI + MCP registry publish automatically)
gh release create vX.Y.Z --title "vX.Y.Z — Description" --notes "Release notes here."

# 8. (Optional) Watch the CI run
gh run watch $(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')

# 9. Verify on PyPI
pip index versions srclight
```

## What's Automated

On `release: [published]`, `.github/workflows/publish.yml` runs two jobs:

1. **PyPI** — builds and publishes via Trusted Publisher (OIDC). No API tokens needed.
2. **MCP Registry** — patches `server.json` version from the git tag, authenticates via GitHub OIDC, publishes. No manual `mcp-publisher` step needed.

Both PyPI and MCP registry publishing are **fully automated**. No manual `mcp-publisher login` or `mcp-publisher publish` required.

## Troubleshooting

### PyPI publish fails
- Check the GitHub Actions run: `gh run view --log`
- Verify the `pypi` environment exists in repo settings
- Verify Trusted Publisher is configured on PyPI for `srclight/srclight`

### MCP registry publish fails
- Check the `mcp-registry` job in the GitHub Actions run
- The server name `io.github.srclight/srclight` requires the `srclight` GitHub org to exist
- OIDC auth requires `id-token: write` permission in the workflow

### MCP registry 400 (duplicate version)
- The version was already published. This usually means CI already ran — no manual publish needed.

### MCP registry 400 (README validation)
- The PyPI package README must contain `<!-- mcp-name: io.github.srclight/srclight -->` (line 1 of README.md)
- PyPI must have the **current version** published — the registry checks the latest version
