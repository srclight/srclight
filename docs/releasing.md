# Releasing Srclight

## Version Locations

Version must be updated in **three files** — all must match:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/srclight/__init__.py` | `__version__ = "X.Y.Z"` |
| `server.json` | `"version": "X.Y.Z"` (top-level and inside `packages[0]`) |

## Semantic Versioning

- **Patch** (0.8.x): bug fixes, docs, metadata
- **Minor** (0.x.0): new features, new MCP tools, new language support
- **Major** (x.0.0): breaking changes to MCP tool signatures or DB schema

## Release Checklist

```bash
# 1. Bump version in all three files
#    pyproject.toml, src/srclight/__init__.py, server.json

# 2. Commit on develop
git add pyproject.toml src/srclight/__init__.py server.json
git commit -m "Bump version to X.Y.Z"

# 3. Merge to master and push
git checkout master
git merge develop --no-edit
git push origin master
git push origin develop

# 4. Create GitHub release (triggers PyPI publish via GitHub Actions)
gh release create vX.Y.Z --title "vX.Y.Z" --notes "Release notes here." --target master

# 5. Wait for PyPI publish to complete
gh run watch $(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')

# 6. Verify on PyPI
pip index versions srclight

# 7. Publish to MCP registry
mcp-publisher publish

# 8. Switch back to develop
git checkout develop
```

## What's Automated

- **PyPI publish**: GitHub Actions workflow (`.github/workflows/publish.yml`) triggers on every GitHub release. Uses Trusted Publisher (OIDC) — no API tokens needed.
- **MCP registry**: Manual for now (`mcp-publisher publish`). Requires a valid token from `mcp-publisher login github`.

## MCP Registry Auth

The `mcp-publisher` CLI authenticates via GitHub OAuth. Token is stored at `~/.mcp_publisher_token`.

```bash
# Login (GitHub device flow)
mcp-publisher login github

# Publish (reads server.json from current directory)
mcp-publisher publish

# Token expires periodically — re-login if you get a 401
```

The server name `io.github.srclight/srclight` requires that the authenticated GitHub user is a **public member** of the `srclight` GitHub org.

## Troubleshooting

### PyPI publish fails
- Check the GitHub Actions run: `gh run view --log`
- Verify the `pypi` environment exists in repo settings
- Verify Trusted Publisher is configured on PyPI for `srclight/srclight`

### MCP registry 403
- Re-login: `mcp-publisher login github`
- Verify org membership is **public** at https://github.com/orgs/srclight/people

### MCP registry 400 (README validation)
- The PyPI package README must contain `<!-- mcp-name: io.github.srclight/srclight -->` (line 1 of README.md)
- This means any version bump that doesn't touch README still needs the tag present
- PyPI must have the **current version** published — the registry checks the latest version
