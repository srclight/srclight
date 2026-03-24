# Playbook: Claude Code Plugin

How to build, maintain, release, and submit the srclight Claude Code plugin.

## Repository

- **Repo**: `srclight/claude-code-plugin` (public, GitHub)
- **Install**: `/plugin marketplace add srclight/claude-code-plugin` then `/plugin install srclight`
- **Official marketplace submission**: via form at `platform.claude.com/plugins/submit`

## Plugin Structure

```
claude-code-plugin/
├── .claude-plugin/plugin.json     # Manifest (name, version, description, author, keywords)
├── .mcp.json                      # MCP server config (flat format, NOT mcpServers wrapper)
├── marketplace.json               # Makes this repo a standalone marketplace
├── hooks/
│   ├── hooks.json                 # SessionStart hook config
│   └── session-start.sh           # Detects .srclight/index.db, guides setup
├── agents/
│   └── codebase-explorer.md       # Autonomous deep research agent
├── skills/
│   ├── codebase-indexing/SKILL.md # Core: tool selection guide + workflows
│   ├── setup/SKILL.md             # /srclight:setup slash command
│   ├── explore-codebase/SKILL.md  # Systematic codebase exploration
│   └── refactor-safely/SKILL.md   # Pre-change safety checklist
├── .gitignore
├── LICENSE                        # MIT
└── README.md
```

## Key Format Rules

These were discovered during development and differ from what you might expect:

### .mcp.json: Flat Format

The official `claude-plugins-official` repo uses **flat format** — server names at top level, NO `mcpServers` wrapper:

```json
{
  "srclight": {
    "command": "uvx",
    "args": ["srclight", "serve", "--transport", "stdio"]
  }
}
```

NOT this (common mistake):
```json
{
  "mcpServers": {
    "srclight": { ... }
  }
}
```

### Transport: stdio, Not SSE

Plugins use **stdio** because Claude Code manages the process lifecycle. SSE requires a server already running on a port. The `.mcp.json` uses `uvx srclight` for zero-install — uv handles the Python package automatically.

This is distinct from the SSE setup described in `docs/usage-guide.md` (systemd service on port 8742). Both can coexist — plugin for single-session stdio, SSE for multi-client scenarios.

### hooks.json: Wrapper Format

Plugin hooks use a wrapper with `hooks` key (unlike settings.json which is flat):

```json
{
  "description": "Optional description",
  "hooks": {
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/session-start.sh",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

Use `${CLAUDE_PLUGIN_ROOT}` for all file paths — never hardcode.

### Skill SKILL.md: Description Format

Third-person, with trigger phrases and `<example>` blocks:

```yaml
description: |
  This skill should be used when the user asks to "specific phrase"...

  <example>
  Context: ...
  user: "..."
  assistant: "..."
  <commentary>Why this triggers.</commentary>
  </example>
```

### Agent Frontmatter

Agents need `name`, `description` (with examples), `model`, `color`, and `tools`.

## Release Process

Simple — no git flow needed for the plugin repo (it's a flat master branch):

```bash
cd ~/Projects/srclight/claude-code-plugin
git add -A && git commit -m "description"
git tag -a vX.Y.Z -m "vX.Y.Z — description"
git push origin master --tags
```

## Submission Process

### Official Marketplace

1. Go to `platform.claude.com/plugins/submit` (requires Anthropic console login)
2. Fill in: plugin link, homepage, name, description, example use cases
3. Check consent box, select "Claude Code" platform, enter MIT license and contact email
4. Submit — Anthropic reviews (no SLA published)

External PRs to `anthropics/claude-plugins-official` are auto-closed. The form is the only path.

### Standalone Marketplace (works immediately)

Users can install without official approval:

```
/plugin marketplace add srclight/claude-code-plugin
/plugin install srclight
```

This works because `marketplace.json` at the repo root lists the plugin.

## Testing

```bash
claude --plugin-dir ~/Projects/srclight/claude-code-plugin
```

Test checklist:
- [ ] Skills load (`/srclight:setup`, trigger phrases for other skills)
- [ ] Agent appears in `/agents`
- [ ] MCP server starts (srclight tools available)
- [ ] SessionStart hook fires (check for index detection message)
- [ ] `/reload-plugins` picks up changes without restart

## Relationship to Other Distribution Channels

| Channel | Transport | Audience | Install method |
|---------|-----------|----------|----------------|
| **Plugin** | stdio via uvx | Claude Code users | `/plugin install srclight` |
| **SSE service** | SSE on port 8742 | Power users, multi-client | `claude mcp add --transport sse` |
| **srclight-app** | Bundled engine | GUI users | Desktop installer |
| **PyPI** | N/A (library) | Developers | `pip install srclight` |
| **MCP Registry** | stdio | MCP ecosystem | Registry discovery |

## Updating the Plugin

When srclight adds new MCP tools or features:

1. Update `skills/codebase-indexing/SKILL.md` tool tables
2. Update `README.md` tool count
3. Update `agents/codebase-explorer.md` tool list if relevant
4. Bump version in `.claude-plugin/plugin.json`
5. Tag and push
