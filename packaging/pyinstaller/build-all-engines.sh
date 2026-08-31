#!/usr/bin/env bash
# ============================================================================
# build-all-engines.sh — Build frozen srclight engines for all 3 platforms
# ============================================================================
#
# Builds PyInstaller engines on Linux (local), macOS (SSH to Mac Mini),
# and optionally Windows (manual step — printed instructions).
#
# Output: srclight-app/dist/engine-{linux,macos,windows}/
# These directories are referenced by loqu8-app.yaml data entries and
# injected into platform-specific archives by release.sh step 4.
#
# Usage:
#   ./packaging/pyinstaller/build-all-engines.sh [options]
#
# Options:
#   --app-dir PATH    Path to srclight-app (default: ../srclight-app relative to repo)
#   --mac-host HOST   Mac Mini SSH host (default: tim@10.1.10.103)
#   --mac-repo PATH   Srclight repo on Mac (default: ~/repos/srclight/srclight)
#   --skip-linux      Skip Linux engine build
#   --skip-macos      Skip macOS engine build
#   --skip-venv       Pass SKIP_VENV=1 to build-engine.sh (use existing venv)
#
# Prerequisites:
#   - Linux: Python 3, pip, tree-sitter-dart installed from GitHub
#   - macOS: SSH access to Mac Mini, srclight repo cloned there
#   - Windows: Python 3 on Windows, run build-engine.sh manually
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
APP_DIR=""
MAC_HOST="tim@10.1.10.103"
MAC_REPO="\$HOME/repos/srclight/srclight"
DO_LINUX=true
DO_MACOS=true
SKIP_VENV=0

# ── Parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --app-dir)    APP_DIR="$2"; shift ;;
        --mac-host)   MAC_HOST="$2"; shift ;;
        --mac-repo)   MAC_REPO="$2"; shift ;;
        --skip-linux) DO_LINUX=false ;;
        --skip-macos) DO_MACOS=false ;;
        --skip-venv)  SKIP_VENV=1 ;;
        --help|-h)
            sed -n '2,/^# =====/p' "$0" | head -30
            exit 0
            ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

# Auto-detect app dir
if [[ -z "$APP_DIR" ]]; then
    # Try sibling directory
    if [[ -d "$REPO_ROOT/../srclight-app" ]]; then
        APP_DIR="$(cd "$REPO_ROOT/../srclight-app" && pwd)"
    else
        echo "ERROR: Cannot find srclight-app. Use --app-dir PATH."
        exit 1
    fi
fi

echo "=== Build All Engines ==="
echo "  Engine repo: $REPO_ROOT"
echo "  App dir:     $APP_DIR"
echo "  Mac host:    $MAC_HOST"
echo ""

ERRORS=0

# ── Linux engine ─────────────────────────────────────────────────────────
if $DO_LINUX; then
    echo "━━━ Linux Engine ━━━"
    cd "$REPO_ROOT"

    SKIP_VENV=$SKIP_VENV "$SCRIPT_DIR/build-engine.sh"

    if [[ -d "$REPO_ROOT/dist/srclight" ]]; then
        DEST="$APP_DIR/dist/engine-linux"
        rm -rf "$DEST"
        mkdir -p "$DEST"
        cp -a "$REPO_ROOT/dist/srclight/." "$DEST/"
        echo "  => Copied to $DEST"
        echo "  => Size: $(du -sh "$DEST" | cut -f1)"
    else
        echo "ERROR: Linux engine build failed — dist/srclight/ not found"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
fi

# ── macOS engine ─────────────────────────────────────────────────────────
if $DO_MACOS; then
    echo "━━━ macOS Engine ━━━"

    # Check Mac Mini reachability
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$MAC_HOST" "echo ok" &>/dev/null; then
        echo "ERROR: Mac Mini ($MAC_HOST) not reachable"
        ERRORS=$((ERRORS + 1))
    else
        echo "  Building on $MAC_HOST..."

        # Sync packaging scripts to Mac
        ssh "$MAC_HOST" "mkdir -p $MAC_REPO/packaging/pyinstaller"
        scp -q "$SCRIPT_DIR/build-engine.sh" "$SCRIPT_DIR/srclight.spec" \
            "$MAC_HOST:$MAC_REPO/packaging/pyinstaller/"

        ssh "$MAC_HOST" bash -l << MACSCRIPT
set -euo pipefail
cd $MAC_REPO

# Ensure tree-sitter-dart is installed (not on PyPI)
if ! python3 -c "import tree_sitter_dart" 2>/dev/null; then
    echo "Installing tree-sitter-dart from GitHub..."
    pip3 install "git+https://github.com/UserNobody14/tree-sitter-dart.git"
fi

SKIP_VENV=${SKIP_VENV} ./packaging/pyinstaller/build-engine.sh

echo "DONE_MAC_BUILD"
MACSCRIPT

        # Copy engine back
        DEST="$APP_DIR/dist/engine-macos"
        rm -rf "$DEST"
        mkdir -p "$DEST"
        scp -rq "$MAC_HOST:$MAC_REPO/dist/srclight/." "$DEST/"
        echo "  => Copied to $DEST"
        echo "  => Size: $(du -sh "$DEST" | cut -f1)"
    fi
    echo ""
fi

# ── Windows engine ───────────────────────────────────────────────────────
echo "━━━ Windows Engine ━━━"
WIN_ENGINE_DIR="$APP_DIR/dist/engine-windows"
if [[ -d "$WIN_ENGINE_DIR" && -f "$WIN_ENGINE_DIR/srclight.exe" ]]; then
    echo "  Found existing Windows engine at $WIN_ENGINE_DIR"
    echo "  Size: $(du -sh "$WIN_ENGINE_DIR" | cut -f1)"
    echo "  To rebuild, delete $WIN_ENGINE_DIR and run manually on Windows."
else
    echo "  Windows engine not found at $WIN_ENGINE_DIR"
    echo ""
    echo "  PyInstaller cannot cross-compile. Build on Windows manually:"
    echo ""
    echo "    cd C:\\Projects\\srclight\\srclight"
    echo "    pip install \"git+https://github.com/UserNobody14/tree-sitter-dart.git\""
    echo "    pip install -e .[docs,pdf]"
    echo "    pip install pyinstaller"
    echo "    pyinstaller packaging\\pyinstaller\\srclight.spec --clean --noconfirm"
    echo ""
    echo "  Then copy dist\\srclight\\ to:"
    echo "    $WIN_ENGINE_DIR"
    echo ""

    # If we're in WSL and Python is available on Windows, offer to try
    if grep -qi microsoft /proc/version 2>/dev/null; then
        WIN_PYTHON=""
        for wp in "/mnt/c/Python312/python.exe" "/mnt/c/Python311/python.exe" \
                   "/mnt/c/Users/$USER/AppData/Local/Programs/Python/Python312/python.exe"; do
            if [[ -f "$wp" ]]; then
                WIN_PYTHON="$wp"
                break
            fi
        done
        if [[ -n "$WIN_PYTHON" ]]; then
            echo "  Found Windows Python: $WIN_PYTHON"
            echo "  To attempt automated build, run:"
            echo "    WIN_PYTHON=\"$WIN_PYTHON\" $0 --skip-linux --skip-macos"
        fi
    fi
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
echo "=== Engine Summary ==="
for plat in linux macos windows; do
    dir="$APP_DIR/dist/engine-$plat"
    if [[ -d "$dir" ]]; then
        SIZE=$(du -sh "$dir" | cut -f1)
        echo "  $plat: $SIZE ✓"
    else
        echo "  $plat: NOT BUILT"
    fi
done

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "ERROR: $ERRORS platform(s) failed. Fix and retry."
    exit 1
fi

echo ""
echo "Next: run release.sh to build installers with engines injected."
echo "  loqu8-dart/tool/release.sh $APP_DIR VERSION --skip-git --skip-upload"
