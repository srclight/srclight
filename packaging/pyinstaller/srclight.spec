# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for srclight.

Produces a one-dir bundle named "srclight" with all tree-sitter grammars,
numpy, MCP server stack (uvicorn/starlette/anyio), and the srclight package.

Usage:
    pyinstaller packaging/pyinstaller/srclight.spec

Or use the companion build-engine.sh script for a full reproducible build.
"""

import os
import sys
import importlib
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
is_linux = sys.platform.startswith("linux")
is_macos = sys.platform == "darwin"
is_windows = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Tree-sitter language packages
#
# Each of these ships a native extension (.so / .dylib / .pyd) that
# PyInstaller won't discover automatically because languages.py loads them
# via importlib.import_module() at runtime.
# ---------------------------------------------------------------------------
TREE_SITTER_PACKAGES = [
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_c",
    "tree_sitter_cpp",
    "tree_sitter_rust",
    "tree_sitter_c_sharp",
    "tree_sitter_java",
    "tree_sitter_go",
    "tree_sitter_php",
    "tree_sitter_dart",
    "tree_sitter_swift",
    "tree_sitter_kotlin",
    "tree_sitter_markdown",
    "tree_sitter_lua",
]

# ---------------------------------------------------------------------------
# Hidden imports
#
# Modules that are imported dynamically (importlib, lazy imports inside
# functions, or conditional imports) and won't be found by static analysis.
# ---------------------------------------------------------------------------
hidden_imports = [
    # Core srclight modules (lazy-imported inside CLI commands / server tools)
    "srclight",
    "srclight.cli",
    "srclight.db",
    "srclight.indexer",
    "srclight.server",
    "srclight.web",
    "srclight.git",
    "srclight.languages",
    "srclight.embeddings",
    "srclight.workspace",
    "srclight.vector_cache",
    "srclight.vector_math",
    "srclight.build",
    "srclight.extractors",
    "srclight.extractors.base",
    "srclight.extractors.text_extractor",
    "srclight.extractors.csv_extractor",
    "srclight.extractors.html_extractor",
    "srclight.extractors.docx_extractor",
    "srclight.extractors.xlsx_extractor",
    "srclight.extractors.pdf_extractor",
    "srclight.extractors.email_extractor",
    "srclight.extractors.image_extractor",

    # Tree-sitter core
    "tree_sitter",

    # All tree-sitter language grammars (loaded via importlib.import_module)
    *TREE_SITTER_PACKAGES,

    # Numeric / array support
    "numpy",
    "numpy.core",
    "numpy.core._methods",
    "numpy.core._dtype_ctypes",

    # MCP server framework
    "mcp",
    "mcp.server",
    "mcp.server.fastmcp",
    "mcp.server.sse",

    # ASGI server stack
    "uvicorn",
    "uvicorn.config",
    "uvicorn.main",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.logging",
    "starlette",
    "starlette.applications",
    "starlette.routing",
    "starlette.requests",
    "starlette.responses",
    "starlette.middleware",
    "starlette.types",
    "anyio",
    "anyio._backends",
    "anyio._backends._asyncio",
    "sniffio",

    # HTTP parsing (uvicorn optional accelerator)
    "httptools",
    "httptools.parser",
    "httptools.parser.parser",

    # Click CLI framework
    "click",
    "click.core",
    "click.decorators",
    "click.types",

    # SQLite (stdlib but sometimes needs explicit inclusion on some platforms)
    "sqlite3",
    "_sqlite3",

    # Standard library modules used by srclight
    "importlib",
    "json",
    "logging",
    "hashlib",
    "struct",
    "urllib",
    "urllib.request",
    "urllib.error",
]

# uvloop: Linux/macOS async event loop accelerator (not available on Windows)
if not is_windows:
    hidden_imports.append("uvloop")

# ---------------------------------------------------------------------------
# Collect native extensions (.so / .dylib / .pyd) from tree-sitter packages
# ---------------------------------------------------------------------------
extra_binaries = []
for pkg_name in TREE_SITTER_PACKAGES:
    try:
        extra_binaries += collect_dynamic_libs(pkg_name)
    except Exception:
        print(f"  WARN: Could not collect dynamic libs for {pkg_name}")

# Also collect tree_sitter core native extension
try:
    extra_binaries += collect_dynamic_libs("tree_sitter")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Collect data files from srclight (the package itself -- inline templates etc.)
# In practice web.py has all HTML/CSS/JS inline, but this catches any future
# data files added to the package.
# ---------------------------------------------------------------------------
extra_datas = []
try:
    extra_datas += collect_data_files("srclight")
except Exception:
    pass

# Collect data files from mcp (may include JSON schemas, etc.)
try:
    extra_datas += collect_data_files("mcp")
except Exception:
    pass

# Collect data files from starlette (may include default error templates)
try:
    extra_datas += collect_data_files("starlette")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [os.path.join(SPECPATH, "entry_point.py")],
    pathex=[os.path.join(SPECPATH, "..", "..", "src")],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # GPU / heavy optional deps -- not bundled in the base build
        "cupy",
        "cupy_cuda12x",
        "paddleocr",
        "paddle",
        "paddlepaddle",
        "pdf2image",
        "pytesseract",
        "torch",
        "tensorflow",
        "tkinter",
        "_tkinter",
        "test",
        "unittest",
    ],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# PYZ (compressed Python bytecode archive)
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data)

# ---------------------------------------------------------------------------
# EXE
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # one-dir mode: binaries go in COLLECT
    name="srclight",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # CLI tool, not windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---------------------------------------------------------------------------
# COLLECT (one-dir bundle)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="srclight",
)
