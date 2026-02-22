"""Build system intelligence layer for Srclight.

Parses CMakeLists.txt, .csproj, package.json, Cargo.toml to extract
build targets, dependencies, and platform conditionals. Also scans
C/C++ files for #ifdef platform guards.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("srclight.build")

# Common platform #ifdef macros
PLATFORM_MACROS = {
    # Windows
    "_WIN32": "windows",
    "_WIN64": "windows",
    "WIN32": "windows",
    "_MSC_VER": "windows",
    "__MINGW32__": "windows",
    "__MINGW64__": "windows",
    # Linux
    "__linux__": "linux",
    "__linux": "linux",
    "__gnu_linux__": "linux",
    # macOS
    "__APPLE__": "apple",
    "__MACH__": "apple",
    "TARGET_OS_MAC": "macos",
    "TARGET_OS_IPHONE": "ios",
    "TARGET_IPHONE_SIMULATOR": "ios-simulator",
    # iOS
    "__IPHONE_OS_VERSION_MIN_REQUIRED": "ios",
    # Android
    "__ANDROID__": "android",
    "ANDROID": "android",
    # FreeBSD/Unix
    "__FreeBSD__": "freebsd",
    "__unix__": "unix",
    # Architecture
    "__x86_64__": "x86_64",
    "__aarch64__": "aarch64",
    "__arm__": "arm",
    "__i386__": "x86",
    # Compilers
    "__clang__": "clang",
    "__GNUC__": "gcc",
}

# CMake platform conditionals
CMAKE_PLATFORM_VARS = {
    "WIN32": "windows",
    "APPLE": "apple",
    "UNIX": "unix",
    "LINUX": "linux",
    "ANDROID": "android",
    "IOS": "ios",
    "CMAKE_SYSTEM_NAME STREQUAL \"Windows\"": "windows",
    "CMAKE_SYSTEM_NAME STREQUAL \"Linux\"": "linux",
    "CMAKE_SYSTEM_NAME STREQUAL \"Darwin\"": "macos",
    "CMAKE_SYSTEM_NAME STREQUAL \"Android\"": "android",
}


def scan_platform_conditionals(repo_root: Path) -> list[dict[str, Any]]:
    """Scan C/C++/C# files for #ifdef platform guards.

    Returns a list of platform-conditional code blocks with file, line,
    platform, and the conditional expression.
    """
    results = []

    # File extensions to scan
    extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".cs", ".m", ".mm"}

    for f in repo_root.rglob("*"):
        if f.suffix not in extensions:
            continue
        # Skip vendored/build dirs
        rel = str(f.relative_to(repo_root))
        if any(part in rel for part in [".git/", "build/", "depends/", "third_party/", ".srclight/", ".codelight/"]):
            continue

        try:
            content = f.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # Match #if, #ifdef, #ifndef, #elif
            m = re.match(r"#\s*(if|ifdef|ifndef|elif)\s+(.+)", stripped)
            if not m:
                continue

            directive = m.group(1)
            expr = m.group(2).strip()

            # Check for platform macros
            platforms = set()
            for macro, platform in PLATFORM_MACROS.items():
                if macro in expr:
                    platforms.add(platform)

            if platforms:
                results.append({
                    "file": rel,
                    "line": i,
                    "directive": f"#{directive}",
                    "expression": expr,
                    "platforms": sorted(platforms),
                })

    return results


def get_platform_variants(repo_root: Path, symbol_name: str) -> list[dict[str, Any]]:
    """Find platform-specific variants of a symbol.

    Scans for the symbol name near #ifdef blocks to find platform-conditional
    implementations.
    """
    results = []
    extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".cs", ".m", ".mm"}

    for f in repo_root.rglob("*"):
        if f.suffix not in extensions:
            continue
        rel = str(f.relative_to(repo_root))
        if any(part in rel for part in [".git/", "build/", "depends/", "third_party/", ".srclight/", ".codelight/"]):
            continue

        try:
            lines = f.read_text(errors="ignore").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        # Find the symbol in this file
        current_platform = None
        platform_stack = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track platform conditionals
            m = re.match(r"#\s*(if|ifdef|ifndef)\s+(.+)", stripped)
            if m:
                expr = m.group(2).strip()
                platforms = set()
                for macro, platform in PLATFORM_MACROS.items():
                    if macro in expr:
                        platforms.add(platform)
                platform_stack.append(sorted(platforms) if platforms else None)
                if platforms:
                    current_platform = sorted(platforms)
            elif re.match(r"#\s*(elif)\s+(.+)", stripped):
                m2 = re.match(r"#\s*elif\s+(.+)", stripped)
                expr = m2.group(1).strip() if m2 else ""
                platforms = set()
                for macro, platform in PLATFORM_MACROS.items():
                    if macro in expr:
                        platforms.add(platform)
                if platform_stack:
                    platform_stack[-1] = sorted(platforms) if platforms else None
                if platforms:
                    current_platform = sorted(platforms)
            elif re.match(r"#\s*else", stripped):
                # Invert — we're in the else branch
                if platform_stack and platform_stack[-1]:
                    current_platform = [f"!{p}" for p in platform_stack[-1]]
                    platform_stack[-1] = current_platform
            elif re.match(r"#\s*endif", stripped):
                if platform_stack:
                    platform_stack.pop()
                current_platform = None
                for ps in reversed(platform_stack):
                    if ps:
                        current_platform = ps
                        break

            # Check if this line defines/contains the symbol
            if symbol_name in line and current_platform:
                results.append({
                    "file": rel,
                    "line": i,
                    "platforms": current_platform,
                    "context": stripped[:120],
                })

    return results


def parse_cmake_targets(repo_root: Path) -> list[dict[str, Any]]:
    """Parse CMakeLists.txt files for build targets.

    Extracts add_library, add_executable targets with their sources
    and linked libraries.
    """
    targets = []

    for cmake_file in repo_root.rglob("CMakeLists.txt"):
        rel = str(cmake_file.relative_to(repo_root))
        if any(part in rel for part in ["build/", "depends/", "third_party/", ".srclight/", ".codelight/"]):
            continue

        try:
            content = cmake_file.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        # Find add_library targets
        for m in re.finditer(
            r"add_library\s*\(\s*(\w+)\s+(STATIC|SHARED|MODULE|INTERFACE|OBJECT)?\s*(.+?)\)",
            content, re.DOTALL,
        ):
            name = m.group(1)
            lib_type = m.group(2) or "STATIC"
            sources_raw = m.group(3).strip()
            # Expand variable references
            sources = _extract_sources(sources_raw, content)
            targets.append({
                "name": name,
                "type": "library",
                "lib_type": lib_type,
                "cmake_file": rel,
                "sources": sources[:20],  # Cap for readability
                "source_count": len(sources),
            })

        # Find add_executable targets
        for m in re.finditer(
            r"add_executable\s*\(\s*(\w+)\s+(.+?)\)",
            content, re.DOTALL,
        ):
            name = m.group(1)
            sources_raw = m.group(2).strip()
            sources = _extract_sources(sources_raw, content)
            targets.append({
                "name": name,
                "type": "executable",
                "cmake_file": rel,
                "sources": sources[:20],
                "source_count": len(sources),
            })

        # Find target_link_libraries
        for m in re.finditer(
            r"target_link_libraries\s*\(\s*(\w+)\s+((?:PUBLIC|PRIVATE|INTERFACE)\s+)?(.+?)\)",
            content, re.DOTALL,
        ):
            target_name = m.group(1)
            deps_raw = m.group(3).strip()
            deps = [d.strip() for d in re.split(r"\s+", deps_raw)
                    if d.strip() and d.strip() not in ("PUBLIC", "PRIVATE", "INTERFACE")]
            # Attach to existing target
            for t in targets:
                if t["name"] == target_name:
                    t.setdefault("dependencies", []).extend(deps)

        # Find platform conditionals in CMake
        for m in re.finditer(
            r"if\s*\(\s*(WIN32|APPLE|UNIX|LINUX|ANDROID|IOS|MINGW)\s*\)",
            content,
        ):
            platform = CMAKE_PLATFORM_VARS.get(m.group(1), m.group(1).lower())
            # Find what's inside this if block
            start = m.end()
            depth = 1
            pos = start
            while pos < len(content) and depth > 0:
                if re.match(r"\s*if\s*\(", content[pos:]):
                    depth += 1
                elif re.match(r"\s*endif\s*\(", content[pos:]):
                    depth -= 1
                pos += 1
            block = content[start:pos]
            # Check for targets in conditional
            for tm in re.finditer(r"(?:add_library|add_executable)\s*\(\s*(\w+)", block):
                for t in targets:
                    if t["name"] == tm.group(1):
                        t.setdefault("platform_conditions", []).append(platform)

    return targets


def parse_csproj_deps(repo_root: Path) -> list[dict[str, Any]]:
    """Parse .csproj files for NuGet package references and project references."""
    results = []

    for csproj in repo_root.rglob("*.csproj"):
        rel = str(csproj.relative_to(repo_root))
        if any(part in rel for part in ["bin/", "obj/", "packages/", ".vs/"]):
            continue

        try:
            content = csproj.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        project_info: dict[str, Any] = {
            "file": rel,
            "name": csproj.stem,
            "packages": [],
            "project_refs": [],
        }

        # PackageReference
        for m in re.finditer(
            r'<PackageReference\s+Include="([^"]+)"(?:\s+Version="([^"]+)")?',
            content,
        ):
            project_info["packages"].append({
                "name": m.group(1),
                "version": m.group(2) or "unspecified",
            })

        # ProjectReference
        for m in re.finditer(r'<ProjectReference\s+Include="([^"]+)"', content):
            project_info["project_refs"].append(m.group(1))

        # TargetFramework
        m = re.search(r"<TargetFramework(?:s)?>(.*?)</TargetFramework(?:s)?>", content)
        if m:
            project_info["target_frameworks"] = m.group(1).split(";")

        if project_info["packages"] or project_info["project_refs"]:
            results.append(project_info)

    return results


def _extract_sources(sources_raw: str, cmake_content: str) -> list[str]:
    """Extract source file paths from CMake variable references and inline lists."""
    sources = []

    # Check for variable reference like ${DICT_LIB_SRC}
    var_refs = re.findall(r"\$\{(\w+)\}", sources_raw)
    for var in var_refs:
        # Try to find the variable definition
        m = re.search(
            rf"set\s*\(\s*{re.escape(var)}\s+(.+?)\)",
            cmake_content, re.DOTALL,
        )
        if m:
            var_content = m.group(1)
            for path in re.findall(r"[\w${}/._-]+\.(?:cpp|cc|cxx|c|h|hpp|cs|m|mm)", var_content):
                # Strip ${CMAKE_CURRENT_SOURCE_DIR}/ prefix
                path = re.sub(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/", "", path)
                sources.append(path)

    # Also check for inline source files
    for path in re.findall(r"[\w${}/._-]+\.(?:cpp|cc|cxx|c|h|hpp|cs|m|mm)", sources_raw):
        path = re.sub(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/", "", path)
        if path not in sources:
            sources.append(path)

    return sources


def get_build_info(repo_root: Path) -> dict[str, Any]:
    """Get comprehensive build system info for a repo.

    Detects build system type (CMake, .csproj, package.json, Cargo.toml)
    and parses what it finds.
    """
    result: dict[str, Any] = {
        "build_systems": [],
        "targets": [],
        "dependencies": [],
    }

    # CMake
    if (repo_root / "CMakeLists.txt").exists():
        result["build_systems"].append("cmake")
        targets = parse_cmake_targets(repo_root)
        result["targets"].extend(targets)

    # .csproj
    csproj_deps = parse_csproj_deps(repo_root)
    if csproj_deps:
        result["build_systems"].append("dotnet")
        result["dependencies"].extend(csproj_deps)

    # package.json
    pkg_json = repo_root / "package.json"
    if pkg_json.exists():
        result["build_systems"].append("npm")
        try:
            import json
            pkg = json.loads(pkg_json.read_text())
            deps = pkg.get("dependencies", {})
            dev_deps = pkg.get("devDependencies", {})
            result["dependencies"].append({
                "file": "package.json",
                "name": pkg.get("name", repo_root.name),
                "packages": [
                    {"name": k, "version": v, "dev": False}
                    for k, v in deps.items()
                ] + [
                    {"name": k, "version": v, "dev": True}
                    for k, v in dev_deps.items()
                ],
            })
        except (json.JSONDecodeError, OSError):
            pass

    # Cargo.toml
    cargo = repo_root / "Cargo.toml"
    if cargo.exists():
        result["build_systems"].append("cargo")

    # pubspec.yaml (Dart/Flutter)
    pubspec = repo_root / "pubspec.yaml"
    if pubspec.exists():
        result["build_systems"].append("flutter")

    return result
