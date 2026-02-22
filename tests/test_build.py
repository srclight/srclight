"""Tests for build system intelligence layer."""

from pathlib import Path

import pytest

from srclight.build import (
    get_build_info,
    get_platform_variants,
    parse_cmake_targets,
    parse_csproj_deps,
    scan_platform_conditionals,
)


@pytest.fixture
def cmake_repo(tmp_path):
    """Create a fake repo with CMakeLists.txt."""
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text("""
cmake_minimum_required(VERSION 3.16)
project(myproject)

set(LIB_SOURCES
    src/main.cpp
    src/utils.cpp
)

add_library(mylib STATIC ${LIB_SOURCES})
add_executable(myapp src/app.cpp)

target_link_libraries(myapp PRIVATE mylib pthread)

if(WIN32)
    add_library(win_helper STATIC src/win_helper.cpp)
endif()
""")

    # Create a source file with platform conditionals
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.cpp").write_text("""
#include <stdio.h>

#ifdef _WIN32
void platform_init() {
    // Windows init
}
#elif __linux__
void platform_init() {
    // Linux init
}
#elif __APPLE__
void platform_init() {
    // macOS/iOS init
}
#endif

void common_func() {
    platform_init();
}
""")

    return tmp_path


@pytest.fixture
def csproj_repo(tmp_path):
    """Create a fake repo with .csproj files."""
    proj = tmp_path / "MyProject.csproj"
    proj.write_text("""
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="Microsoft.Extensions.Logging" Version="8.0.0" />
    <ProjectReference Include="../SharedLib/SharedLib.csproj" />
  </ItemGroup>
</Project>
""")
    return tmp_path


def test_parse_cmake_targets(cmake_repo):
    targets = parse_cmake_targets(cmake_repo)
    names = {t["name"] for t in targets}
    assert "mylib" in names
    assert "myapp" in names

    # Check mylib details
    mylib = next(t for t in targets if t["name"] == "mylib")
    assert mylib["type"] == "library"
    assert mylib["lib_type"] == "STATIC"

    # Check myapp dependencies
    myapp = next(t for t in targets if t["name"] == "myapp")
    assert "mylib" in myapp.get("dependencies", [])


def test_scan_platform_conditionals(cmake_repo):
    conditionals = scan_platform_conditionals(cmake_repo)
    assert len(conditionals) >= 3

    platforms_found = set()
    for c in conditionals:
        platforms_found.update(c["platforms"])

    assert "windows" in platforms_found
    assert "linux" in platforms_found
    assert "apple" in platforms_found


def test_get_platform_variants(cmake_repo):
    variants = get_platform_variants(cmake_repo, "platform_init")
    assert len(variants) >= 2
    # Should find platform_init under different #ifdef blocks
    platform_sets = [tuple(v["platforms"]) for v in variants]
    assert any("windows" in ps for ps in platform_sets)
    assert any("linux" in ps for ps in platform_sets)


def test_parse_csproj_deps(csproj_repo):
    deps = parse_csproj_deps(csproj_repo)
    assert len(deps) == 1
    proj = deps[0]
    assert proj["name"] == "MyProject"

    pkg_names = [p["name"] for p in proj["packages"]]
    assert "Newtonsoft.Json" in pkg_names
    assert "Microsoft.Extensions.Logging" in pkg_names

    assert len(proj["project_refs"]) == 1
    assert "SharedLib" in proj["project_refs"][0]
    assert proj["target_frameworks"] == ["net8.0"]


def test_get_build_info_cmake(cmake_repo):
    info = get_build_info(cmake_repo)
    assert "cmake" in info["build_systems"]
    assert len(info["targets"]) >= 2


def test_get_build_info_empty(tmp_path):
    info = get_build_info(tmp_path)
    assert info["build_systems"] == []
    assert info["targets"] == []
