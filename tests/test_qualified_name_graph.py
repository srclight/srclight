"""A qualified C++ method name must reach both halves of the method.

A method declared in its class and defined outside it is two symbols: the
declaration, named `moveNeedle` with qualified name `Panel_c::moveNeedle`,
and the definition, named `Panel_c::moveNeedle`. Calls written
`panel->moveNeedle()` land on the declaration, while the body scanned for
callees is the definition's. Resolving the qualified name to the definition
alone, get_callees found the method's calls and get_callers found nobody.
"""
import json

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer

PANEL_H = """\
class Panel_c {
public:
    void moveNeedle();
};
"""

PANEL_CPP = """\
#include "panel.h"

void Panel_c::moveNeedle() {
    clearGauge();
}

void clearGauge() {
}

void runPanel(Panel_c* panel) {
    panel->moveNeedle();
}
"""


@pytest.fixture
def served(tmp_path, monkeypatch):
    from srclight import server as server_mod

    root = tmp_path / "repo"
    root.mkdir()
    (root / "panel.h").write_text(PANEL_H)
    (root / "panel.cpp").write_text(PANEL_CPP)
    db_path = root / ".srclight" / "index.db"
    db_path.parent.mkdir()
    db = Database(db_path)
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=root)).index()
    db.close()

    # _get_db() walks up from the CWD — never let a test reach a real index.
    monkeypatch.chdir(root)
    monkeypatch.setattr(server_mod, "_workspace_name", None)
    server_mod.configure(db_path=db_path, repo_root=root)
    yield server_mod
    server_mod._close_databases()
    server_mod.configure(db_path=None, repo_root=None)


def _names(payload: str, key: str) -> set[str]:
    return {entry["name"] for entry in json.loads(payload).get(key, [])}


def test_get_callers_finds_callers_by_the_qualified_name(served):
    assert "runPanel" in _names(served.get_callers("Panel_c::moveNeedle"), "callers")


def test_get_callees_still_finds_callees_by_the_qualified_name(served):
    assert "clearGauge" in _names(served.get_callees("Panel_c::moveNeedle"), "callees")


def test_the_bare_name_keeps_resolving_as_before(served):
    assert "runPanel" in _names(served.get_callers("moveNeedle"), "callers")
