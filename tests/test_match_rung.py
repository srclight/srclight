"""The match ladder: a pure function of (query, name), no database.

It lives in db.py because BOTH modes score with it — workspace mode and the
single-repo mode the published plugin runs.

A lower rung never outranks a higher one, however strong its statistics. The
scoring bug this replaces let bm25 and a hardcoded LIKE constant decide across
tiers, so a symbol FTS5 found could sort below one only LIKE could find.
"""

import pytest

from srclight.db import RUNG_NONE, match_rung


@pytest.mark.parametrize("query,name,expected", [
    # exact, case-sensitive, is the top rung
    ("checkPoint", "checkPoint", 0),
    # exact but for case
    ("checkpoint", "checkPoint", 1),
    ("CHECKPOINT", "checkpoint", 1),
    # exact once the identifier is folded to words
    ("check point", "checkPoint", 2),
    ("check_point", "checkPoint", 2),
    # every token present as a word-part, in the name's order
    ("wal checkpoint", "WalCheckpointer", 3),
    ("user wal", "UserWalCheckpointer", 3),
    # every token present, wrong order
    ("checkpoint wal", "WalCheckpointer", 4),
    # substring only
    ("checkpoint", "getcheckpointing", 5),
    # no match on the name at all
    ("checkpoint", "unrelated", RUNG_NONE),
])
def test_rung_assignment(query, name, expected):
    assert match_rung(query, name) == expected


def test_a_lower_rung_never_outranks_a_higher_one():
    """The ordering property, stated directly."""
    q = "checkpoint"
    ordered = ["checkpoint", "checkPoint", "check point", "WalCheckpoint",
               "getcheckpointing", "unrelated"]
    rungs = [match_rung(q, n) for n in ordered]
    assert rungs == sorted(rungs), f"ladder not monotonic: {list(zip(ordered, rungs))}"


def test_exact_match_is_case_insensitive_in_practice():
    """`if name == query` was case-sensitive, so checkPoint got no bonus at all."""
    assert match_rung("checkpoint", "checkPoint") < match_rung("checkpoint", "WalCheckpoint")
