# tests/test_graph_precision.py
"""The harness's classifier: is a name occurrence real code or only comment/string?

Line-based heuristic (not AST): a reference counts as code if the name appears
on any line OUTSIDE of '#'/'//' comment tails and string literals. Conservative
by design — when in doubt it says "code" (we are measuring FALSE edges; only a
clear comment/string-only occurrence may count as false).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from measure_graph_precision import classify_reference


def test_plain_call_is_code():
    assert classify_reference("def f():\n    helper()\n", "helper") == "code"


def test_name_only_in_comment_is_flagged():
    assert classify_reference("def f():\n    # calls helper eventually\n    pass\n",
                              "helper") == "comment_or_string_only"


def test_name_only_in_string_is_flagged():
    assert classify_reference('def f():\n    return "helper"\n', "helper") == "comment_or_string_only"


def test_name_in_code_AND_comment_is_code():
    assert classify_reference("def f():\n    helper()  # helper does x\n", "helper") == "code"


def test_absent_name():
    assert classify_reference("def f(): pass\n", "helper") == "absent"
