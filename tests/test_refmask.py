# tests/test_refmask.py
"""Masking must remove names in comments/strings and NOTHING else, offset-stable."""
from srclight.refmask import mask_noncode


def test_python_hash_comment_masked():
    out = mask_noncode("x = helper()  # calls helper\n", "python")
    assert "helper()" in out                  # the code call survives
    assert out.count("helper") == 1           # the comment copy is gone
    assert len(out) == len("x = helper()  # calls helper\n")


def test_python_string_masked():
    out = mask_noncode('name = "helper"\n', "python")
    assert "helper" not in out


def test_python_docstring_masked_across_lines():
    src = 'def f():\n    """uses helper\n    a lot"""\n    helper()\n'
    out = mask_noncode(src, "python")
    assert out.count("helper") == 1           # only the real call remains
    assert out.count("\n") == src.count("\n")  # newlines preserved


def test_c_block_comment_masked():
    src = "int f() {\n/* helper does x */\nreturn helper();\n}\n"
    out = mask_noncode(src, "cpp")
    assert out.count("helper") == 1


def test_cpp_line_comment_masked():
    out = mask_noncode("int y = helper(); // helper again\n", "cpp")
    assert out.count("helper") == 1


def test_escaped_quote_does_not_derail():
    out = mask_noncode('s = "a \\" quote"; helper();\n', "cpp")
    assert "helper()" in out


def test_python_hash_not_treated_as_comment_in_c_include():
    out = mask_noncode('#include "helper.h"\nhelper();\n', "cpp")
    assert "#include" in out                  # preprocessor line survives masking
    assert out.count("helper") >= 1           # the call survives


def test_lua_line_comment_masked():
    src = "local x = helper()  -- calls helper\n"
    out = mask_noncode(src, "lua")
    assert "helper()" in out                  # the code call survives
    assert out.count("helper") == 1           # the comment copy is gone
    assert len(out) == len(src)


def test_lua_block_comment_masked_across_lines():
    src = "--[[ helper does x\n     and y ]]\nhelper()\n"
    out = mask_noncode(src, "lua")
    assert out.count("helper") == 1
    assert out.count("\n") == src.count("\n")  # newlines preserved


def test_lua_long_string_masked():
    src = "local s = [[helper]]\nhelper()\n"
    out = mask_noncode(src, "lua")
    assert out.count("helper") == 1


def test_lua_length_operator_is_not_a_comment():
    """`#` counts a table in Lua — it must not blank the rest of the line."""
    out = mask_noncode("local n = #items + helper()\n", "lua")
    assert "helper()" in out


def test_lua_floor_division_is_not_a_comment():
    """`//` divides in Lua — the grammar reads it as an operator, so must we."""
    out = mask_noncode("local q = total // helper()\n", "lua")
    assert "helper()" in out


def test_long_brackets_stay_inside_lua():
    """`[[` opens a string in Lua only; elsewhere it is a nested index.

    An unclosed one blanks everything after it, so letting the rule reach
    another language would erase whole symbol bodies.
    """
    out = mask_noncode("SELECT ARRAY[[1,2] FROM t;\nSELECT helper();\n", "sql")
    assert "helper()" in out


def test_lua_long_bracket_with_levels_masked():
    """`[=[ ]=]` and `--[==[ ]==]` are the same construct at another level."""
    out = mask_noncode("local s = [==[helper]==]\n--[=[ helper ]=]\nhelper()\n", "lua")
    assert out.count("helper") == 1
