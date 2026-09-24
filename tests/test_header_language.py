"""A `.h` file is C unless it holds C++ — anywhere in it, not only in its head.

Detection read the first 4096 bytes. A header that opens with pages of
C-compatible declarations and reaches its first class further down was
parsed as C, where the class and its inline methods do not exist.
"""
from srclight.languages import detect_language

C_PREAMBLE = "".join(f"inline int helper{i}(int v) {{ return v + {i}; }}\n" for i in range(200))


def _header(tmp_path, text: str):
    path = tmp_path / "late.h"
    path.write_text(text)
    return path


def test_a_class_far_below_the_head_makes_a_header_cpp(tmp_path):
    path = _header(tmp_path, C_PREAMBLE + "class Stack_c {\npublic:\n    void push();\n};\n")
    assert len(C_PREAMBLE) > 4096
    assert detect_language(path) == "cpp"


def test_a_namespace_or_template_far_below_the_head_makes_a_header_cpp(tmp_path):
    assert detect_language(_header(tmp_path, C_PREAMBLE + "namespace tools {\n}\n")) == "cpp"
    assert detect_language(_header(tmp_path, C_PREAMBLE + "template <typename T>\nT twice(T v);\n")) == "cpp"


def test_a_derived_struct_far_below_the_head_makes_a_header_cpp(tmp_path):
    text = C_PREAMBLE + "struct Poly_c : public Base_c {\n    int n;\n};\n"
    assert detect_language(_header(tmp_path, text)) == "cpp"


def test_a_plain_c_header_stays_c(tmp_path):
    text = C_PREAMBLE + "struct point { int x; int y; };\n/* a class of errors */\nint classify(int v);\n"
    assert detect_language(_header(tmp_path, text)) == "c"
