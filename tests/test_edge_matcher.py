"""The name matcher behind the call graph.

_build_edges scans every symbol body for references to known symbol names. The
matching has to survive C++ name shapes that are easy to get wrong: qualified
chains, destructors, and names carrying characters no tokeniser can split on.

The end-to-end tests pin behaviour observed on the alternation implementation,
so they stay meaningful across any change of matching strategy. The property
test compares the matcher against a reference alternation built in the test
itself — the shapes it covers are the ones the extractor does not produce from
a small fixture, but which a real C++ codebase does.
"""

import random
import re

import pytest

from srclight.db import Database
from srclight.indexer import IndexConfig, Indexer, build_name_matcher


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def cpp_project(tmp_path):
    """C++ names the matcher has to get right. Every name clears the noise
    filters of _build_edges (length >= 4, not in NOISE_NAMES)."""
    src = tmp_path / "cppproj"
    src.mkdir()

    (src / "widget.hpp").write_text('''\
struct WidgetPanel {
    int slot_value;
};

struct PanelRegistry {
    static int lookup_slot(void);
};

struct ExtensionHost {
    static PanelRegistry registry_for(void);
};
''')

    (src / "widget.cpp").write_text('''\
#include "widget.hpp"

WidgetPanel::~WidgetPanel() {
    slot_value = 0;
}

int mask_value(void) {
    return 7;
}

int clear_mask(void) {
    return ~mask_value();
}

int outer_caller(void) {
    return ExtensionHost::PanelRegistry::lookup_slot();
}
''')

    return src


def _calls(db, source_name: str) -> set[str]:
    return {
        row["name"]
        for row in db.conn.execute(
            """SELECT t.name AS name FROM symbol_edges e
               JOIN symbols s ON e.source_id = s.id
               JOIN symbols t ON e.target_id = t.id
               WHERE e.edge_type = 'calls' AND s.name = ?""",
            (source_name,),
        )
    }


@pytest.fixture
def indexed(db, cpp_project):
    Indexer(db, IndexConfig(root=cpp_project)).index(cpp_project)
    return db


def test_qualified_chain_yields_every_known_segment(indexed):
    """`A::B::c()` references A, B and c when each is a known symbol.

    The chain is not itself a symbol name, so the matcher has to walk it
    segment by segment rather than treat it as one opaque token.
    """
    assert {"ExtensionHost", "PanelRegistry", "lookup_slot"} <= _calls(indexed, "outer_caller")


def test_destructor_body_does_not_reference_its_own_class(indexed):
    """`WidgetPanel::~WidgetPanel` must not yield a bare `WidgetPanel`.

    The destructor's own qualified name appears in its body, and it is the
    longest name matching at that position, so it is consumed whole and
    discarded as a self-reference. A matcher that handles destructors in a
    separate pass emits the class name as well and invents an edge.
    """
    assert "WidgetPanel" not in _calls(indexed, "WidgetPanel::~WidgetPanel")


def test_bitwise_complement_does_not_hide_the_name(indexed):
    """`~mask_value()` is a complement applied to a call, not a destructor.

    A matcher that lets a token start with `~` swallows the identifier and
    loses the reference — far more common in C than destructors.
    """
    assert "mask_value" in _calls(indexed, "clear_mask")


# --- property: the matcher agrees with a reference alternation ---------------

def _reference_matcher(names):
    """The alternation _build_edges used before, kept as the oracle.

    One regex, alternatives ordered longest-first, so at any position the
    longest name wins and shorter names inside it never surface.
    """
    ordered = sorted(names, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in ordered) + r")\b")
    return lambda content: set(pattern.findall(content))


REFERENCE_CASES = [
    pytest.param(
        {"Registry", "Registry::Lookup", "Extension"},
        "Extension::Registry::Lookup();",
        id="longest-qualified-name-wins-over-its-prefix",
    ),
    pytest.param(
        {"Widget", "Widget::~Widget"},
        "Widget::~Widget() { reset(); }",
        id="destructor-consumes-the-class-name",
    ),
    pytest.param(
        {"mask_value", "Widget::~Widget"},
        "return ~mask_value();",
        id="complement-is-not-a-destructor",
    ),
    pytest.param(
        # A name no tokeniser can split on, overlapping names that it can.
        {"ObjectExtension", "Register", "ObjectExtension::Register<T>::Ident"},
        "return ObjectExtension::Register<T>::Ident;",
        id="untokenisable-name-shadows-the-names-inside-it",
    ),
    pytest.param(
        # `\b` after `=` needs a word character next, so a name ending in
        # punctuation cannot match before `(`. Both matchers see only the class.
        {"MessageBox", "MessageBox::operator+="},
        "MessageBox::operator+=(other);",
        id="punctuation-terminated-name-cannot-match-before-a-bracket",
    ),
    pytest.param(
        {"MessageBox", "MessageBox::operator+="},
        "MessageBox::operator+=x;",
        id="punctuation-terminated-name-matches-before-a-word-character",
    ),
    pytest.param(
        # Accepting the first name must leave the rest of the chain findable.
        {"Registry<T>::Lookup", "Lookup::Inner", "Inner::Leaf"},
        "return Registry<T>::Lookup::Inner::Leaf;",
        id="search-resumes-inside-a-chain-after-a-match",
    ),
    pytest.param(
        {"Foo<T>::bar", "bar::baz", "baz"},
        "Foo<T>::bar::baz();",
        id="search-resumes-after-a-template-qualified-name",
    ),
    pytest.param(
        # No boundary before `handler` here, in either case.
        {"handler"},
        "cafehandler; 123handler; caféhandler;",
        id="a-name-inside-a-larger-word-is-not-a-reference",
    ),
    pytest.param(
        {"Widget"},
        "x = 1Widget;",
        id="a-name-after-a-digit-is-not-a-reference",
    ),
    pytest.param(
        # The boundary is between `~` and the class, so the bare name matches
        # even when what precedes `::~` is unknown.
        {"Widget", "Alias"},
        "p->Alias::~Widget(); MyWidget::~Widget();",
        id="explicit-destructor-call-yields-the-class",
    ),
    pytest.param(
        {"Buffer", "Buffer::Inner", "Buffer::Inner::Leaf"},
        "Buffer::Inner::Leaf a; Buffer::Inner b; Buffer c;",
        id="nested-chains-of-decreasing-length",
    ),
    pytest.param(
        {"handle", "handler", "handle_event"},
        "handle_event(); handler(); handle();",
        id="names-that-are-prefixes-of-each-other",
    ),
    pytest.param(
        # Reached from the run after the `~`, and still bound by the boundary
        # before the `~` itself.
        {"~Widget", "~Widget<T>", "Widget", "~WidgetPool"},
        "x~Widget; x~Widget<T>; x~WidgetPool; ~Widget(); p->~Widget(); x~Widgets;",
        id="destructor-names-are-reached-from-their-class-run",
    ),
    pytest.param(
        # Prefixes of different lengths before the same run.
        {"1. Setup", ". Setup", "Setup", "~Setup", "~~Setup"},
        "a1. Setup; x. Setup; y~Setup; z~~Setup; Setup;",
        id="prefixed-names-before-the-same-run",
    ),
    pytest.param(
        # A prefix ending in a digit: no run starts after it, so the name
        # is searched for directly, beside one reached from its run.
        {"~1x", "~x", "~"},
        "a~1x; b~x; c~;",
        id="prefix-ending-in-a-digit",
    ),
]


@pytest.mark.parametrize("names,content", REFERENCE_CASES)
def test_matcher_agrees_with_reference_alternation(names, content):
    assert build_name_matcher(names)(content) == _reference_matcher(names)(content)


def test_longest_wins_among_names_that_share_a_start_position():
    """Names unreachable from an identifier run still compete by length.

    Several of them can begin at the same index, and the walk takes the first
    that matches — so they have to be ordered longest first. Emitted one at a
    time they inherit whatever order the name set iterated in, which for strings
    varies between processes, so the wrong one wins at random.

    Every candidate has to END on a boundary too, or the shorter ones are
    rejected there and the length ordering is never exercised — which is why
    the names are `::` chains: each one stops in front of a `:` or the `;`.
    """
    for depth in range(2, 12):
        names = {"::seg" * n for n in range(1, depth + 1)}
        content = "y" + "::seg" * depth + ";"
        assert build_name_matcher(names)(content) == _reference_matcher(names)(content)


def test_matcher_agrees_with_reference_on_random_input():
    """Fuzz the two against each other.

    The cases above are the shapes that were known to be hard. This looks for
    the ones that are not: names and bodies are assembled from the same pool of
    fragments, so collisions, prefixes and partial chains occur often.
    """
    rng = random.Random(20240607)
    # Non-ASCII and punctuation-leading fragments matter: a name whose first
    # character is not an ASCII letter cannot be reached from an identifier run,
    # and several such names can start at the same position — which is where
    # ordering is easiest to lose.
    fragments = [
        "Widget", "Registry", "Lookup", "Inner", "handler", "value", "Wid", "handle",
        "émetteur", "Ünicode", "envoyer",
    ]
    punctuation = ["::", "::~", "<T>::", "(", ")", ";", " ", ".", "->", "~", "1", "é", "_", "&"]

    pieces = fragments + ["::", "<T>::", "::~", "&", "."]
    for _ in range(2000):
        names = {
            "".join(rng.choice(pieces) for _ in range(rng.randint(1, 3)))
            for _ in range(rng.randint(1, 6))
        }
        names = {n for n in names if len(n) >= 4}
        if not names:
            continue
        content = "".join(
            rng.choice(fragments + punctuation) for _ in range(rng.randint(4, 40))
        )
        assert build_name_matcher(names)(content) == _reference_matcher(names)(content), (
            f"names={names!r} content={content!r}"
        )
